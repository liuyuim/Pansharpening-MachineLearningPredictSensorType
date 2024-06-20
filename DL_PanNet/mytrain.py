#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a re-implementation of training code of this paper:
# J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding, J. Paisley. "PanNet: A deep network architecture for pan-sharpening", ICCV,2017. 
# author: Junfeng Yang
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.layers as ly
import os
import scipy.io as sio
import mat73


def get_edge(data):
    """get high-frequency."""
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape)==3:
            rs[i,:,:] = data[i,:,:] - cv2.boxFilter(data[i,:,:],-1,(5,5))#方框滤波
        else:
            rs[i,:,:,:] = data[i,:,:,:] - cv2.boxFilter(data[i,:,:,:],-1,(5,5))
    return rs


def get_batch(train_data, bs):
    """get training patches."""
    
    gt    = train_data['gt'][...]   # ground truth N*H*W*C
    pan   = train_data['pan'][...]  # Pan image N*H*W
    ms_lr = train_data['ms'][...]   # low resolution MS image
    lms   = train_data['lms'][...]  # MS image interpolation to Pan scale

    # gt shape: (100, 64, 64, 8)
    # pan shape: (100, 64, 64)
    # ms_lr shape: (100, 16, 16, 8)
    # lms shape: (100, 64, 64, 8)


    #print('gt shape:', gt.shape)
    #print('pan shape:', pan.shape)
    #print('ms_lr shape:', ms_lr.shape)
    #print('lms shape:', lms.shape)
    
    gt    = np.array(gt, dtype=np.float32) / 2047.  ### normalization, WorldView L = 11
    pan   = np.array(pan, dtype=np.float32) /2047.
    ms_lr = np.array(ms_lr, dtype=np.float32) / 2047.
    lms   = np.array(lms, dtype=np.float32) /2047.
    
    N = gt.shape[0] #shape[0]是行数，shape[1]就是列数
    batch_index = np.random.randint(0,N,size=bs)
    
    gt_batch    = gt[batch_index,:,:,:]
    pan_batch   = pan[batch_index,:,:]
    ms_lr_batch = ms_lr[batch_index,:,:,:]
    lms_batch   = lms[batch_index,:,:,:]
    
    pan_hp_batch = get_edge(pan_batch)
    pan_hp_batch = pan_hp_batch[:,:,:,np.newaxis] # expand to N*H*W*1 #在这一位置增加一个一维
    
    ms_hp_batch = get_edge(ms_lr_batch)

    # gt_batch shape: (batch_size, 64, 64, 8)
    # pan_hp_batch shape: (batch_size, 64, 64, 1)
    # ms_hp_batch shape: (batch_size, 16, 16, 8)
    # lms_batch shape: (batch_size, 64, 64, 8)
    
    return gt_batch, lms_batch, pan_hp_batch, ms_hp_batch


def vis_ms(data, num_spectral=8):
    if num_spectral == 8:
        _,b,g,_,r,_,_,_ = tf.split(data,num_spectral,axis = 3)
        print(b.shape)
        print(g.shape)
        print(r.shape)

    if num_spectral == 4:
        _,b,g,r = tf.split(data,num_spectral,axis = 3)
    
    vis = tf.concat([r,g,b],axis=3)#用来拼接张量的函数
    return vis


# PanNet model
def PanNet(ms, pan, num_spectral=8, num_res=4, num_fm=32, reuse=False):
    
    weight_decay = 1e-5

    with tf.compat.v1.variable_scope('net'):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
            
        
        ms = ly.conv2d_transpose(ms,num_spectral,8,4,activation_fn = None, weights_initializer = ly.variance_scaling_initializer(), 
                                 weights_regularizer = ly.l2_regularizer(weight_decay))
        ms = tf.concat([ms,pan],axis=3)

        rs = ly.conv2d(ms, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.relu)
        
        for i in range(num_res):
            rs1 = ly.conv2d(rs, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), #正则化，初始化
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.relu)
            rs1 = ly.conv2d(rs1, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = None)
            rs = tf.add(rs,rs1)
        
        rs = ly.conv2d(rs, num_outputs = num_spectral, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = None)
        return rs


if __name__ =='__main__':

    tf.compat.v1.reset_default_graph()

    channel_size     = 4    # channel size

    train_batch_size = 8    # training batch size
    test_batch_size  = 8    # validation batch size
    image_size       = 64   # patch size
    iterations       = 255000 #30   # total number of iterations to use.
    
    restore          = False   # load model or not
    method           = 'Adam'  # training method: Adam or SGD

    model_directory  = './models'                            # directory to save trained model to.
    train_data_name  = './training_data/train.mat'           # training data
    test_data_name   = './training_data/validation.mat'      # validation data
    
    #train_data_name  = './training_data/wv3/train.mat'           # training data
    #test_data_name   = './training_data/wv3/validation.mat'      # validation data
############## loading data

    #train_data = sio.loadmat(train_data_name)
    #test_data  = sio.loadmat(test_data_name)

    train_data = mat73.loadmat(train_data_name)
    test_data  = mat73.loadmat(test_data_name)
    
    
############## placeholder for training
    
    gt     = tf.compat.v1.placeholder(dtype=tf.float32, shape=[train_batch_size,image_size,image_size,channel_size])
    
    lms    = tf.compat.v1.placeholder(dtype=tf.float32, shape=[train_batch_size,image_size,image_size,channel_size])

    ms_hp  = tf.compat.v1.placeholder(dtype=tf.float32, shape=[train_batch_size,image_size//4,image_size//4,channel_size])

    pan_hp = tf.compat.v1.placeholder(dtype=tf.float32, shape=[train_batch_size,image_size,image_size,1])
    

############# placeholder for testing

    test_gt     = tf.compat.v1.placeholder(dtype=tf.float32,shape=[test_batch_size,image_size,image_size,channel_size])

    test_lms    = tf.compat.v1.placeholder(dtype=tf.float32,shape=[test_batch_size,image_size,image_size,channel_size])

    test_ms_hp  = tf.compat.v1.placeholder(dtype=tf.float32,shape=[test_batch_size,image_size//4,image_size//4,channel_size])

    test_pan_hp = tf.compat.v1.placeholder(dtype=tf.float32,shape=[test_batch_size,image_size,image_size,1])



######## network architecture
    mrs = PanNet(ms_hp, pan_hp, num_spectral=4)
    mrs = tf.add(mrs, lms)
    
    test_rs = PanNet(test_ms_hp, test_pan_hp, num_spectral=4, reuse=True)
    test_rs = test_rs + test_lms


######## loss function
    #函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
    mse = tf.reduce_mean(tf.square(mrs - gt))
    test_mse = tf.reduce_mean(tf.square(test_rs - test_gt))

##### Loss summary

    mse_loss_sum = tf.compat.v1.summary.scalar("mse_loss",mse)

    test_mse_sum = tf.compat.v1.summary.scalar("test_loss",test_mse)

    lms_sum = tf.compat.v1.summary.image("lms",tf.clip_by_value(vis_ms(lms,num_spectral=4),0,1))

    mrs_sum = tf.compat.v1.summary.image("rs",tf.clip_by_value(vis_ms(mrs,num_spectral=4),0,1))

    label_sum = tf.compat.v1.summary.image("label",tf.clip_by_value(vis_ms(gt,num_spectral=4),0,1))
    
    all_sum = tf.compat.v1.summary.merge([mse_loss_sum,mrs_sum,label_sum,lms_sum])
    
#########   optimal    Adam or SGD
         
    t_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='net')  

    
    if method == 'Adam':
        g_optim = tf.compat.v1.train.AdamOptimizer(0.001, beta1 = 0.9).minimize(mse, var_list=t_vars)
    else:
        global_steps = tf.Variable(0,trainable = False)
        lr = tf.train.exponential_decay(0.1,global_steps,decay_steps = 50000, decay_rate = 0.1)
        clip_value = 0.1/lr
        optim = tf.train.MomentumOptimizer(lr,0.9)
        gradient, var   = zip(*optim.compute_gradients(mse,var_list = t_vars))
        gradient, _ = tf.clip_by_global_norm(gradient,clip_value)
        g_optim = optim.apply_gradients(zip(gradient,var),global_step = global_steps)
        
##### GPU setting

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session(config=config)

####

    init = tf.compat.v1.global_variables_initializer()

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
 
        if restore:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess,ckpt.model_checkpoint_path)
            

        for i in range(iterations):

            train_gt, train_lms, train_pan_hp, train_ms_hp = get_batch(train_data, bs=train_batch_size)

            _,mse_loss,merged = sess.run(
                [g_optim,mse,all_sum],
                feed_dict={
                    gt: train_gt,
                    lms: train_lms,
                    ms_hp: train_ms_hp,
                    pan_hp: train_pan_hp
                    }
            )


            if i % 10 == 0:

                print("Iter: " + str(i) + " MSE: " + str(mse_loss))


            if i%1000 == 0 and i!=0:
                test_gt_batch, test_lms_batch, test_pan_hp_batch, test_ms_hp_batch = get_batch(test_data, bs = test_batch_size)
                
                test_mse_loss,merged = sess.run([test_mse,test_mse_sum],   
                                               feed_dict = {test_gt : test_gt_batch, test_lms : test_lms_batch,
                                                            test_ms_hp : test_ms_hp_batch, test_pan_hp : test_pan_hp_batch})

                print ("Iter: " + str(i) + " validation_MSE: " + str(test_mse_loss))      
                
            if i % 10000 == 0 and i != 0:             
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess,model_directory+'/model-'+str(i)+'.ckpt')
                print ("Save Model")