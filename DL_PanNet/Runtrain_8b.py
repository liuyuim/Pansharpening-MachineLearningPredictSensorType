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



# get high-frequency
def get_edge(data):  
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape)==3:
            rs[i,:,:] = data[i,:,:] - cv2.boxFilter(data[i,:,:],-1,(5,5))
        else:
            rs[i,:,:,:] = data[i,:,:,:] - cv2.boxFilter(data[i,:,:,:],-1,(5,5))
    return rs



 # get training patches
def get_batch(train_data,bs): 
    
    gt = train_data['gt'][...]    ## ground truth N*H*W*C
    pan = train_data['pan'][...]  #### Pan image N*H*W
    ms_lr = train_data['ms'][...] ### low resolution MS image
    lms   = train_data['lms'][...]   #### MS image interpolation to Pan scale
    
    gt = np.array(gt,dtype = np.float32) / 2047.  ### normalization, WorldView L = 11
    pan = np.array(pan, dtype = np.float32) /2047.
    ms_lr = np.array(ms_lr, dtype = np.float32) / 2047.
    lms  = np.array(lms, dtype = np.float32) /2047.
    

    
    N = gt.shape[0]
    batch_index = np.random.randint(0,N,size = bs)
    
    gt_batch = gt[batch_index,:,:,:]
    pan_batch = pan[batch_index,:,:]
    ms_lr_batch = ms_lr[batch_index,:,:,:]
    lms_batch  = lms[batch_index,:,:,:]
    
    pan_hp_batch = get_edge(pan_batch)
    pan_hp_batch = pan_hp_batch[:,:,:,np.newaxis] # expand to N*H*W*1
    
    ms_hp_batch = get_edge(ms_lr_batch)
    
    
    return gt_batch, lms_batch, pan_hp_batch, ms_hp_batch


def vis_ms(data):
    _,b,g,_,r,_,_,_ = tf.split(data,8,axis = 3)
    vis = tf.concat([r,g,b],axis = 3)
    return vis



# PanNet structures
def PanNet(ms, pan, num_spectral = 8, num_res = 4, num_fm = 32, reuse=False):
    
    weight_decay = 1e-5
    with tf.variable_scope('net'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        
        ms = ly.conv2d_transpose(ms,num_spectral,8,4,activation_fn = None, weights_initializer = ly.variance_scaling_initializer(), 
                                 weights_regularizer = ly.l2_regularizer(weight_decay))
        ms = tf.concat([ms,pan],axis=3)

        rs = ly.conv2d(ms, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.relu)
        
        for i in range(num_res):
            rs1 = ly.conv2d(rs, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.relu)
            rs1 = ly.conv2d(rs1, num_outputs = num_fm, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = None)
            rs = tf.add(rs,rs1)
        
        rs = ly.conv2d(rs, num_outputs = num_spectral, kernel_size = 3, stride = 1, 
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = None)
        return rs


# if __name__ =='__main__':
def run_main(model_directory,train_data_name,test_data_name):

    tf.compat.v1.reset_default_graph() 

    train_batch_size = 32 # training batch size
    test_batch_size = 32  # validation batch size
    image_size = 64      # patch size
    iterations = 255000 # total number of iterations to use.
    # model_directory = './models' # directory to save trained model to.
    # train_data_name = './training_data/train.mat'  # training data
    # test_data_name  = './training_data/validation.mat'   # validation data
    restore = False  # load model or not
    method = 'Adam'  # training method: Adam or SGD
    
############## loading data
    # train_data = sio.loadmat(train_data_name)
    # test_data = sio.loadmat(test_data_name)
    train_data = mat73.loadmat(train_data_name)
    test_data  = mat73.loadmat(test_data_name)
    
    
############## placeholder for training
    gt = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,8])
    
    lms = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,8])
    ms_hp = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size//4,image_size//4,8])
    pan_hp = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,1])
    

############# placeholder for testing
    test_gt = tf.placeholder(dtype = tf.float32,shape = [test_batch_size,image_size,image_size,8])

    test_lms = tf.placeholder(dtype = tf.float32,shape = [test_batch_size,image_size,image_size,8])
    test_ms_hp = tf.placeholder(dtype = tf.float32,shape = [test_batch_size,image_size//4,image_size//4,8])
    test_pan_hp = tf.placeholder(dtype = tf.float32,shape = [test_batch_size,image_size,image_size,1])



######## network architecture
    mrs = PanNet(ms_hp,pan_hp)
    mrs = tf.add(mrs,lms)
    
    test_rs = PanNet(test_ms_hp,test_pan_hp,reuse = True)
    test_rs = test_rs + test_lms


######## loss function
    mse = tf.reduce_mean(tf.square(mrs - gt))
    test_mse = tf.reduce_mean(tf.square(test_rs - test_gt))

##### Loss summary
    mse_loss_sum = tf.summary.scalar("mse_loss",mse)

    test_mse_sum = tf.summary.scalar("test_loss",test_mse)

    lms_sum = tf.summary.image("lms",tf.clip_by_value(vis_ms(lms),0,1))
    mrs_sum = tf.summary.image("rs",tf.clip_by_value(vis_ms(mrs),0,1))

    label_sum = tf.summary.image("label",tf.clip_by_value(vis_ms(gt),0,1))
    
    all_sum = tf.summary.merge([mse_loss_sum,mrs_sum,label_sum,lms_sum])
    
#########   optimal    Adam or SGD
         
    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'net')    

    
    if method == 'Adam':
        g_optim = tf.train.AdamOptimizer(0.001, beta1 = 0.9) \
                          .minimize(mse, var_list=t_vars)

    else:
        global_steps = tf.Variable(0,trainable = False)
        lr = tf.train.exponential_decay(0.1,global_steps,decay_steps = 50000, decay_rate = 0.1)
        clip_value = 0.1/lr
        optim = tf.train.MomentumOptimizer(lr,0.9)
        gradient, var   = zip(*optim.compute_gradients(mse,var_list = t_vars))
        gradient, _ = tf.clip_by_global_norm(gradient,clip_value)
        g_optim = optim.apply_gradients(zip(gradient,var),global_step = global_steps)
        
##### GPU setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

####

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:  
        sess.run(init)
 
        if restore:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess,ckpt.model_checkpoint_path)
            

        
        for i in range(iterations):

            train_gt, train_lms, train_pan_hp, train_ms_hp = get_batch(train_data, bs = train_batch_size)
            _,mse_loss,merged = sess.run([g_optim,mse,all_sum],feed_dict = {gt: train_gt, lms: train_lms,
                                         ms_hp: train_ms_hp, pan_hp: train_pan_hp})


            if i % 100 == 0:

                print ("Iter: " + str(i) + " MSE: " + str(mse_loss))


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