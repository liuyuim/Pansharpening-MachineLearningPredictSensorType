#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of training code of this paper:
# J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding, J. Paisley. "PanNet: A deep network architecture for pan-sharpening", ICCV,2017. 
# author: Junfeng Yang

import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import scipy.io as sio
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def PanNet(ms, pan, num_spectral = 8, num_res = 4, num_fm = 32, reuse=False):
    
    weight_decay = 1e-5
    with tf.compat.v1.variable_scope('net'):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
            
        
        ms = ly.conv2d_transpose(ms,num_spectral,8,4,activation_fn = None, weights_initializer = ly.variance_scaling_initializer(), 
                                 biases_initializer = None,
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

def get_edge(data): # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) ==3:
        for i in range(data.shape[2]):
            rs[:,:,i] = data[:,:,i] -cv2.boxFilter(data[:,:,i],-1,(5,5))
    else:
        rs = data - cv2.boxFilter(data,-1,(5,5))
    return rs

if __name__=='__main__':

    # test_data = 'G:\AFusionGroup\Shiyan\shiyan20230421\GF1_Data\Test_FuDel/51.mat'
    test_data = 'F:/AFusionGroup/Shiyan/shiyan20231202/GF2_Data/Test_Fu1024/1.mat'

    # model_directory = 'G:\AFusionGroup\Shiyan\shiyan20230421\WV2_PanNet/SaveModel/'
    model_directory = 'F:/AFusionGroup/Shiyan/shiyan20231202/GF2_PanNet/SaveModel/'

    tf.compat.v1.reset_default_graph()
    
    data = sio.loadmat(test_data)
    # data = mat73.loadmat(test_data)
    
    ms = data['ms'][...]      # MS image
    ms = np.array(ms,dtype = np.float32) /2047.
    print('ms shape:', ms.shape)

    lms = data['lms'][...]    # up-sampled LRMS image
    lms = np.array(lms, dtype = np.float32) /2047.
    print('lms shape:', lms.shape)

    pan  = data['pan'][...]  # PAN image
    pan  = np.array(pan,dtype = np.float32) /2047.
    print('pan shape:', pan.shape)
    
    ms_hp = get_edge(ms)   # high-frequency parts of MS image
    ms_hp = ms_hp[np.newaxis,:,:,:]
    print('ms_hp shape:', ms_hp.shape)

    pan_hp = get_edge(pan) # high-frequency parts of PAN image
    pan_hp = pan_hp[np.newaxis,:,:,np.newaxis]
    print('pan_hp shape:', pan_hp.shape)

    h = pan.shape[0] # height
    w = pan.shape[1] # width


    lms  = lms[np.newaxis,:,:,:]
    
    # placeholder for tensor
    p_hp = tf.compat.v1.placeholder(shape=[1,h,w,1],dtype=tf.float32)
    # m_hp = tf.compat.v1.placeholder(shape=[1,h/4,w/4,8],dtype=tf.float32)
    m_hp = tf.compat.v1.placeholder(shape=[1,h/4,w/4,4],dtype=tf.float32)
    # lms_p = tf.compat.v1.placeholder(shape=[1,h,w,8],dtype=tf.float32)
    lms_p = tf.compat.v1.placeholder(shape=[1,h,w,4],dtype=tf.float32)

    # 输出张量用于调试
    print('p_hp shape:', p_hp.shape)
    print('m_hp shape:', m_hp.shape)
    print('lms_p shape:', lms_p.shape)


    rs = PanNet(m_hp,p_hp,num_spectral=4) # output high-frequency parts
    
    mrs = tf.add(rs,lms_p) 
    
    output = tf.clip_by_value(mrs,0,1) # final output

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    
    with tf.compat.v1.Session() as sess:  
        sess.run(init)
        
        # loading  model       
        if tf.train.get_checkpoint_state(model_directory):  
           ckpt = tf.train.latest_checkpoint(model_directory)
           saver.restore(sess, ckpt)
           print ("load new model")

        else:
           ckpt = tf.train.get_checkpoint_state(model_directory + "pre-trained/")
           saver.restore(sess,ckpt.model_checkpoint_path) # pre-trained model                                                                   
           print ("load pre-trained model")                            
        


        final_output = sess.run(output,feed_dict = {p_hp:pan_hp, m_hp:ms_hp, lms_p:lms})

        sio.savemat('./result/output.mat', {'output':final_output[0,:,:,:]})
       # mat73.savemat('./result/output.mat', {'output':final_output[0,:,:,:]})