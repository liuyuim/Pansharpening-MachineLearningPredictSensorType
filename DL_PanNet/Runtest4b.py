#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import os, sys, fnmatch

import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import scipy.io as sio
import cv2

import mat73

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1' # 默认，显示所有信息 
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error 

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #上面代码仅仅能去除普通的pycharm的警告信息，对于tf产生的警告信息根本没用，如果使用的是tf1.0不用 加compat.v1 该代码是直接将提示信息设置为只提醒error类型的信息，不提醒warning类型





"""
# 原test.py代码       

"""


"""
# This is a re-implementation of training code of this paper:
# J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding, J. Paisley. "PanNet: A deep network architecture for pan-sharpening", ICCV,2017. 
# author: Junfeng Yang
"""
# import tensorflow as tf
# import tensorflow.contrib.layers as ly
# import numpy as np
# import scipy.io as sio
# import cv2
# import os
# import mat73
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# The GPU id to use, either "0" or "1" or "2" or "3"
#os.environ[“CUDA_VISIBLE_DEVICES”] = “1,0” 
#设置当前使用的GPU设备为1,0号两个设备,名称依次为'/gpu:0'、'/gpu:1'。表示优先使用1号设备,然后使用0号设备

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

# if __name__=='__main__':
def run_main(test_data,model_directory,testOutput_data):

    # test_data = './training_data/GF1_test.mat'

    # model_directory = './models_GF1/'

    tf.compat.v1.reset_default_graph()
    
    data = sio.loadmat(test_data)
    # data = mat73.loadmat(test_data)
    
    ms = data['ms'][...]      # MS image
    # ms = np.array(ms,dtype = np.float32) /2047.
    ms = np.array(ms,dtype = np.float32) 
    # print('ms shape:', ms.shape)

    lms = data['lms'][...]    # up-sampled LRMS image
    # lms = np.array(lms, dtype = np.float32) /2047.
    lms = np.array(lms, dtype = np.float32) 
    # print('lms shape:', lms.shape)

    pan  = data['pan'][...]  # PAN image
    # pan  = np.array(pan,dtype = np.float32) /2047.
    pan  = np.array(pan,dtype = np.float32) 
    # print('pan shape:', pan.shape)
    
    ms_hp = get_edge(ms)   # high-frequency parts of MS image
    ms_hp = ms_hp[np.newaxis,:,:,:]
    # print('ms_hp shape:', ms_hp.shape)

    pan_hp = get_edge(pan) # high-frequency parts of PAN image
    pan_hp = pan_hp[np.newaxis,:,:,np.newaxis]
    # print('pan_hp shape:', pan_hp.shape)

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
    # print('p_hp shape:', p_hp.shape)
    # print('m_hp shape:', m_hp.shape)
    # print('lms_p shape:', lms_p.shape)


    rs = PanNet(m_hp,p_hp,num_spectral=4) # output high-frequency parts
    
    mrs = tf.add(rs,lms_p) 
    
    # output = tf.clip_by_value(mrs,0,1) # final output
    output = tf.clip_by_value(mrs,0,2047) # final output

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

        sio.savemat(testOutput_data, {'output':final_output[0,:,:,:]})
       # mat73.savemat('./result/output.mat', {'output':final_output[0,:,:,:]})


"""
# 改造的代码，适用于批量跑数据
# 

"""
#批量运行测试代码 
if __name__=='__main__':# 正常融合  二级目录结构  
    
    print("run DL_PanNet Fusion... \n")
    
    Datapath = 'G:\AFusionGroup\Benchmark-FuLianJin\Test_DR'
    saveDir = 'G:\AFusionGroup\Benchmark-FuLianJin\Test_DRFusion'
    # sensor_list = ['GF1_*','GF2_*','QB_*','WV2_*','WV3_*'] # sensor_list 和 model_directory_list 两个列表内的元素需要一一对应
    sensor_list = ['GF2_*'] # sensor_list 和 model_directory_list 两个列表内的元素需要一一对应
    # model_directory_list = ['./models/models_GF1/', './models/models_GF2/', './models/models_QB/', './models/models_WV2/', './models/models_WV3/',]
    model_directory_list = ['./models/models_GF2/']
    #预测实验统计评估值的分布特征
    # model_directory_list = ['./models/models_GF1/','./models/models_GF1/','./models/models_GF1/','./models/models_GF1/','./models/models_GF1/',]
    
    
    # 判断 传感器类型数量 和 对应的model数量 是否匹配
    if(len(sensor_list) != len(model_directory_list)):
        print("检测到 传感器类型数量 和 对应的model数量 不匹配 程序退出。\n请检查 sensor_list 和 model_directory_list两个列表 ")
        sys.exit(0) # sys.exit(0) 该方法中包含一个参数status，默认为0，表示正常退出，也可以为1，表示异常退出。

    for i in range(len(sensor_list)): # for i in range(5): range里只有一个数值，表示从零开始到这个数值-1的数字。0 1 2 3 4 
        sensor = sensor_list[i]
        model_directory = model_directory_list[i]
		# ErjiDir_list = fnmatch.filter(os.listdir(Datapath), 'GF1_*ErjiDir')
        ErjiDir_list = fnmatch.filter(os.listdir(Datapath), sensor) # 根据一级目录遍历并筛选出二级目录列表

        for ErjiDir in ErjiDir_list:
            
            TestData_path = os.path.join(Datapath,ErjiDir) #拼接出二级目录的路径 ..\DataDL_1_TestData\GF1_GengDi
            print("--------------\n ")
            print("【正在处理的二级目录】",TestData_path,"【sensor】",sensor,"【model_directory】",model_directory)
            MatName_list = fnmatch.filter(os.listdir(TestData_path), '*.mat') #对二级目录遍历并筛选出.mat文件名存到列表
            # path_list.sort(key=lambda x:int(x[:-4])) #新加入的一行做的事情是--对每个文件名将句号前的字符串转化为数字，然后以数字为key来进行排序。
            # path_list.sort(key=lambda x:int(x.split('.')[0])) #只需考虑句号前面的数字顺序了
            
            saveErjiDir_path = os.path.join(saveDir,ErjiDir) #拼接出 保存二级目录 的路径 ..\DataDL_PannetOutput\GF1_GengDi
            if not os.path.exists(saveErjiDir_path): #创建 保存二级目录
                os.makedirs(saveErjiDir_path)

            for MatName in MatName_list:
                # f = open(os.path.join(path,filename),'rb')
                test_data = os.path.join(TestData_path,MatName) #用 二级目录的路径 和 mat文件名 拼接出 测试mat文件的路径 ..\DataDL_1_TestData\GF1_GengDi\j1p2.mat
                # print('测试mat文件的路径:',test_data)
                testOutput_data = os.path.join(saveDir,ErjiDir,MatName) #用自定义的保存文件夹和mat文件名拼接出待保存mat文件的路径  ..\DataDL_PannetOutput\GF1_GengDi\j1p2.mat 
                # print('待保存mat文件的路径:',testOutput_data)
                run_main(test_data,model_directory,testOutput_data)
                print("保存...  ",testOutput_data)

    print("===========\n")
    print("run DL_PanNet Fusion 已全部处理完毕，请到此文件夹查看：",saveDir)






















