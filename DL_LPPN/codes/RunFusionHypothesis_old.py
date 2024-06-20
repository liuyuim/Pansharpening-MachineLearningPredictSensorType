'''
用脚本批量跑
conda activate LYpy3.7tf1.14np1.16
cd/d E:\LiuYu\FusionEvaluateExperiment\MethodDL_LPPN\codes
python RunFusionHypothesis.py
'''

# ---------------------------------------------------------------
# Copyright (c) 2021, Cheng Jin, Liang-Jian Deng, Ting-Zhu Huang,
# Gemine Vivone, All rights reserved.
#
# This work is licensed under GNU Affero General Public License 
# v3.0 International To view a copy of this license, see the 
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird and GaoFen-2), please change the corresponding
# inputs.
# ---------------------------------------------------------------

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf
import GF_model
import time
import h5py

import os, sys, fnmatch
import RunTestGF1, RunTestGF2, RunTestQB, RunTestWV2b4,RunTestWV2b8, RunTestWV3b4, RunTestWV3b8, RunTestWV4b4 
# tf.reset_default_graph()

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# # 隐去warning
# # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1' # 默认，显示所有信息 
# # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error 
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error 
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #上面代码仅仅能去除普通的pycharm的警告信息，对于tf产生的警告信息根本没用，如果使用的是tf1.0不用 加compat.v1 该代码是直接将提示信息设置为只提醒error类型的信息，不提醒warning类型

# """
# # 原test.py代码       
# """

# # if __name__ == '__main__':
# def run_main(test_data,model_directory,testOutput_data):

#     # test_data = './test_data/GaoFen-2.mat'
#     # model_directory = './pretrained/models_GF1'

#     tf.reset_default_graph()

#     data = sio.loadmat(test_data)
#     #data = h5py.File(test_data)
    
#     # data normalization

#     ms = data['ms'][...]  # MS image
#     ms = np.array(ms, dtype=np.float32) / 1023. # 2047 for WorldView-3, QuickBird datasets, 1023 for GaoFen-2 datasets
#     ms = ms[np.newaxis, :, :, :]

#     lms = data['lms'][...]  # up-sampled LRMS image
#     lms = np.array(lms, dtype=np.float32) / 1023. # 2047 for WorldView-3, QuickBird datasets, 1023 for GaoFen-2 datasets
#     lms = lms[np.newaxis, :, :, :]

#     pan = data['pan'][...]  # pan image
#     pan = np.array(pan, dtype=np.float32) / 1023. # 2047 for WorldView-3, QuickBird datasets, 1023 for GaoFen-2 datasets
#     pan = pan[np.newaxis, :, :, np.newaxis]

#     h = pan.shape[1]  # height
#     w = pan.shape[2]  # width

#     # placeholder for tensor
#     pan_p = tf.placeholder(shape=[1, h, w, 1], dtype=tf.float32)
#     # ms_p = tf.placeholder(shape=[1, h / 4, w / 4, 8], dtype=tf.float32)    
#     ms_p = tf.placeholder(shape=[1, h / 4, w / 4, 4], dtype=tf.float32)    
#     # lms_p = tf.placeholder(shape=[1, h, w, 8], dtype=tf.float32)
#     lms_p = tf.placeholder(shape=[1, h, w, 4], dtype=tf.float32)

#     # 输出张量用于调试
#     print('p_hp shape:', pan_p.shape)
#     print('m_hp shape:', ms_p.shape)
#     print('lms_p shape:', lms_p.shape)

#     output_pyramid = GF_model.LPPN(pan_p, ms_p)
    
#     output  = tf.clip_by_value(output_pyramid[4], 0, 1)  # final output
#     # 输出张量用于调试
#     print('p_hp shape:', pan_p.shape)
#     print('m_hp shape:', ms_p.shape)
#     print('lms_p shape:', lms_p.shape)

#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         sess.run(init)
#         # loading  model
#         if tf.train.get_checkpoint_state(model_directory):
#             ckpt = tf.train.latest_checkpoint(model_directory)
#             saver.restore(sess, ckpt)
#             print("load new model")

#         else:
#             ckpt = tf.train.get_checkpoint_state(model_directory + "pre-trained/")
#             saver.restore(sess,
#                           ckpt.model_checkpoint_path)
#             print("load pre-trained model")
#         start_time = time.time()
#         final_output = sess.run(output, feed_dict={pan_p: pan, lms_p: lms, ms_p: ms})
#         end_time = time.time()
#         print('running time: ', end_time-start_time)
#         # sio.savemat('./result/output_GF.mat', {'output_LPPN': final_output[0, :, :, :]})
#         sio.savemat(testOutput_data, {'output_LPPN': final_output[0, :, :, :]})


"""
# 改造的代码，适用于批量跑数据
# from mytest_Run import run_main
"""

#批量运行测试代码
def run_TestHypothesis(Datapath,saveDir,model_directory,HypothesisDir_list):
    print("run DL_LPPN Fusion Forecast ... \n")
    
    # 判断 传感器种类数 和 对应的model数 是否匹配
    # if(len(sensor_list) != len(model_directory_list)):
    #     print("检测到 传感器种类数 和 对应的model数 不匹配 程序退出。\n检查run_TestHypothesis()函数中 sensor_list 和 model_directory_list两个列表 ")
    #     sys.exit(0) # sys.exit(0) 该方法中包含一个参数status，默认为0，表示正常退出，也可以为1，表示异常退出。

    # for i in range(5): range里只有一个数值，表示从零开始到这个数值-1的数字。0 1 2 3 4 
    # for i in range(len(sensor_list)):
    #     sensor = sensor_list[i]
    
    # for j in range(len(HypothesisDir_list)): 
    #     model_directory = model_directory_list[j]
    #     HypothesisDir = HypothesisDir_list[j]
    
    # ##################################### 用第 1 个模型测试 ####################################
    HypothesisDir = HypothesisDir_list[0]
    # ErjiDir_list = fnmatch.filter(os.listdir(Datapath), sensor) # 根据一级目录遍历出筛选二级目录列表
    # for ErjiDirName in ErjiDir_list:
        
    # TestData_path = os.path.join(Datapath,ErjiDirName) #拼接出二级目录的路径 ..\DataDL_1_TestData\Xsensor_
    TestData_path = Datapath #拼接出二级目录的路径 ..\DataDL_1_TestData\Xsensor_
    print("--------------\n ")
    print("【正在处理的目录】",TestData_path,"【sensor】ToBeTested","【model_directory】")
    MatName_list = fnmatch.filter(os.listdir(TestData_path), '*.mat') #对二级目录遍历并筛选出.mat文件名存到列表
    # path_list.sort(key=lambda x:int(x[:-4])) #新加入的一行做的事情是--对每个文件名将句号前的字符串转化为数字，然后以数字为key来进行排序。
    # path_list.sort(key=lambda x:int(x.split('.')[0])) #只需考虑句号前面的数字顺序了
        
    MakeDirs_path = os.path.join(saveDir,HypothesisDir) #拼接出 需要创建的目录 的路径 ..\DataDL_PannetOutput\GF1
    if not os.path.exists(MakeDirs_path): #创建 需要创建的目录
        os.makedirs(MakeDirs_path)

    for MatName in MatName_list:
        # f = open(os.path.join(path,filename),'rb')
        test_data = os.path.join(TestData_path,MatName) #用 二级目录的路径 和 mat文件名 拼接出 测试mat文件的路径 ..\DataDL_1_TestData\Xsensor_\j1p2.mat
        # print('测试mat文件的路径:',test_data)
        testOutput_data = os.path.join(saveDir,HypothesisDir,MatName) #用自定义的保存文件夹和mat文件名拼接出待保存mat文件的路径  ..\DataDL_PannetOutput\Xsensor_\j1p2.mat 
        # print('待保存mat文件的路径:',testOutput_data)
        RunTestGF1.run_main(test_data,testOutput_data,model_directory)
        print("保存...  ",testOutput_data)

    # ##################################### 用第 2 个模型测试 ####################################
    HypothesisDir = HypothesisDir_list[1]
    # ErjiDir_list = fnmatch.filter(os.listdir(Datapath), sensor) # 根据一级目录遍历出筛选二级目录列表
    # for ErjiDirName in ErjiDir_list:
        
    # TestData_path = os.path.join(Datapath,ErjiDirName) #拼接出二级目录的路径 ..\DataDL_1_TestData\Xsensor_
    TestData_path = Datapath #拼接出二级目录的路径 ..\DataDL_1_TestData\Xsensor_
    print("--------------\n ")
    print("【正在处理的目录】",TestData_path,"【sensor】ToBeTested","【model_directory】")
    MatName_list = fnmatch.filter(os.listdir(TestData_path), '*.mat') #对二级目录遍历并筛选出.mat文件名存到列表
    # path_list.sort(key=lambda x:int(x[:-4])) #新加入的一行做的事情是--对每个文件名将句号前的字符串转化为数字，然后以数字为key来进行排序。
    # path_list.sort(key=lambda x:int(x.split('.')[0])) #只需考虑句号前面的数字顺序了
        
    MakeDirs_path = os.path.join(saveDir,HypothesisDir) #拼接出 需要创建的目录 的路径 ..\DataDL_PannetOutput\GF1
    if not os.path.exists(MakeDirs_path): #创建 需要创建的目录
        os.makedirs(MakeDirs_path)

    for MatName in MatName_list:
        # f = open(os.path.join(path,filename),'rb')
        test_data = os.path.join(TestData_path,MatName) #用 二级目录的路径 和 mat文件名 拼接出 测试mat文件的路径 ..\DataDL_1_TestData\Xsensor_\j1p2.mat
        # print('测试mat文件的路径:',test_data)
        testOutput_data = os.path.join(saveDir,HypothesisDir,MatName) #用自定义的保存文件夹和mat文件名拼接出待保存mat文件的路径  ..\DataDL_PannetOutput\Xsensor_\j1p2.mat 
        # print('待保存mat文件的路径:',testOutput_data)
        RunTestQB.run_main(test_data,testOutput_data)
        print("保存...  ",testOutput_data)

    

if __name__=='__main__':
    # 将Datapath中的数据分布按假设的若干种传感器模型训练，并保存到对应的文件夹中
    # 两个列表内的元素需要一一对应    Tmp\ExampleTestData_Sensor\GF1_1\TestData_DR
    
    HypothesisDir_list = ['HypothesisInGF1model', 'HypothesisInQBmodel']
    Datapath = 'G:\AFusionGroup\Shiyan\shiyan20230421\GF1_Data\Test_Fu'
    saveDir = 'G:\AFusionGroup\Shiyan\shiyan20230421\GF1_LPPN\HypothesisOutput'
    
    model_directory = 'G:\AFusionGroup\Shiyan\shiyan20230421\GF1_LPPN\SaveModel/'
    run_TestHypothesis(Datapath,saveDir,model_directory,HypothesisDir_list)

    model_directory = 'G:\AFusionGroup\Shiyan\shiyan20230421\QB_LPPN\SaveModel/'
    run_TestHypothesis(Datapath,saveDir,model_directory,HypothesisDir_list)


