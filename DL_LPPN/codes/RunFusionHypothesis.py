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
import RunTestGF1, RunTestGF2, RunTestQB, RunTestIK, RunTestJL1, RunTestWV2b4, RunTestWV3b4, RunTestWV4b4 


"""
# 改造的代码，适用于批量跑数据
# from mytest_Run import run_main
"""

#批量运行测试代码
def run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list):
    print("run DL_PanNet Fusion Forecast ... \n")
    
    # 判断 传感器种类数 和 对应的model数 是否匹配
    if(len(model_directory_list) != len(HypothesisDir_list)):
        print("检测到 传感器种类数 和 对应的model数 不匹配 程序退出。\n检查run_TestHypothesis()函数中 sensor_list 和 model_directory_list两个列表 ")
        sys.exit(0) # sys.exit(0) 该方法中包含一个参数status，默认为0，表示正常退出，也可以为1，表示异常退出。
    
    
    for j in range(len(model_directory_list)): 
        model_directory = model_directory_list[j]
        HypothesisDir = HypothesisDir_list[j]
                           
        # TestData_path = os.path.join(Datapath,ErjiDir) #拼接出二级目录的路径 ..\DataDL_1_TestData\Xsensor_
        TestData_path = Datapath #拼接出二级目录的路径 ..\DataDL_1_TestData\Xsensor_
        print("--------------\n ")
        print("【正在处理的目录】",TestData_path,"【sensor】ToBeTested","【model_directory】",model_directory)
        MatName_list = fnmatch.filter(os.listdir(TestData_path), '*.mat') #对二级目录遍历并筛选出.mat文件名存到列表
        # path_list.sort(key=lambda x:int(x[:-4])) #新加入的一行做的事情是--对每个文件名将句号前的字符串转化为数字，然后以数字为key来进行排序。
        # path_list.sort(key=lambda x:int(x.split('.')[0])) #只需考虑句号前面的数字顺序了
        
        #创建HypothesisDir目录
        MakeDirs_path = os.path.join(saveDir,HypothesisDir) #拼接出 需要创建的目录 的路径 ..\DataDL_PannetOutput\GF1
        if not os.path.exists(MakeDirs_path): #创建目录
            os.makedirs(MakeDirs_path)

        for MatName in MatName_list:
            # f = open(os.path.join(path,filename),'rb')
            test_data = os.path.join(TestData_path,MatName) #用 二级目录的路径 和 mat文件名 拼接出 测试mat文件的路径 ..\DataDL_1_TestData\Xsensor_\j1p2.mat
            # print('测试mat文件的路径:',test_data)
            testOutput_data = os.path.join(saveDir,HypothesisDir,MatName) #用自定义的保存文件夹和mat文件名拼接出待保存mat文件的路径  ..\DataDL_PannetOutput\Xsensor_\j1p2.mat 
            # print('待保存mat文件的路径:',testOutput_data)

            # # 判断传入的
            # test_data = mat73.loadmat(test_data)  # HxWxC=256x256x8#
            # locals(Runtest)['run_main'](test_data,model_directory,testOutput_data)
            if fnmatch.fnmatch(HypothesisDir, '*GF1*'):
                print("检测HypothesisDir...使用的融合代码是 RunTestGF1 ... ")
                RunTestGF1.run_main(test_data,model_directory,testOutput_data)
            elif fnmatch.fnmatch(HypothesisDir, '*GF2*'):
                print("检测HypothesisDir...使用的融合代码是 RunTestGF2 ... ")        
                RunTestGF2.run_main(test_data,model_directory,testOutput_data)
            elif fnmatch.fnmatch(HypothesisDir, '*IK*'): 
                print("检测HypothesisDir...使用的融合代码是 RunTestIK ... ")   
                RunTestIK.run_main(test_data,model_directory,testOutput_data)
            elif fnmatch.fnmatch(HypothesisDir, '*JL1*'): 
                print("检测HypothesisDir...使用的融合代码是 RunTestJL1 ... ")   
                RunTestJL1.run_main(test_data,model_directory,testOutput_data)
            elif fnmatch.fnmatch(HypothesisDir, '*QB*'):
                print("检测HypothesisDir...使用的融合代码是 RunTestQB ... ")        
                RunTestQB.run_main(test_data,model_directory,testOutput_data)
            elif fnmatch.fnmatch(HypothesisDir, '*WV2*'):
                print("检测HypothesisDir...使用的融合代码是 RunTestWV2b4 ... ")   
                RunTestWV2b4.run_main(test_data,model_directory,testOutput_data)
            elif fnmatch.fnmatch(HypothesisDir, '*WV3*'):
                print("检测HypothesisDir...使用的融合代码是 RunTestWV3b4 ... ")   
                RunTestWV3b4.run_main(test_data,model_directory,testOutput_data)
            elif fnmatch.fnmatch(HypothesisDir, '*WV4*'):
                print("检测HypothesisDir...使用的融合代码是 RunTestWV4b4 ... ")   
                RunTestWV4b4.run_main(test_data,model_directory,testOutput_data)

            print("保存...  ",testOutput_data)

    print("==================\n")
    print("run DL_PanNet FusionForEvaluationAndForecast 已全部处理完毕，请到此文件夹查看：",saveDir)

# conda activate LYpy3.7tf1.14np1.16
# cd/d C:\Users\LiuYu\Desktop\FusionEvaluateExperiment\DL_LPPN\codes
# python RunFusionHypothesis.py

 
if __name__=='__main__':
    # 将Datapath中的数据分布按假设的若干种传感器模型训练，并保存到对应的文件夹中
    # 两个列表内的元素需要一一对应    Tmp\ExampleTestData_Sensor\GF1_1\TestData_DR
    
    # 两个列表内的元素需要一一对应    
    model_directory_list = [
                            'O:/HC550WDC16TO/Shiyan/20231201/GF1_LPPN/SaveModel',  
                            # 'O:/HC550WDC16TO/Shiyan/20231201/GF2_LPPN/SaveModel',  
                            'O:/HC550WDC16TO/Shiyan/20231201/IK_LPPN/SaveModel',
                            # 'O:/HC550WDC16TO/Shiyan/20231201/JL1_LPPN/SaveModel',
                            'O:/HC550WDC16TO/Shiyan/20231201/WV2_LPPN/SaveModel',   
                            'O:/HC550WDC16TO/Shiyan/20231201/WV3_LPPN/SaveModel',
                            'O:/HC550WDC16TO/Shiyan/20231201/WV4_LPPN/SaveModel',
                            'O:/HC550WDC16TO/Shiyan/20231201/QB_LPPN/SaveModel'
                            ]
    HypothesisDir_list = [
                        'HypothesisInGF1model', 
                        # 'HypothesisInGF2model', 
                        'HypothesisInIKmodel',
                        # 'HypothesisInJL1model', 
                        'HypothesisInWV2model', 
                        'HypothesisInWV3model',
                        'HypothesisInWV4model',
                        'HypothesisInQBmodel'
                        ]
    
    SensorNames = ['GF1', 'IK', 'QB', 'WV2', 'WV3', 'WV4'] # SensorNames = ['GF1', 'GF2', 'JL1', 'QB', 'WV2', 'WV3'] ['GF1', 'IK', 'QB', 'WV2', 'WV3', 'WV4']
    for sensor_name in SensorNames:

        Sensor_Data = sensor_name + '_Data'
        Sensor_Net = sensor_name + '_LPPN'        
        Base_dir = r'O:/HC550WDC16TO/Shiyan/20231201'
        Sizes = ["256","128","64","32"] # Sizes = ["1024","512","256","128","64","32"]
        for Size in Sizes:
            
            Test_Fu = 'Test_Fu' + Size        
            HypothesisOutput_Fu = 'HypothesisOutput_Fu' + Size # HypothesisOutput_Fu = 'HypothesisOutput_Fu' + Size

            Datapath = os.path.join(Base_dir, Sensor_Data, Test_Fu)
            saveDir = os.path.join(Base_dir, Sensor_Net, HypothesisOutput_Fu)
            run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)
    
    # Datapath = 'F:/AFusionGroup/Shiyan/shiyan20230901/GF1_Data/Test_Fu1024'
    # saveDir = 'F:/AFusionGroup/Shiyan/shiyan20230901/GF1_LPPN/HypothesisOutput_Fu1024'    
    # run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)