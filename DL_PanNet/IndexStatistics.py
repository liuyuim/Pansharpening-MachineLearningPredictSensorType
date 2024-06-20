#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a re-implementation of training code of this paper:
# J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding, J. Paisley. "PanNet: A deep network architecture for pan-sharpening", ICCV,2017. 
# author: Junfeng Yang

conda activate LYpy3.7tf1.14np1.16
cd/d E:\TQW\FusionEvaluateExperiment\MethodDL_PanNet
cd/d E:\TQW\FusionEvaluateExperiment_A\MethodDL_PanNet
python RuntrainGF1.py
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
# 引入本目录下其他py文件，调用函数
import Runtrain
import RunFusionHypothesis


if __name__ =='__main__':    

    # 运行Train，定义training data，validation data路径，将训练好的model保存
    model_directory  = '../../Tmp/IndexStatistics100/GF1_SaveModel'                            # directory to save trained model to.
    train_data_name  = '../../Tmp/IndexStatistics100/GF1_TrainingData_output/new.mat'           # training data
    test_data_name   = '../../Tmp/IndexStatistics100/GF1_ValidationData_output/new.mat'      # validation data
    Runtrain.run_main(model_directory,train_data_name,test_data_name)

   # 将Datapath中的数据分布按假设的若干种传感器模型训练，并保存到对应的文件夹中
    Datapath = '..\..\Tmp\IndexStatistics100\GF1_1\TestData_Fu'
    saveDir = '..\..\Tmp\IndexStatistics100\GF1_1\HypothesisOutput'
    
    # 两个列表内的元素需要一一对应    
    model_directory_list = ['../../Tmp/IndexStatistics100/GF1_SaveModel/', 
                            '../../Tmp/IndexStatistics100/GF2_SaveModel/', 
                            '../../Tmp/IndexStatistics100/QB_SaveModel/', 
                            '../../Tmp/IndexStatistics100/WV2_SaveModel/', 
                            '../../Tmp/IndexStatistics100/WV3_SaveModel/',]
    modelDirName_list = ['HypothesisInGF1model', 'HypothesisInGF2model', 'HypothesisInQBmodel', 'HypothesisInWV2model', 'HypothesisInWV3model']
    
    RunFusionHypothesis.run_Test(Datapath,saveDir,model_directory_list,modelDirName_list)