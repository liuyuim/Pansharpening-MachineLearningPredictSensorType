#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# conda activate LYpy3.7tf1.14np1.16
# cd/d O:\HC550WDC16TO\FusionEvaluateExperiment\DL_PanNet
# python RuntrainGF1.py



import os 
import Runtrain, Runtrain_4b1023, Runtrain_4b2047


if __name__ =='__main__':
        
    # num_spectral = 4  # 波段数 4 or 8
    # model_directory  = 'G:/AFusionGroup/Shiyan/shiyan20230421/WV4_PanNet/SaveModel'                     # directory to save trained model to.
    # train_data_name  = 'E:/LiuYu/FusionEvaluateExperiment/MethodDL_PanNet/training_data/GF2_train_NHWC.mat'       # training data
    # test_data_name   = 'E:/LiuYu/FusionEvaluateExperiment/MethodDL_PanNet/training_data/GF2_validation_NHWC.mat'       # validation data
    # Runtrain.run_main(model_directory,train_data_name,test_data_name,num_spectral)
    
    
    # model_directory  = 'O:/HC550WDC16TO/Shiyan/20231201/GF1_PanNet/SaveModel'                     # directory to save trained model to.
    # train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231201/GF1_Data/Train/N_H_W_C.mat'       # training data
    # test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231201/GF1_Data/Validation/N_H_W_C.mat'       # validation data
    # Runtrain_4b1023.run_main(model_directory,train_data_name,test_data_name)
    # Runtrain_8b.run_main(model_directory,train_data_name,test_data_name)
        
    # model_directory  = 'O:/HC550WDC16TO/Shiyan/20231201/IK_PanNet/SaveModel'                     # directory to save trained model to.
    # train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231201/IK_Data/Train/N_H_W_C.mat'       # training data
    # test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231201/IK_Data/Validation/N_H_W_C.mat'       # validation data
    # Runtrain_4b2047.run_main(model_directory,train_data_name,test_data_name)

    # model_directory  = 'O:/HC550WDC16TO/Shiyan/20231201/QB_PanNet/SaveModel'                     # directory to save trained model to.
    # train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231201/QB_Data/Train/N_H_W_C.mat'       # training data
    # test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231201/QB_Data/Validation/N_H_W_C.mat'       # validation data
    # Runtrain_4b2047.run_main(model_directory,train_data_name,test_data_name)



    model_directory  = 'O:/HC550WDC16TO/Shiyan/20231202/GF1_PanNet/SaveModel'                     # directory to save trained model to.
    train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231202/GF1_Data/Train/N_H_W_C.mat'       # training data
    test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231202/GF1_Data/Validation/N_H_W_C.mat'       # validation data
    Runtrain_4b1023.run_main(model_directory,train_data_name,test_data_name)
        
    model_directory  = 'O:/HC550WDC16TO/Shiyan/20231202/GF2_PanNet/SaveModel'                     # directory to save trained model to.
    train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231202/GF2_Data/Train/N_H_W_C.mat'       # training data
    test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231202/GF2_Data/Validation/N_H_W_C.mat'       # validation data
    Runtrain_4b1023.run_main(model_directory,train_data_name,test_data_name)

    model_directory  = 'O:/HC550WDC16TO/Shiyan/20231202/JL1_PanNet/SaveModel'                     # directory to save trained model to.
    train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231202/JL1_Data/Train/N_H_W_C.mat'       # training data
    test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231202/JL1_Data/Validation/N_H_W_C.mat'       # validation data
    Runtrain_4b2047.run_main(model_directory,train_data_name,test_data_name)

    model_directory  = 'O:/HC550WDC16TO/Shiyan/20231202/QB_PanNet/SaveModel'                     # directory to save trained model to.
    train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231202/QB_Data/Train/N_H_W_C.mat'       # training data
    test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231202/QB_Data/Validation/N_H_W_C.mat'       # validation data
    Runtrain_4b2047.run_main(model_directory,train_data_name,test_data_name)