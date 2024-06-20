#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# conda activate LYpy3.7tf1.14np1.16
# cd/d O:\HC550WDC16TO\FusionEvaluateExperiment\DL_PanNet
# python RuntrainWV2.py



import os 
import Runtrain, Runtrain_4b1023, Runtrain_4b2047


if __name__ =='__main__':
        
        
    # model_directory  = 'O:/HC550WDC16TO/Shiyan/20231201/WV2_PanNet/SaveModel'                     # directory to save trained model to.
    # train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231201/WV2_Data/Train/N_H_W_C.mat'       # training data
    # test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231201/WV2_Data/Validation/N_H_W_C.mat'       # validation data
    # Runtrain_4b2047.run_main(model_directory,train_data_name,test_data_name)
    # # Runtrain_8b.run_main(model_directory,train_data_name,test_data_name)
    
    # model_directory  = 'O:/HC550WDC16TO/Shiyan/20231201/WV3_PanNet/SaveModel'                     # directory to save trained model to.
    # train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231201/WV3_Data/Train/N_H_W_C.mat'       # training data
    # test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231201/WV3_Data/Validation/N_H_W_C.mat'       # validation data
    # Runtrain_4b2047.run_main(model_directory,train_data_name,test_data_name)

    # model_directory  = 'O:/HC550WDC16TO/Shiyan/20231201/WV4_PanNet/SaveModel'                     # directory to save trained model to.
    # train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231201/WV4_Data/Train/N_H_W_C.mat'       # training data
    # test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231201/WV4_Data/Validation/N_H_W_C.mat'       # validation data
    # Runtrain_4b2047.run_main(model_directory,train_data_name,test_data_name)



    model_directory  = 'O:/HC550WDC16TO/Shiyan/20231202/WV2_PanNet/SaveModel'                     # directory to save trained model to.
    train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231202/WV2_Data/Train/N_H_W_C.mat'       # training data
    test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231202/WV2_Data/Validation/N_H_W_C.mat'       # validation data
    Runtrain_4b2047.run_main(model_directory,train_data_name,test_data_name)
    # Runtrain_8b.run_main(model_directory,train_data_name,test_data_name)
    
    model_directory  = 'O:/HC550WDC16TO/Shiyan/20231202/WV3_PanNet/SaveModel'                     # directory to save trained model to.
    train_data_name  = 'O:/HC550WDC16TO/Shiyan/20231202/WV3_Data/Train/N_H_W_C.mat'       # training data
    test_data_name   = 'O:/HC550WDC16TO/Shiyan/20231202/WV3_Data/Validation/N_H_W_C.mat'       # validation data
    Runtrain_4b2047.run_main(model_directory,train_data_name,test_data_name)