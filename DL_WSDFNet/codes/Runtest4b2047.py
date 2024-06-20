'''
用脚本批量跑
conda activate LYpy3.8pt1.9.0
cd/d E:\LiuYu\FusionEvaluateExperiment\MethodDL_WSDFNet\codes
python Runtest4b.py

'''
# ------------------------------------------------------------------
# Copyright (c) 2021, Zi-Rong Jin, Tian-Jing Zhang, Cheng Jin, and 
# Liang-Jian Deng, All rights reserved.
#
# This work is licensed under GNU Affero General Public License
# v3.0 International To view a copy of this license, see the
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird and GaoFen-2), please change the corresponding
# inputs.
# ------------------------------------------------------------------

import torch
from evaluate4b import compute_index
from scipy import io as sio
from model4b import WSDFNet

import h5py

import os, sys, fnmatch

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

"""
# 原test.py代码       

"""

def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8
    # data = h5py.File(file_path)  # HxWxC=256x256x8
    # data = mat73.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    lms = torch.from_numpy(
        data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy((data['ms'] / 2047.0)
                          ).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy((data['pan'] / 2047.0))   # HxW = 256x256

    return lms, ms, pan


# def load_gt_compared(file_path):
#     # data = sio.loadmat(file_path)  # HxWxC=256x256x8
#     data = mat73.loadmat(file_path)  # HxWxC=256x256x8

#     # tensor type:
#     test_gt = torch.from_numpy(data['gt'] / 2047.0)  # CxHxW = 8x256x256

#     return test_gt

def run_main(test_data,model_directory,testOutput_data):
    
    # file_path = "test_data/WorldView-3_1.mat"
    file_path = test_data
    test_lms, test_ms, test_pan = load_set(file_path)
    test_lms = test_lms.cuda().unsqueeze(dim=0).float()

    # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
    test_ms = test_ms.cuda().unsqueeze(dim=0).float()
    test_pan = test_pan.cuda().unsqueeze(dim=0).unsqueeze(
        dim=1).float()  # convert to tensor type: 1x1xHxW
    # test_gt = load_gt_compared(file_path)  # compared_result
    # test_gt = (test_gt * 2047).cuda().double()
    model = WSDFNet().cuda()
    # model.load_state_dict(torch.load('./pretrained/WSDFNET_500.pth'))
    model.load_state_dict(torch.load(os.path.join(model_directory,'WSDFNET_500.pth')))

    model.eval()
    with torch.no_grad():
        output3 = model(test_pan, test_lms)
        result_our = torch.squeeze(output3).permute(1, 2, 0)
        sr = torch.squeeze(output3).permute(
            1, 2, 0).cpu().detach().numpy()  # HxWxC
        result_our = result_our * 2047
        result_our = result_our.type(torch.DoubleTensor).cuda()

        # sio.savemat('../results/WorldView-3_1_wsdfnet.mat', {'wsdfnet_output': sr})
        sio.savemat(testOutput_data, {'output': sr})

        # our_SAM, our_ERGAS = compute_index(test_gt, result_our, 4)
        # # print loss for each epoch
        # print('WSDFNet_output_SAM: {}'.format(our_SAM))
        # print('WSDFNet_output_ERGAS: {}'.format(our_ERGAS))



"""
# 改造的代码，适用于批量跑数据
# 

"""
#批量运行测试代码 
if __name__=='__main__':# 正常融合  二级目录结构  
    
    print("run DL_PanNet Fusion... \n")
    
    Datapath = '。\GF1_Data\Test_Fu'
    saveDir = '。\GF1_Data\Fusion'
    # sensor_list = ['GF1_*','GF2_*','QB_*','WV2_*','WV3_*'] # sensor_list 和 model_directory_list 两个列表内的元素需要一一对应
    sensor_list = ['GF1_*','GF2_*','QB_*','WV2_*','WV3_*'] # sensor_list 和 model_directory_list 两个列表内的元素需要一一对应
    # model_directory_list = ['./models/models_GF1/', './models/models_GF2/', './models/models_QB/', './models/models_WV2/', './models/models_WV3/',]
    model_directory_list = ['./models/models_GF1/', './models/models_GF2/', './models/models_QB/', './models/models_WV2/', './models/models_WV3/',]
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

