
import os, sys, fnmatch
import cv2

import Runtest4b1023,Runtest4b2047

def run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list):
    print("run DL Fusion Forecast ... \n")
    
    # 判断 传感器种类数 和 对应的model数 是否匹配
    if(len(model_directory_list) != len(HypothesisDir_list)):
        print("检测到 传感器种类数 和 对应的model数 不匹配 程序退出。\n检查run_TestHypothesis()函数中 HypothesisDir_list 和 model_directory_list两个列表 ")
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

            # # 判断传入的mat波段数
            # test_data = mat73.loadmat(test_data)  # HxWxC=256x256x8
            # if cv2.channels(test_data) == 4 :
            #     print("4波段\n")
            #     Runtest4b.run_main(test_data,model_directory,testOutput_data)
            # elif cv2.channels(test_data) == 8 :
            #     print("8波段\n")
            #     Runtest8b.run_main(test_data,model_directory,testOutput_data)
            
            # Runtest4b1023.run_main(test_data,model_directory,testOutput_data)
            # Runtest4b2047.run_main(test_data,model_directory,testOutput_data)
        
            if fnmatch.fnmatch(HypothesisDir, '*GF*'):
                print("检测HypothesisDir是GF1/GF2...使用的融合代码是 Runtest4b1023 ... ")
                Runtest4b1023.run_main(test_data,model_directory,testOutput_data)            
            elif (fnmatch.fnmatch(HypothesisDir, '*A*')
                  |fnmatch.fnmatch(HypothesisDir, '*IK*')
                  |fnmatch.fnmatch(HypothesisDir, '*JL*')
                  |fnmatch.fnmatch(HypothesisDir, '*QB*')                       
                  |fnmatch.fnmatch(HypothesisDir, '*WV*')):
                print("检测HypothesisDir是A/IK/JL/QB/WV...使用的融合代码是 Runtest4b2047 ... ")
                Runtest4b2047.run_main(test_data,model_directory,testOutput_data)            
            else:
                print("无法匹配HypothesisDir...  ")
                break            
            print("保存...  ",testOutput_data)

    print("==================\n")
    print("已全部处理完毕，请到此文件夹查看：",saveDir)


# 用脚本批量跑
# conda activate LYpy3.8pt1.9.0
# cd/d O:\HC550WDC16TO\FusionEvaluateExperiment\DL_WSDFNet\codes
# python RunFusionHypothesis.py

    
if __name__=='__main__': # 按多种模型生成假设结果，并保存到对应的假设结果文件夹中  

    # 两个列表内的元素需要一一对应    （注意列表每项的逗号）
    model_directory_list = [
                            'O:/HC550WDC16TO/Shiyan/20231202/GF1_WSDFNet/SaveModel/', 
                            'O:/HC550WDC16TO/Shiyan/20231202/GF2_WSDFNet/SaveModel/', 
                            # 'O:/HC550WDC16TO/Shiyan/20231202/IK_WSDFNet/SaveModel/', 
                            'O:/HC550WDC16TO/Shiyan/20231202/JL1_WSDFNet/SaveModel/',
                            'O:/HC550WDC16TO/Shiyan/20231202/WV2_WSDFNet/SaveModel/', 
                            'O:/HC550WDC16TO/Shiyan/20231202/WV3_WSDFNet/SaveModel/',
                            # 'O:/HC550WDC16TO/Shiyan/20231202/WV4_WSDFNet/SaveModel/',
                            'O:/HC550WDC16TO/Shiyan/20231202/QB_WSDFNet/SaveModel/'
                            # 'O:/HC550WDC16TO/Shiyan/20231202/A_WSDFNet/SaveModel/'
                            ]
    HypothesisDir_list = [
                            'HypothesisInGF1model', 
                            'HypothesisInGF2model', 
                            # 'HypothesisInIKmodel',
                            'HypothesisInJL1model',
                            'HypothesisInWV2model', 
                            'HypothesisInWV3model', 
                            # 'HypothesisInWV4model',
                            'HypothesisInQBmodel'
                            # 'HypothesisInAmodel'
                          ]    
        
    SensorNames = ['GF1', 'GF2', 'JL1', 'QB', 'WV2', 'WV3']  # ['GF1', 'GF2', 'JL1', 'QB', 'WV2', 'WV3'] [ 'GF1', 'IK', 'QB', 'WV2', 'WV3', 'WV4']
    for sensor_name in SensorNames:

        Sensor_Data = sensor_name + '_Data'
        Sensor_Net = sensor_name + '_WSDFNet'        
        Base_dir = r'O:/HC550WDC16TO/Shiyan/20231202'
        Sizes = [ "256","128","64","32" ] # Sizes = ["1024","512","256","128","64","32"]
        for Size in Sizes:
            
            Test_Fu = 'Test_DR' + Size        
            # HypothesisOutput_Fu = 'HypothesisOutput_Fu' + Size   # HypothesisOutput_Fu = 'HypothesisOutput_Fu' + Size  AmodelOutput_Fu
            HypothesisOutput_Fu = 'HypothesisOutput_DR' + Size   # HypothesisOutput_Fu = 'HypothesisOutput_Fu' + Size

            Datapath = os.path.join(Base_dir, Sensor_Data, Test_Fu)
            saveDir = os.path.join(Base_dir, Sensor_Net, HypothesisOutput_Fu)
            run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

# # 1024
#     Datapath = 'F:\AFusionGroup\Shiyan\shiyan20231202\GF1_Data\Test_Fu1024'
#     saveDir = 'F:\AFusionGroup\Shiyan\shiyan20231202\GF1_WSDFNet\HypothesisOutput_Fu1024'
#     run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)
