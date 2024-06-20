'''
用脚本批量跑
conda activate LYpy3.7tf1.14np1.16
cd/d E:\LiuYu\FusionEvaluateExperiment\MethodDL_PanNet
python RunFusionHypothesis.py

'''
import os, sys, fnmatch
import cv2
import mat73
import Runtest4b, Runtest8b




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

            # # 判断传入的mat波段数
            # test_data = mat73.loadmat(test_data)  # HxWxC=256x256x8
            # if cv2.channels(test_data) == 4 :
            #     print("4波段\n")
            #     Runtest4b.run_main(test_data,model_directory,testOutput_data)
            # elif cv2.channels(test_data) == 8 :
            #     print("8波段\n")
            #     Runtest8b.run_main(test_data,model_directory,testOutput_data)
            Runtest4b.run_main(test_data,model_directory,testOutput_data)
            
            print("保存...  ",testOutput_data)

    print("==================\n")
    print("run DL_PanNet FusionForEvaluationAndForecast 已全部处理完毕，请到此文件夹查看：",saveDir)

            
if __name__=='__main__': # 按多种模型生成假设结果，并保存到对应的假设结果文件夹中  

    
    # 两个列表内的元素需要一一对应    
    model_directory_list = [# 'C:/Home/AFusionGroup/Shiyan/shiyan20230706/GF1_PanNet/SaveModel',
                            # 'C:/Home/AFusionGroup/Shiyan/shiyan20230706/GF2_PanNet/SaveModel/', 
                            'C:/Home/AFusionGroup/Shiyan/shiyan20230706/IK_PanNet/SaveModel/', 
                            'C:/Home/AFusionGroup/Shiyan/shiyan20230706/QB_PanNet/SaveModel/', 
                            # 'C:/Home/AFusionGroup/Shiyan/shiyan20230706/WV2_PanNet/SaveModel/',
                            # 'C:/Home/AFusionGroup/Shiyan/shiyan20230706/WV3_PanNet/SaveModel/',
                            'C:/Home/AFusionGroup/Shiyan/shiyan20230706/WV4_PanNet/SaveModel/',
                            ]
    HypothesisDir_list = [# 'HypothesisInGF1model', 
                        # 'HypothesisInGF2model',
                        'HypothesisInIKmodel',
                        'HypothesisInQBmodel', 
                        # 'HypothesisInWV2model',
                        # 'HypothesisInWV3model',
                        'HypothesisInWV4model',
                        ]
    
    Datapath = 'C:/Home/AFusionGroup/Shiyan/shiyan20230706/IK_Data/Test_Fu'
    saveDir = 'C:/Home/AFusionGroup/Shiyan/shiyan20230706/IK_PanNet/HypothesisOutput_Fu'
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

    Datapath = 'F:/AFusionGroup\Shiyan\shiyan20230601\GF2_Data\Test_Fu'
    saveDir = 'F:/AFusionGroup\Shiyan\shiyan20230601\GF2_PanNet\HypothesisOutput_Fu'    
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

    Datapath = 'F:/AFusionGroup\Shiyan\shiyan20230601\QB_Data\Test_Fu'
    saveDir = 'F:/AFusionGroup\Shiyan\shiyan20230601\QB_PanNet\HypothesisOutput_Fu'    
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

    Datapath = 'F:/AFusionGroup\Shiyan\shiyan20230601\WV2_Data\Test_Fu'
    saveDir = 'F:/AFusionGroup\Shiyan\shiyan20230601\WV2_PanNet\HypothesisOutput_Fu'    
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

    Datapath = 'F:/AFusionGroup\Shiyan\shiyan20230601\WV3_Data\Test_Fu'
    saveDir = 'F:/AFusionGroup\Shiyan\shiyan20230601\WV3_PanNet\HypothesisOutput_Fu'    
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)
    
    
    Datapath = 'F:/AFusionGroup\Shiyan\shiyan20230601\GF1_Data\Test_DR'
    saveDir = 'F:/AFusionGroup\Shiyan\shiyan20230601\GF1_PanNet\HypothesisOutput_DR'    
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

    Datapath = 'F:/AFusionGroup\Shiyan\shiyan20230601\GF2_Data\Test_DR'
    saveDir = 'F:/AFusionGroup\Shiyan\shiyan20230601\GF2_PanNet\HypothesisOutput_DR'    
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

    Datapath = 'F:/AFusionGroup\Shiyan\shiyan20230601\QB_Data\Test_DR'
    saveDir = 'F:/AFusionGroup\Shiyan\shiyan20230601\QB_PanNet\HypothesisOutput_DR'    
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

    Datapath = 'F:/AFusionGroup\Shiyan\shiyan20230601\WV2_Data\Test_DR'
    saveDir = 'F:/AFusionGroup\Shiyan\shiyan20230601\WV2_PanNet\HypothesisOutput_DR'    
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

    Datapath = 'F:/AFusionGroup\Shiyan\shiyan20230601\WV3_Data\Test_DR'
    saveDir = 'F:/AFusionGroup\Shiyan\shiyan20230601\WV3_PanNet\HypothesisOutput_DR'    
    run_TestHypothesis(Datapath,saveDir,model_directory_list,HypothesisDir_list)

