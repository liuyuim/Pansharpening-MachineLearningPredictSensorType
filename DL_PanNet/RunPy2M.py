'''
用脚本预测影像传感器类型，并用预测结果进行融合
conda activate lytf1.14py3.7np1.16
cd/d E:\LiuYu\FusionEvaluateExperiment\MethodDL_PanNet
python RunPy2M.py
'''
import os, sys
import matlab.engine
import RunFusionHypothesis

eng = matlab.engine.start_matlab() #启动matlab engine 
# eng.cd('E:\LiuYu\FusionEvaluateExperiment\MethodDL_PanNet',nargout=0)

# 制作数据集用于融合，路径写死在matlab代码中
eng.Py2M_RemakeMat ('..\..\Tmp\Data_Pairs_Ground\ATestTmp')

# 将Datapath中的数据分布按假设的若干种传感器模型融合，结果保存到对应的文件夹中 # 两个列表内的元素需要一一对应 
Datapath = '..\Tmp\DataDLForecast\TestData_Fu'
saveDir = '..\Tmp\DataDLForecast\PannetOutputToForecast'
model_directory_list = ['./models/models_GF1/', './models/models_GF2/', './models/models_QB/', './models/models_WV2/', './models/models_WV3/',]
modelDirName_list = ['HypothesisInGF1model', 'HypothesisInGF2model', 'HypothesisInQBmodel', 'HypothesisInWV2model', 'HypothesisInWV3model']
RunFusionHypothesis.run_Test(Datapath,saveDir,model_directory_list,modelDirName_list) # os.system("python ./RunFusionForecast.py")

# 使用matlab代码进行预测
sensorPre = eng.Py2M_ForecastFuDR()
print("【matlab返回的预测结果:】",sensorPre) 

# 按照预测的传感器类型进行融合
Datapath = '..\DataDLForecast_Tmp\TestData_Fu'
saveDir = '..\DataDLForecast_Tmp\PannetOutput'

if sensorPre == 'sensorGF1':
    model_directory_list = ['./models/models_GF1/']
    modelDirName_list = ['FusionInGF1model']
elif sensorPre == 'sensorGF2':    
    model_directory_list = ['./models/models_GF2/']
    modelDirName_list = ['FusionInGF2model']
elif sensorPre == 'sensorQB':    
    model_directory_list = ['./models/models_QB/']
    modelDirName_list = ['FusionInQBmodel']
elif sensorPre == 'sensorWV2':    
    model_directory_list = ['./models/models_WV2/']
    modelDirName_list = ['FusionInWV2model']
elif sensorPre == 'sensorWV3':    
    model_directory_list = ['./models/models_WV3/']
    modelDirName_list = ['FusionInWV3model']
else:
    print("检测到 传感器种类 不匹配 程序退出。\n检查run_Test()函数中 sensor_list 和 model_directory_list两个列表 ")
    sys.exit(0) # sys.exit(0) 该方法中包含一个参数status，默认为0，表示正常退出，也可以为1，表示异常退出。


RunFusionHypothesis.run_Test(Datapath,saveDir,model_directory_list,modelDirName_list)
# eng.quit()

# if __name__=='__main__':
#     # run_Train()
#     run_Test()


























