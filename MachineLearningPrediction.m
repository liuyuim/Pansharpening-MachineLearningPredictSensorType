%% 5.4 评价结果自助打包
clc; clear; close all; addpath(genpath('.\Fx\'));

NumImgStart = 1; NumImgEnd = 100; % 文件夹中按自然数排序第多少张的范围，不是文件名范围
for Sizes = [1024]  %% Size,512,256,128,64,32
    Size = num2str(Sizes);
    
    NetNames = {'WSDFNet'}; %'PanNet','LPPN','WSDFNet'
    for i = 1:numel(NetNames)
        NetName = NetNames{i};
        
        SensorNames = {'GF1','QB','WV2','WV3'}; %% Sensor 'GF1','GF2','IK','JL1','QB','WV2','WV3','WV4'
        for j = 1:numel(SensorNames)
            Sensor_Net = strcat(SensorNames{j},'_',NetName); % 或者    Sensor_Data = SensorNames{i} + "_Data";Sensor_Data = strcat(SensorNames{i}, '_Data');  
            Evaluate_Fu = ['Evaluate_Fu',Size];
            EvaluationDir = fullfile('F:\Demo\Data_MLPrediction\OurDatasetResult',Sensor_Net,Evaluate_Fu); %NBUDatasetResult OurDatasetResult
            saveDirName = [Evaluate_Fu,'_G1QW2W3']; % _G1QW2W3 _G1QW4
            Evaluation2RepackMatrix (SensorNames,EvaluationDir,saveDirName,NumImgStart,NumImgEnd) %'F:\Demo\Data_Evaluation\IK_WSDFNet\Evaluate_Fu128',...
          
        end
    end
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 【7 机器学习方法对指标优化】
% matlab的libsvm工具箱安装教程 在Fx下对应函数有说明
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% kmeans K均值  ML_kmeans4 是 包括三个投影图的
clc; clear; close all hidden; addpath(genpath('.\Fx\'));
Size = [1024];Size = num2str(Size);

EvaluationDirList={['F:\Demo\Data_MLPrediction\NBUDatasetResult\GF1_WSDFNet\Evaluate_Fu',Size,'_G1QW4'],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\QB_WSDFNet\Evaluate_Fu',Size,'_G1QW4'],...                                       
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV4_WSDFNet\Evaluate_Fu',Size,'_G1QW4']};
TrainProportion = 0.8 ; TestProportion = 0.2 ; % Rand占比
columnSample = [3];


ML_kmeans4(EvaluationDirList,TrainProportion,columnSample)

%% 7.1 Self数据训练，Self数据预测（G1 G2 J1 Q W2 W3）
clc; clear; close all hidden; addpath(genpath('.\Fx\'));
Size = 1024; Size = num2str(Size);
EvaluationDirList={['F:\Demo\Data_MLPrediction\NBUDatasetResult\GF1_WSDFNet\Evaluate_Fu',Size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\IK_WSDFNet\Evaluate_Fu',Size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\QB_WSDFNet\Evaluate_Fu',Size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV2_WSDFNet\Evaluate_Fu',Size],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV3_WSDFNet\Evaluate_Fu',Size],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV4_WSDFNet\Evaluate_Fu',Size]};

TrainProportion = 0.8 ; TestProportion = 0.2 ; % Rand占比
columnSample = [1,2,3,18,19,20,21];



% SVM支持向量机
ML_SVMRandData1(EvaluationDirList,TrainProportion,columnSample)

% SVM支持向量机 交叉验证超参数调优
% ML_SVMcgRandData1(EvaluationDirList,TrainProportion,columnSample)

% RF随机森林
% ML_RFRandData1(EvaluationDirList,TrainProportion,columnSample)

% RF随机森林 超参数调优
% ML_RFcgRandData1(EvaluationDirList,TrainProportion,columnSample) % 随机抽取比例的训练集+自定义指标

% BP神经网络
% ML_BPRandData1(EvaluationDirList,TrainProportion,columnSample)


%% 7.2 NBU数据训练，Self数据预测 (G1 Q W2 W3)
clc; clear; close all hidden; addpath(genpath('.\Fx\'));
Size = 1024; Size = num2str(Size);
EvaluationDirListTr={['F:\Demo\Data_MLPrediction\NBUDatasetResult\GF1_WSDFNet\Evaluate_Fu',Size,'_G1QW2W3'],...                     
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\QB_WSDFNet\Evaluate_Fu',Size,'_G1QW2W3'],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV2_WSDFNet\Evaluate_Fu',Size,'_G1QW2W3'],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV3_WSDFNet\Evaluate_Fu',Size,'_G1QW2W3']};

EvaluationDirList={['F:\Demo\Data_MLPrediction\OurDatasetResult\GF1_WSDFNet\Evaluate_Fu',Size,'_G1QW2W3'],...                     
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\QB_WSDFNet\Evaluate_Fu',Size,'_G1QW2W3'],...
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\WV2_WSDFNet\Evaluate_Fu',Size,'_G1QW2W3'],...
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\WV3_WSDFNet\Evaluate_Fu',Size,'_G1QW2W3']};

TrainProportion = 0.8 ; TestProportion = 0.2 ; % Rand占比
columnSample = [1,2,3,18,19,20,21]; %设置指标所在列， columnSample = 0; 代表取所有

% SVM支持向量机
% ML_SVMRandData2(EvaluationDirListTr,TrainProportion,EvaluationDirList,TestProportion,columnSample)

% SVM支持向量机 交叉验证超参数调优
% ML_SVMcgRandData2(EvaluationDirListTr,TrainProportion,EvaluationDirList,TestProportion,columnSample)

% RF随机森林
ML_RFRandData2(EvaluationDirListTr,TrainProportion,EvaluationDirList,TestProportion,columnSample)





%% 7.3 手动指定指标 作为机器学习特征 预测准确率
% （指定columnSample，十次 均值+-方差，结果是列）

clc; clear; close all hidden; addpath(genpath('.\Fx\'));
NetName = 'WSDFNet'; % PanNet WSDFNet WSDFNet 
TrainProportion = 0.8 ; TestProportion = 0.2 ; % Rand占比
columnSampleVector = [1,2,3,18,19,20,21];  % columnSample = [1,2,3,18,19,20,21];

% 
MeanVarList_Sum = MeanVarList_IndexManual(NetName,TrainProportion,TestProportion,columnSampleVector);
fprintf("程序结束，请在 工作区 点击对应变量查看，变量中从左到右依次是循环的Size的对应结果。");

%% 手动指定指标 作为机器学习特征 每种传感器取三幅图片投票法预测后再统计
clc; clear; close all hidden; addpath(genpath('.\Fx\'));
NetName = 'WSDFNet'; % PanNet WSDFNet WSDFNet 
TrainProportion = 0.8 ; TestProportion = 0.2 ; % Rand占比
columnSampleVector = [1,2,3,18,19,20,21];  % columnSample = [1,2,3,18,19,20,21];

% 
MeanVarList_Vote3time (NetName,TrainProportion,TestProportion,columnSampleVector);
fprintf("程序结束，请在 工作区 点击对应变量查看，变量中从左到右依次是循环的Size的对应结果。");

%% 升级版 循环指定计算单指标与多指标 作为机器学习特征 预测准确率（十次 均值+-方差，结果保存成table）

clc; clear; close all hidden; addpath(genpath('.\Fx\'));
NetName = 'WSDFNet'; % PanNet WSDFNet WSDFNet 
TrainProportion = 0.8 ; TestProportion = 0.2 ; % Rand占比
columnSampleVector = [1,2,3,18,19,20,21];  % columnSample = [1,2,3,18,19,20,21];
MeanVarList_Sum_table = [];

for i = 1:length(columnSampleVector)+1
    if i == length(columnSampleVector)+1
        columnSample = columnSampleVector; % 在循环单个指标结束后，最后使用整个向量
    else
        columnSample = columnSampleVector(i);
    end
    
    MeanVarList_Sum = MeanVarList_IndexAuto (NetName,TrainProportion,TestProportion,columnSample);
    MeanVarList_Sum_table = vertcat(MeanVarList_Sum_table,MeanVarList_Sum);
end

% 响铃
for i = 1:5
    sound(sin(1:3000));
    pause(2); % 可选，可添加延迟以控制每次播放之间的间隔
end