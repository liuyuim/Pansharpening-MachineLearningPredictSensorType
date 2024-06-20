

function [MeanVarList_Sum] = MeanVarList_IndexManual (NetName,TrainProportion,TestProportion,columnSampleVector)
%% 7.1 NBU数据训练，NBU数据预测  for循环size

MeanVarList_01SVM = []; MeanVarList_01RF = [];

for Sizes = [1024]  % size = [1024,512,256,128,64,32]
size = num2str(Sizes);
EvaluationDirList={['F:\Demo\Data_MLPrediction\NBUDatasetResult\GF1_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\IK_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\QB_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV2_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV3_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV4_',NetName,'\Evaluate_Fu',size]};
% F:\Demo\Data_MLPrediction\NBUDatasetResult\GF1_
% SVM支持向量机 十次 均值+-方差
    for i = 1:10
%         accuracy = ML_SVMcgRandData1(EvaluationDirList,TrainProportion,columnSampleVector);
        accuracy = ML_SVMcgRandData1(EvaluationDirList,TrainProportion,columnSampleVector);
        accuracy = accuracy(1 , :);
        accuracyList(i,:) = accuracy;
        close all hidden;
    end
% Mean = num2str(mean(accuracyList), '%.2f'); % Var = num2str(var(accuracyList), '%.2f');
% MeanVar = [Mean,'±',Var] ;
% formatSpec = ' Mean = %s ！\n';  fprintf(formatSpec, Mean);
% accuracyList = cellstr(accuracyList);% 将字符串数组转换为单元格数组
MeanVarList_01SVM = horzcat(MeanVarList_01SVM, accuracyList);% 将单元格数组连接在一起

% RF随机森林 十次 均值+-方差
    for i = 1:10
        accuracy = ML_RFRandData1(EvaluationDirList,TrainProportion,columnSampleVector);
        accuracy = accuracy(1 , :);
        accuracyList(i,:) = accuracy;
        close all hidden;
    end
% Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
% MeanVar = [Mean,'±',Var] ;
% formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
% accuracyList = cellstr(accuracyList);% 将字符串数组转换为单元格数组
MeanVarList_01RF = horzcat(MeanVarList_01RF, accuracyList);% 将单元格数组连接在一起

end






%% 7.2 Self数据训练，Self数据预测  for循环size

MeanVarList_02SVM = []; MeanVarList_02RF = [];

for Sizes = [1024]  % size = [1024,512,256,128,64,32]
size = num2str(Sizes);
EvaluationDirList={['F:\Demo\Data_MLPrediction\OurDatasetResult\GF1_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\GF2_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\JL1_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\QB_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\WV2_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\WV3_',NetName,'\Evaluate_Fu',size]};

% SVM支持向量机 十次 均值+-方差
    for i = 1:10
        accuracy = ML_SVMcgRandData1(EvaluationDirList,TrainProportion,columnSampleVector);
        accuracy = accuracy(1 , :);
        accuracyList(i,:) = accuracy;
        close all hidden;
    end
% Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
% MeanVar = [Mean,'±',Var] ;
% formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
% MeanVarCell = cellstr(MeanVar);% 将字符串数组转换为单元格数组
MeanVarList_02SVM = horzcat(MeanVarList_02SVM, accuracyList);% 将单元格数组连接在一起
% 
% RF随机森林 十次 均值+-方差
    for i = 1:10
        accuracy = ML_RFRandData1(EvaluationDirList,TrainProportion,columnSampleVector);
        accuracy = accuracy(1 , :);
        accuracyList(i,:) = accuracy;
        close all hidden;
    end
% Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
% MeanVar = [Mean,'±',Var] ;
% formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
% MeanVarCell = cellstr(MeanVar);% 将字符串数组转换为单元格数组
MeanVarList_02RF = horzcat(MeanVarList_02RF, accuracyList);% 将单元格数组连接在一起
% 
end



%% 合并

MeanVarList_Sum = [MeanVarList_01SVM, MeanVarList_01RF, MeanVarList_02SVM, MeanVarList_02RF];




