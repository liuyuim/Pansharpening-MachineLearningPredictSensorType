

function [] = MeanVarList_Vote3time (NetName,TrainProportion,TestProportion,columnSampleVector)
%% 7.1 NBU数据训练，NBU数据预测  for循环size

NumT_SumSVM = 0; NumF_SumSVM = 0; NumT_SumRF = 0; NumF_SumRF = 0;
for Sizes = [32]  % size = [1024,512,256,128,64,32]
size = num2str(Sizes);
EvaluationDirList={['F:\Demo\Data_MLPrediction\NBUDatasetResult\GF1_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\IK_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\QB_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV2_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV3_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV4_',NetName,'\Evaluate_Fu',size]};
% D:\Shiyan\20231201\GF1_
% SVM支持向量机 十次 均值+-方差
    for i = 1:10
        [NumT,NumF] = ML_SVMcgRandData1Vote3time(EvaluationDirList,TrainProportion,columnSampleVector);
        NumT_SumSVM = NumT_SumSVM + NumT;
        NumF_SumSVM = NumF_SumSVM + NumF;
        close all hidden;
    end
% Mean = num2str(mean(accuracyList), '%.2f'); % Var = num2str(var(accuracyList), '%.2f');
% MeanVar = [Mean,'±',Var] ;
% formatSpec = ' Mean = %s ！\n';  fprintf(formatSpec, Mean);
% accuracyList = cellstr(accuracyList);% 将字符串数组转换为单元格数组
accuracy_NBU_SVM = num2str( NumT_SumSVM / (NumT_SumSVM + NumF_SumSVM), '%.4f');


% RF随机森林 十次 均值+-方差
    for i = 1:10
        [NumT,NumF] = ML_RFRandData1Vote3time(EvaluationDirList,TrainProportion,columnSampleVector);
        NumT_SumRF = NumT_SumRF + NumT;
        NumF_SumRF = NumF_SumRF + NumF;
        close all hidden;
    end
% Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
% MeanVar = [Mean,'±',Var] ;
% formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
% accuracyList = cellstr(accuracyList);% 将字符串数组转换为单元格数组
accuracy_NBU_RF = num2str( NumT_SumRF / (NumT_SumRF + NumF_SumRF), '%.4f');
end






%% 7.2 Self数据训练，Self数据预测  for循环size

for Sizes = [32]  % size = [1024,512,256,128,64,32]
size = num2str(Sizes);
EvaluationDirList={['F:\Demo\Data_MLPrediction\OurDatasetResult\GF1_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\GF2_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\JL1_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\QB_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\WV2_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\WV3_',NetName,'\Evaluate_Fu',size]};


% SVM支持向量机 十次 均值+-方差
    for i = 1:10
        [NumT,NumF] = ML_SVMcgRandData1Vote3time(EvaluationDirList,TrainProportion,columnSampleVector);
        NumT_SumSVM = NumT_SumSVM + NumT;
        NumF_SumSVM = NumF_SumSVM + NumF;
        close all hidden;
    end
% Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
% MeanVar = [Mean,'±',Var] ;
% formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
% MeanVarCell = cellstr(MeanVar);% 将字符串数组转换为单元格数组
accuracy_Self_SVM = num2str( NumT_SumSVM / (NumT_SumSVM + NumF_SumSVM), '%.4f');

% 
% RF随机森林 十次 均值+-方差
    for i = 1:10
        [NumT,NumF] = ML_RFRandData1Vote3time(EvaluationDirList,TrainProportion,columnSampleVector);
        NumT_SumRF = NumT_SumRF + NumT;
        NumF_SumRF = NumF_SumRF + NumF;
        close all hidden;
    end
% Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
% MeanVar = [Mean,'±',Var] ;
% formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
% MeanVarCell = cellstr(MeanVar);% 将字符串数组转换为单元格数组
accuracy_Self_RF = num2str( NumT_SumRF / (NumT_SumRF + NumF_SumRF), '%.4f');
 

% 
end



%% 合并

formatSpec = ' NBU_SVM  PanSize 预测 accuracy  = %s ！\n';  fprintf(formatSpec, accuracy_NBU_SVM);  
formatSpec = ' NBU_RF PanSize 预测 accuracy  = %s ！\n';  fprintf(formatSpec, accuracy_NBU_RF);  
formatSpec = ' Self_SVM  PanSize 预测 accuracy  = %s ！\n';  fprintf(formatSpec, accuracy_Self_SVM);  
formatSpec = ' Self_RF  PanSize 预测 accuracy  = %s ！\n';  fprintf(formatSpec, accuracy_Self_RF); 

