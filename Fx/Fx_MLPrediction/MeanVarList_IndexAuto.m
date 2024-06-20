

function [MeanVarList_Sum] = MeanVarList_IndexAuto (NetName,TrainProportion,TestProportion,columnSample)
%% 7.1 Self数据训练，Self数据预测  for循环size

MeanVarList_01SVM = []; MeanVarList_01RF = [];

for Sizes = [1024,512,256,128,64,32]  % size = 1024;
size = num2str(Sizes);
EvaluationDirList={['F:\Demo\Data_MLPrediction\NBUDatasetResult\GF1_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\IK_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\QB_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV2_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV3_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\NBUDatasetResult\WV4_',NetName,'\Evaluate_Fu',size]};

% SVM支持向量机 十次 均值+-方差
    for i = 1:10
        accuracy = ML_SVMcgRandData1(EvaluationDirList,TrainProportion,columnSample);
        accuracy = accuracy(1 , :);
        accuracyList(i,:) = accuracy;
        close all hidden;
    end
Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
MeanVar = [Mean,'±',Var] ;
formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
MeanVarCell = cellstr(MeanVar);% 将字符串数组转换为单元格数组
MeanVarList_01SVM = horzcat(MeanVarList_01SVM, MeanVarCell);% 将单元格数组连接在一起

% RF随机森林 十次 均值+-方差
    for i = 1:10
        accuracy = ML_RFRandData1(EvaluationDirList,TrainProportion,columnSample);
        accuracy = accuracy(1 , :);
        accuracyList(i,:) = accuracy;
        close all hidden;
    end
Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
MeanVar = [Mean,'±',Var] ;
formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
MeanVarCell = cellstr(MeanVar);% 将字符串数组转换为单元格数组
MeanVarList_01RF = horzcat(MeanVarList_01RF, MeanVarCell);% 将单元格数组连接在一起

end
fprintf("程序结束，请在 工作区 点击对应变量查看，变量中从左到右依次是循环的Size的对应结果。");





%% 7.1 Self数据训练，Self数据预测  for循环size

MeanVarList_02SVM = []; MeanVarList_02RF = [];

for Sizes = [1024,512,256,128,64,32]  % size = 1024;
size = num2str(Sizes);
EvaluationDirList={['F:\Demo\Data_MLPrediction\OurDatasetResult\GF1_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\GF2_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\JL1_',NetName,'\Evaluate_Fu',size],... 
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\QB_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\WV2_',NetName,'\Evaluate_Fu',size],...
                    ['F:\Demo\Data_MLPrediction\OurDatasetResult\WV3_',NetName,'\Evaluate_Fu',size]};

% SVM支持向量机 十次 均值+-方差
    for i = 1:10
        accuracy = ML_SVMcgRandData1(EvaluationDirList,TrainProportion,columnSample);
        accuracy = accuracy(1 , :);
        accuracyList(i,:) = accuracy;
        close all hidden;
    end
Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
MeanVar = [Mean,'±',Var] ;
formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
MeanVarCell = cellstr(MeanVar);% 将字符串数组转换为单元格数组
MeanVarList_02SVM = horzcat(MeanVarList_02SVM, MeanVarCell);% 将单元格数组连接在一起

% RF随机森林 十次 均值+-方差
    for i = 1:10
        accuracy = ML_RFRandData1(EvaluationDirList,TrainProportion,columnSample);
        accuracy = accuracy(1 , :);
        accuracyList(i,:) = accuracy;
        close all hidden;
    end
Mean = num2str(mean(accuracyList), '%.2f');  Var = num2str(var(accuracyList), '%.2f');
MeanVar = [Mean,'±',Var] ;
formatSpec = ' Mean ± Var = %s ！\n';  fprintf(formatSpec, MeanVar);
MeanVarCell = cellstr(MeanVar);% 将字符串数组转换为单元格数组
MeanVarList_02RF = horzcat(MeanVarList_02RF, MeanVarCell);% 将单元格数组连接在一起

end
fprintf("程序结束，请在 工作区 点击对应变量查看，变量中从左到右依次是循环的Size的对应结果。");


%% 合并

MeanVarList_Sum = [MeanVarList_01SVM; MeanVarList_01RF; MeanVarList_02SVM; MeanVarList_02RF];




