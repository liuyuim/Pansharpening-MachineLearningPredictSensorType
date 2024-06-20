%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                        说明                               %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% matlab的libsvm工具箱安装教程
% 官网 https://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html
% 32才需要编译，64位不需要编译。我是64位。
% (1)直接将解压后的libsvm-3.32文件夹放到matlab的toolbox中，可以从快捷方式右键打开文件所在位置
% (2)在matlab中"设置路径"——"添加并包含子文件夹"——"保存"——"关闭"
% (3)在matlab中"预设"——"常规""——"更新工具箱路径缓存"——"确定"
% 然后就可以验证了。验证的三条语句：
% 例如我的路径是这个，你们要换成自己的C:\Program Files\MATLAB\R2023a\toolbox\libsvm-3.32
% >> [heart_scale_label,heart_scale_inst]=libsvmread('C:\Program Files\MATLAB\R2023a\toolbox\libsvm-3.32\heart_scale');
% >> model = svmtrain(heart_scale_label,heart_scale_inst, '-c 1 -g 0.07');
% >> [predict_label, accuracy, dec_values] =svmpredict(heart_scale_label, heart_scale_inst, model);
% 会出现
% optimization finished, #iter = 134
% nu = 0.433785
% obj = -101.855060, rho = 0.426412
% nSV = 130, nBSV = 107
% Total nSV = 130
% Accuracy = 86.6667% (234/270) (classification)

% 以上内容参考https://www.bilibili.com/video/BV1HT4y12727/?spm_id_from=333.337.search-card.all.click&vd_source=eb9a5338e2b00e3e19c0a29e3a5acfa5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = ML_SVMRandData2ComboFeature(EvaluationDirList1,TrainProportion,EvaluationDirList2,TestProportion,columnSample)
format longG
    %% 分特征数据和标签数据,分训练集测试集 利用MLMatrixRead获取特征不要index_id
    % 分特征数据和标签数据,作为训练集
    [features1,label1,train_id,~,index_id] = MLMatrixRead(EvaluationDirList1,TrainProportion,columnSample);
    % 分特征数据和标签数据,作为测试集
    [features2,label2,test_id,~,index_id] = MLMatrixRead(EvaluationDirList2,TestProportion,columnSample);
    
    % 归一化
    temp1 = mapminmax(features1',0,1);%归一化到0-1之间
    featuresn1 = temp1';%转置为列向
    temp2 = mapminmax(features2',0,1);%归一化到0-1之间
    featuresn2 = temp2';%转置为列向
    

    %% 特征组合
    [ComboSave] = GenerateFeaturesCombinations(columnSample);
    NumComboSave = size(ComboSave,1);
    accuracyMatrix = cell(NumComboSave, 4);
    best_accuracy = 0;
    best_columnSample = {};
    

    for i = 1:NumComboSave
        columnSample = ComboSave{i,:};        

        %% 利用MLMatrixRead获取index，不要features,label,train_id,test_id
        [~,~,~,~,index_id] = MLMatrixRead(EvaluationDirList1,TrainProportion,columnSample);

        %% 构建支持向量机
        % cmd=['-c 100 -g 0.1 -s 0 -t 2'];%支持向量机参数设置
        cmd=['-c 100 -g 0.1'];%支持向量机参数设置
        model = svmtrain(label1(train_id),featuresn1(train_id,index_id),cmd); %对数据进行训练
        % model=libsvmtrain(class,featuresn,cmd);
        %% 使用支持向量机预测
        [predict_label, accuracy, decision_values]=svmpredict(label2(test_id),featuresn2(test_id,index_id),model);
    
        %% 预测结果作图
        % figure
        % plot(label(test_id),'bo')
        % hold on
        % plot(predict_label,'r*')
        % grid on
        % xlabel('样本序号')
        % ylabel('类型')
        % legend('实际类型','预测类型')
        % set(gca,'fontsize',12)
        % 
        %% 混淆矩阵
        % figure
        % cm = confusionchart(label(test_id), predict_label);
        % cm.Title = 'Confusion Matrix for Test Data';
        % cm.ColumnSummary = 'column-normalized';
        % cm.RowSummary = 'row-normalized';
        % 
        %%
        % % formatSpec = '训练集比重 %s！%d个图像中第%d个！\n';
        % fprintf("——————————————————————————————\n");
        % TrainProportion = num2str(TrainProportion);train_id = num2str(train_id);
        % formatSpec = '训练集比重 %s,训练集(train_id)随机抽取的列有 %s,测试集抽取的列有 %s ！\n';
        % fprintf(formatSpec, TrainProportion, train_id, test_id);
        
        %%
        accuracy = accuracy(1 , :);
        accuracyMatrix{i,1} = columnSample;
        accuracyMatrix{i,2} = train_id;
        accuracyMatrix{i,3} = test_id;
        accuracyMatrix{i,4} = accuracy;

        while accuracy > best_accuracy
            best_accuracy = accuracy;
            best_columnSample = columnSample;
        end
    
    end
    
    %% 保存       
    saveDir = fullfile(fileparts(fileparts(EvaluationDirList2{1})),'AccuracyMatrix');
    if ~exist(saveDir,'dir')%待保存的图像文件夹不存在，就建文件夹
        mkdir(saveDir)
    end
    saveName = strcat("ML_SVMRandData2ComboFeature",  string(datetime, 'yyyy-MM-dd-HH-mm-ss'), '.mat');
    saveName = fullfile(saveDir,saveName);

    save(saveName, 'accuracyMatrix');  

    TrainProportion = num2str(TrainProportion);
    best_accuracy = num2str(best_accuracy);
    % best_accuracy = sprintf('%.2f%%', best_accuracy);
    best_columnSample = num2str(best_columnSample);
    formatSpec = '训练集比重 %s, 当组合指标为 %s 时, 获得best_accuracy为 %s ,所有尝试记录见accuracyMatrix %s！\n';
    fprintf(formatSpec, TrainProportion, best_columnSample, best_accuracy,saveName);
end