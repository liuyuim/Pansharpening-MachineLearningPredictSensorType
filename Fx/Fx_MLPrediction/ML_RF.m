%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                        说明                               %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% https://www.bilibili.com/video/BV1Rs4y1c7tM/?spm_id_from=333.337.search-card.all.click&vd_source=eb9a5338e2b00e3e19c0a29e3a5acfa5
% 26:36

function [] = ML_RF(EvaluationDirList)

    features=[];
    class=[];
    numData = numel(EvaluationDirList);
    for i = 1:numData
        EvaluationDir = EvaluationDirList{i};
        saveName = fullfile(EvaluationDir,'MatrixAll_Fu.mat');
        Matrix_Fu = load(saveName).Matrix_Fu;
        Matrix_Fu = permute(Matrix_Fu,[3,2,1]);    
        Matrix_Fu_Horizontal(:,:,i) = [Matrix_Fu(:,:,1) Matrix_Fu(:,:,2) Matrix_Fu(:,:,3)];
        %特征
        features = vertcat(features, Matrix_Fu_Horizontal(:,:,i)); %使用 vertcat 将第二个矩阵垂直追加到第一个矩阵。
        %类型
        class = vertcat(class, ones(100,1)*i);
    end
    %%
    % train_id=[1:80, 101:180, 201:280];%训练样本序号
    % test_id=[81:100, 181:200, 281:300];%测试样本序号
    train_id=[1:80, 101:180, 201:280, 301:380];%训练样本序号
    test_id=[81:100, 181:200, 281:300, 381:400];%测试样本序号
    % train_id=[1:50, 151:200, 301:350, 451:500, 601:650, 751:800];%训练样本序号
    % test_id=[51:150, 201:300, 351:450, 501:600, 651:750, 801:900];%测试样本序号
    train_data = features(train_id,:);%训练集
    label_train = class(train_id); %训练标签
    test_data = features(test_id,:); %测试集
    label_test = class(test_id); %测试标签
    %%
    ntree=100;
    %构建参数优化回归模型
    RF_Model = TreeBagger(ntree,train_data,label_train,'Method','classification','OOBPredictorImportance','on');
    
    %%
    % 进行预测
    predict_label= predict(RF_Model, test_data);
    %%
    for i=1:length(predict_label)
        predict_label1(i)=str2double(predict_label{i,1});
    end
    
    %%
    num=0;
    for i= 1:length(label_test)
        a=predict_label{i,1};
        if str2double(a)==label_test(i,1)
            num=num+1;
        end
    end
    %%
    accuracy=num/length(label_test);
    disp('RF分类正确率为')
    disp(accuracy)
    %%
    imp = RF_Model.OOBPermutedPredictorDeltaError;
    figure;
    bar(imp);
    title('Curvature Test');
    ylabel('Predictor importance estimates');
    xlabel('Predictors');
    h = gca;
    h.TickLabelInterpreter = 'none';
    
    figure
    cm = confusionchart(predict_label1, label_test);
    cm.Title = ' Confusion Matrix for Test Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    view(RF_Model.Trees{1},'Mode','graph')
end