%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                        说明                               %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% https://www.bilibili.com/video/BV1Rs4y1c7tM/?spm_id_from=333.337.search-card.all.click&vd_source=eb9a5338e2b00e3e19c0a29e3a5acfa5
% 26:36

function [] = ML_RFRand2(EvaluationDirList1,TrainProportion,EvaluationDirList2,TestProportion,columnSample)

    % 分特征数据和标签数据,作为训练集
    [features1,label1,train_id,~,index_id] = MLMatrixRead(EvaluationDirList1,TrainProportion,columnSample);
    % 分特征数据和标签数据,作为测试集
    [features2,label2,test_id,~,index_id] = MLMatrixRead(EvaluationDirList2,TestProportion,columnSample);

    train_data = features1(train_id,index_id);%训练集
    label_train = label1(train_id); %训练标签
    test_data = features2(test_id,index_id); %测试集
    label_test = label2(test_id); %测试标签
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

    %%
    % formatSpec = '训练集比重 %s！%d个图像中第%d个！\n';
    fprintf("——————————————————————————————\n");
    TrainProportion = num2str(TrainProportion);train_id = num2str(train_id);
    formatSpec = '训练集比重 %s,训练集(train_id)随机抽取的列有 %s ！\n';
    fprintf(formatSpec, TrainProportion, train_id);
end