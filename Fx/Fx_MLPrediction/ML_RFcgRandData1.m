%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                        说明                               %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% https://www.bilibili.com/video/BV1Rs4y1c7tM/?spm_id_from=333.337.search-card.all.click&vd_source=eb9a5338e2b00e3e19c0a29e3a5acfa5
% 26:36

function [accuracy] = ML_RFcgRand(EvaluationDirList,TrainProportion,columnSample)

    % 分特征数据和标签数据,分训练集测试集
    [features,label,train_id,test_id,index_id] = MLMatrixRead(EvaluationDirList,TrainProportion,columnSample);

    train_data = features(train_id,index_id); %训练集
    label_train = label(train_id); %训练标签
    % train_data = [train_data,label_train];
    test_data = features(test_id,index_id); %测试集
    label_test = label(test_id); %测试标签
    %%
    ntree = 100;
    %
    % X = table(features(train_id,index_id));
    X = array2table(train_data);
    Y = array2table(label_train);
    rng('default'); % For reproducibility

    maxMinLS = 40; %20; %随机森林中树的复杂度（深度）。复杂的树会过拟合，简单的树拟合效果不佳。 因此，指定每个叶片的最小观测次数不超过20次。
    minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
    numPTS = optimizableVariable('numPTS',[1,size(X,2)-1],'Type','integer'); %当树生长时，每个节点上要采样的预测数量。指定从1到所有预测期进行抽样。
    hyperparametersRF = [minLS; numPTS]; %bayesopt函数用于实现贝叶斯优化，使用该函数要求将上述指定调整参数以optimizableVariable对象传递。

    % results = bayesopt(@(params)oobErrRF(params,X,ntree),hyperparametersRF, 'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);
    % bestOOBErr = results.MinObjective;
    % bestHyperparameters = results.XAtMinObjective;

    results = bayesopt(@(params)classificationErrorRF(params,X,Y),hyperparametersRF, 'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);

    bestClassificationError = results.MinObjective
    bestHyperparameters = results.XAtMinObjective

    %构建参数优化回归模型
    RF_Model = TreeBagger(ntree,train_data,label_train,'Method','classification','MinLeafSize',bestHyperparameters.minLS,'NumPredictorstoSample',bestHyperparameters.numPTS,'OOBPredictorImportance','on');
    % RF_Model = TreeBagger(ntree,train_data,label_train,'Method','classification','MinLeafSize',bestHyperparameters.minLS,'NumPredictorstoSample',bestHyperparameters.numPTS);
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