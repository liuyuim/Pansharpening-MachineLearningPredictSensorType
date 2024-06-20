%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                        说明                               %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% https://www.bilibili.com/video/BV1Rs4y1c7tM/?spm_id_from=333.337.search-card.all.click&vd_source=eb9a5338e2b00e3e19c0a29e3a5acfa5
% 26:36

function [NumT,NumF] = ML_RFRandData1Vote3time(EvaluationDirList,TrainProportion,columnSample)

    % 分特征数据和标签数据,分训练集测试集
    [features,label,train_id,test_id,index_id] = MLMatrixRead(EvaluationDirList,TrainProportion,columnSample);

    train_data = features(train_id,index_id);%训练集
    label_train = label(train_id); %训练标签
    test_data = features(test_id,index_id); %测试集
    label_test = label(test_id); %测试标签
    %%
    ntree = 100;

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
    % accuracy=num/length(label_test);
    accuracy=(num/length(label_test)) * 100; % 将小数点右移两位
    disp('RF分类正确率为(%)')
    disp(accuracy)
    
    %%
    % 定义每个子集的大小
subset_size = 20;
% 初始化正确和错误的预测数量
NumT = 0;
NumF = 0;
% 循环遍历不同的子集
for subset_start = 1:subset_size:numel(test_id)
    subset_end = min(subset_start+subset_size-1, numel(test_id));
    
    % 从当前子集中随机选择3个样本
%     num_samples = subset_end - subset_start + 1;    random_indices = randperm(num_samples, 3) + subset_size; % 20*6=120里随机选择3个样本
    random_indices = randsample(subset_start:subset_end, 3); % 从subset_start到subset_end区间取三个数
    
    % 获取这3个样本的索引
    sample_indices = test_id(random_indices);
    
    % 获取这3个样本的预测结果
    predicted_labels = predict_label1(random_indices);%
    predicted_labels = predicted_labels';
    true_labels = label(sample_indices);

    % 进行投票统计
    unique_labels = unique(predicted_labels);
    votes = zeros(size(unique_labels));
    for i = 1:numel(unique_labels)
        votes(i) = sum(predicted_labels == unique_labels(i));
    end

    % 找出得票最多的标签
    [max_votes, max_index] = max(votes);

    % 最终预测结果
    final_prediction = unique_labels(max_index);

    % 检查最终结果是否与真实标签匹配   
    if max_votes >= 2 %统一的票数
        % 检查是否有至少一个预测结果与真实标签匹配
        if any(final_prediction == unique(true_labels))
            disp(['子集起始索引为', num2str(subset_start), '的预测正确']);
            NumT = NumT + 1;
        else
            disp(['子集起始索引为', num2str(subset_start), '的预测错误']);
            NumF = NumF + 1;
        end
    else
        disp(['子集起始索引为', num2str(subset_start), '的预测错误']);
        NumF = NumF + 1;
    end
end
    
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