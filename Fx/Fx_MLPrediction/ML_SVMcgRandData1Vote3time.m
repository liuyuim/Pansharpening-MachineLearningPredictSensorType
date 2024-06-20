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
% Accuracy = 86.6667% (234/270) (labelification)

% 以上内容参考https://www.bilibili.com/video/BV1HT4y12727/?spm_id_from=333.337.search-card.all.click&vd_source=eb9a5338e2b00e3e19c0a29e3a5acfa5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [NumT,NumF] = ML_SVMcgRandData1Vote3time(EvaluationDirList,TrainProportion,columnSample)

    % 分特征数据和标签数据,分训练集测试集
    [features,label,train_id,test_id,index_id] = MLMatrixRead(EvaluationDirList,TrainProportion,columnSample);
    
    % 归一化 
    temp=mapminmax(features',0,1);%归一化到0-1之间
    featuresn=temp';%转置为列向
    
    %% 构建支持向量机
    % cmd=['-c 100 -g 0.1 -s 0 -t 2'];%支持向量机参数设置
    % [bestacc,bestc,bestg] = SVMcgForClass(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
    [bestacc,bestc,bestg] = SVMcgForClass(label(train_id),featuresn(train_id,index_id),-8,8,-8,8,4,1,1,4.5); %交叉验证
    
    cmd = [' -c ',num2str( bestc ),' -g ',num2str(bestg),'-s 0 -t 2']; %支持向量机参数设置
    model=svmtrain(label(train_id),featuresn(train_id,index_id),cmd); %对数据进行训练
    % model=libsvmtrain(label,featuresn,cmd);
    
    %% 使用支持向量机预测
    
    [predict_label, accuracy, decision_values]=svmpredict(label(test_id),featuresn(test_id,index_id),model);
    % [predict_label,accuracy, prob_estimates]=svmpredict(label(test_id),featuresn(test_id,:),model,'-b probability_estimates');
    
% 假设你已经有了预测结果 label(test_id)
% 假设你已经有了真实标签 true_labels(test_id)

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
    predicted_labels = predict_label(random_indices);%
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
    
    
    %% 计算TP TN PP PN
    % TP=0; TN=0; FP=0; FN=0; %%真正例 真反例 假正例 假反例
    % IF=[predict_label label(test_id)];
    % for i = 1:length(predict_label)
    %     if (IF(i,2)==1&TF(i,1)==1)
    %         TP=TP+1;
    %     end
    %     if (IF(i,2)==1&TF(i,1)==0)
    %         FN=FN+1;
    %     end
    %     if (IF(i,2)==0&TF(i,1)==0)
    %         TN=TN+1;
    %     end
    %     if (IF(i,2)==0&IF(i,1)==1)
    %         FP=FP+1;
    %     end
    % end

    %% 计算Acc. Spe. Sen.
    % Acc=((TP+TN)/(TP+TN+FP+FN))*100; %%准确度
    % Sen=(TP/(TP+FN))*100; %%灵敏度
    % Spe=(TN/(TN+FP))*100; %%特效度

    %% 预测结果作图
    figure
    plot(label(test_id),'bo')
    hold on
    plot(predict_label,'r*')
    grid on
    xlabel('样本序号')
    ylabel('类型')
    legend('实际类型','预测类型')
    firstline = 'The labelify of breast cancer';
    % secondline = ['TP=',num2str (TP),' ','TN=',num2str (TN),' ','FP=',num2str(FP),' ','FN=',nun2str (FN),' '];
    % thirdline = ['Acc.=',nun2str(Acc),'',' ','Spe.=',nun2str(Spe),'%',' ','Sen.=', nun2str(Sen),'%'];
    % title({firstline:secondline:thirdline},'Fontsize',12);
    title({firstline},'Fontsize',12);

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

    %% 混涌矩阵
    figure
    cm = confusionchart(label(test_id), predict_label);
    cm.Title = 'Confusion Matrix for Test Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';

    %%
    % formatSpec = '训练集比重 %s！%d个图像中第%d个！\n';
    fprintf("——————————————————————————————\n");
    TrainProportion = num2str(TrainProportion);train_id = num2str(train_id);
    formatSpec = '训练集比重 %s,训练集(train_id)随机抽取的列有 %s ！\n';
    fprintf(formatSpec, TrainProportion, train_id);
end