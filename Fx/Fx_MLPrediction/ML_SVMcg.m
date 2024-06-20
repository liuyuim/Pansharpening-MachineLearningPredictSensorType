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


function [] = ML_SVMcg(EvaluationDirList)

    %% 读取数据
    features=[];
    label=[];
    numData = numel(EvaluationDirList);
    for i = 1:numData
        EvaluationDir = EvaluationDirList{i};
        saveName = fullfile(EvaluationDir,'MatrixAll_Fu.mat');
        Matrix_Fu = load(saveName).Matrix_Fu;
        Matrix_Fu = permute(Matrix_Fu,[3,2,1]);    
        Matrix_Fu_Horizontal(:,:,i) = [Matrix_Fu(:,:,1) Matrix_Fu(:,:,2) Matrix_Fu(:,:,3)];
        %特征
        features = vertcat(features, Matrix_Fu_Horizontal(:,:,i));  %使用 vertcat 将第二个矩阵垂直追加到第一个矩阵。
        %类型
        label = vertcat(label, ones(100,1)*i); 
    end
        
    %% 归一化
    temp=mapminmax(features',0,1);%归一化到0-1之间
    featuresn=temp';%转置为列向
    train_id=[1:80, 101:180, 201:280];%训练样本序号
    test_id=[81:100, 181:200, 281:300];%测试样本序号
    
    %% 构建支持向量机
    % cmd=['-c 100 -g 0.1 -s 0 -t 2'];%支持向量机参数设置
    % [bestacc,bestc,bestg] = SVMcgForClass(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
    [bestacc,bestc,bestg] = SVMcgForClass(label(train_id),featuresn(train_id,:),-8,8,-8,8,4,1,1,4.5); %交叉验证
    
    cmd = [' -c ',num2str( bestc ),' -g ',num2str(bestg),'-s 0 -t 2']; %支持向量机参数设置
    model=svmtrain(label(train_id),featuresn(train_id,:),cmd); %对数据进行训练
    % model=libsvmtrain(label,featuresn,cmd);
    
    %% 使用支持向量机预测
    [predict_label,accuracy, decision_values]=svmpredict(label(test_id),featuresn(test_id,:),model);
    % [predict_label,accuracy, prob_estimates]=svmpredict(label(test_id),featuresn(test_id,:),model,'-b probability_estimates');
    
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
end