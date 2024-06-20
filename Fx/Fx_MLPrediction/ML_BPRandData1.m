%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                        说明                               %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% https://www.bilibili.com/video/BV1x24y1F74j/?p=36&spm_id_from=pageDriver
% 27:36

function [] = ML_BPRandData1(EvaluationDirList,TrainProportion,columnSample)

    % 分特征数据和标签数据,分训练集测试集
    [features,label,train_id,test_id,index_id] = MLMatrixRead(EvaluationDirList,TrainProportion,columnSample);

    train_data = features(train_id,index_id);%训练集
    train_label = label(train_id); %训练标签
    test_data = features(test_id,index_id); %测试集
    test_label = label(test_id); %测试标签
    
    train_data = train_data';
    train_label = train_label';
    test_data = test_data';
    
    %%
    %归一化
    % zscore
    [train_data_regular,train_data_maxmin] = mapminmax(train_data);
    [train_label_regular,train_label_maxmin] = mapminmax(train_label);
    %创建网络
    %%调用形式
    % [6,4]第一/二层隐藏层神经元个数,'logsig','tansig',激活函数'purelin'线性全连接输出
    % 隐藏层神经元个数的确定：Nh=Ns/(No+Ni)*a  Ns为训练样本集数量,No为输出神经元（标签一维即是1）,Ni为输入神经元个数（特征维15即是15）,a常常取值2~10
    net=newff(train_data_regular,train_label_regular,[6,4],{'logsig','tansig','purelin'});
    %%
    %常用激活函数 logsig~sigmoid tansig~tanh purelin 线性传递函数 poslin~reLU
    %%激活函数的设置
    % compet - Competitive transfer function.
    % elliotsig - Elliot sigmoid transfer function.
    % hardlim - Positive hard limit transfer function.
    % hardlims - Symmetric hard limit transfer function.
    % logsig - Logarithmic sigmoid transfer function.
    % netinv - Inverse transfer function.
    % poslin - Positive linear transfer function.
    % purelin - Linear transfer function.
    % radbas - Radial basis transfer function.
    % radhasn - Radial hasis normalized transfer fiinction
    % satlin - Positive saturating linear transfer function.
    % satlins - Symmetric saturating linear transfer function.
    % softmax - Soft max transfer function.
    % tansig - Symmetric sigmoid transfer function.
    % tribas - Triangular basis transfer function.
    % net=newff(train_data_regular,train_label_regular,[6,3,3],{'logsig','tansig','logsig','purelin','tansig'
    % net=newff(train_data_regular,train_label_regular,6,{'logsig','logsig'});
    % net=newff(train_data_regular,train_label_regular,6,{'logsig','purelin'});
    % net=newff(train_data_regular,train_label_regular,6,{'logsig','tansig'});
    % %设置训练次数
    % net.trainParam.epochs = 50000;
    % %设置收敛误差
    % net.trainParam.goal=0.000001;
    % newff(P,T,S,TF,BTF,BLF,PF,IPF,OPF,DDF) takes optional inputs,
    %   TF Transfer function of ith layer. Default is 'tansig' for
    %   hidden layers, and "purelin' for output layer.
    
    %训练网络
    [net,~]=train(net,train_data_regular,train_label_regular);
    %%
    %将输入数据归一化
    test_data_regular = mapminmax('apply',test_data,train_data_maxmin);
    
    %%
    %放入到网络输出数据
    test_label_regular=sim(net,test_data_regular);
    %%
    %将得到的数据反归一化得到预测数据
    BP_predict=mapminmax('reverse',test_label_regular,train_label_maxmin);
    BP_predict=BP_predict';
    %计算平均绝对值误差MAPE
    errors_nn=sum(abs(BP_predict-test_label)./(test_label))/length(test_label);
    figure(1)
    color=[111,168,86;128,199,252;112,138,248;184,84,246]/255;
    plot(test_label,'Color',color(2,:),'LineWidth',1)
    hold on
    plot(BP_predict,'*','Color',color(1,:))
    hold on
    titlestr=['MATLAB自带BP神经网络',' 误差为: ' ,num2str(errors_nn)];
    title(titlestr)
    
    %%
    % formatSpec = '训练集比重 %s！%d个图像中第%d个！\n';
    fprintf("——————————————————————————————\n");
    TrainProportion = num2str(TrainProportion);train_id = num2str(train_id);
    formatSpec = '训练集比重 %s,训练集(train_id)随机抽取的列有(其余列是测试集) %s ！\n';
    fprintf(formatSpec, TrainProportion, train_id);

end