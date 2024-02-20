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

% 
% Input:
% Proportion 定义比例，= 1时选取所有
% columnSample 指标所在的列号作为样例

% Output：
% features 选哪些列,
% label 生成的不变的,
% rand_id 随机数组,
% surplus_id 随机数组后剩余的数组,
% index_id 根据columnSample扩展出所有入选的列号


function [features,label,rand_id,surplus_id,index_id] = MLMatrixRead(EvaluationDirList,Proportion,columnSample)

    while (Proportion < 0 || Proportion > 1)
        fprintf("Proportion取值范围(0,1)!");
        break;
    end
    
    
    %% 读取数据
    % data = xlsread('Metrics.xlsx','Sheet1','A1:N178');
    % Numcolumns = size(Matrix_Fu_GF_Horizontal,1);
    % features = [Matrix_Fu_Horizontal(1:Numcolumns,:,1); Matrix_Fu_Horizontal(1:Numcolumns,:,2); Matrix_Fu_Horizontal(1:Numcolumns,:,3)]; %特征
    % features = vertcat(features,Matrix_Fu_Horizontal(:,:,i));
    % class = [ones(100,1); ones(100,1)*2; ones(100,1)*3]; %类型  

    features=[];
    label=[];    
    numHypothesis = numel(EvaluationDirList);
    for i = 1:numHypothesis
        EvaluationDir = EvaluationDirList{i};
        saveName = fullfile(EvaluationDir,'MatrixAll_Fu.mat');
        Matrix_Fu = load(saveName).Matrix_Fu;
        
        % 判断下EvaluationDir数量和Matrix_Fu里HypothesisDir数量是否一致
        if ~isequal(numHypothesis,size(Matrix_Fu, 1))
            fprintf("EvaluationDir数量和Matrix_Fu里HypothesisDir数量不一致");
            break;
        end
        % 维度转换
        Matrix_Fu = permute(Matrix_Fu,[3,2,1]);    
        % 得到Test1的Matrix_Fu如下所示   
        %   /   ┌index1... indexN(HypoN)┐ 
        %  /    │p1   *         *       │
        % ┌  index1... indexN(Hypo1)┐───┘ 
        % │p1   *         *         │
        % └pN───────────────────────┘
        % 
        % Matrix_Fu_Horizontal(:,:,i) = [Matrix_Fu(:,:,1) Matrix_Fu(:,:,2) Matrix_Fu(:,:,3)];Matrix_Fu_Horizontal(:,:,i) = 
        Matrix_Fu_Horizontal=[];   % Matrix_Fu_Horizontal当临时中间变量，每个测试集循环完就清零
            for j = 1:numHypothesis                
                Matrix_Fu_Horizontal = horzcat(Matrix_Fu_Horizontal,Matrix_Fu(:,:,j));
            end
        % 得到Test1的Matrix_Fu_Horizontal如下所示   
        % ┌  index1... indexN(Hypo1) index1... indexN(HypoN)┐
        % │p1   *         *         │  *          *         │
        % └pN───────────────────────────────────────────────┘
        
        %特征
        features = vertcat(features, Matrix_Fu_Horizontal); % 将后个矩阵垂直追加到前个矩阵。   
        
        % 得到的features如下所示   
        % ┌  index1... indexN(Hypo1) index1... indexN(HypoN)┐
        % │p1   *         *         │  *          *         │—— Test1
        % └pN───────────────────────────────────────────────┘
        % ...
        % ...
        % │p1   *         *         │  *          *         │—— TestN
        % └pN───────────────────────────────────────────────┘
        %类型
        % label = vertcat(label, ones(100,1)*i);(Matrix_Fu,[3,2,1])
        label = vertcat(label, ones(size(Matrix_Fu,1),1)*i);
    end
    
    %% 分训练集测试集
    
    % train_id=[1:50, 151:200, 301:350, 451:500, 601:650, 751:800];%训练样本序号
    % test_id=[51:150, 201:300, 351:450, 501:600, 651:750, 801:900];%测试样本序号
    % train_id=[1:80, 101:180, 201:280];%训练样本序号
    % test_id=[81:100, 181:200, 281:300];%测试样本序号

    %%在50行的矩阵中抽取40行%by cyl
    % M=rand(50,10);%随机生成50*10的矩阵
    % n=randsample(50,40,'false');%随机在1-50个数值之间抽取40个数，并放入n中
    % A=M(n,:)%A为抽取40行后的矩阵
    % c=1:50;%定义c为1-50的列矩阵
    % c(n)=[];%去掉抽取的n行
    % B=M(c,:)%B为抽取剩下的矩阵

    % Proportion是0~1之间
    if (Proportion ~= 1)
        PerTestNum = size(Matrix_Fu,1); % 每组测试集图片数量
        nSample=randsample(PerTestNum,Proportion * PerTestNum,'false');%随机在。。个数值之间抽取。。个数，并放入n中
        nSample = nSample';
        rand_id = nSample;
        for i = 1 : numHypothesis-1
            nSample = nSample + PerTestNum;
            rand_id = [rand_id,nSample];% 训练集图片索引
        end
        
        endIdx = numHypothesis * PerTestNum;  % 所有索引最后一个号码 = 假设文件夹个数 * 每组测试集图片数量
        allIndices = 1:endIdx;% 获取当前组的所有图片索引    
        surplus_id = setdiff(allIndices, rand_id);% 计算测试集图片索引
    
    % Proportion是0~1之间
    elseif(Proportion == 1)
        rand_id = ':';
        surplus_id = ':';
    end
    
    %% 数据集指标索引

    PerColumnNum = size(Matrix_Fu,2); % 每组数据集指标数量    
    if (columnSample == 0)
        index_id = ':';

    else
        index_id = columnSample;
        for i = 1 : numHypothesis-1
            columnSample = columnSample + PerColumnNum;
            index_id = [index_id,columnSample];% 数据集指标索引
        end
    end
end