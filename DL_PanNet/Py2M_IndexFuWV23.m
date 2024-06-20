function [ sensorPre] = Py2M_ForecastFuDR ()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 【评价指标准确率统计】
% 统计每一个指标 最优值是不是出现在 应该的传感器影像上
% 对比方法采用 Pansharpening Tool ver 1.3中的方法；
% 对比指标主要是
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
close all
% IndexFuDR
% fprintf("======对若干假设模型下融合影像评价并返回预测结果！====== \n");

% WV2
TestOutputYijiPath='..\..\Tmp\IndexStatistics100_jianshe\WV2_1\HypothesisOutput\'; %测试集经过深度学习test代码得出的五个假设融合结果，
TestDataYijiPath='..\..\Tmp\IndexStatistics100_jianshe\WV2_1\TestData_Fu\'; %测试集
saveDir = '..\..\Tmp\IndexStatistics100_jianshe\WV2_1\PannetEvaluate\';%设置对应保存路径
IndexFuDR (TestOutputYijiPath,TestDataYijiPath,saveDir);
% IndexStatisticsFuDR
Index_Sensor = "WV2";
IndexStatisticsFuDR (saveDir);

% WV3
TestOutputYijiPath='..\..\Tmp\IndexStatistics100_jianshe\WV3_1\HypothesisOutput\'; %测试集经过深度学习test代码得出的五个假设融合结果，
TestDataYijiPath='..\..\Tmp\IndexStatistics100_jianshe\WV3_1\TestData_Fu\'; %测试集
saveDir = '..\..\Tmp\IndexStatistics100_jianshe\WV3_1\PannetEvaluate\';%设置对应保存路径
IndexFuDR (TestOutputYijiPath,TestDataYijiPath,saveDir);
% IndexStatisticsFuDR
Index_Sensor = "WV3";
IndexStatisticsFuDR (saveDir);
%% 
% 创建包含局部函数的脚本local functions in scripts
    function f_Index = IndexFuDR (TestOutputYijiPath,TestDataYijiPath,saveDir)
    
    % TestData是不变的，利用它得到NumImgs
    TestDataErjiDir_Path = fullfile(TestDataYijiPath); %.\DataDL_1TestData
    TestData_list = dir([TestDataErjiDir_Path,'\','*.mat']) ;        
    NumImgs = size(TestData_list,1);

    % 用TestOutputYijiPath 遍历出二级目录名，共用 GF1_GengDi GF1_LinDi   GF1_WeiBiaoDuoLei ...
    ErjiDir_list = dir(TestOutputYijiPath) ;  % 二级目录列表
    ErjiDir_list_Nums = size(ErjiDir_list,1);  % 二级目录个数 包括 .和..
    
    % 定义一个矩阵
    Matrix_Fu = zeros(ErjiDir_list_Nums-2,5,NumImgs); % Matrix_Fu = zeros(5,5,100);

    for i_ErjiDir = 3 : ErjiDir_list_Nums
        %列出当前二级文件夹内所有的mat
        TestOutputErjiDir_Path = fullfile(TestOutputYijiPath,ErjiDir_list(i_ErjiDir).name); %.\DataDL_PannetOutput\GF1_GengDi
        TestOutput_list = dir([TestOutputErjiDir_Path,'\','*.mat']) ;
        
        % 在当前二级目录处理每一个mat

        for i_NumImgs = 1:NumImgs
        
            formatSpec = '正在处理二级目录 %s！%d个图像中第%d个！\n';
            fprintf(formatSpec,ErjiDir_list(i_ErjiDir).name, NumImgs, i_NumImgs);
    
            % 校验 当前从 TestOutput和TestData文件夹 分别取出的 mat文件名 是否一致
            %验证两者是否一致
            if ~isequal(TestOutput_list(i_NumImgs).name, TestData_list(i_NumImgs).name)
                fprintf("当前从 TestOutput和TestData文件夹分别取出的 mat文件名 不一致");
                break;
            end
            
            % 然后再正常运行
            
            %把mat文件加载进来
            TestOutputPath = [TestOutput_list(i_NumImgs).folder,'\',TestOutput_list(i_NumImgs).name]; %TestOutput_list列表中的第i个目录和文件名拼成要加载的mat路径 如 E:\LiuYu\FusionEvaluateExperiment\DataDL_PannetOutput\j1p1.mat
            TestOutputDate = load(TestOutputPath); 
            TestDataPath = [TestData_list(i_NumImgs).folder,'\',TestData_list(i_NumImgs).name]; %TestData_list列表中的第i个目录和文件名拼成要加载的mat路径  E:\LiuYu\FusionEvaluateExperiment\DataDLPre_1TestData\j1p1.mat
            TestData = load(TestDataPath); 
            
                         
            cd  '..\Toolbox\Pansharpening Tool ver 1.3\'
        
            
          %%
            %Full resoution results
%             I_MS_LR = double(imgData.MS); % MS image;
%             I_MS =  double(imgData.MS_Up);% MS image upsampled to the PAN size;
%             I_PAN = double(imgData.Pan); %Pan
            I_F = double(TestOutputDate.output); %I_F:                Fused Image;
            I_MS_LR = double(TestData.ms); % I_MS_LR:            MS image;
            I_MS =  double(TestData.lms);%  I_MS:               MS image upsampled to the PAN size;
            I_PAN = double(TestData.pan); % I_PAN:              Panchromatic image;
            Paras = TestData.Paras;
            
            % Threshold values out of dynamic range
            thvalues = 0;
            
            L = ceil(log2(double(max(I_PAN(:)))+1));% Radiometric Resolution
            
            %     Params = imgData.Paras;
            %     Paras.ratio = Scale;%分辨率
            %     Paras.sensor = SensorName;%传感器类型
            %     Paras.intre = 'bicubic';%插值方式
            
            
            sensor = Paras.sensor;
            im_tag =  Paras.sensor;
            ratio = Paras.ratio;
        
            t2=tic;
            [D_lambda_DL,D_S_DL,QNRI_DL,SAM_DL,SCC_DL] = indexes_evaluation_FS(I_F,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,im_tag,ratio);
            MatrixResult_Fu(1,:) = [D_lambda_DL,D_S_DL,QNRI_DL,SAM_DL,SCC_DL];
            time_=toc(t2);
            fprintf('Elaboration time Fu: %.2f [sec]\n',time_);
        
        
      
%             %Reduced resoution results
%         %     I_GT = double(imgData.MS); %ground truth
%         %     I_PAN = double(imgData.Pan_LR);% low resolution Pan image
%         %     I_MS  = double(imgData.MS_LR_Up);% low resolution MS image upsampled at  low resolution  PAN scale;
%         %     I_MS_LR = double(imgData.MS_LR);% low resolution MS image
%             
%             I_F = double(TestOutputDate.output); %  ;
%             I_GT = double(TestData.gt); %ground truth
%             I_PAN = double(TestData.pan); % low resolution Pan image
% %           这两个没用上
%             I_MS =  double(TestData.lms);  % low resolution MS image upsampled at  low resolution  PAN scale;
%             I_MS_LR = double(TestData.ms); % low resolution MS image
%         
%            
%             % Initialization of the function parameters
%             % Threshold values out of dynamic range
%              
%             sensor = Paras.sensor;
%             im_tag =  Paras.sensor;
%             ratio = Paras.ratio;
%         
%             L = ceil(log2(double(max(I_PAN(:)))+1));% Radiometric Resolution
%             Qblocks_size = 32;
%         
%             flag_cut_bounds = 0;%不进行裁切
%             dim_cut = 0;%裁剪的大小不设置
%             thvalues = 0;
%         
%             t2=tic;
%             [Q_avg_DL, SAM_DL, ERGAS_DL, SCC_GT_DL, Q_DL] = indexes_evaluation(I_F,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%             MatrixResult_DR(1,:) = [Q_DL,Q_avg_DL,SAM_DL,ERGAS_DL,SCC_GT_DL];
%             time_=toc(t2);
%             fprintf('Elaboration time DR: %.2f [sec]\n',time_);

            cd ../../MethodDL_PanNet %返回主文件夹
            
%           MatrixResults_Fu(:,:,i)= cat(1, MatrixResults_Fu, MatrixResult_Fu);;%利用cat联结（按第几维来联结，被联结的图，要联结的）
            MatrixResults_Fu(1,:,i_NumImgs) = MatrixResult_Fu;
         
            % 保存每组图像的融合结果
            
%             saveName = fullfile(saveDir,ErjiDir_list(ErjiDir_i).name,[num2str(i),'.mat']);
            
            saveErjiDir_Path = fullfile(saveDir,ErjiDir_list(i_ErjiDir).name,['\']); %.\DataDL_PannetEvaluate\GF1_GengDi
            if ~exist(saveErjiDir_Path,'dir')%待保存的图像文件夹不存在，就建文件夹
                mkdir(saveErjiDir_Path)            
            end

            saveName = fullfile(saveDir,ErjiDir_list(i_ErjiDir).name,TestOutput_list(i_NumImgs).name);
            save(saveName, 'TestOutputPath','MatrixResult_Fu');
            
        end
    
        Matrix_Fu(i_ErjiDir-2,:,:) = MatrixResults_Fu;
        saveName = fullfile(saveDir,'MatrixAll_Fu.mat'); % saveName = fullfile(saveDir,ErjiDir_list(i_ErjiDir).name,'all.mat');
        save(saveName, 'TestOutputPath','Matrix_Fu');   
%         clearvars MatrixResults_Fu MatrixResults_DR % 需要循环使用临时变量名时，把变量清除一下，在下一个二级目录重新创建这两个变量，避免旧数据污染
        Matrix_Fu(i_ErjiDir-1,:,:) = 0; %[0;;NumImgs]%通过在现有索引范围之外插入新矩阵来扩展其大小。

    end
    
    fprintf('已保存 %s mat文件！并将该二级目录统计结果打印xlsx \n ', saveName);
end

%% 
% 创建包含局部函数的脚本local functions in scripts
    function f_IndexStatistics = IndexStatisticsFuDR (saveDir)
        saveName = fullfile(saveDir,'MatrixAll_Fu.mat');
        %Matrix_Fu = saveName;
        Matrix_Fu = load(saveName).Matrix_Fu;
        % TestData是不变的，利用它得到NumImgs
        TestDataErjiDir_Path = fullfile(TestDataYijiPath); %.\DataDL_1TestData
        TestData_list = dir([TestDataErjiDir_Path,'\','*.mat']) ;        
        NumImgs = size(TestData_list,1);

        % 对5参数循环
        fprintf('对5参数循环排序，判断，统计。判断是否准确！ \n');        
        for i_Indexs = 1:5 % D_lambda, D_S, QNRI, SAM, SCC       

            % 对100幅影像循环排序，判断，统计
            Index_SensorCorrectNum = 0; %计数器清零
            for i_NumImgs = 1:NumImgs                 
                Index_GF1 = Matrix_Fu(1,i_Indexs,i_NumImgs);
%                 Index_GF2 = Matrix_Fu(2,i_Indexs,i_NumImgs);
                Index_QB  = Matrix_Fu(2,i_Indexs,i_NumImgs);
                Index_WV2 = Matrix_Fu(3,i_Indexs,i_NumImgs);
                Index_WV3 = Matrix_Fu(4,i_Indexs,i_NumImgs);
               
                % 按相同顺序对向量进行排序。创建两个在对应元素中包含相关数据的行向量。https://ww2.mathworks.cn/help/matlab/ref/sort.html#d124e1289785
                X=[Index_GF1,Index_QB,Index_WV2,Index_WV3];
                Y={'GF1','QB','WV2','WV3'};
%                 ,'GF2'Index_GF2,
                % 首先对向量 X 进行排序，然后按照与 X 相同的顺序对向量 Y 进行排序。
                [~,I]=sort(X); %[Xsorted,I]=sort(X);
                Ysorted=Y(I);
                % char(Ysorted(1)) % 由小到大排序，第一个最小，再转换成字符串类型 % char(Ysorted(end)) 由小到大排序，最后一个最大，再转换成字符串类型 
                
                %验证两者是否一致 
                switch(i_Indexs)
                    % D_lambda
                    case 1
                        Index_SensorSort = char(Ysorted(1));
                        if isequal(Index_Sensor,Index_SensorSort)
                        Index_SensorCorrectNum = Index_SensorCorrectNum + 1;
                        end
                    % D_S
                    case 2
                        Index_SensorSort = char(Ysorted(1));
                        if isequal(Index_Sensor,Index_SensorSort)
                        Index_SensorCorrectNum = Index_SensorCorrectNum + 1;
                        end
                    % QNRI
                    case 3
                        Index_SensorSort = char(Ysorted(end));
                        if isequal(Index_Sensor,Index_SensorSort)
                        Index_SensorCorrectNum = Index_SensorCorrectNum + 1;
                        end
                    % SAM
                    case 4
                        Index_SensorSort = char(Ysorted(1));
                        if isequal(Index_Sensor,Index_SensorSort)
                        Index_SensorCorrectNum = Index_SensorCorrectNum + 1;
                        end
                    % SCC
                    case 5
                        Index_SensorSort = char(Ysorted(end));
                        if isequal(Index_Sensor,Index_SensorSort)
                        Index_SensorCorrectNum = Index_SensorCorrectNum + 1;
                        end    
                    % otherwise
                    otherwise                
                        fprintf("对5参数循环出错！");
                        break;

                end
                
                        
            end
            % 对100幅影像循环排序，判断，统计 后，计算正确率
            Index_SensorCorrectSum(i_Indexs) = Index_SensorCorrectNum;
            Index_SensorCorrectRate(i_Indexs) = double(Index_SensorCorrectNum / NumImgs);

        end
        % 对5参数循环 后，保存数据为mat
        saveName = fullfile(saveDir,'Index_SensorCorrectRate_Fu.mat'); 
        save(saveName, 'Index_SensorCorrectSum','Index_SensorCorrectRate');
        fprintf("已完成对本次样本影像参数循环排序，判断，统计，请到%s查看！\n  ", saveDir);
        fprintf('D_lambda,D_S,QNRI,SAM,SCC 准确个数计数，准确率分别为： \n');
        disp(Index_SensorCorrectSum);
        disp(Index_SensorCorrectRate);
    end

end