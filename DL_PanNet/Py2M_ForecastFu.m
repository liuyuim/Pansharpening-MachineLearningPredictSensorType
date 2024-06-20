function [ sensorPre] = Py2M_ForecastFuDR ()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 【5 批量评价和预测统计】
% 采用DeepLearning数据集进行融合实验，在高低两个分辨率尺度上对融合结果进行评价
% 对比方法采用 Pansharpening Tool ver 1.3中的方法；
% 对比指标主要是
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf("======对若干假设模型下融合影像评价并返回预测结果！====== \n");
TestOutputYijiPath='..\..\Tmp\DataDLForecast\PannetOutputToForecast\';

TestDataYijiPath='..\..\Tmp\DataDLForecast\TestData_Fu\'; 
saveDir = '..\..\Tmp\DataDLForecast2\PannetEvaluate\';%设置对应保存路径
ForecastFuDR (TestOutputYijiPath,TestDataYijiPath,saveDir);
fprintf("本次评价已完成，返回预测结果，脚本程序结束！\n");

%% 
% 创建包含局部函数的脚本local functions in scripts
function f_Forecast = ForecastFuDR (TestOutputYijiPath,TestDataYijiPath,saveDir)
    
        
    % 用TestOutputYijiPath 遍历出二级目录名，共用 GF1_GengDi GF1_LinDi   GF1_WeiBiaoDuoLei ...
    ErjiDir_list = dir(TestOutputYijiPath) ;  % 二级目录列表
    ErjiDir_list_Nums = size(ErjiDir_list,1);  % 二级目录个数 包括 .和..
    
    for i_ErjiDir = 3 : ErjiDir_list_Nums
        %列出当前二级文件夹内所有的mat
        TestOutputErjiDir_Path = fullfile(TestOutputYijiPath,ErjiDir_list(i_ErjiDir).name); %.\DataDL_PannetOutput\GF1_GengDi
        TestOutput_list = dir([TestOutputErjiDir_Path,'\','*.mat']) ;
        
        TestDataErjiDir_Path = fullfile(TestDataYijiPath); %.\DataDL_1TestData
        TestData_list = dir([TestDataErjiDir_Path,'\','*.mat']) ;
        
        NumImgs = size(TestOutput_list,1);

        % 在当前二级目录处理每一个mat

        for i = 1:NumImgs
        
            formatSpec = '正在处理二级目录 %s！%d个图像中第%d个！\n';
            fprintf(formatSpec,ErjiDir_list(i_ErjiDir).name, NumImgs, i);
    
            % 校验 当前从 TestOutput和TestData文件夹 分别取出的 mat文件名 是否一致
                       
            %验证两者是否一致
            if ~isequal(TestOutput_list(i).name, TestData_list(i).name)
                fprintf("当前从 TestOutput和TestData文件夹分别取出的 mat文件名 不一致");
                break;
            end
            
            % 然后再正常运行
            
            %把mat文件加载进来
            TestOutputPath = [TestOutput_list(i).folder,'\',TestOutput_list(i).name]; %TestOutput_list列表中的第i个目录和文件名拼成要加载的mat路径 如 E:\LiuYu\FusionEvaluateExperiment\DataDL_PannetOutput\j1p1.mat
            TestOutputDate = load(TestOutputPath); 
            TestDataPath = [TestData_list(i).folder,'\',TestData_list(i).name]; %TestData_list列表中的第i个目录和文件名拼成要加载的mat路径  E:\LiuYu\FusionEvaluateExperiment\DataDLPre_1TestData\j1p1.mat
            TestData = load(TestDataPath); 
            
            % load('./Params.mat')
             
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
            
            MatrixResults_Fu(:,:,i)= MatrixResult_Fu;
%             MatrixResults_DR(:,:,i) = MatrixResult_DR;
         
            % 保存每组图像的融合结果
            
%             saveName = fullfile(saveDir,ErjiDir_list(ErjiDir_i).name,[num2str(i),'.mat']);
            
            saveErjiDir_Path = fullfile(saveDir,ErjiDir_list(i_ErjiDir).name,['\']); %.\DataDL_PannetEvaluate\GF1_GengDi
            if ~exist(saveErjiDir_Path,'dir')%待保存的图像文件夹不存在，就建文件夹
                mkdir(saveErjiDir_Path)            
            end

            saveName = fullfile(saveDir,ErjiDir_list(i_ErjiDir).name,TestOutput_list(i).name);
            save(saveName, 'TestOutputPath','MatrixResult_Fu');
            
        end
    
        %开始统计

        %计算均值
        Mean_Fu = mean(MatrixResults_Fu,3);
%         Mean_DR = mean(MatrixResults_DR,3);
        
        %计算中值 有了平均值，为什么还要中值?平均值容易受极端数据影响。
        median_Fu = median(MatrixResults_Fu,3);
%         median_DR = median(MatrixResults_DR,3);
        
        %计算最大元素和最小元素
        max_Fu = max(MatrixResults_Fu,[],3);
%         max_DR = max(MatrixResults_DR,[],3);
        min_Fu = min(MatrixResults_Fu,[],3);
%         min_DR = min(MatrixResults_DR,[],3);

        % 计算95%置信区间
        ZX95 = [mean(MatrixResults_Fu,3)-1.96*(std(MatrixResults_Fu,0,3)/sqrt(NumImgs)) mean(MatrixResults_Fu,3)+1.96*(std(MatrixResults_Fu,0,3)/sqrt(NumImgs))];
        ZX95_Fu = [ZX95(:,1) ZX95(:,6) ZX95(:,2) ZX95(:,7) ZX95(:,3) ZX95(:,8) ZX95(:,4) ZX95(:,9) ZX95(:,5) ZX95(:,10)];
        
%         ZX95 = [mean(MatrixResults_DR,3)-1.96*(std(MatrixResults_DR,0,3)/sqrt(NumImgs)) mean(MatrixResults_DR,3)+1.96*(std(MatrixResults_DR,0,3)/sqrt(NumImgs))];
%         ZX95_DR = [ZX95(:,1) ZX95(:,6) ZX95(:,2) ZX95(:,7) ZX95(:,3) ZX95(:,8) ZX95(:,4) ZX95(:,9) ZX95(:,5) ZX95(:,10)];
%         
        
        saveName = fullfile(saveDir,ErjiDir_list(i_ErjiDir).name,'all.mat');
        save(saveName, 'TestOutputPath','MatrixResults_Fu','Mean_Fu','median_Fu','max_Fu','min_Fu','ZX95_Fu');
        
        %fprintf('已保存 评价 mat文件！计算均值,置信区间！');saveName;        
        fprintf('已保存 %s mat文件！并将该二级目录统计结果all.mat打印xlsx \n ', saveName);


        % 输出到xlsx
        Xlsx_Title = [ErjiDir_list(i_ErjiDir).name; "Mean_Fu"; "Mean_DR"; "median_Fu"; "median_DR"; "max_Fu"; "max_DR"; "min_Fu"; "min_DR"; "ZX95_Fu"; "ZX95_DR"; ]; 
        Xlsx_zhi(2,:) =  Mean_Fu;
%         Xlsx_zhi(3,:) =  Mean_DR;
        Xlsx_zhi(4,:) =  median_Fu;
%         Xlsx_zhi(5,:) =  median_DR;
        Xlsx_zhi(6,:) =  max_Fu;
%         Xlsx_zhi(7,:) =  max_DR;
        Xlsx_zhi(8,:) =  min_Fu;
%         Xlsx_zhi(9,:) =  min_DR;
%         Xlsx_zhi(9,:) =  ZX95_Fu;
%         Xlsx_zhi(10,:) =  ZX95_DR;

        saveXlsxName = fullfile(saveDir,'report.xlsx');
        XlsxLocate_ = num2str((i_ErjiDir-3)*20+1);
        XlsxLocateA = ['A',XlsxLocate_]; 
        XlsxLocateB = ['B',XlsxLocate_]; 
        xlswrite(saveXlsxName, Xlsx_Title,'sheet1',XlsxLocateA);
        xlswrite(saveXlsxName, Xlsx_zhi,'sheet1',XlsxLocateB);

        
        %判断
        fprintf('计算该假设模型下融合影像评价结果的相似度...\n ---------------------------- \n');
        [ScoreGF1_Max_FuMin_Fu,ScoreGF2_Max_FuMin_Fu,ScoreQB_Max_FuMin_Fu,ScoreWV2_Max_FuMin_Fu,ScoreWV3_Max_FuMin_Fu]=deal(0);%初始化为0 
        switch(ErjiDir_list(i_ErjiDir).name)
            % HypothesisInGF1model
            case 'HypothesisInGF1model'
                % 检查五参数上下限
                if max_Fu(:,1)<0.0447 && min_Fu(:,1)>0.0053  % D_lambda
                    ScoreGF1_Max_FuMin_Fu = ScoreGF1_Max_FuMin_Fu + 1;end
                if max_Fu(:,2)<0.4048 && min_Fu(:,2)>0.1308  % D_S
                    ScoreGF1_Max_FuMin_Fu = ScoreGF1_Max_FuMin_Fu + 1;end
                if max_Fu(:,3)<0.8575 && min_Fu(:,3)>0.5802  % QNRI
                    ScoreGF1_Max_FuMin_Fu = ScoreGF1_Max_FuMin_Fu + 1;end
                if max_Fu(:,4)<1.3546 && min_Fu(:,4)>0.9757  % SAM
                    ScoreGF1_Max_FuMin_Fu = ScoreGF1_Max_FuMin_Fu + 1;end
                if max_Fu(:,5)<0.6750 && min_Fu(:,5)>0.5709  % SCC
                    ScoreGF1_Max_FuMin_Fu = ScoreGF1_Max_FuMin_Fu + 1;end
                
                % 检查 Mean_Fu 相似度
                Mean_Fu_Reference = [0.02185856,0.230439425,0.752884307,1.156726476,0.613180165];
                ScoreGF1_Mean_Fu = sum(abs((Mean_Fu+Mean_Fu_Reference)/(Mean_Fu-Mean_Fu_Reference)));
                % 检查 median_Fu 相似度
                median_Fu_Reference = [0.020792261,0.19562825,0.786973289,1.168947477,0.602604151];
                ScoreGF1_median_Fu = sum(abs((median_Fu+median_Fu_Reference)/(median_Fu-median_Fu_Reference)));
                sensorGF1 = ScoreGF1_Max_FuMin_Fu + ScoreGF1_Mean_Fu + ScoreGF1_median_Fu;

            % HypothesisInGF2model
            case 'HypothesisInGF2model'
                if max_Fu(:,1)<0.1212 && min_Fu(:,1)>0.0514  % D_lambda
                    ScoreGF2_Max_FuMin_Fu = ScoreGF2_Max_FuMin_Fu + 1;end
                if max_Fu(:,2)<0.9059 && min_Fu(:,2)>0.7439  % D_S
                    ScoreGF2_Max_FuMin_Fu = ScoreGF2_Max_FuMin_Fu + 1;end
                if max_Fu(:,3)<0.2319 && min_Fu(:,3)>0.0864  % QNRI
                    ScoreGF2_Max_FuMin_Fu = ScoreGF2_Max_FuMin_Fu + 1;end
                if max_Fu(:,4)<1.8317 && min_Fu(:,4)>0.8198  % SAM
                    ScoreGF2_Max_FuMin_Fu = ScoreGF2_Max_FuMin_Fu + 1;end
                if max_Fu(:,5)<0.6812 && min_Fu(:,5)>0.5411  % SCC
                    ScoreGF2_Max_FuMin_Fu = ScoreGF2_Max_FuMin_Fu + 1;end
                
                % 检查 Mean_Fu 相似度
                Mean_Fu_Reference = [0.080751336,0.857860676,0.130227697,1.253855358,0.615218815];
                ScoreGF2_Mean_Fu = sum(abs((Mean_Fu+Mean_Fu_Reference)/(Mean_Fu-Mean_Fu_Reference)));
                % 检查 median_Fu 相似度
                median_Fu_Reference = [0.085596062,0.879337963,0.113337035,1.258134822,0.632646351];
                ScoreGF2_median_Fu = sum(abs((median_Fu+median_Fu_Reference)/(median_Fu-median_Fu_Reference)));
                sensorGF2 = ScoreGF2_Max_FuMin_Fu + ScoreGF2_Mean_Fu + ScoreGF2_median_Fu;

            % HypothesisInQBmodel
            case 'HypothesisInQBmodel'
                if max_Fu(:,1)<0.1620 && min_Fu(:,1)>0.0432  % D_lambda
                    ScoreQB_Max_FuMin_Fu = ScoreQB_Max_FuMin_Fu + 1;end
                if max_Fu(:,2)<0.8695 && min_Fu(:,2)>0.5065  % D_S
                    ScoreQB_Max_FuMin_Fu = ScoreQB_Max_FuMin_Fu + 1;end
                if max_Fu(:,3)<0.4136 && min_Fu(:,3)>0.1247  % QNRI
                    ScoreQB_Max_FuMin_Fu = ScoreQB_Max_FuMin_Fu + 1;end
                if max_Fu(:,4)<1.7640 && min_Fu(:,4)>1.3573  % SAM
                    ScoreQB_Max_FuMin_Fu = ScoreQB_Max_FuMin_Fu + 1;end
                if max_Fu(:,5)<0.8495 && min_Fu(:,5)>0.7781  % SCC
                    ScoreQB_Max_FuMin_Fu = ScoreQB_Max_FuMin_Fu + 1;end
                
                % 检查 Mean_Fu 相似度
                Mean_Fu_Reference = [0.089665016,0.739719188,0.233220815,1.589680116,0.81929679];
                ScoreQB_Mean_Fu = sum(abs((Mean_Fu+Mean_Fu_Reference)/(Mean_Fu-Mean_Fu_Reference)));
                % 检查 median_Fu 相似度
                median_Fu_Reference = [0.085019488,0.77469407,0.214284735,1.606005896,0.82265559];
                ScoreQB_median_Fu = sum(abs((median_Fu+median_Fu_Reference)/(median_Fu-median_Fu_Reference)));
                sensorQB = ScoreQB_Max_FuMin_Fu + ScoreQB_Mean_Fu + ScoreQB_median_Fu;

            % HypothesisInWV2model
            case 'HypothesisInWV2model'
                if max_Fu(:,1)<0.1169 && min_Fu(:,1)>0.0158  % D_lambda
                    ScoreWV2_Max_FuMin_Fu = ScoreWV2_Max_FuMin_Fu + 1;end
                if max_Fu(:,2)<0.6428 && min_Fu(:,2)>0.3491  % D_S
                    ScoreWV2_Max_FuMin_Fu = ScoreWV2_Max_FuMin_Fu + 1;end
                if max_Fu(:,3)<0.5785 && min_Fu(:,3)>0.3515  % QNRI
                    ScoreWV2_Max_FuMin_Fu = ScoreWV2_Max_FuMin_Fu + 1;end
                if max_Fu(:,4)<2.7797 && min_Fu(:,4)>1.7232  % SAM
                    ScoreWV2_Max_FuMin_Fu = ScoreWV2_Max_FuMin_Fu + 1;end
                if max_Fu(:,5)<0.7538 && min_Fu(:,5)>0.5436  % SCC
                    ScoreWV2_Max_FuMin_Fu = ScoreWV2_Max_FuMin_Fu + 1;end
                
                % 检查 Mean_Fu 相似度
                Mean_Fu_Reference = [0.056429528,0.517484918,0.452895078,2.207158142,0.621170841];
                ScoreWV2_Mean_Fu = sum(abs((Mean_Fu+Mean_Fu_Reference)/(Mean_Fu-Mean_Fu_Reference)));
                % 检查 median_Fu 相似度
                median_Fu_Reference = [0.043496768,0.554197045,0.41749045,2.104662228,0.621713368];
                ScoreWV2_median_Fu = sum(abs((median_Fu+median_Fu_Reference)/(median_Fu-median_Fu_Reference)));
                sensorWV2 = ScoreWV2_Max_FuMin_Fu + ScoreWV2_Mean_Fu + ScoreWV2_median_Fu;

            % HypothesisInWV3model
            case 'HypothesisInWV3model'
                if max_Fu(:,1)<0.1107 && min_Fu(:,1)>0.0137  % D_lambda
                    ScoreWV3_Max_FuMin_Fu = ScoreWV3_Max_FuMin_Fu + 1;end
                if max_Fu(:,2)<0.8170 && min_Fu(:,2)>0.4388  % D_S
                    ScoreWV3_Max_FuMin_Fu = ScoreWV3_Max_FuMin_Fu + 1;end
                if  max_Fu(:,3)<0.5488 && min_Fu(:,3)>0.1789  % QNRI
                    ScoreWV3_Max_FuMin_Fu = ScoreWV3_Max_FuMin_Fu + 1;end
                if  max_Fu(:,4)<2.7441 && min_Fu(:,4)>1.0446  % SAM
                    ScoreWV3_Max_FuMin_Fu = ScoreWV3_Max_FuMin_Fu + 1;end
                if  max_Fu(:,5)<0.8389 && min_Fu(:,5)>0.6111  % SCC  
                    ScoreWV3_Max_FuMin_Fu = ScoreWV3_Max_FuMin_Fu + 1;end
                
                % 检查 Mean_Fu 相似度
                Mean_Fu_Reference = [0.042378648,0.618918708,0.364139748,2.00501318,0.740135162];
                ScoreWV3_Mean_Fu = sum(abs((Mean_Fu+Mean_Fu_Reference)/(Mean_Fu-Mean_Fu_Reference)));
                % 检查 median_Fu 相似度
                median_Fu_Reference = [0.034090398,0.589069887,0.400721038,2.164430929,0.727635602];
                ScoreWV3_median_Fu = sum(abs((median_Fu+median_Fu_Reference)/(median_Fu-median_Fu_Reference)));
                sensorWV3 = ScoreWV3_Max_FuMin_Fu + ScoreWV3_Mean_Fu + ScoreWV3_median_Fu;

            otherwise                
                fprintf("不能预测为何种遥感影像，请检查二级文件夹是否匹配！");
                break;
        end


        % 需要循环使用临时变量名时，把变量清除一下，在下一个二级目录重新创建这两个变量，避免旧数据污染
        clearvars MatrixResults_Fu MatrixResults_DR

%         SavePath_strsplit = strsplit(TestOutput_list(i).folder,'\'); %把路径用split以\分割
%         SavePath_strsplit_char = char(SavePath_strsplit(5)); %分割完的结果转成字符型作为二级目录名
%         SaveName = fullfile(saveDir,SavePath_strsplit_char); %实际保存mat的绝对路径是由上面定义的
%               
%         SaveName = fullfile(saveDir,'all.mat');
%         save(saveName, 'MatrixResults_Fu', 'MatrixImage_DR','ImgPaths');
%         save(SaveName, 'MatrixResults_Fu', 'MatrixResults_DR');
    
       
    end
    %排序
    fprintf('对若干假设模型下融合影像评价结果的相似度排序。选出相似度最高的！ \n');
    % 按相同顺序对向量进行排序。创建两个在对应元素中包含相关数据的行向量。https://ww2.mathworks.cn/help/matlab/ref/sort.html#d124e1289785
    X=[sensorGF1,sensorGF2,sensorQB,sensorWV2,sensorWV3];
    Y={'sensorGF1','sensorGF2','sensorQB','sensorWV2','sensorWV3'};
    % 首先对向量 X 进行排序，然后按照与 X 相同的顺序对向量 Y 进行排序。
    [~,I]=sort(X); %[Xsorted,I]=sort(X);
    Ysorted=Y(I);
    sensorPre = char(Ysorted(end)); % 由小到大排序，最后一个最大，再转换成字符串类型
    
    formatSpec = "目录%s已完成评价，请到%s查看！\n 预测遥感影像为【%s传感器】\n ";
    fprintf(formatSpec,TestOutputYijiPath,saveDir,sensorPre);
    fprintf('sensorGF1,sensorGF2,sensorQB,sensorWV2,sensorWV3 分数分别为： \n');
    disp(X);
end

end