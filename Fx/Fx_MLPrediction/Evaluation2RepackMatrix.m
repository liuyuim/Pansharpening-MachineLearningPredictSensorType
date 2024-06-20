%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 遍历所有项
%     items = dir(EvaluationData); % 获取顶层文件夹下的所有项
%     for i = 1:length(items)
%         item = items(i);
% 
%         % 跳过 . 和 .. 特殊目录
%         if strcmp(item.name, '.') || strcmp(item.name, '..')
%             continue;
%         end
% 
%         % 如果当前项是文件夹，进行处理
%         if item.isdir
%             % 输出文件夹的名称
%             fprintf('文件夹：%s\n', item.name);
%             HypoDirList = {};  % 初始化HypoDirList
%             % 在此处添加你希望执行的操作            
%             for j = 1:length(Hypo)
%                 % 使用contains函数检查文件夹名是否包含Hypo中的关键词                
%                 if contains(item.name, Hypo{j})
%                     HypoDirList{end+1} = item.name;
%                 end
%             end
% 
%         end
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ ] = Evaluation2RepackMatrix (SensorNames,EvaluationDir,saveDirName,NumImgStart,NumImgEnd)

% for i = 1:numel(EvaluationDir)
    % EvaluationDir = EvaluationDirList{i};
    %% 对每个 EvaluationDir 中每个 HypoDir 处理
    % numHypoDir = numel(HypoDirList);
    Matrix_Fu = []; 
    for k = 1:numel(SensorNames)
        HypoDir = ['HypothesisIn',SensorNames{k},'model'];  
        EvaluationHypoDir = fullfile(EvaluationDir,HypoDir);
               
        % 得到NumImgs    
        EvaluationMat_list = dir([EvaluationHypoDir,'\','*.mat']) ; % EvaluationMat_list = dir([EvaluationHypoDir,'**/*.mat']) ; % '**/*.mat'
        
        % 获取文件名并排序
        EvaluationMatNames = {EvaluationMat_list.name};
        SortedMatNames = sort_nat(EvaluationMatNames); % SortedMatNames = natsort(EvaluationMatNames); 
        
        % 设置xlsx文件名
        XlsxName = strcat("EvaluateReport", HypoDir, string(datetime, 'yyyy-MM-dd-HH-mm-ss'), '.xlsx');
        
        %% 在二级目录处理每一个mat
        % 判断文件总数够不够
        NumImgs = NumImgEnd - NumImgStart +  1 ; 
        NumImgSum = numel(EvaluationMat_list);
        if (NumImgs > NumImgSum)
            formatSpec = '目录 %s！总共有图像%d个，您要提取%d个，数量不够！... \n';
            fprintf(formatSpec,EvaluationHypoDir, NumImgSum, NumImgs);
            return ; %break;
        end
    
        for i_NumImgs = NumImgStart:NumImgEnd
        
            formatSpec = '开始处理目录 %s！%d个图像中第%d个！... \n';
            fprintf(formatSpec,EvaluationHypoDir, NumImgs, i_NumImgs);
    
            %把mat文件加载进来
            % EvaluationMatPath = [EvaluationMat_list(i_NumImgs).folder,'\',EvaluationMat_list(i_NumImgs).name]; %TestOutput_list列表中的第i个目录和文件名拼成要加载的mat路径 如 E:\LiuYu\FusionEvaluateExperiment\DataDL_PannetOutput\j1p1.mat
            EvaluationMatPath = fullfile(EvaluationHypoDir, SortedMatNames{i_NumImgs});
            load(EvaluationMatPath); 
            
            MatrixResults_Fu(1,:,i_NumImgs) = MatrixResult_Fu;
            
            %% 保存
            saveDir = fullfile(fileparts(EvaluationDir),saveDirName); %fileparts取上一级目录
            if ~exist(saveDir,'dir')%待保存的图像文件夹不存在，就建文件夹
                mkdir(saveDir)
            end
    
            %输出到xlsx https://ww2.mathworks.cn/help/matlab/ref/writematrix.html
            
            saveXlsxName = fullfile(saveDir,XlsxName);
            XlsxTitle = [ "D_lambda", "D_S", "QNRI", "SAM", "SCC", ...
                "Q_index", "SAM_index", "ERGAS_index", "sCC", "Q2n_index", ...
                "RB", "RV", "RSD", "RMSE_", "ERGAS_", "QAVE_", "CCMean", "SD", "entropy_", "CEMean", "SFMean", "Path"]; % 指标标题
            % xlswrite(saveXlsxName, XlsxTitle,['sheet',i_ErjiDir-2],'A1');
            
    
            writematrix(XlsxTitle,saveXlsxName)
            writematrix(MatrixResult_Fu,saveXlsxName,'WriteMode','append')
            
            writematrix(FusionImgPath,saveXlsxName,'Range',['x',num2str(i_NumImgs+1)]);   % F
    
            formatSpec = '已将当前图片评价结果保存至目录 %s！\n';
            fprintf(formatSpec,saveDir);
        end
        
        %% 开始统计
    
        %计算均值
        Mean_Fu = mean(MatrixResults_Fu,3);
        writematrix('均值',saveXlsxName,'WriteMode','append')
        writematrix(Mean_Fu,saveXlsxName,'WriteMode','append')        
        %计算中值 
        median_Fu = median(MatrixResults_Fu,3);
        writematrix('中值',saveXlsxName,'WriteMode','append')
        writematrix(median_Fu,saveXlsxName,'WriteMode','append')             
        %计算最大元素和最小元素
        max_Fu = max(MatrixResults_Fu,[],3);    
        writematrix('最大值',saveXlsxName,'WriteMode','append')
        writematrix(max_Fu,saveXlsxName,'WriteMode','append')  
        min_Fu = min(MatrixResults_Fu,[],3);
        writematrix('最小值',saveXlsxName,'WriteMode','append')
        writematrix(min_Fu,saveXlsxName,'WriteMode','append')  
        % 计算95%置信区间
        % ZX95 = [mean(MatrixResults_Fu,3)-1.96*(std(MatrixResults_Fu,0,3)/sqrt(NumImgs)) mean(MatrixResults_Fu,3)+1.96*(std(MatrixResults_Fu,0,3)/sqrt(NumImgs))];
        % ZX95_Fu = [ZX95(:,1) ZX95(:,6) ZX95(:,2) ZX95(:,7) ZX95(:,3) ZX95(:,8) ZX95(:,4) ZX95(:,9) ZX95(:,5) ZX95(:,10)];
        % writematrix('95%置信区间',saveXlsxName,'WriteMode','append')
        % writematrix(ZX95_Fu,saveXlsxName,'WriteMode','append')  
        
    
        % Matrix_Fu = zeros(ErjiDir_list_Nums-2,14,NumImgs); % Matrix_Fu = zeros(5,5,100);        
        % Matrix_Fu(1,size(MatrixResults_Fu,2),NumImgs) = 0; %插入新矩阵来扩展其大小。
        % Matrix_Fu(numHypoDir,:,:) = MatrixResults_Fu;
    
        Matrix_Fu = vertcat(Matrix_Fu,MatrixResults_Fu);
        saveName = fullfile(saveDir,'MatrixAll_Fu.mat'); % saveName = fullfile(saveDir,ErjiDir_list(i_ErjiDir).name,'all.mat');
        save(saveName, 'saveDir','Matrix_Fu');   
        % clearvars MatrixResults_Fu MatrixResults_DR % 需要循环使用临时变量名时，把变量清除一下，在下一个二级目录重新创建这两个变量，避免旧数据污染
        
        % end
    end    
    fprintf('已保存 %s mat文件！并将该二级目录统计结果打印xlsx \n ', saveName);
end


