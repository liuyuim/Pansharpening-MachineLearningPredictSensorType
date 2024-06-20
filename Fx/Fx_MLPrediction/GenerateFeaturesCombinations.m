% clc;clear;close all;addpath(genpath('.\Fx\'));
% 
% columnSample = [1, 2, 3, 4];
% comboSave = []; % 创建一个空的单元格数组来存储组合
% % k = 1; % 选择的元素数量
% for k = 1:length(columnSample)
%     combo = zeros(1, k);
%     indexNum = length(columnSample);
%     generateCombinations(columnSample, k, combo, indexNum);
%     % combinations = [combinations; subCombinations];
% end
% 
% 
% 
% function [comboSave] = generateCombinations(arr, k, combo, indexNum)
% 
% 
%     if k == 0
%         disp(combo);
%         comboSave = vertcat(comboSave,combo);
%     elseif indexNum == 0
%         return;
%     else
%         combo(k) = arr(indexNum);
%         generateCombinations(arr, k - 1, combo, indexNum - 1);
%         generateCombinations(arr, k, combo, indexNum - 1);
%     end
% 
%     % 将所有组合保存到一个文件
%     save('comboSave.mat', 'comboSave');   
% end


% clc;clear;close all;addpath(genpath('.\Fx\'));
function [comboSave] = GenerateFeaturesCombinations(columnSample)
% columnSample = [1, 2, 3, 4];

% 创建一个初始的空单元格数组
comboSave = cell(0, 1);

for k = 1:length(columnSample)
    combo = zeros(1, k);
    indexNum = length(columnSample);
    % 传递comboSave并累积组合
    subCombinations = generateCombinations(columnSample, k, combo, indexNum);
    comboSave = [comboSave; subCombinations];
end

% 将所有组合保存到一个文件
% save('comboSave.mat', 'comboSave');
