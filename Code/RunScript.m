%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                        program entry                      %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all hidden; addpath(genpath('.\Fx\'));
Size = 1024;  Size = num2str(Size); % Size = 1024,512,256,128,64,32
EvaluationDirList={['D:\Sensor-Type-Prediction-for-Pansharpening\Experiment\GF1_PanNet\Evaluate_Fu',Size],... 
                    ['D:\Sensor-Type-Prediction-for-Pansharpening\Experiment\IK_PanNet\Evaluate_Fu',Size],... 
                    ['D:\Sensor-Type-Prediction-for-Pansharpening\Experiment\QB_PanNet\Evaluate_Fu',Size],... 
                    ['D:\Sensor-Type-Prediction-for-Pansharpening\Experiment\WV2_PanNet\Evaluate_Fu',Size],...
                    ['D:\Sensor-Type-Prediction-for-Pansharpening\Experiment\WV3_PanNet\Evaluate_Fu',Size],...
                    ['D:\Sensor-Type-Prediction-for-Pansharpening\Experiment\WV4_PanNet\Evaluate_Fu',Size]};

TrainProportion = 0.8 ; TestProportion = 0.2 ; % Rand占比
columnSample = [1,2,3,18,19,20,21];

% SVM
ML_SVMRandData1(EvaluationDirList,TrainProportion,columnSample)

% SVM (Cross-validate hyperparameter tuning)
% ML_SVMcgRandData1(EvaluationDirList,TrainProportion,columnSample)

% RF
% ML_RFRandData1(EvaluationDirList,TrainProportion,columnSample)

