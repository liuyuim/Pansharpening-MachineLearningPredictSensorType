clc
clear
close all

PathDir='..\Tmp\shiyan230418_AllLandTypes\A_GF1_TestData\TestData_BenchmarkOutput\'; %Demo_CreateFusionPairs生成的mat文件目录
% DL_RunScriptFUDR.m % Train,vali
% Py2M_RemakeMat (PathDir); %生成全分辨率的test影像用于最终融合,生成降分辨率的test影像用于预测

% Runtrain.py
% python 中运行融合 RunFusionHypothesis.py
 
% Py2M_ForecastFu(); % 预测 
 
 Py2M_IndexFu(); %统计5指标 准确率

