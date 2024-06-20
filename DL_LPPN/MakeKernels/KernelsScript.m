%% 4.1 网络所需的准备

% LPPN MakeKernel
clc;clear;close all;addpath(genpath('.\Fx\'));


ratio = 4 ;
N = 7 ;
nBands = 4 ;

% sensor = 'IKONOS';
% Kernel_MTF_MS(sensor,ratio,N,nBands);
% Kernel_MTF_PAN(sensor,ratio,N);

sensor = 'WV4';
Kernel_MTF_MS(sensor,ratio,N,nBands);
Kernel_MTF_PAN(sensor,ratio,N);

