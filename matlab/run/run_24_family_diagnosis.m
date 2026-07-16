% 九维连续估计后的五物理部件族诊断补充实验。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
experiment_24_family_diagnosis();
