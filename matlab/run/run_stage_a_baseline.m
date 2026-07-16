% 阶段 A1：健康基准兼容层与纯 MATLAB LSBoost 对照。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
experiment_stage_a_baseline();
