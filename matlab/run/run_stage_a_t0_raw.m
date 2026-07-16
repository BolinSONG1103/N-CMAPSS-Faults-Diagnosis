% 阶段 A2：raw_scan 与 MATLAB T0 原始数字核验。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
experiment_stage_a_t0_raw();
