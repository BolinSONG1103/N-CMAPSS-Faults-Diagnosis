% 阶段 A3：E3 退化严重度与同子集配对诊断。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
experiment_stage_a_e3();
