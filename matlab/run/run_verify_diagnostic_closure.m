% 验证诊断闭环输出、unit 隔离、未知故障折和正文图。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
verify_diagnostic_closure();
