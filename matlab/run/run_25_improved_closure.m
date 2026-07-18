% 第3章决策层改进对比正式入口（v2 基线 vs v3 单元自适应+时序平滑）。
% 复用 t4 锁定序列与 split 缓存，不覆盖 v2；输出至 outputs/t7_improved_closure。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
experiment_25_improved_closure(false);
