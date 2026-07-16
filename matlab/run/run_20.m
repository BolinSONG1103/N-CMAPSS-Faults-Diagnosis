% T2 正式入口：数据驱动对照 × 组合外推。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir); addpath(fullfile(matlabRoot,'src'));
v=jsondecode(fileread(fullfile(matlabRoot,'outputs','t0','t0_verdict.json')));
assert(v.overall_pass,'T0 未通过，禁止启动 T2。');
experiment_20(false);
