% T1 正式入口：工况自适应 H(w)，双窗口模式。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir); addpath(fullfile(matlabRoot,'src'));
v=jsondecode(fileread(fullfile(matlabRoot,'outputs','t0','t0_verdict.json')));
assert(v.overall_pass,'T0 未通过，禁止启动 T1。');
experiment_19b(false);
