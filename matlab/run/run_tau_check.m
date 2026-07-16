% T3 正式入口：tau_required 与 sqrt(N) 的定量确证。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir); addpath(fullfile(matlabRoot,'src'));
v=jsondecode(fileread(fullfile(matlabRoot,'outputs','t0','t0_verdict.json')));
assert(v.overall_pass,'T0 未通过，禁止启动 T3。');
experiment_tau();
