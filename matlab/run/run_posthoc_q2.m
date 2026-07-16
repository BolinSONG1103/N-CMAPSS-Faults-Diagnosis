% Q2s 事后诊断入口；不改变正式预注册裁决。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir); addpath(fullfile(matlabRoot,'src'));
experiment_posthoc_q2();
