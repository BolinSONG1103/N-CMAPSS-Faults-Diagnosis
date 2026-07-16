% 阶段 B/C 最终一致性、图表和复现纪律核查。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
verify_stage_bc();
