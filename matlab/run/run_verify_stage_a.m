% 阶段 A 完整性核验。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
verify_stage_a();
