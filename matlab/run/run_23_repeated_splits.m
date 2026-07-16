% 五个 unit 划分下重建健康基准、H、标定层和正式闭环。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
experiment_23_repeated_splits();
