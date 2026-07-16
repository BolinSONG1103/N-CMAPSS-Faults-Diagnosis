% T4/T5 正式入口。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir); addpath(fullfile(matlabRoot,'src'));
build_chapter3_results();
