% 读取正式闭环 CSV 并生成四组正文结果图。
clear; clc;
runDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
build_chapter3_diagnostic_figures();
