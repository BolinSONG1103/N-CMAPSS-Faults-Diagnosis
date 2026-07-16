% T0 闸门：MATLAB 原生共享管线数值验收。
% 正式实验 T1–T5 只有在本脚本两条判据都通过后才允许启动。

clear; clc;
runDir = fileparts(mfilename('fullpath'));
matlabRoot = fileparts(runDir);
addpath(fullfile(matlabRoot,'src'));
outDir = fullfile(matlabRoot,'outputs','t0');
if ~isfolder(outDir), mkdir(outDir); end
diaryFile = fullfile(outDir,'t0_run.log');
if isfile(diaryFile), delete(diaryFile); end
diary(diaryFile);
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>

cfg = ncmapss_lib.config(false);
ncmapss_lib.print_preregistered('T0',cfg);
fprintf('  FAST_MODE = %d（T0 必须使用正式配置）\n',cfg.FAST_MODE);
fprintf('  基准实现 = %s\n',cfg.BASELINE_BACKEND);
fprintf('  说明：全部实验入口为 MATLAB；为逐位保持任务书指定的 sklearn HistGB，\n');
fprintf('        健康基准使用 MATLAB-Python 窄兼容层，原生 LSBoost 仅作敏感性对照。\n');

stream = ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
cacheFile = fullfile(matlabRoot,'cache','pipeline_cache_exact.mat');
% 首次运行从原始数据重建；失败后的代码级修复可复用同一已落盘管线，
% 避免把数分钟的 13 模型训练误当成新实验。
forceRebuild = false;
pipe = ncmapss_lib.build_pipeline(cfg,stream,cacheFile,forceRebuild);

condRelErr = abs(pipe.cond_Hn-cfg.COND_TARGET)/cfg.COND_TARGET;
condPass = condRelErr<=cfg.COND_REL_TOL;
fprintf('\nT0-a: cond(H_n)=%.6f, 相对误差=%.4f%% -> %s\n', ...
    pipe.cond_Hn,100*condRelErr,pass_text(condPass));

fprintf('\n加载 DS03/test 并复算 N=1000, lambda=%.1f 的 D ...\n',cfg.LAM_MAIN);
ds03 = fullfile(cfg.DATA_DIR,'N-CMAPSS_DS03-012.h5');
tbl = ncmapss_lib.load_per_cycle(ds03,'test',cfg,stream,[]);
[tbl,~] = ncmapss_lib.add_corrected(tbl,pipe.ref);
tbl = ncmapss_lib.attach_residuals(pipe,tbl);
units = sort(unique(tbl.unit));
trueFaults = {'HPT_eff_mod','LPT_eff_mod','LPT_flow_mod'};
trueIdx = find(ismember(cfg.THETA9,trueFaults));

rows = table('Size',[numel(units),8], ...
    'VariableTypes',{'double','double','double','double','double','double','double','double'}, ...
    'VariableNames',{'unit','N','lambda','skill','rmse','rmse_early','false_alarm','detect_corr'});
for i = 1:numel(units)
    [wins,~] = ncmapss_lib.build_windows(tbl,units(i),'full',1000,cfg,stream);
    R = wins.R; TH = wins.TH; T = size(R,1);
    Cum = ncmapss_lib.build_cum(T,9);
    M = ncmapss_lib.build_M(pipe.Hn,T);
    thetaHat = ncmapss_lib.solve_D(M,Cum,R,cfg.LAM_MAIN,T,9);
    met = ncmapss_lib.compute_metrics(thetaHat,TH,trueIdx);
    rows{i,:} = [units(i),1000,cfg.LAM_MAIN,met.skill,met.rmse, ...
        met.rmse_early,met.false_alarm,met.detect_corr];
    fprintf('  unit %d: skill=%.7f\n',round(units(i)),met.skill);
end

% 铁律：先落盘逐机原始结果，再聚合与裁决。
rawFile = fullfile(outDir,'t0_ds03_raw.csv');
writetable(rows,rawFile,'Encoding','UTF-8');
skillMean = mean(rows.skill);
skillAbsErr = abs(skillMean-cfg.T0_SKILL_TARGET);
skillPass = skillAbsErr<=cfg.T0_SKILL_ABS_TOL;
fprintf('T0-b: 技能分均值=%.9f, 目标=%.9f, 绝对误差=%.6f -> %s\n', ...
    skillMean,cfg.T0_SKILL_TARGET,skillAbsErr,pass_text(skillPass));

verdict = struct();
verdict.stage = 'T0';
verdict.fast_mode = cfg.FAST_MODE;
verdict.baseline_backend = pipe.baseline_backend;
verdict.cond_target = cfg.COND_TARGET;
verdict.cond_actual = pipe.cond_Hn;
verdict.cond_relative_error = condRelErr;
verdict.cond_pass = condPass;
verdict.skill_target = cfg.T0_SKILL_TARGET;
verdict.skill_actual = skillMean;
verdict.skill_absolute_error = skillAbsErr;
verdict.skill_pass = skillPass;
verdict.overall_pass = condPass && skillPass;
verdict.raw_csv = rawFile;
jsonFile = fullfile(outDir,'t0_verdict.json');
fid = fopen(jsonFile,'w','n','UTF-8');
assert(fid>=0,'无法写入 %s',jsonFile);
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(verdict,'PrettyPrint',true),'char');

fprintf('\n%s\n',repmat('=',1,96));
fprintf('T0 总裁决: %s\n',pass_text(verdict.overall_pass));
fprintf('逐机 CSV: %s\n',rawFile);
fprintf('裁决 JSON: %s\n',jsonFile);
fprintf('%s\n',repmat('=',1,96));

function s = pass_text(tf)
if tf, s='PASS'; else, s='FAIL'; end
end
