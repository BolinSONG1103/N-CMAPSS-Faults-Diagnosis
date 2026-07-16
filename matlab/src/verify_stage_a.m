function verdict=verify_stage_a()
%VERIFY_STAGE_A 阶段 A 产物、裁决和绘图完整性核验。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir);
stageDir=fullfile(matlabRoot,'outputs','stage_a');

a1=jsondecode(fileread(fullfile(stageDir,'a1_baseline','verdict_a1.json')));
a2=jsondecode(fileread(fullfile(stageDir,'a2_t0','verdict_a2.json')));
q2=jsondecode(fileread(fullfile(stageDir,'a3_q2','verdict_a3_q2.json')));
e3=jsondecode(fileread(fullfile(stageDir,'a3_e3','verdict_a3_e3.json')));

assert(~a1.automatic_switch_allowed,'A1 不应允许自动切换到 LSBoost。');
assert(a1.cond_relative_drift>.01,'A1 cond 漂移应超过 1%%。');
assert(a2.pass && a2.absolute_difference<=a2.tolerance,'A2 T0 原始值核验失败。');
assert(q2.formal_Q2s_remains_false && q2.shrinkage_explanation_supported, ...
    'Q2 正式裁决或事后机制结论异常。');
assert(e3.formal_E3_remains_false && e3.specific_claim_ood_degradation_is_larger_rejected, ...
    'E3 正式裁决或严重度诊断结论异常。');

base=readtable(fullfile(stageDir,'a1_baseline','baseline_comparison_summary.csv'),'TextType','string');
t0=readtable(fullfile(stageDir,'a2_t0','t0_raw_comparison.csv'),'TextType','string');
q2raw=readtable(fullfile(stageDir,'a3_q2','q2_lambda_raw.csv'));
sev=readtable(fullfile(stageDir,'a3_e3','severity_per_unit.csv'),'TextType','string');
pair=readtable(fullfile(stageDir,'a3_e3','same_subset_paired_summary.csv'),'TextType','string');
assert(any(base.backend=="Z_zero") && any(t0.method=="Z_zero"), ...
    'A1/A2 表缺少 Z_zero。');
assert(all(q2raw.Z_zero_skill==0),'Q2 原始表的 Z_zero 哨兵异常。');
assert(any(sev.arm=="Z_zero") && any(pair.arm=="Z_zero"),'E3 表缺少 Z_zero。');

png=dir(fullfile(stageDir,'**','*.png'));
pdf=dir(fullfile(stageDir,'**','*.pdf'));
svg=dir(fullfile(stageDir,'**','*.svg'));
assert(numel(png)==12 && numel(pdf)==12 && numel(svg)==12, ...
    '阶段 A 图形应为 12 组 PNG/PDF/SVG 三联产物。');
minDpi=inf;
for i=1:numel(png)
    info=imfinfo(fullfile(png(i).folder,png(i).name));
    scale=1;
    if isfield(info,'ResolutionUnit') && strcmpi(info.ResolutionUnit,'meter')
        scale=.0254;
    elseif isfield(info,'ResolutionUnit') && strcmpi(info.ResolutionUnit,'centimeter')
        scale=2.54;
    end
    minDpi=min([minDpi,scale*double(info.XResolution),scale*double(info.YResolution)]);
end
assert(minDpi>=599,'阶段 A PNG 分辨率低于 600 dpi。');

verdict=struct('stage','A','overall_pass',true,'a1_isolated_dependency',true, ...
    'a2_t0_pass',a2.pass,'q2_formal_unchanged',q2.formal_Q2s_remains_false, ...
    'q2_shrinkage_supported',q2.shrinkage_explanation_supported, ...
    'e3_formal_unchanged',e3.formal_E3_remains_false, ...
    'e3_old_severity_explanation_rejected', ...
        e3.specific_claim_ood_degradation_is_larger_rejected, ...
    'figure_sets',numel(png),'figure_formats',{{'png','pdf','svg'}}, ...
    'minimum_png_dpi',minDpi,'stage_B_authorization_received',false);
fid=fopen(fullfile(stageDir,'verification.json'),'w','n','UTF-8'); assert(fid>=0);
c=onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(verdict,'PrettyPrint',true),'char');
fprintf('Stage A verification PASS: %d figure sets, minimum PNG dpi %.1f.\n',numel(png),minDpi);
end
