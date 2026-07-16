function report=verify_diagnostic_closure()
%VERIFY_DIAGNOSTIC_CLOSURE 验证闭环输出、unit 隔离和图表可追溯性。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); root=fileparts(matlabRoot);
outDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure');
figDir=fullfile(root,'chapter3_results','diagnostic_figures');
manifest=readtable(fullfile(matlabRoot,'outputs','protocol','unit_split_manifest.csv'),'TextType','string');

assert(all(ismember(["train","calibration","test_id"],unique(manifest.role))));
files=unique(manifest.file);
for i=1:numel(files)
    q=manifest.file==files(i); z=manifest(q,:);
    assert(numel(unique(z.unit))==height(z),'同一文件的 unit 被分配到多个角色。');
    assert(any(z.role=="train")&&any(z.role=="calibration")&&any(z.role=="test_id"));
end

v=jsondecode(fileread(fullfile(outDir,'verdict21.json')));
assert(~v.fast_mode && ~v.test_truth_used_for_selection && ~v.stage_is_engineering_limit);
assert(strcmp(v.unknown_protocol,'leave_one_fault_family_out'));

lambda=readtable(fullfile(outDir,'lambda_calibration.csv'));
lambdaSelected=as_logical_mask(lambda.selected,'lambda.selected');
assert(nnz(lambdaSelected)==1,'lambda_calibration.csv 必须且只能标记一个已选 lambda。');
lambdaValue=lambda.lambda(lambdaSelected);
assert(is_close(lambdaValue,v.lambda_selected_on_calibration), ...
    'CSV 与 verdict21.json 中记录的已选 lambda 不一致。');
persistence=readtable(fullfile(outDir,'persistence_calibration.csv'));
persistenceSelected=as_logical_mask(persistence.selected,'persistence.selected');
assert(nnz(persistenceSelected)==1,'persistence_calibration.csv 必须且只能标记一个已选 K。');
assert(persistence.K(persistenceSelected)==v.persistence, ...
    'CSV 与 verdict21.json 中记录的已选 K 不一致。');

cycle=readtable(fullfile(outDir,'diagnostic_cycle_raw.csv'),'TextType','string');
units=readtable(fullfile(outDir,'diagnostic_unit_metrics.csv'),'TextType','string');
assert(all(ismember(["test_id","ood_combo"],unique(cycle.regime))));
assert(all(ismember(["test_id","ood_combo"],unique(units.regime))));
assert(all(cycle.pred_stage>=0 & cycle.pred_stage<=3));
assert(all(cycle.truth_stage>=0 & cycle.truth_stage<=3));

unknown=readtable(fullfile(outDir,'unknown_family_summary.csv'),'TextType','string');
assert(height(unknown)==5,'必须保留五个 leave-one-fault-family-out 折。');
assert(all(unknown.AUROC>=0&unknown.AUROC<=1));
assert(all(unknown.AUPRC>=0&unknown.AUPRC<=1));

robustDir=fullfile(matlabRoot,'outputs','t5_robustness_ablation');
ablation=readtable(fullfile(robustDir,'ablation_unit_metrics.csv'),'TextType','string');
robustness=readtable(fullfile(robustDir,'robustness_unit_metrics.csv'),'TextType','string');
baseline=readtable(fullfile(robustDir,'baseline_cycle_sensitivity.csv'),'TextType','string');
assert(all(ismember(["D_full","B_unconstrained","sign_only","D_no_shrink"], ...
    unique(ablation.method))));
assert(all(ismember(["noise","bias","missing_channel"],unique(robustness.perturbation))));
assert(numel(unique(robustness.channel(robustness.perturbation=="missing_channel")))==13);
assert(all(ismember([1 3 5],unique(baseline.n_reference_cycles))));

figManifest=readtable(fullfile(figDir,'diagnostic_figure_manifest.csv'),'TextType','string');
assert(height(figManifest)==5);
for i=1:height(figManifest)
    png=fullfile(figDir,figManifest.file_stem(i)+".png");
    pdf=fullfile(figDir,figManifest.file_stem(i)+".pdf");
    assert(isfile(png)&&isfile(pdf),'缺少诊断闭环图：%s',figManifest.file_stem(i));
    info=imfinfo(png); assert(info.XResolution>=590,'PNG 分辨率不足。');
end

report=struct('pass',true,'split_files',numel(files),'split_rows',height(manifest), ...
    'test_units',height(units),'cycle_component_rows',height(cycle), ...
    'unknown_folds',height(unknown),'figure_sets',height(figManifest), ...
    'ablation_methods',numel(unique(ablation.method)),'missing_channel_folds',13, ...
    'test_truth_used_for_selection',false);
write_json(fullfile(outDir,'verification.json'),report);
fprintf('诊断闭环验证通过：%d 台测试发动机，%d 个未知故障折，%d 组结果图。\n', ...
    height(units),height(unknown),height(figManifest));
end

function mask=as_logical_mask(value,name)
% readtable 会将 CSV 中的 logical 列读成数值 0/1；统一恢复为逻辑掩码。
if islogical(value)
    mask=value;
elseif isnumeric(value)
    assert(all(isfinite(value))&&all(value==0|value==1), ...
        '%s 必须仅包含 0/1 或 logical 值。',name);
    mask=value~=0;
elseif isstring(value)||iscellstr(value)||iscategorical(value)
    text=lower(strtrim(string(value)));
    assert(all(ismember(text,["0","1","false","true"])), ...
        '%s 必须仅包含 0/1 或 true/false。',name);
    mask=text=="1"|text=="true";
else
    error('%s 的数据类型无法转换为逻辑掩码。',name);
end
mask=mask(:);
end

function tf=is_close(a,b)
% 容忍 CSV/JSON 十进制序列化带来的末位舍入差异。
scale=max([1;abs(double(a(:)));abs(double(b(:)))]);
tf=isscalar(a)&&isscalar(b)&&isfinite(a)&&isfinite(b) ...
    && abs(double(a)-double(b))<=1e-10*scale;
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
