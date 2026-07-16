function summary=experiment_23_repeated_splits(keepIntermediate)
%EXPERIMENT_23_REPEATED_SPLITS 五个 unit 划分下重建全部闭环并汇总稳定性。
% 默认只保留跨划分汇总证据；各 seed 的可再生中间目录在全部汇总成功后删除。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
if nargin<1, keepIntermediate=false; end
seeds=314159:314163; rows=cell(0,1); ri=0;
lambdaRows=cell(numel(seeds),1); persistenceRows=cell(numel(seeds),1);
stageRows=cell(numel(seeds),1); unknownRows=cell(numel(seeds),1);
calibrationRows=cell(numel(seeds),1); seedDirs=strings(numel(seeds),1);
for seed=seeds
    fprintf('\nRepeated split %d/%d: seed=%d\n',seed-seeds(1)+1,numel(seeds),seed);
    experiment_21_diagnostic_closure(false,seed);
    if seed==314159
        outDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure');
    else
        outDir=fullfile(matlabRoot,'outputs',sprintf('t4_diagnostic_closure_seed_%d',seed));
    end
    iseed=seed-seeds(1)+1; seedDirs(iseed)=string(outDir);
    t=readtable(fullfile(outDir,'diagnostic_unit_metrics.csv'),'TextType','string');
    t.split_seed=repmat(seed,height(t),1);
    ri=ri+1; rows{ri}=t; %#ok<AGROW>

    lambdaTable=readtable(fullfile(outDir,'lambda_calibration.csv'));
    persistenceTable=readtable(fullfile(outDir,'persistence_calibration.csv'));
    assert(nnz(lambdaTable.selected~=0)==1,'每个划分必须且只能选择一个 lambda。');
    assert(nnz(persistenceTable.selected~=0)==1,'每个划分必须且只能选择一个 K。');
    lambdaRows{iseed}=with_seed(lambdaTable,seed);
    persistenceRows{iseed}=with_seed(persistenceTable,seed);
    stageRows{iseed}=with_seed(readtable(fullfile(outDir,'stage_component_sensitivity.csv'), ...
        'TextType','string'),seed);
    unknownRows{iseed}=with_seed(readtable(fullfile(outDir,'unknown_family_summary.csv'), ...
        'TextType','string'),seed);
    v=jsondecode(fileread(fullfile(outDir,'verdict21.json')));
    assert(~v.fast_mode && ~v.test_truth_used_for_selection, ...
        '重复划分只能汇总非烟测且未使用测试真值选参的结果。');
    calibrationRows{iseed}=table(seed,v.lambda_selected_on_calibration,v.persistence, ...
        v.stage_components,v.unknown_threshold,v.n_calibration_units,v.n_test_units, ...
        'VariableNames',{'split_seed','lambda','persistence','stage_components', ...
        'unknown_threshold','n_calibration_units','n_test_units'});
end
raw=vertcat(rows{:}); outDir=fullfile(matlabRoot,'outputs','t6_repeated_splits');
if ~isfolder(outDir), mkdir(outDir); end
writetable(raw,fullfile(outDir,'repeated_split_unit_metrics.csv'),'Encoding','UTF-8');
writetable(vertcat(calibrationRows{:}),fullfile(outDir, ...
    'repeated_split_calibration.csv'),'Encoding','UTF-8');
writetable(vertcat(lambdaRows{:}),fullfile(outDir, ...
    'repeated_split_lambda_calibration.csv'),'Encoding','UTF-8');
writetable(vertcat(persistenceRows{:}),fullfile(outDir, ...
    'repeated_split_persistence_calibration.csv'),'Encoding','UTF-8');
writetable(vertcat(stageRows{:}),fullfile(outDir, ...
    'repeated_split_stage_sensitivity.csv'),'Encoding','UTF-8');
writetable(vertcat(unknownRows{:}),fullfile(outDir, ...
    'repeated_split_unknown_summary.csv'),'Encoding','UTF-8');

metrics=["detection_FAR","detection_DR","detection_F1","isolation_macroF1", ...
    "stage_weighted_kappa","stage_MAE"];
groups=unique(raw.regime); out=cell(0,1); oi=0;
for ig=1:numel(groups)
    for im=1:numel(metrics)
        seedMeans=zeros(numel(seeds),1);
        for is=1:numel(seeds)
            q=raw.regime==groups(ig)&raw.split_seed==seeds(is);
            seedMeans(is)=mean(raw.(metrics(im))(q),'omitnan');
        end
        oi=oi+1; out{oi}=table(groups(ig),metrics(im),numel(seeds),mean(seedMeans), ...
            std(seedMeans),min(seedMeans),max(seedMeans), ...
            'VariableNames',{'regime','metric','n_splits','mean','std','min','max'}); %#ok<AGROW>
    end
end
summary=vertcat(out{:});
assert(height(summary)==2*numel(metrics) && all(summary.n_splits==numel(seeds)), ...
    '重复划分汇总的场景、指标或划分数量不完整。');
writetable(summary,fullfile(outDir,'repeated_split_summary.csv'),'Encoding','UTF-8');

verdict=struct('analysis_type','five_unit_isolated_repeated_splits', ...
    'split_seeds',seeds,'n_splits',numel(seeds),'all_fast_mode_false',true, ...
    'test_truth_used_for_selection',false,'intermediate_seed_directories_kept', ...
    logical(keepIntermediate));
write_json(fullfile(outDir,'verdict23.json'),verdict);

if ~keepIntermediate
    % 主划分是正文结果源；只清理由本实验生成的四个临时 seed 目录。
    for i=2:numel(seedDirs)
        path=char(seedDirs(i));
        if isfolder(path), rmdir(path,'s'); end
    end
end
end

function t=with_seed(t,seed)
t.split_seed=repmat(seed,height(t),1);
t=movevars(t,'split_seed','Before',1);
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0);
c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
