function summary=experiment_23_repeated_splits()
%EXPERIMENT_23_REPEATED_SPLITS 五个 unit 划分下重建全部闭环并汇总稳定性。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
seeds=314159:314163; rows=cell(0,1); ri=0;
for seed=seeds
    fprintf('\nRepeated split %d/%d: seed=%d\n',seed-seeds(1)+1,numel(seeds),seed);
    experiment_21_diagnostic_closure(false,seed);
    if seed==314159
        outDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure');
    else
        outDir=fullfile(matlabRoot,'outputs',sprintf('t4_diagnostic_closure_seed_%d',seed));
    end
    t=readtable(fullfile(outDir,'diagnostic_unit_metrics.csv'),'TextType','string');
    t.split_seed=repmat(seed,height(t),1);
    ri=ri+1; rows{ri}=t; %#ok<AGROW>
end
raw=vertcat(rows{:}); outDir=fullfile(matlabRoot,'outputs','t6_repeated_splits');
if ~isfolder(outDir), mkdir(outDir); end
writetable(raw,fullfile(outDir,'repeated_split_unit_metrics.csv'),'Encoding','UTF-8');

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
writetable(summary,fullfile(outDir,'repeated_split_summary.csv'),'Encoding','UTF-8');
end
