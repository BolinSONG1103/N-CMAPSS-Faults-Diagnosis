function verdict=experiment_24_family_diagnosis()
%EXPERIMENT_24_FAMILY_DIAGNOSIS 九维连续估计后的五物理部件族诊断。
% 映射与指标由 docs/部件族诊断补充协议.md 在运行前冻结。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
cfg=ncmapss_lib.config(false); dcfg=chapter3_diagnostic_lib.config(cfg);
baseDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure');
outDir=fullfile(matlabRoot,'outputs','t7_family_diagnosis');
if ~isfolder(outDir), mkdir(outDir); end

S=load(fullfile(baseDir,'closure_sequences.mat'),'calRaw','idRaw','oodRaw');
D=load(fullfile(baseDir,'diagnostic_model.mat'),'lambdaSelected');
P=load(fullfile(matlabRoot,'cache','pipeline_cache_split_v2.mat'),'pipe');
assert(isfield(P.pipe,'Hn')&&~isempty(P.pipe.Hn),'主划分缓存缺少 Hn。');

cal=solve_sequences(S.calRaw,P.pipe.Hn,D.lambdaSelected);
familyModel=chapter3_family_lib.fit(cal,dcfg);
thresholds=table(chapter3_family_lib.names().',familyModel.family_threshold.', ...
    repmat(dcfg.alpha,5,1),'VariableNames',{'family','threshold','alpha'});
writetable(thresholds,fullfile(outDir,'family_threshold_calibration.csv'),'Encoding','UTF-8');
writetable(familyModel.persistence_selection, ...
    fullfile(outDir,'family_persistence_calibration.csv'),'Encoding','UTF-8');

test=solve_sequences([S.idRaw,S.oodRaw],P.pipe.Hn,D.lambdaSelected);
[cycleRaw,unitMetrics,familyMetrics,detectConf]=evaluate_sequences(test,familyModel);
writetable(cycleRaw,fullfile(outDir,'family_cycle_raw.csv'),'Encoding','UTF-8');
writetable(unitMetrics,fullfile(outDir,'family_unit_metrics.csv'),'Encoding','UTF-8');
writetable(familyMetrics,fullfile(outDir,'family_metrics.csv'),'Encoding','UTF-8');
writetable(detectConf,fullfile(outDir,'family_detection_confusion.csv'),'Encoding','UTF-8');
summary=bootstrap_summary(unitMetrics,2000,cfg.SEED+2410);
writetable(summary,fullfile(outDir,'family_unit_bootstrap_summary.csv'),'Encoding','UTF-8');

assert(height(familyMetrics)==10,'五个部件族必须在两个测试场景中全部保留。');
assert(all(ismember(["test_id","ood_combo"],unique(unitMetrics.regime))));
assert(nnz(familyModel.persistence_selection.selected)==1);
verdict=struct('analysis_type','five_physical_family_multilabel_diagnosis', ...
    'fast_mode',false,'source_continuous_model','diagnostic_closure_v2', ...
    'lambda_locked_from_v2_calibration',D.lambdaSelected, ...
    'family_names',{cellstr(chapter3_family_lib.names())}, ...
    'family_groups',{{1,[2 3],[4 5],[6 7],[8 9]}}, ...
    'severity_definition','RMS_of_normalized_parameters', ...
    'persistence',familyModel.persistence,'test_truth_used_for_selection',false, ...
    'stage_model_changed',false,'unknown_protocol_changed',false, ...
    'protocol','docs/部件族诊断补充协议.md');
write_json(fullfile(outDir,'verdict24.json'),verdict);
fprintf('五部件族诊断完成：%d 台发动机，K=%d，测试真值未参与选参。\n', ...
    height(unitMetrics),familyModel.persistence);
end

function out=solve_sequences(raw,Hn,lambda)
out=raw;
for i=1:numel(raw)
    s=raw{i}; T=size(s.R,1); M=ncmapss_lib.build_M(Hn,T);
    Cum=ncmapss_lib.build_cum(T,size(Hn,2));
    s.theta_hat=ncmapss_lib.solve_D(M,Cum,s.R,lambda,T,size(Hn,2));
    out{i}=s;
end
end

function [cycleRaw,unitMetrics,familyMetrics,detectConf]=evaluate_sequences(seqs,model)
names=chapter3_family_lib.names(); cycleRows=cell(0,1); unitRows=cell(numel(seqs),1); ri=0;
for i=1:numel(seqs)
    s=seqs{i}; p=chapter3_family_lib.predict(s.theta_hat,model);
    yLabels=chapter3_family_lib.truth_labels(s.hs,s.fault_idx,numel(s.cycles));
    y=struct('family_label',yLabels,'fault',~logical(s.hs));
    met=chapter3_family_lib.sequence_metrics(p,y,s.cycles);
    row=struct2table(met); row.regime=string(s.regime); row.subset=string(s.subset);
    row.unit=s.unit; unitRows{i}=row;
    for t=1:numel(s.cycles)
        for g=1:numel(names)
            ri=ri+1; cycleRows{ri}=table(string(s.regime),string(s.subset),s.unit, ...
                s.cycles(t),names(g),g,logical(s.hs(t)),p.severity(t,g), ...
                yLabels(t,g),p.family_label(t,g), ...
                'VariableNames',{'regime','subset','unit','cycle','family', ...
                'family_index','hs','family_severity','truth_label','pred_label'}); %#ok<AGROW>
        end
    end
end
cycleRaw=vertcat(cycleRows{:}); unitMetrics=vertcat(unitRows{:});
unitMetrics=movevars(unitMetrics,{'regime','subset','unit'},'Before',1);
familyMetrics=family_summary(cycleRaw,names);
detectConf=detection_confusion(cycleRaw);
end

function out=family_summary(raw,names)
regimes=unique(raw.regime); rows=cell(numel(regimes)*numel(names),1); ri=0;
for ir=1:numel(regimes)
    for g=1:numel(names)
        q=raw.regime==regimes(ir)&raw.family_index==g;
        keys=unique(raw(q,{'subset','unit'}),'rows'); vals=zeros(height(keys),4);
        for k=1:height(keys)
            z=q&raw.subset==keys.subset(k)&raw.unit==keys.unit(k);
            m=chapter3_diagnostic_lib.binary_metrics(raw.pred_label(z),raw.truth_label(z));
            vals(k,:)=[m.precision,m.DR,m.F1,m.FAR];
        end
        ri=ri+1; rows{ri}=table(regimes(ir),names(g),g,height(keys), ...
            mean(vals(:,1)),mean(vals(:,2)),mean(vals(:,3)),mean(vals(:,4)), ...
            'VariableNames',{'regime','family','family_index','n_units', ...
            'precision','recall','F1','FAR'});
    end
end
out=vertcat(rows{:});
end

function out=detection_confusion(raw)
d=unique(raw(:,{'regime','subset','unit','cycle','hs'}),'rows'); d.truth_fault=~d.hs;
p=groupsummary(raw,{'regime','subset','unit','cycle'},'max','pred_label');
d=innerjoin(d,p(:,{'regime','subset','unit','cycle','max_pred_label'}), ...
    'Keys',{'regime','subset','unit','cycle'}); d.pred_fault=logical(d.max_pred_label);
regimes=unique(d.regime); rows=cell(numel(regimes),1);
for ir=1:numel(regimes)
    z=d(d.regime==regimes(ir),:); C=zeros(2);
    for i=1:height(z)
        C(double(z.truth_fault(i))+1,double(z.pred_fault(i))+1)= ...
            C(double(z.truth_fault(i))+1,double(z.pred_fault(i))+1)+1;
    end
    [a,b]=ndgrid(0:1,0:1); rows{ir}=table(repmat(regimes(ir),4,1),a(:),b(:),C(:), ...
        'VariableNames',{'regime','truth','prediction','count'});
end
out=vertcat(rows{:});
end

function out=bootstrap_summary(t,B,seed)
metrics={'detection_FAR','detection_DR','detection_F1','detection_delay', ...
    'unit_false_alarm','pre_onset_lead','family_microF1','family_macroF1', ...
    'family_hamming_loss'};
old=rng; cleaner=onCleanup(@() rng(old)); %#ok<NASGU>
rng(seed,'twister'); regimes=unique(t.regime); rows=cell(0,1); ri=0;
for ig=1:numel(regimes)
    q=t(t.regime==regimes(ig),:);
    for im=1:numel(metrics)
        x=q.(metrics{im}); x=x(isfinite(x)); avg=NaN; lo=NaN; hi=NaN;
        if ~isempty(x)
            boots=zeros(B,1);
            for b=1:B, boots(b)=mean(x(randi(numel(x),numel(x),1))); end
            avg=mean(x); lo=prctile(boots,2.5); hi=prctile(boots,97.5);
        end
        ri=ri+1; rows{ri}=table(regimes(ig),string(metrics{im}),numel(x),avg,lo,hi, ...
            'VariableNames',{'regime','metric','n_units','mean','ci_low','ci_high'}); %#ok<AGROW>
    end
end
out=vertcat(rows{:});
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0);
c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
