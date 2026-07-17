function verdict=experiment_24_family_diagnosis()
%EXPERIMENT_24_FAMILY_DIAGNOSIS 将 v2 锁定的九参数标签合并为五物理部件族。
% 只读取正式 CSV/JSON；不重新反演、不重新标定，也不依赖 matlab/cache。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
baseDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure');
outDir=fullfile(matlabRoot,'outputs','t7_family_diagnosis');
if ~isfolder(outDir), mkdir(outDir); end

raw=readtable(fullfile(baseDir,'diagnostic_cycle_raw.csv'),'TextType','string');
v21=jsondecode(fileread(fullfile(baseDir,'verdict21.json')));
verification=jsondecode(fileread(fullfile(baseDir,'verification.json')));
assert(verification.pass && ~v21.fast_mode && ~v21.test_truth_used_for_selection, ...
    '实验24只能读取已经验证通过的 v2 正式结果。');

[cycleRaw,unitMetrics,familyMetrics,detectConf]=evaluate_locked_labels(raw);
verify_detection_unchanged(unitMetrics,fullfile(baseDir,'diagnostic_unit_metrics.csv'));

mapping=table(chapter3_family_lib.names().', ...
    ["1";"[2 3]";"[4 5]";"[6 7]";"[8 9]"], ...
    repmat("OR_of_locked_v2_parameter_labels",5,1), ...
    'VariableNames',{'family','parameter_indices','decision_rule'});
writetable(mapping,fullfile(outDir,'family_mapping.csv'),'Encoding','UTF-8');
writetable(cycleRaw,fullfile(outDir,'family_cycle_raw.csv'),'Encoding','UTF-8');
writetable(unitMetrics,fullfile(outDir,'family_unit_metrics.csv'),'Encoding','UTF-8');
writetable(familyMetrics,fullfile(outDir,'family_metrics.csv'),'Encoding','UTF-8');
writetable(detectConf,fullfile(outDir,'family_detection_confusion.csv'),'Encoding','UTF-8');
summary=bootstrap_summary(unitMetrics,2000,42+2410);
writetable(summary,fullfile(outDir,'family_unit_bootstrap_summary.csv'),'Encoding','UTF-8');

assert(height(familyMetrics)==10,'五个部件族必须在两个测试场景中全部保留。');
verdict=struct('analysis_type','five_physical_family_label_aggregation', ...
    'fast_mode',false,'source','t4_diagnostic_closure/diagnostic_cycle_raw.csv', ...
    'source_verification_pass',logical(verification.pass), ...
    'family_names',{cellstr(chapter3_family_lib.names())}, ...
    'family_groups',{{1,[2 3],[4 5],[6 7],[8 9]}}, ...
    'decision_rule','OR_of_locked_v2_parameter_labels', ...
    'severity_definition','RMS_of_locked_v2_normalized_parameter_estimates', ...
    'thresholds_recalibrated',false,'persistence_reselected',false, ...
    'continuous_inverse_recomputed',false,'detection_parity_with_v2',true, ...
    'test_truth_used_for_selection',false,'stage_model_changed',false, ...
    'unknown_protocol_changed',false,'protocol','docs/部件族诊断补充协议.md');
write_json(fullfile(outDir,'verdict24.json'),verdict);
fprintf('五部件族诊断完成：%d 台发动机；检测结果与 v2 逐发动机一致。\n',height(unitMetrics));
end

function [cycleRaw,unitMetrics,familyMetrics,detectConf]=evaluate_locked_labels(raw)
names=chapter3_family_lib.names();
keys=unique(raw(:,{'regime','subset','unit'}),'rows');
cycleRows=cell(0,1); unitRows=cell(height(keys),1); ri=0;
for i=1:height(keys)
    q=raw.regime==keys.regime(i)&raw.subset==keys.subset(i)&raw.unit==keys.unit(i);
    z=raw(q,:); cycles=sort(unique(z.cycle)); T=numel(cycles);
    thetaHat=zeros(T,9); truthParameter=false(T,9); predParameter=false(T,9);
    hs=false(T,1);
    for t=1:T
        for j=1:9
            pick=z.cycle==cycles(t)&z.component==j;
            assert(nnz(pick)==1,'每个 unit/cycle/parameter 必须且只能有一条记录。');
            thetaHat(t,j)=z.theta_hat(pick);
            truthParameter(t,j)=z.truth_label(pick)~=0;
            predParameter(t,j)=z.pred_label(pick)~=0;
            if j==1, hs(t)=z.hs(pick)~=0; end
        end
    end
    truthFamily=chapter3_family_lib.aggregate_labels(truthParameter);
    predFamily=chapter3_family_lib.aggregate_labels(predParameter);
    severity=chapter3_family_lib.severity(thetaHat);
    pred=struct('family_label',predFamily,'fault',any(predFamily,2));
    truth=struct('family_label',truthFamily,'fault',~hs);
    met=chapter3_family_lib.sequence_metrics(pred,truth,cycles);
    row=struct2table(met); row.regime=keys.regime(i); row.subset=keys.subset(i);
    row.unit=keys.unit(i); unitRows{i}=row;
    for t=1:T
        for g=1:5
            ri=ri+1; cycleRows{ri}=table(keys.regime(i),keys.subset(i),keys.unit(i), ...
                cycles(t),names(g),g,hs(t),severity(t,g),truthFamily(t,g),predFamily(t,g), ...
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

function verify_detection_unchanged(current,path)
old=readtable(path,'TextType','string');
metrics={'detection_FAR','detection_DR','detection_precision','detection_F1','detection_delay'};
assert(height(current)==height(old),'实验24与 v2 的发动机数量不一致。');
for i=1:height(current)
    q=old.regime==current.regime(i)&old.subset==current.subset(i)&old.unit==current.unit(i);
    assert(nnz(q)==1,'无法唯一匹配 v2 unit 结果。');
    for m=1:numel(metrics)
        a=current.(metrics{m})(i); b=old.(metrics{m})(q);
        assert((isnan(a)&&isnan(b)) || abs(a-b)<=1e-12*max([1,abs(a),abs(b)]), ...
            '部件族聚合意外改变了 v2 检测指标：%s。',metrics{m});
    end
end
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
