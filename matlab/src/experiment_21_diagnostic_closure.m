function verdict=experiment_21_diagnostic_closure(fastMode,splitSeed)
%EXPERIMENT_21_DIAGNOSTIC_CLOSURE 健康参数估计后的诊断闭环正式实验。
% 训练、标定、分布内测试按发动机 unit 完全隔离；DS02/DS03 仅作未见组合测试。

if nargin<1, fastMode=true; end
if nargin<2, splitSeed=314159; end
srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
cfg=ncmapss_lib.config(fastMode); dcfg=chapter3_diagnostic_lib.config(cfg);
cfg.SPLIT_SEED=splitSeed;
if fastMode
    maxUnits=1; windows=1000; lambdaGrid=cfg.LAMBDA_GRID(1:2:end);
    outDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure_smoke');
else
    maxUnits=inf; windows=1000; lambdaGrid=cfg.LAMBDA_GRID;
    if splitSeed==314159
        outDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure');
    else
        outDir=fullfile(matlabRoot,'outputs',sprintf('t4_diagnostic_closure_seed_%d',splitSeed));
    end
end
if ~isfolder(outDir), mkdir(outDir); end
logFile=fullfile(outDir,'run.log'); if isfile(logFile), delete(logFile); end
diary(logFile); cleaner=onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('%s\n第3章诊断闭环，FAST_MODE=%d\n%s\n',repmat('=',1,96),fastMode,repmat('=',1,96));
fprintf('冻结协议：unit 三划分；lambda 仅由 calibration 选择；alpha=%.3f；K 候选=%s。\n', ...
    dcfg.alpha,mat2str(dcfg.persistence_grid));

protocolDir=fullfile(matlabRoot,'outputs','protocol');
if splitSeed==314159
    manifestName='unit_split_manifest.csv'; cacheName='pipeline_cache_split_v2.mat';
else
    manifestName=sprintf('unit_split_manifest_seed_%d.csv',splitSeed);
    cacheName=sprintf('pipeline_cache_split_v2_seed_%d.mat',splitSeed);
end
manifest=ncmapss_lib.make_unit_split_manifest(cfg,fullfile(protocolDir,manifestName));
stream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
pipe=ncmapss_lib.build_pipeline(cfg,stream, ...
    fullfile(matlabRoot,'cache',cacheName),false,manifest);

dataStream=ncmapss_lib.make_stream(cfg.SEED+21,cfg.RNG_BACKEND);
calRaw=load_id_role(cfg,pipe,manifest,'calibration',windows,maxUnits,dataStream);
[lambdaSelected,lambdaTable]=select_lambda(calRaw,pipe.Hn,lambdaGrid);
writetable(lambdaTable,fullfile(outDir,'lambda_calibration.csv'),'Encoding','UTF-8');
fprintf('calibration 锁定 lambda = %.8g。\n',lambdaSelected);

calPred=attach_predictions(calRaw,pipe.Hn,lambdaSelected);
model=chapter3_diagnostic_lib.fit(calPred,dcfg,pipe.Hn);
writetable(model.persistence_selection,fullfile(outDir,'persistence_calibration.csv'),'Encoding','UTF-8');
save(fullfile(outDir,'diagnostic_model.mat'),'model','lambdaSelected','manifest','-v7.3');

idRaw=load_id_role(cfg,pipe,manifest,'test_id',windows,maxUnits,dataStream);
oodRaw=load_external(cfg,pipe,windows,maxUnits,dataStream);
save(fullfile(outDir,'closure_sequences.mat'),'calRaw','idRaw','oodRaw','-v7.3');
testPred=[attach_predictions(idRaw,pipe.Hn,lambdaSelected), ...
    attach_predictions(oodRaw,pipe.Hn,lambdaSelected)];
[cycleRaw,unitMetrics,componentMetrics,stageConf,detectConf]=evaluate_sequences( ...
    testPred,model,pipe.Hn,cfg);
writetable(cycleRaw,fullfile(outDir,'diagnostic_cycle_raw.csv'),'Encoding','UTF-8');
writetable(unitMetrics,fullfile(outDir,'diagnostic_unit_metrics.csv'),'Encoding','UTF-8');
writetable(componentMetrics,fullfile(outDir,'component_metrics.csv'),'Encoding','UTF-8');
writetable(stageConf,fullfile(outDir,'stage_confusion.csv'),'Encoding','UTF-8');
writetable(detectConf,fullfile(outDir,'detection_confusion.csv'),'Encoding','UTF-8');

summary=unit_bootstrap_summary(unitMetrics,2000,cfg.SEED+2210);
writetable(summary,fullfile(outDir,'unit_bootstrap_summary.csv'),'Encoding','UTF-8');
stageSensitivity=stage_sensitivity(calPred,testPred,model,dcfg,pipe.Hn,cfg);
writetable(stageSensitivity,fullfile(outDir,'stage_component_sensitivity.csv'),'Encoding','UTF-8');
[unknownFolds,unknownPoints]=unknown_family_experiment(calRaw,idRaw,pipe.Hn,dcfg,cfg);
writetable(unknownFolds,fullfile(outDir,'unknown_family_summary.csv'),'Encoding','UTF-8');
writetable(unknownPoints,fullfile(outDir,'unknown_family_points.csv'),'Encoding','UTF-8');

verdict=struct('analysis_type','unit_isolated_diagnostic_closure', ...
    'fast_mode',fastMode,'split_signature',pipe.split_signature, ...
    'split_seed',splitSeed, ...
    'lambda_selected_on_calibration',lambdaSelected, ...
    'component_threshold',model.component_threshold, ...
    'persistence',model.persistence,'stage_components',model.stage.K, ...
    'unknown_threshold',model.unknown_threshold, ...
    'n_calibration_units',numel(calRaw),'n_test_units',numel(testPred), ...
    'test_truth_used_for_selection',false,'stage_is_engineering_limit',false, ...
    'unknown_protocol','leave_one_fault_family_out');
write_json(fullfile(outDir,'verdict21.json'),verdict);
fprintf('闭环原始结果已落盘：%d 条 cycle-component 记录，%d 台测试发动机。\n', ...
    height(cycleRaw),height(unitMetrics));
end

function seqs=load_id_role(cfg,pipe,manifest,role,N,maxUnits,stream)
seqs={};
for i=1:size(cfg.IDENT_FILES,1)
    file=cfg.IDENT_FILES{i,1}; faults=cfg.IDENT_FILES{i,2};
    units=sort(ncmapss_lib.units_for_role(manifest,file,role));
    if isfinite(maxUnits), units=units(1:min(maxUnits,numel(units))); end
    q=load_file_units(cfg,pipe,file,'dev',units,faults,[ncmapss_lib.subset_tag(file) '_id'], ...
        role,N,stream);
    seqs=[seqs,q]; %#ok<AGROW>
end
end

function seqs=load_external(cfg,pipe,N,maxUnits,stream)
seqs={};
for i=1:size(cfg.VALID_FILES,1)
    subset=cfg.VALID_FILES{i,1}; file=cfg.VALID_FILES{i,2}; faults=cfg.VALID_FILES{i,3};
    path=fullfile(cfg.DATA_DIR,file);
    aNames=ncmapss_lib.get_names(path,'A_var'); A=double(h5read(path,'/A_test')).';
    units=sort(unique(A(:,strcmp(aNames,'unit'))));
    if isfinite(maxUnits), units=units(1:min(maxUnits,numel(units))); end
    q=load_file_units(cfg,pipe,file,'test',units,faults,subset,'ood_combo',N,stream);
    seqs=[seqs,q]; %#ok<AGROW>
end
end

function seqs=load_file_units(cfg,pipe,file,split,units,faults,subset,regime,N,stream)
if isempty(units), seqs={}; return; end
d=ncmapss_lib.load_per_cycle(fullfile(cfg.DATA_DIR,file),split,cfg,stream,units);
[d,~]=ncmapss_lib.add_corrected(d,pipe.ref); d=ncmapss_lib.attach_residuals(pipe,d);
idxTrue=find(ismember(cfg.THETA9,faults)); seqs=cell(1,numel(units));
for iu=1:numel(units)
    u=units(iu); wins=ncmapss_lib.build_windows(d,u,'full',N,cfg,stream); w=wins(1);
    du=d(d.unit==u,:); cycles=sort(unique(du.cycle)); hs=false(numel(cycles),1);
    for t=1:numel(cycles)
        z=du.hs(du.cycle==cycles(t)); hs(t)=mean(z)>=.5;
    end
    seqs{iu}=struct('regime',regime,'subset',subset,'file',file,'unit',double(u), ...
        'cycles',double(cycles(:)),'hs',hs,'R',w.R,'theta_true',w.TH, ...
        'fault_idx',idxTrue,'N',N);
end
end

function [selected,out]=select_lambda(seqs,Hn,grid)
rows=cell(numel(grid)*numel(seqs),1); ri=0;
for il=1:numel(grid)
    for i=1:numel(seqs)
        s=seqs{i}; T=size(s.R,1); M=ncmapss_lib.build_M(Hn,T); Cum=ncmapss_lib.build_cum(T,size(Hn,2));
        th=ncmapss_lib.solve_D(M,Cum,s.R,grid(il),T,size(Hn,2));
        ri=ri+1; rows{ri}=table(grid(il),string(s.subset),s.unit, ...
            mean((th-s.theta_true).^2,'all'), ...
            'VariableNames',{'lambda','subset','unit','MSE'});
    end
end
raw=vertcat(rows{:});
[G,lambda]=findgroups(raw.lambda); meanMSE=splitapply(@mean,raw.MSE,G);
unitCount=splitapply(@numel,raw.MSE,G); out=table(lambda,unitCount,meanMSE);
[~,i]=min(out.meanMSE); selected=out.lambda(i); out.selected=out.lambda==selected;
end

function seqs=attach_predictions(raw,Hn,lambda)
seqs=raw;
for i=1:numel(raw)
    s=raw{i}; T=size(s.R,1); M=ncmapss_lib.build_M(Hn,T); Cum=ncmapss_lib.build_cum(T,size(Hn,2));
    s.theta_hat=ncmapss_lib.solve_D(M,Cum,s.R,lambda,T,size(Hn,2)); seqs{i}=s;
end
end

function [cycleRaw,unitMetrics,componentMetrics,stageConf,detectConf]=evaluate_sequences(seqs,model,Hn,cfg)
cycleRows=cell(0,1); unitRows=cell(numel(seqs),1); ci=0;
for i=1:numel(seqs)
    s=seqs{i}; pred=chapter3_diagnostic_lib.predict(s.theta_hat,s.R,Hn,model);
    truth=chapter3_diagnostic_lib.truth(s.theta_true,s.hs,model.stage, ...
        model.truth_active_eps,s.fault_idx);
    met=chapter3_diagnostic_lib.sequence_metrics(pred,truth,s.cycles);
    unitRows{i}=struct2table(with_keys(met,s));
    for t=1:numel(s.cycles)
        for j=1:numel(cfg.THETA9)
            ci=ci+1;
            cycleRows{ci,1}=table(string(s.regime),string(s.subset),s.unit,s.cycles(t), ...
                string(cfg.THETA9{j}),j,s.hs(t),s.theta_true(t,j),s.theta_hat(t,j), ...
                truth.component_label(t,j),pred.component_label(t,j), ...
                truth.component_stage(t,j),pred.component_stage(t,j), ...
                truth.stage(t),pred.stage(t),pred.reconstruction_score(t),pred.unknown(t), ...
                'VariableNames',{'regime','subset','unit','cycle','parameter','component', ...
                'hs','theta_true','theta_hat','truth_label','pred_label', ...
                'truth_component_stage','pred_component_stage','truth_stage','pred_stage', ...
                'reconstruction_score','unknown'});
        end
    end
end
cycleRaw=vertcat(cycleRows{:}); unitMetrics=vertcat(unitRows{:});
componentMetrics=component_summary(cycleRaw,cfg);
stageConf=grouped_confusion(cycleRaw(:,{'regime','subset','unit','cycle','truth_stage','pred_stage'}), ...
    'truth_stage','pred_stage',0:3);
d=unique(cycleRaw(:,{'regime','subset','unit','cycle','hs'}),'rows');
d.truth_fault=~d.hs;
p=groupsummary(cycleRaw,{'regime','subset','unit','cycle'},'max','pred_label');
d=innerjoin(d,p(:,{'regime','subset','unit','cycle','max_pred_label'}), ...
    'Keys',{'regime','subset','unit','cycle'});
d.pred_fault=logical(d.max_pred_label);
detectConf=grouped_confusion(d,'truth_fault','pred_fault',[0 1]);
end

function s=with_keys(met,seq)
s=met; s.regime=string(seq.regime); s.subset=string(seq.subset); s.unit=seq.unit;
end

function out=component_summary(raw,cfg)
rows=cell(0,1); ri=0; regimes=unique(raw.regime);
for ir=1:numel(regimes)
    for j=1:numel(cfg.THETA9)
        q=raw.regime==regimes(ir)&raw.component==j;
        keys=unique(raw(q,{'subset','unit'}),'rows'); vals=zeros(height(keys),4);
        for k=1:height(keys)
            z=q&raw.subset==keys.subset(k)&raw.unit==keys.unit(k);
            m=chapter3_diagnostic_lib.binary_metrics(raw.pred_label(z),raw.truth_label(z));
            vals(k,:)=[m.precision,m.DR,m.F1,m.FAR];
        end
        ri=ri+1; rows{ri}=table(regimes(ir),string(cfg.THETA9{j}),j,height(keys), ...
            mean(vals(:,1)),mean(vals(:,2)),mean(vals(:,3)),mean(vals(:,4)), ...
            'VariableNames',{'regime','parameter','component','n_units', ...
            'precision','recall','F1','FAR'});
    end
end
out=vertcat(rows{:});
end

function out=confusion_table(t,trueName,predName,levels)
t=unique(t,'rows'); C=zeros(numel(levels));
for i=1:height(t)
    a=find(levels==double(t.(trueName)(i)),1); b=find(levels==double(t.(predName)(i)),1);
    if ~isempty(a)&&~isempty(b), C(a,b)=C(a,b)+1; end
end
[a,b]=ndgrid(levels,levels); out=table(a(:),b(:),C(:), ...
    'VariableNames',{'truth','prediction','count'});
end

function out=grouped_confusion(t,trueName,predName,levels)
groups=unique(t.regime); rows=cell(numel(groups),1);
for i=1:numel(groups)
    z=confusion_table(t(t.regime==groups(i),:),trueName,predName,levels);
    z.regime=repmat(groups(i),height(z),1); z=movevars(z,'regime','Before',1); rows{i}=z;
end
out=vertcat(rows{:});
end

function out=unit_bootstrap_summary(t,B,seed)
metrics={'detection_FAR','detection_DR','detection_F1','detection_delay', ...
    'isolation_microF1','isolation_macroF1','hamming_loss', ...
    'stage_weighted_kappa','stage_MAE','unknown_rate'};
rows=cell(0,1); ri=0; regimes=unique(t.regime); old=rng; cleanup=onCleanup(@() rng(old)); %#ok<NASGU>
rng(seed,'twister');
for ig=1:numel(regimes)
    q=t(t.regime==regimes(ig),:);
    for im=1:numel(metrics)
        x=q.(metrics{im}); x=x(isfinite(x));
        if isempty(x)
            avg=NaN; lo=NaN; hi=NaN;
        else
            boots=zeros(B,1);
            for b=1:B, boots(b)=mean(x(randi(numel(x),numel(x),1))); end
            avg=mean(x); lo=prctile(boots,2.5); hi=prctile(boots,97.5);
        end
        ri=ri+1; rows{ri}=table(regimes(ig),string(metrics{im}),numel(x),avg,lo,hi, ...
            'VariableNames',{'regime','metric','n_units','mean','ci_low','ci_high'});
    end
end
out=vertcat(rows{:});
end

function out=stage_sensitivity(calPred,testPred,model,dcfg,Hn,cfg)
x=collect_true_severity(calPred,dcfg.truth_active_eps,dcfg.stage_samples_per_unit); rows=cell(0,1); ri=0;
for K=dcfg.stage_component_sensitivity
    m=model; m.stage=chapter3_diagnostic_lib.fit_stage_gmm(x,K,dcfg.gmm_replicates,dcfg.seed+K);
    vals=zeros(numel(testPred),2);
    for i=1:numel(testPred)
        s=testPred{i}; p=chapter3_diagnostic_lib.predict(s.theta_hat,s.R,Hn,m);
        y=chapter3_diagnostic_lib.truth(s.theta_true,s.hs,m.stage,m.truth_active_eps,s.fault_idx);
        [vals(i,1),vals(i,2)]=chapter3_diagnostic_lib.ordinal_metrics(p.stage,y.stage);
    end
    for ir=1:2
        regimes={'test_id','ood_combo'}; q=strcmp(cellfun(@(s)s.regime,testPred,'UniformOutput',false),regimes{ir});
        ri=ri+1; rows{ri}=table(K,string(regimes{ir}),sum(q),mean(vals(q,1),'omitnan'), ...
            mean(vals(q,2),'omitnan'),'VariableNames', ...
            {'stage_components','regime','n_units','weighted_kappa','stage_MAE'});
    end
end
out=vertcat(rows{:});
end

function x=collect_true_severity(seqs,epsActive,maxPerUnit)
x=zeros(0,1);
for i=1:numel(seqs)
    z=max(0,-seqs{i}.theta_true); v=z(z>epsActive);
    if numel(v)>maxPerUnit, v=v(round(linspace(1,numel(v),maxPerUnit))); end
    x=[x;v(:)]; %#ok<AGROW>
end
end

function [folds,points]=unknown_family_experiment(calRaw,testRaw,Hn,dcfg,cfg)
families={"HPT_eff",1;"Fan",[2 3];"HPC",[4 5];"LPT",[6 7];"LPC",[8 9]};
foldRows=cell(0,1); pointRows=cell(0,1); fi=0; pi=0;
for f=1:size(families,1)
    name=families{f,1}; omitted=families{f,2}; knownIdx=setdiff(1:size(Hn,2),omitted);
    calKnown=calRaw(cellfun(@(s)isempty(intersect(s.fault_idx,omitted)),calRaw));
    testKnown=testRaw(cellfun(@(s)isempty(intersect(s.fault_idx,omitted)),testRaw));
    testUnknown=testRaw(cellfun(@(s)~isempty(intersect(s.fault_idx,omitted)),testRaw));
    if isempty(calKnown)||isempty(testKnown)||isempty(testUnknown), continue; end
    foldLambda=select_reduced_lambda(calKnown,Hn(:,knownIdx),knownIdx,cfg.LAMBDA_GRID);
    calScore=cellfun(@(s)reduced_unit_score(s,Hn,knownIdx,foldLambda),calKnown);
    tau=prctile(calScore,100*(1-dcfg.unknown_alpha));
    allTest=[testKnown,testUnknown]; label=[false(numel(testKnown),1);true(numel(testUnknown),1)];
    score=cellfun(@(s)reduced_unit_score(s,Hn,knownIdx,foldLambda),allTest).';
    [aucRoc,aucPr]=chapter3_diagnostic_lib.binary_auc(score,label);
    pred=score>tau; knownAccept=mean(~pred(~label)); unknownReject=mean(pred(label));
    fi=fi+1; foldRows{fi}=table(string(name),string(mat2str(omitted)),foldLambda,tau, ...
        numel(calKnown),numel(testKnown),numel(testUnknown),aucRoc,aucPr,knownAccept,unknownReject, ...
        'VariableNames',{'family','omitted_components','lambda','threshold','n_calibration_known', ...
        'n_test_known','n_test_unknown','AUROC','AUPRC','known_accept_rate','unknown_reject_rate'});
    for i=1:numel(allTest)
        pi=pi+1; pointRows{pi}=table(string(name),string(allTest{i}.subset),allTest{i}.unit, ...
            label(i),score(i),tau,pred(i),'VariableNames', ...
            {'family','subset','unit','is_unknown','score','threshold','rejected'});
    end
end
folds=vertcat(foldRows{:}); points=vertcat(pointRows{:});
end

function selected=select_reduced_lambda(seqs,H,knownIdx,grid)
loss=zeros(numel(grid),1);
for il=1:numel(grid)
    unitLoss=zeros(numel(seqs),1);
    for i=1:numel(seqs)
        s=seqs{i}; T=size(s.R,1); n=size(H,2);
        th=ncmapss_lib.solve_D(ncmapss_lib.build_M(H,T), ...
            ncmapss_lib.build_cum(T,n),s.R,grid(il),T,n);
        unitLoss(i)=mean((th-s.theta_true(:,knownIdx)).^2,'all');
    end
    loss(il)=mean(unitLoss);
end
[~,i]=min(loss); selected=grid(i);
end

function score=reduced_unit_score(s,Hn,knownIdx,lambda)
H=Hn(:,knownIdx); T=size(s.R,1); M=ncmapss_lib.build_M(H,T); Cum=ncmapss_lib.build_cum(T,numel(knownIdx));
th=ncmapss_lib.solve_D(M,Cum,s.R,lambda,T,numel(knownIdx));
q=chapter3_diagnostic_lib.reconstruction_score(s.R,th,H); score=prctile(q,95);
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
