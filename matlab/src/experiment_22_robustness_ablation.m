function verdict=experiment_22_robustness_ablation(fastMode)
%EXPERIMENT_22_ROBUSTNESS_ABLATION 约束消融与传感器扰动鲁棒性。

if nargin<1, fastMode=true; end
srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
cfg=ncmapss_lib.config(fastMode); dcfg=chapter3_diagnostic_lib.config(cfg);
baseDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure');
if fastMode, baseDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure_smoke'); end
outDir=fullfile(matlabRoot,'outputs','t5_robustness_ablation');
if fastMode, outDir=fullfile(matlabRoot,'outputs','t5_robustness_ablation_smoke'); end
if ~isfolder(outDir), mkdir(outDir); end

S=load(fullfile(baseDir,'closure_sequences.mat'),'calRaw','idRaw','oodRaw');
D=load(fullfile(baseDir,'diagnostic_model.mat'),'model','lambdaSelected');
P=load(fullfile(matlabRoot,'cache','pipeline_cache_split_v2.mat'),'pipe'); pipe=P.pipe;
if isempty(pipe.Hn), error('chapter3:MissingHn','split pipeline 缓存缺少 Hn。'); end
if isempty(pipe.base) && isfield(pipe,'base_paths')
    pipe.base=ncmapss_lib.load_python_models(pipe.base_paths);
end
testRaw=[S.idRaw,S.oodRaw];

fprintf('预注册消融：full D、无约束 B、仅非正、D 无收缩；每个方法单独用 calibration 标定决策层。\n');
methods=["D_full","B_unconstrained","sign_only","D_no_shrink"];
abRows=cell(0,1); ai=0;
for im=1:numel(methods)
    cal=solve_sequences(S.calRaw,pipe.Hn,D.lambdaSelected,methods(im));
    model=chapter3_diagnostic_lib.fit(cal,dcfg,pipe.Hn);
    test=solve_sequences(testRaw,pipe.Hn,D.lambdaSelected,methods(im));
    for i=1:numel(test)
        ai=ai+1; abRows{ai}=score_unit(test{i},model,pipe.Hn,methods(im)); %#ok<AGROW>
    end
end
ablation=vertcat(abRows{:});
writetable(ablation,fullfile(outDir,'ablation_unit_metrics.csv'),'Encoding','UTF-8');

fprintf('预注册扰动：标准化残差高斯噪声、单通道固定偏置、逐通道缺失；阈值不重新标定。\n');
rbRows=cell(0,1); ri=0; seeds=cfg.SEED+(1:5);
if fastMode, seeds=seeds(1); end
noiseLevels=[0 .25 .5 1]; biasLevels=[.25 .5 1];
for seed=seeds
    for a=noiseLevels
        z=perturb_and_solve(testRaw,pipe.Hn,D.lambdaSelected,'noise',a,seed);
        for i=1:numel(z), ri=ri+1; rbRows{ri}=score_robust(z{i},D.model,pipe.Hn, ...
                'noise',a,seed,NaN); end %#ok<AGROW>
    end
    for a=biasLevels
        z=perturb_and_solve(testRaw,pipe.Hn,D.lambdaSelected,'bias',a,seed);
        for i=1:numel(z), ri=ri+1; rbRows{ri}=score_robust(z{i},D.model,pipe.Hn, ...
                'bias',a,seed,z{i}.perturbed_channel); end %#ok<AGROW>
    end
end
for channel=1:size(pipe.Hn,1)
    z=missing_channel_solve(testRaw,pipe.Hn,D.lambdaSelected,channel);
    for i=1:numel(z), ri=ri+1; rbRows{ri}=score_robust(z{i},D.model,pipe.Hn, ...
            'missing_channel',1,0,channel); end %#ok<AGROW>
end
robustness=vertcat(rbRows{:});
writetable(robustness,fullfile(outDir,'robustness_unit_metrics.csv'),'Encoding','UTF-8');
summary=summarize_robustness(robustness);
writetable(summary,fullfile(outDir,'robustness_summary.csv'),'Encoding','UTF-8');

baseline=baseline_cycle_sensitivity(cfg,pipe,D.model,D.lambdaSelected,fastMode);
writetable(baseline,fullfile(outDir,'baseline_cycle_sensitivity.csv'),'Encoding','UTF-8');

verdict=struct('analysis_type','preregistered_ablation_and_sensor_perturbation', ...
    'fast_mode',fastMode,'methods',{cellstr(methods)},'noise_sigma_units',noiseLevels, ...
    'bias_sigma_units',biasLevels,'missing_channel_folds',size(pipe.Hn,1), ...
    'baseline_reference_cycles',[1 3 5], ...
    'perturbation_seeds',seeds,'thresholds_recalibrated_for_perturbation',false, ...
    'decision_layer_recalibrated_per_ablation',true);
write_json(fullfile(outDir,'verdict22.json'),verdict);
end

function out=solve_sequences(raw,Hn,lambda,method)
out=raw;
for i=1:numel(raw)
    s=raw{i}; s.theta_hat=solve_one(s.R,Hn,lambda,method); out{i}=s;
end
end

function th=solve_one(R,Hn,lambda,method)
T=size(R,1); n=size(Hn,2);
switch string(method)
    case "D_full"
        th=ncmapss_lib.solve_D(ncmapss_lib.build_M(Hn,T), ...
            ncmapss_lib.build_cum(T,n),R,lambda,T,n);
    case "D_no_shrink"
        th=ncmapss_lib.solve_D(ncmapss_lib.build_M(Hn,T), ...
            ncmapss_lib.build_cum(T,n),R,0,T,n);
    case "B_unconstrained"
        th=ncmapss_lib.solve_B(Hn,R,lambda);
    case "sign_only"
        A=[-Hn;sqrt(lambda)*eye(n)]; opts=optimset('Display','off'); th=zeros(T,n);
        for t=1:T
            d=lsqnonneg(A,[R(t,:).';zeros(n,1)],opts); th(t,:)=-d.';
        end
    otherwise
        error('chapter3:UnknownAblation','未知消融方法 %s。',method);
end
end

function row=score_unit(s,model,Hn,method)
p=chapter3_diagnostic_lib.predict(s.theta_hat,s.R,Hn,model);
y=chapter3_diagnostic_lib.truth(s.theta_true,s.hs,model.stage,model.truth_active_eps,s.fault_idx);
m=chapter3_diagnostic_lib.sequence_metrics(p,y,s.cycles);
other=setdiff(1:size(s.theta_true,2),s.fault_idx);
row=table(string(method),string(s.regime),string(s.subset),s.unit, ...
    sqrt(mean((s.theta_hat-s.theta_true).^2,'all')),mean(abs(s.theta_hat(:,other)),'all'), ...
    m.detection_FAR,m.detection_DR,m.detection_F1,m.isolation_macroF1, ...
    m.stage_weighted_kappa,m.stage_MAE, ...
    'VariableNames',{'method','regime','subset','unit','RMSE','inactive_component_error', ...
    'detection_FAR','detection_DR','detection_F1','isolation_macroF1','stage_kappa','stage_MAE'});
end

function out=perturb_and_solve(raw,Hn,lambda,kind,amplitude,seed)
out=raw; old=rng; cleanup=onCleanup(@() rng(old)); %#ok<NASGU> 
rng(seed,'twister');
for i=1:numel(raw)
    s=raw{i}; R=s.R;
    if strcmp(kind,'noise')
        R=R+amplitude*randn(size(R)); s.perturbed_channel=NaN;
    else
        channel=randi(size(R,2)); signBias=2*(rand>.5)-1;
        R(:,channel)=R(:,channel)+signBias*amplitude; s.perturbed_channel=channel;
    end
    s.R_perturbed=R; s.theta_hat=solve_one(R,Hn,lambda,'D_full'); out{i}=s;
end
end

function out=missing_channel_solve(raw,Hn,lambda,channel)
out=raw; keep=setdiff(1:size(Hn,1),channel); H=Hn(keep,:);
for i=1:numel(raw)
    s=raw{i}; s.R_perturbed=s.R; s.theta_hat=solve_one(s.R(:,keep),H,lambda,'D_full');
    s.perturbed_channel=channel; out{i}=s;
end
end

function row=score_robust(s,model,Hn,kind,amplitude,seed,channel)
if isfield(s,'R_perturbed'), R=s.R_perturbed; else, R=s.R; end
% 缺失通道只影响反演，拒识分数仍在完整可观测通道上计算，不进入本实验指标。
p=chapter3_diagnostic_lib.predict(s.theta_hat,s.R,Hn,model);
y=chapter3_diagnostic_lib.truth(s.theta_true,s.hs,model.stage,model.truth_active_eps,s.fault_idx);
m=chapter3_diagnostic_lib.sequence_metrics(p,y,s.cycles);
row=table(string(kind),amplitude,seed,channel,string(s.regime),string(s.subset),s.unit, ...
    sqrt(mean((s.theta_hat-s.theta_true).^2,'all')),m.detection_F1,m.isolation_macroF1, ...
    m.stage_weighted_kappa,m.stage_MAE,mean((R-s.R).^2,'all'), ...
    'VariableNames',{'perturbation','amplitude','seed','channel','regime','subset','unit', ...
    'RMSE','detection_F1','isolation_macroF1','stage_kappa','stage_MAE','input_MSE'});
end

function out=summarize_robustness(t)
[G,kind,amplitude,regime]=findgroups(t.perturbation,t.amplitude,t.regime);
out=table(kind,amplitude,regime,splitapply(@numel,t.RMSE,G), ...
    splitapply(@mean,t.RMSE,G),splitapply(@mean,t.detection_F1,G), ...
    splitapply(@mean,t.isolation_macroF1,G),splitapply(@(x)mean(x,'omitnan'),t.stage_kappa,G), ...
    'VariableNames',{'perturbation','amplitude','regime','n_unit_runs', ...
    'mean_RMSE','mean_detection_F1','mean_isolation_macroF1','mean_stage_kappa'});
end

function out=baseline_cycle_sensitivity(cfg,pipe,model,lambda,fastMode)
manifest=readtable(fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'outputs','protocol','unit_split_manifest.csv'),'TextType','string');
rows=cell(0,1); ri=0; refs=[1 3 5]; maxUnits=inf; if fastMode, maxUnits=1; end
for nRef=refs
    localCfg=cfg; localCfg.N_REF_CYCLES=nRef;
    stream=ncmapss_lib.make_stream(cfg.SEED+902,cfg.RNG_BACKEND);
    seqs={};
    for i=1:size(cfg.IDENT_FILES,1)
        file=cfg.IDENT_FILES{i,1}; faults=cfg.IDENT_FILES{i,2};
        units=sort(ncmapss_lib.units_for_role(manifest,file,'test_id'));
        if isfinite(maxUnits), units=units(1:min(maxUnits,numel(units))); end
        seqs=[seqs,load_sensitivity_file(localCfg,pipe,file,'dev',units,faults, ...
            [ncmapss_lib.subset_tag(file) '_id'],'test_id',stream)]; %#ok<AGROW>
    end
    for i=1:size(cfg.VALID_FILES,1)
        file=cfg.VALID_FILES{i,2}; faults=cfg.VALID_FILES{i,3}; subset=cfg.VALID_FILES{i,1};
        path=fullfile(cfg.DATA_DIR,file); aNames=ncmapss_lib.get_names(path,'A_var');
        A=double(h5read(path,'/A_test')).'; units=sort(unique(A(:,strcmp(aNames,'unit'))));
        if isfinite(maxUnits), units=units(1:min(maxUnits,numel(units))); end
        seqs=[seqs,load_sensitivity_file(localCfg,pipe,file,'test',units,faults, ...
            subset,'ood_combo',stream)]; %#ok<AGROW>
    end
    solved=solve_sequences(seqs,pipe.Hn,lambda,'D_full');
    for i=1:numel(solved)
        r=score_unit(solved{i},model,pipe.Hn,'D_full'); r.n_reference_cycles=nRef;
        ri=ri+1; rows{ri}=r; %#ok<AGROW>
    end
end
out=vertcat(rows{:});
end

function seqs=load_sensitivity_file(cfg,pipe,file,split,units,faults,subset,regime,stream)
if isempty(units), seqs={}; return; end
d=ncmapss_lib.load_per_cycle(fullfile(cfg.DATA_DIR,file),split,cfg,stream,units);
[d,~]=ncmapss_lib.add_corrected(d,pipe.ref); d=ncmapss_lib.attach_residuals(pipe,d);
idxTrue=find(ismember(cfg.THETA9,faults)); seqs=cell(1,numel(units));
for iu=1:numel(units)
    u=units(iu); w=ncmapss_lib.build_windows(d,u,'full',1000,cfg,stream); w=w(1);
    du=d(d.unit==u,:); cycles=sort(unique(du.cycle)); hs=false(numel(cycles),1);
    for t=1:numel(cycles), hs(t)=mean(du.hs(du.cycle==cycles(t)))>=.5; end
    seqs{iu}=struct('regime',regime,'subset',subset,'file',file,'unit',double(u), ...
        'cycles',double(cycles(:)),'hs',hs,'R',w.R,'theta_true',w.TH, ...
        'fault_idx',idxTrue,'N',1000);
end
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
