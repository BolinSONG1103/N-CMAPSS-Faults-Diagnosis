function verdict = experiment_20(fastMode)
% T2：数据驱动对照臂 × 跨故障组合外推。

if nargin<1, fastMode=true; end
srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
cfg=ncmapss_lib.config(fastMode); ncmapss_lib.set_plot_defaults();
if fastMode
    windows=[10 1000]; epochs=8; maxUnits=2; unitLimit=3;
    outDir=fullfile(matlabRoot,'outputs','t2_dd_smoke');
else
    windows=[1 10 100 1000]; epochs=40; maxUnits=inf; unitLimit=inf;
    outDir=fullfile(matlabRoot,'outputs','t2_dd');
end
if ~isfolder(outDir), mkdir(outDir); end
logFile=fullfile(outDir,'run.log'); if isfile(logFile), delete(logFile); end
diary(logFile); dc=onCleanup(@() diary('off')); %#ok<NASGU>
L=16; hidden=64; opts=struct('Epochs',epochs,'Batch',256,'LearnRate',1e-3,'Hidden',hidden);
fprintf('%s\nT2 MATLAB：数据驱动对照 × 组合外推，FAST_MODE=%d\n%s\n', ...
    repmat('=',1,100),fastMode,repmat('=',1,100));
fprintf('预注册：E1 最佳数据驱动>=D-0.05；E2 DS02/03 D逐臂全胜；E3 非线性落差>D+0.05。\n');

pipeStream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
pipe=ncmapss_lib.build_pipeline(cfg,pipeStream,fullfile(matlabRoot,'cache','pipeline_cache_exact.mat'),false);
assert(abs(pipe.cond_Hn-cfg.COND_TARGET)/cfg.COND_TARGET<=cfg.COND_REL_TOL,'T0 指纹失效。');
dataStream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);

% 全部 dev 单元进入候选训练机队；DS05/DS07 最后两台留出做分布内评测。
trainSeqs={}; indist=struct('DS05_id',{{}},'DS07_id',{{}});
for i=1:size(cfg.IDENT_FILES,1)
    file=cfg.IDENT_FILES{i,1}; params=cfg.IDENT_FILES{i,2};
    d=ncmapss_lib.load_per_cycle(fullfile(cfg.DATA_DIR,file),'dev',cfg,dataStream,[]);
    [d,~]=ncmapss_lib.add_corrected(d,pipe.ref); d=ncmapss_lib.attach_residuals(pipe,d);
    units=sort(unique(d.unit)); if isfinite(unitLimit), units=units(1:min(unitLimit,numel(units))); end
    tag=''; if strcmp(file,'N-CMAPSS_DS05.h5'), tag='DS05_id'; end
    if strcmp(file,'N-CMAPSS_DS07.h5'), tag='DS07_id'; end
    held=[]; if ~isempty(tag), held=units(max(1,numel(units)-1):end); end
    for u=units.'
        seq=unit_sequence(d(d.unit==u,:),cfg);
        item=struct('per',seq,'file',file,'unit',double(u),'faults',{params});
        if ismember(u,held), indist.(tag){end+1}=item; else, trainSeqs{end+1}=item; end %#ok<AGROW>
    end
    fprintf('%-25s 训练%d台 / 留出%d台\n',file,numel(units)-numel(held),numel(held));
    clear d
end

order=ncmapss_lib.randperm_stream(dataStream,numel(trainSeqs),numel(trainSeqs)); trainSeqs=trainSeqs(order);
nVal=max(1,floor(numel(trainSeqs)/5)); valSeqs=trainSeqs(1:nVal); trSeqs=trainSeqs(nVal+1:end);
fprintf('按发动机切分：训练%d台，验证%d台。\n',numel(trSeqs),numel(valSeqs));

[models,linear,counts,trainInfo]=train_arms(trSeqs,valSeqs,windows,L,opts,dataStream,cfg.SEED);
writetable(struct2table(counts),fullfile(outDir,'parameter_counts.csv'),'Encoding','UTF-8');
save(fullfile(outDir,'models.mat'),'models','linear','trainInfo','-v7.3');

% 分布内 + 组合外推评测。
raw=evaluate_all(models,linear,indist,pipe,cfg,dataStream,windows,L,maxUnits);
writetable(raw,fullfile(outDir,'dd_raw.csv'),'Encoding','UTF-8'); % 先落盘
[verdict,summary,gaps,fa]=judge(raw,false);

% E1 保护条款：正式版若失败，自动执行只用 N=1000 的最有利变体。
if ~fastMode && ~verdict.E1
    fprintf('\nE1 未通过，执行预注册的 N=1000 最有利条件重试 ...\n');
    [modelsBest,linearBest,countsBest,trainBest]=train_arms(trSeqs,valSeqs,1000,L,opts,dataStream,cfg.SEED+100);
    rawBest=evaluate_all(modelsBest,linearBest,indist,pipe,cfg,dataStream,1000,L,maxUnits);
    writetable(rawBest,fullfile(outDir,'dd_bestcase_N1000_raw.csv'),'Encoding','UTF-8');
    [retry,summaryBest,gapsBest,faBest]=judge(rawBest,true);
    retry.parameter_counts=countsBest; retry.train_info=trainBest;
    verdict.E1_retry_N1000=retry.E1;
    verdict.retry_executed=true;
    verdict.retry=retry;
    writetable(summaryBest,fullfile(outDir,'bestcase_summary.csv'),'Encoding','UTF-8');
    writetable(gapsBest,fullfile(outDir,'bestcase_gaps.csv'),'Encoding','UTF-8');
    writetable(faBest,fullfile(outDir,'bestcase_false_alarm.csv'),'Encoding','UTF-8');
else
    verdict.E1_retry_N1000=verdict.E1;
    verdict.retry_executed=false;
end

writetable(summary,fullfile(outDir,'skill_summary.csv'),'Encoding','UTF-8');
writetable(gaps,fullfile(outDir,'generalization_gaps.csv'),'Encoding','UTF-8');
writetable(fa,fullfile(outDir,'false_alarm_ood.csv'),'Encoding','UTF-8');
write_json(fullfile(outDir,'verdict20.json'),verdict);
plot_t2(outDir,summary,gaps);
fprintf('\nE1=%d，E2=%d，E3=%d，E1重试=%d\n',verdict.E1,verdict.E2,verdict.E3,verdict.E1_retry_N1000);
end

function [models,linear,counts,info]=train_arms(trSeqs,valSeqs,windows,L,opts,stream,seed)
init=RandStream('mt19937ar','Seed',seed);
[Xv,Yv]=sample_sequences(valSeqs,windows,L,stream);
XvCurrent=squeeze(Xv(:,end,:)); XvWP=haar_features(Xv,2);
sampleMLP=@() sample_features(trSeqs,windows,L,stream,'current');
sampleWP=@() sample_features(trSeqs,windows,L,stream,'wp');
[models.MLP,info.mlp_val]=chapter3_models.train_ffn('E_MLP','mlp',sampleMLP, ...
    XvCurrent,Yv,13,opts,init);
[models.WPMixer,info.wp_val]=chapter3_models.train_ffn('E_WPMixer','wp',sampleWP, ...
    XvWP,Yv,size(XvWP,2),opts,init);

[Xt,Yt]=sample_sequences(trSeqs,windows,L,stream);
cur=squeeze(Xt(:,end,:)); flat=reshape(Xt,size(Xt,1),[]);
linear.E_Lin=[cur,ones(size(cur,1),1)]\Yt;
linear.E_MixLinear=[flat,ones(size(flat,1),1)]\Yt;
counts(1)=struct('arm','E_Lin','parameters',numel(linear.E_Lin));
counts(2)=struct('arm','E_MLP','parameters',chapter3_models.parameter_count(models.MLP));
counts(3)=struct('arm','E_MixLinear','parameters',numel(linear.E_MixLinear));
counts(4)=struct('arm','E_WPMixer','parameters',chapter3_models.parameter_count(models.WPMixer));
for i=1:numel(counts), fprintf('    %-12s 参数量=%d\n',counts(i).arm,counts(i).parameters); end
end

function [X,Y]=sample_features(seqs,windows,L,stream,kind)
[Xs,Y]=sample_sequences(seqs,windows,L,stream);
if strcmp(kind,'current'), X=squeeze(Xs(:,end,:)); else, X=haar_features(Xs,2); end
end

function [X,Y]=sample_sequences(seqs,windows,L,stream)
nObs=0; for i=1:numel(seqs), nObs=nObs+size(seqs{i}.per.theta,1); end
X=zeros(nObs,L,13,'single'); Y=zeros(nObs,9,'single'); cursor=0;
for i=1:numel(seqs)
    per=seqs{i}.per; T=size(per.theta,1);
    wi=ncmapss_lib.randperm_stream(stream,numel(windows),1); N=windows(wi);
    rbar=zeros(T,13,'single');
    for t=1:T
        samples=per.R_samples{t}; k=min(N,size(samples,1));
        idx=ncmapss_lib.randperm_stream(stream,size(samples,1),k);
        rbar(t,:)=mean(samples(idx,:),1);
    end
    padded=[zeros(L-1,13,'single');rbar];
    for t=1:T
        cursor=cursor+1; X(cursor,:,:)=padded(t:t+L-1,:); Y(cursor,:)=per.theta(t,:);
    end
end
end

function seq=unit_sequence(d,cfg)
cycles=sort(unique(d.cycle));
ref=d(ismember(d.cycle,cycles(1:min(cfg.N_REF_CYCLES,numel(cycles)))),:);
b=mean(table2array(ref(:,cfg.RESID_COLS)),1); th0=mean(table2array(ref(:,cfg.THETA9))./cfg.THETA_SPAN,1);
seq.R_samples=cell(numel(cycles),1); seq.theta=zeros(numel(cycles),9,'single');
for t=1:numel(cycles)
    sub=d(d.cycle==cycles(t),:);
    seq.R_samples{t}=single(table2array(sub(:,cfg.RESID_COLS))-b);
    seq.theta(t,:)=single(table2array(sub(1,cfg.THETA9))./cfg.THETA_SPAN-th0);
end
end

function raw=evaluate_all(models,linear,indist,pipe,cfg,stream,windows,L,maxUnits)
rows={};
tags={'DS05_id','DS07_id'};
for i=1:numel(tags)
    list=indist.(tags{i});
    for j=1:numel(list)
        rows{end+1,1}=eval_item(list{j},tags{i},'in_dist',models,linear,pipe,cfg,stream,windows,L); %#ok<AGROW>
    end
end
for iv=1:size(cfg.VALID_FILES,1)
    tag=cfg.VALID_FILES{iv,1}; file=cfg.VALID_FILES{iv,2}; faults=cfg.VALID_FILES{iv,3};
    d=ncmapss_lib.load_per_cycle(fullfile(cfg.DATA_DIR,file),'test',cfg,stream,[]);
    [d,~]=ncmapss_lib.add_corrected(d,pipe.ref); d=ncmapss_lib.attach_residuals(pipe,d);
    units=sort(unique(d.unit)); if isfinite(maxUnits), units=units(1:min(maxUnits,numel(units))); end
    for u=units.'
        item=struct('per',unit_sequence(d(d.unit==u,:),cfg),'unit',double(u),'faults',{faults});
        rows{end+1,1}=eval_item(item,tag,'ood_combo',models,linear,pipe,cfg,stream,windows,L); %#ok<AGROW>
    end
end
raw=vertcat(rows{:});
end

function out=eval_item(item,tag,regime,models,linear,pipe,cfg,stream,windows,L)
per=item.per; T=size(per.theta,1); theta=double(per.theta); idxTrue=find(ismember(cfg.THETA9,item.faults));
out=table();
for N=windows
    rbar=zeros(T,13);
    for t=1:T
        s=per.R_samples{t}; k=min(N,size(s,1)); idx=ncmapss_lib.randperm_stream(stream,size(s,1),k); rbar(t,:)=mean(s(idx,:),1);
    end
    X=make_seq(rbar,L); cur=squeeze(X(:,end,:)); flat=reshape(X,T,[]); wp=haar_features(X,2);
    pred=struct(); pred.E_Lin=[cur,ones(T,1)]*linear.E_Lin;
    pred.E_MLP=chapter3_models.predict_ffn(models.MLP,'mlp',cur);
    pred.E_MixLinear=[flat,ones(T,1)]*linear.E_MixLinear;
    pred.E_WPMixer=chapter3_models.predict_ffn(models.WPMixer,'wp',wp);
    M=ncmapss_lib.build_M(pipe.Hn,T); Cum=ncmapss_lib.build_cum(T,9);
    pred.D=ncmapss_lib.solve_D(M,Cum,rbar,cfg.LAM_MAIN,T,9);
    arms=[{'Z_zero'},fieldnames(pred).'];
    for ia=1:numel(arms)
        arm=arms{ia}; if strcmp(arm,'Z_zero'), th=zeros(size(theta)); else, th=pred.(arm); end
        met=ncmapss_lib.compute_metrics(th,theta,idxTrue);
        out=[out;table({regime},{tag},item.unit,N,{arm},met.skill,met.rmse_early,met.false_alarm,met.detect_corr, ...
            'VariableNames',{'regime','subset','unit','N','arm','skill','rmse_early','false_alarm','detect_corr'})]; %#ok<AGROW>
    end
end
end

function X=make_seq(rbar,L)
T=size(rbar,1); padded=[zeros(L-1,13);rbar]; X=zeros(T,L,13,'single');
for t=1:T, X(t,:,:)=single(padded(t:t+L-1,:)); end
end

function feat=haar_features(X,level)
bands={permute(X,[1 3 2])};
for lv=1:level
    next={};
    for b=1:numel(bands)
        z=bands{b}; lo=(z(:,:,1:2:end)+z(:,:,2:2:end))/sqrt(2); hi=(z(:,:,1:2:end)-z(:,:,2:2:end))/sqrt(2);
        next{end+1}=lo; next{end+1}=hi; %#ok<AGROW>
    end
    bands=next;
end
feat=[]; for b=1:numel(bands), feat=[feat,reshape(bands{b},size(X,1),[])]; end %#ok<AGROW>
end

function [v,summary,gaps,fa]=judge(raw,isRetry)
arms={'D','E_Lin','E_MLP','E_MixLinear','E_WPMixer'};
q=raw(~strcmp(raw.arm,'Z_zero'),:);
summary=groupsummary(q,{'regime','subset','arm'},'mean','skill');
wide=unstack(summary,'mean_skill','arm');
in=wide(strcmp(wide.regime,'in_dist'),:); ood=wide(strcmp(wide.regime,'ood_combo'),:);
skIn=zeros(1,numel(arms)); skOod=skIn;
for i=1:numel(arms), skIn(i)=mean(in.(arms{i})); skOod(i)=mean(ood.(arms{i})); end
bestDD=max(skIn(2:end)); E1=bestDD>=skIn(1)-.05;
E2=true; for r=1:height(ood), for i=2:numel(arms), E2=E2 && ood.D(r)>ood.(arms{i})(r); end, end
gap=skIn-skOod; E3=all(gap(3:5)>gap(1)+.05);
gaps=table(arms.',skIn.',skOod.',gap.','VariableNames',{'arm','in_dist','ood_combo','gap'});
fa=groupsummary(raw(strcmp(raw.regime,'ood_combo')&~strcmp(raw.arm,'Z_zero'),:),{'subset','arm'},'mean','false_alarm');
v=struct('E1',E1,'E2',E2,'E3',E3,'retry_variant',isRetry, ...
    'best_data_driven_in_dist',bestDD,'D_in_dist',skIn(1),'D_gap',gap(1), ...
    'gaps',table2struct(gaps),'E1_valid_for_ood',E1);
end

function plot_t2(outDir,summary,gaps)
wide=unstack(summary,'mean_skill','arm'); in=wide(strcmp(wide.regime,'in_dist'),:); ood=wide(strcmp(wide.regime,'ood_combo'),:);
arms=gaps.arm; vin=zeros(numel(arms),1); vood=vin;
for i=1:numel(arms), vin(i)=mean(in.(arms{i})); vood(i)=mean(ood.(arms{i})); end
f=figure('Visible','off','Position',[100 100 1050 560]); b=bar([vin vood]); hold on; yline(0,'k-'); grid on;
set(gca,'XTickLabel',arms); ylabel('技能分 S'); legend(b,{'分布内','组合外推'},'Location','best'); title('数据驱动对照与组合外推');
for i=1:numel(arms), text(i,max(vin(i),vood(i))+.03,sprintf('%+.2f',gaps.gap(i)),'HorizontalAlignment','center'); end
exportgraphics(f,fullfile(outDir,'fig_extrapolation.png'),'Resolution',180); close(f);
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
