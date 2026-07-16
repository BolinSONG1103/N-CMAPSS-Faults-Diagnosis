function verdict = experiment_19b(fastMode)
% T1：工况自适应 H(w)=H0*diag(g(w))，整周期/高TRA双窗口正式评测。

if nargin<1, fastMode=true; end
srcDir = fileparts(mfilename('fullpath'));
matlabRoot = fileparts(srcDir);
addpath(srcDir);
cfg = ncmapss_lib.config(fastMode);
ncmapss_lib.set_plot_defaults();
if fastMode
    windows=[100 1000]; lambdaGrid=logspace(0,2,5); maxUnits=2;
    epochs=6; kAblate=[1 4]; nTrainMax=60000;
    outDir=fullfile(matlabRoot,'outputs','t1_moe_smoke');
else
    windows=[1 10 100 1000]; lambdaGrid=logspace(-1,3,9); maxUnits=inf;
    epochs=30; kAblate=[1 2 4 8]; nTrainMax=300000;
    outDir=fullfile(matlabRoot,'outputs','t1_moe');
end
if ~isfolder(outDir), mkdir(outDir); end
diaryFile=fullfile(outDir,'run.log'); if isfile(diaryFile), delete(diaryFile); end
diary(diaryFile); diaryCleanup=onCleanup(@() diary('off')); %#ok<NASGU>

Kmain=4; modes={'full','highTRA'}; evalArms={'H0','TRA4','MoE_K4','FreeMLP','MoE_dir'};
opts=struct('Epochs',epochs,'Batch',4096,'LearnRate',3e-3,'Hidden',32,'DirPenalty',1e-3);
fprintf('%s\nT1 MATLAB：工况自适应 H(w)，FAST_MODE=%d\n%s\n', ...
    repmat('=',1,100),fastMode,repmat('=',1,100));
fprintf('预注册：Q1>=5%%；Q2s占优率>=95%%且中位gap>=0.01；Q3<0.01；Q4容差0.005。\n');

pipeStream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
cacheFile=fullfile(matlabRoot,'cache','pipeline_cache_exact.mat');
pipe=ncmapss_lib.build_pipeline(cfg,pipeStream,cacheFile,false);
fprintf('cond(H_n)=%.6f\n',pipe.cond_Hn);
assert(abs(pipe.cond_Hn-cfg.COND_TARGET)/cfg.COND_TARGET<=cfg.COND_REL_TOL,'T0指纹失效。');

% 样本级门控训练集：重新使用 seed=42 读取与 T0 相同的辨识行。
dataStream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
Ws=cell(5,1); THs=cell(5,1); Rs=cell(5,1); Us=cell(5,1);
for i=1:size(cfg.IDENT_FILES,1)
    file=cfg.IDENT_FILES{i,1};
    d=ncmapss_lib.load_for_ident(fullfile(cfg.DATA_DIR,file),cfg,dataStream);
    [d,~]=ncmapss_lib.add_corrected(d,pipe.ref);
    d=d(d.hs==0,:);
    Ws{i}=single(table2array(d(:,cfg.OP_COLS)));
    THs{i}=single(table2array(d(:,cfg.THETA9))./cfg.THETA_SPAN);
    Rs{i}=single(ncmapss_lib.residual(pipe.base,pipe.corrected_cols,cfg.OP_COLS,d)./pipe.resid_std);
    Us{i}=d.unit+1000*i;
    fprintf('训练池 %-25s %d 行\n',file,height(d));
end
W=vertcat(Ws{:}); TH=vertcat(THs{:}); R=vertcat(Rs{:}); U=vertcat(Us{:});
clear Ws THs Rs Us d
if size(W,1)>nTrainMax
    sel=ncmapss_lib.randperm_stream(dataStream,size(W,1),nTrainMax);
    W=W(sel,:); TH=TH(sel,:); R=R(sel,:); U=U(sel,:);
end
units=unique(U); order=ncmapss_lib.randperm_stream(dataStream,numel(units),numel(units));
units=units(order); nVal=max(1,floor(.25*numel(units))); valUnits=units(1:nVal);
isVal=ismember(U,valUnits);
wMu=mean(W(~isVal,:),1); wSd=std(W(~isVal,:),1,1)+1e-8;
Wz=(W-wMu)./wSd;
Wtr=Wz(~isVal,:); THtr=TH(~isVal,:); Rtr=R(~isVal,:);
Wva=Wz(isVal,:); THva=TH(isVal,:); Rva=R(isVal,:);
fprintf('门控训练/验证：%d/%d 行，按发动机切分。\n',size(Wtr,1),size(Wva,1));

initStream=RandStream('mt19937ar','Seed',cfg.SEED);
models=struct(); scores=struct();
[models.H0,scores.H0]=chapter3_models.train_moe('H0',1,'const',false,pipe.Hn, ...
    Wtr,THtr,Rtr,Wva,THva,Rva,opts,initStream);
for K=kAblate
    field=sprintf('MoE_K%d',K);
    [models.(field),scores.(field)]=chapter3_models.train_moe(field,K,'moe',false, ...
        pipe.Hn,Wtr,THtr,Rtr,Wva,THva,Rva,opts,initStream);
end
[models.FreeMLP,scores.FreeMLP]=chapter3_models.train_moe('FreeMLP',1,'free',false, ...
    pipe.Hn,Wtr,THtr,Rtr,Wva,THva,Rva,opts,initStream);
[models.MoE_dir,scores.MoE_dir]=chapter3_models.train_moe('MoE_dir',Kmain,'moe',true, ...
    pipe.Hn,Wtr,THtr,Rtr,Wva,THva,Rva,opts,initStream);

dH=chapter3_models.moe_effective_H(models.MoE_dir,pipe.Hn)-pipe.Hn;
colCos=zeros(1,9);
for j=1:9
    colCos(j)=dot(pipe.Hn(:,j),pipe.Hn(:,j)+dH(:,j))/ ...
        max(norm(pipe.Hn(:,j))*norm(pipe.Hn(:,j)+dH(:,j)),1e-12);
end
dirRatio=norm(dH,'fro')/norm(pipe.Hn,'fro');
fprintf('MoE_dir: ||dH||/||H0||=%.6f，最差方向余弦=%.6f\n',dirRatio,min(colCos));

% TRA 四分位硬门控对照。
tra=W(~isVal,1); traQ=quantile(tra,[.25 .5 .75]); gBins=zeros(4,9);
for b=1:4
    lo=-inf; hi=inf; if b>1, lo=traQ(b-1); end; if b<4, hi=traQ(b); end
    mask=tra>lo & tra<=hi;
    gBins(b,:)=fit_gain_ls(pipe.Hn,double(TH(~isVal,:)),double(R(~isVal,:)),mask).';
end
fprintf('TRA4 fan_flow: %.3f -> %.3f；LPT_eff: %.3f -> %.3f\n', ...
    gBins(1,3),gBins(4,3),gBins(1,6),gBins(4,6));

% 门控的物理分区统计。
Atrain=chapter3_models.moe_alphas(models.MoE_K4,double(Wtr));
expertTRA=sum(Atrain.*double(W(~isVal,1)),1)./max(sum(Atrain,1),1e-12);
expertShare=mean(Atrain,1);
expertGain=double(extractdata(chapter3_models.softplus(models.MoE_K4.experts))).';
[expertTRA,expertOrder]=sort(expertTRA); expertShare=expertShare(expertOrder); expertGain=expertGain(expertOrder,:);
gateTable=array2table([(1:Kmain).',expertTRA.',expertShare.',expertGain], ...
    'VariableNames',[{'expert','weighted_TRA','mean_share'},cfg.THETA9]);
writetable(gateTable,fullfile(outDir,'gate_physics.csv'),'Encoding','UTF-8');
save(fullfile(outDir,'models.mat'),'models','scores','wMu','wSd','traQ','gBins','-v7.3');

% 双窗口评测。
dummy=struct('skill',0,'rmse',0,'rmse_early',0,'false_alarm',0,'detect_corr',0);
rows=repmat(make_row('','',0,0,'',NaN,dummy),0,1);
rowi=0; fallbackTotal=0; totalCycles=0;
for iv=1:size(cfg.VALID_FILES,1)
    tag=cfg.VALID_FILES{iv,1}; file=cfg.VALID_FILES{iv,2}; faults=cfg.VALID_FILES{iv,3};
    d=ncmapss_lib.load_per_cycle(fullfile(cfg.DATA_DIR,file),'test',cfg,dataStream,[]);
    [d,~]=ncmapss_lib.add_corrected(d,pipe.ref); d=ncmapss_lib.attach_residuals(pipe,d);
    testUnits=sort(unique(d.unit)); if isfinite(maxUnits), testUnits=testUnits(1:min(maxUnits,end)); end
    idxTrue=find(ismember(cfg.THETA9,faults));
    fprintf('\n%s test units=%s\n',tag,mat2str(testUnits.'));
    for iu=1:numel(testUnits)
        u=testUnits(iu);
        for im=1:numel(modes)
            [wins,fb]=ncmapss_lib.build_windows(d,u,modes{im},windows,cfg,dataStream);
            fallbackTotal=fallbackTotal+fb.highTRA;
            if strcmp(modes{im},'highTRA'), totalCycles=totalCycles+numel(unique(d.cycle(d.unit==u))); end
            for iw=1:numel(wins)
                Rw=wins(iw).R; T=size(Rw,1); theta=wins(iw).TH;
                Cum=ncmapss_lib.build_cum(T,9);
                rowi=rowi+1; rows(rowi)=make_row(modes{im},tag,u,wins(iw).N,'Z_zero',NaN, ...
                    ncmapss_lib.compute_metrics(zeros(size(theta)),theta,idxTrue)); %#ok<AGROW>
                for ia=1:numel(evalArms)
                    arm=evalArms{ia}; Hbase=pipe.Hn; Hlist=cell(T,1);
                    if strcmp(arm,'MoE_dir'), Hbase=chapter3_models.moe_effective_H(models.MoE_dir,pipe.Hn); end
                    for t=1:T
                        op=wins(iw).OPS{t};
                        if strcmp(arm,'TRA4')
                            bin=1+sum(op(:,1)>traQ,2); gains=gBins(bin,:);
                        elseif strcmp(arm,'H0')
                            gains=ones(size(op,1),9);
                        else
                            gains=chapter3_models.moe_gains(models.(arm),(double(op)-double(wMu))./double(wSd));
                        end
                        Hlist{t}=Hbase.*mean(gains,1);
                    end
                    M=ncmapss_lib.build_M_var(Hlist,T);
                    for lam=lambdaGrid
                        thetaHat=ncmapss_lib.solve_D_var(M,Cum,Rw,lam,T,9);
                        met=ncmapss_lib.compute_metrics(thetaHat,theta,idxTrue);
                        rowi=rowi+1; rows(rowi)=make_row(modes{im},tag,u,wins(iw).N,arm,lam,met); %#ok<AGROW>
                    end
                end
            end
        end
        fprintf('  unit %d 完成\n',round(u));
    end
end
raw=struct2table(rows);
writetable(raw,fullfile(outDir,'moe_raw.csv'),'Encoding','UTF-8'); % 先落盘

agg=groupsummary(raw(~strcmp(raw.arm,'Z_zero'),:),{'mode','subset','N','lambda','arm'},'mean','skill');
wide=unstack(agg,'mean_skill','arm');
wide.gap=wide.MoE_K4-wide.H0;
fullGap=wide.gap(strcmp(wide.mode,'full')); highGap=wide.gap(strcmp(wide.mode,'highTRA'));
q1Gain=1-scores.MoE_K4/scores.H0; Q1=q1Gain>=.05;
Q2f=median(fullGap)<.01;
rateHigh=mean(highGap>0); medHigh=median(highGap); Q2s=rateHigh>=.95 && medHigh>=.01;
lamPick=lambdaGrid(abs(log(lambdaGrid/cfg.LAM_MAIN))==min(abs(log(lambdaGrid/cfg.LAM_MAIN)))); lamPick=lamPick(1);
main=raw((raw.lambda==lamPick | strcmp(raw.arm,'Z_zero')),:);
mainAgg=groupsummary(main,{'mode','subset','arm'},'mean',{'skill','false_alarm','detect_corr','rmse_early'});
tags=unique(raw.subset); q3ok=[]; q4ok=[]; q6diff=[];
for im=1:numel(modes)
    for it=1:numel(tags)
        q3ok(end+1)=get_skill(mainAgg,modes{im},tags{it},'MoE_dir')-get_skill(mainAgg,modes{im},tags{it},'MoE_K4')<.01; %#ok<AGROW>
        q4ok(end+1)=get_skill(mainAgg,modes{im},tags{it},'MoE_K4')>=get_skill(mainAgg,modes{im},tags{it},'FreeMLP')-.005; %#ok<AGROW>
        if strcmp(modes{im},'highTRA')
            q6diff(end+1)=get_skill(mainAgg,modes{im},tags{it},'MoE_K4')-get_skill(mainAgg,modes{im},tags{it},'TRA4'); %#ok<AGROW>
        end
    end
end
Q3=all(q3ok); Q4=all(q4ok); fallbackRate=fallbackTotal/max(totalCycles,1);
verdict=struct('FAST_MODE',fastMode,'Q1',Q1,'Q1_forward_gain',q1Gain, ...
    'Q2f_mechanism_consistent',Q2f,'Q2f_median_gap',median(fullGap), ...
    'Q2s',Q2s,'Q2s_dominance_rate',rateHigh,'Q2s_median_gap',medHigh, ...
    'Q3',Q3,'Q4',Q4,'Q6_moe_minus_TRA4',q6diff, ...
    'direction_change_ratio',dirRatio,'direction_cosine_min',min(colCos), ...
    'highTRA_fallback_rate',fallbackRate,'core_pass',Q1&&Q2s&&Q3&&Q4);
write_json(fullfile(outDir,'verdict19b.json'),verdict);
writetable(mainAgg,fullfile(outDir,'main_lambda_summary.csv'),'Encoding','UTF-8');
writetable(wide,fullfile(outDir,'pointwise_gap.csv'),'Encoding','UTF-8');

plot_t1(outDir,wide,gateTable,models.MoE_K4,wMu,wSd,tra,W,cfg,expertOrder);
fprintf('\nQ1=%d (gain %.3f), Q2s=%d (rate %.1f%%, median %.4f), Q3=%d, Q4=%d\n', ...
    Q1,q1Gain,Q2s,100*rateHigh,medHigh,Q3,Q4);
fprintf('highTRA 回退率 %.3f%%；T1 core_pass=%d\n',100*fallbackRate,verdict.core_pass);
end

function g=fit_gain_ls(H,TH,R,mask)
TH=TH(mask,:); R=R(mask,:);
AtA=(TH.'*TH).*(H.'*H); Aty=sum(TH.*(R*H),1).';
dead=diag(AtA)<1e-8*max(max(diag(AtA)),1e-12); ridge=1e-6*trace(AtA)/size(H,2);
g=(AtA+ridge*eye(size(H,2)))\Aty; g(dead)=1; g=min(max(g,.05),5);
end

function r=make_row(mode,subset,unit,N,arm,lambda,m)
r=struct('mode',mode,'subset',subset,'unit',double(unit),'N',double(N),'arm',arm, ...
    'lambda',double(lambda),'skill',m.skill,'rmse',m.rmse,'rmse_early',m.rmse_early, ...
    'false_alarm',m.false_alarm,'detect_corr',m.detect_corr);
end

function s=get_skill(tbl,mode,subset,arm)
mask=strcmp(tbl.mode,mode)&strcmp(tbl.subset,subset)&strcmp(tbl.arm,arm);
s=tbl.mean_skill(find(mask,1));
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end

function plot_t1(outDir,wide,gateTable,model,wMu,wSd,tra,W,cfg,expertOrder)
f=figure('Visible','off','Position',[100 100 1400 550]); tl=tiledlayout(1,2);
for im=1:2
    ax=nexttile; mode={'full','highTRA'}; m=mode{im}; hold(ax,'on');
    tags=unique(wide.subset); Ns=unique(wide.N);
    for it=1:numel(tags), for in=1:numel(Ns)
        q=strcmp(wide.mode,m)&strcmp(wide.subset,tags{it})&wide.N==Ns(in);
        if any(q), semilogx(ax,wide.lambda(q),wide.gap(q),'-o','DisplayName',sprintf('%s N=%d',tags{it},Ns(in))); end
    end, end
    yline(ax,0,'--k'); grid(ax,'on'); xlabel(ax,'lambda'); ylabel(ax,'MoE-H0 技能分'); title(ax,m); legend(ax,'Location','best','FontSize',7);
end
title(tl,'工况自适应增益：整周期与高TRA窗口'); exportgraphics(f,fullfile(outDir,'fig_dual_mode.png'),'Resolution',180); close(f);

traGrid=linspace(quantile(tra,.01),quantile(tra,.99),200).'; base=repmat(mean(double(W),1),200,1); base(:,1)=traGrid;
z=(base-double(wMu))./double(wSd); A=chapter3_models.moe_alphas(model,z); G=chapter3_models.moe_gains(model,z);
f=figure('Visible','off','Position',[100 100 1400 520]); tiledlayout(1,2); ax=nexttile; plot(ax,traGrid,A(:,expertOrder),'LineWidth',1.8); grid(ax,'on'); xlabel(ax,'TRA'); ylabel(ax,'alpha_k'); title(ax,'MoE 专家分区');
ax=nexttile; idx=[3 6 2 1]; plot(ax,traGrid,G(:,idx),'LineWidth',1.8); yline(ax,1,':k'); grid(ax,'on'); xlabel(ax,'TRA'); ylabel(ax,'g_j(w)'); title(ax,'部件增益'); legend(ax,cfg.THETA9(idx),'Interpreter','none');
exportgraphics(f,fullfile(outDir,'fig_gate_physics.png'),'Resolution',180); close(f);
writetable(gateTable,fullfile(outDir,'gate_physics.csv'),'Encoding','UTF-8');
end
