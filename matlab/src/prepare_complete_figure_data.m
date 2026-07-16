function audit=prepare_complete_figure_data(force)
%PREPARE_COMPLETE_FIGURE_DATA 仅为完整图集补存缺失的中间数据与 theta_hat。
% 不重跑任何 P/Q/E 实验；所有模型、H 和超参数均读取锁定缓存与正式输出。

if nargin<1, force=false; end
srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
outDir=fullfile(matlabRoot,'outputs','complete_figures','data');
if ~isfolder(outDir), mkdir(outDir); end
paths=struct( ...
    'A1',fullfile(outDir,'a1_similarity_correction.csv'), ...
    'A2',fullfile(outDir,'a2_baseline_fit.csv'), ...
    'B3',fullfile(outDir,'b3_lambda_trajectory.csv'), ...
    'B4',fullfile(outDir,'b4_multiunit_component_error.csv'), ...
    'C34',fullfile(outDir,'c3_c4_extrapolation_trajectory.csv'));

needRep=force || ~isfile(paths.A1) || ~isfile(paths.A2) || ~isfile(paths.B3) || ~isfile(paths.C34);
needMulti=force || ~isfile(paths.B4);
cfg=ncmapss_lib.config(false); pipeStream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
pipe=ncmapss_lib.build_pipeline(cfg,pipeStream,fullfile(matlabRoot,'cache','pipeline_cache_exact.mat'),false);
assert(abs(pipe.cond_Hn-cfg.COND_TARGET)/cfg.COND_TARGET<=cfg.COND_REL_TOL,'锁定 H 指纹失效。');

if needRep
    unit=13; path=fullfile(cfg.DATA_DIR,'N-CMAPSS_DS03-012.h5');
    d=ncmapss_lib.load_per_cycle(path,'test',cfg,pipeStream,unit);
    [d,~]=ncmapss_lib.add_corrected(d,pipe.ref); d=ncmapss_lib.attach_residuals(pipe,d);
    cycles=sort(unique(d.cycle)); early=d(ismember(d.cycle,cycles(1:min(3,numel(cycles)))),:);
    if force || ~isfile(paths.A1), build_similarity(early,unit,paths.A1); end
    if force || ~isfile(paths.A2), build_baseline_fit(pipe,early,unit,paths.A2); end
    if force || ~isfile(paths.B3), build_lambda_trajectories(pipe,d,unit,cfg,pipeStream,paths.B3); end
    if force || ~isfile(paths.C34), build_extrapolation_trajectories(pipe,d,unit,cfg,matlabRoot,paths.C34); end
    clear d early
end

if needMulti, build_multiunit_error(pipe,cfg,matlabRoot,paths.B4); end

names=string(fieldnames(paths)); values=strings(numel(names),1); exists=false(numel(names),1);
for i=1:numel(names), values(i)=string(paths.(names(i))); exists(i)=isfile(values(i)); end
audit=table(names,values,exists,'VariableNames',{'data_id','path','exists'});
assert(all(audit.exists),'完整图集补存数据不完整。');
fprintf('Complete-figure data ready: %d files.\n',height(audit));
end

function build_similarity(d,unit,outPath)
rawNames={"Nf","Nc","Wf","T24","T30","T48","T50", ...
    "P15","P21","P24","Ps30","P40","P50"};
correctedNames={"Nf_c","Nc_c","Wf_c","T24_c","T30_c","T48_c","T50_c", ...
    "P15_c","P21_c","P24_c","Ps30_c","P40_c","P50_c"};
driverNames={"theta_c","theta_c","compound","theta_c","theta_c","theta_c","theta_c", ...
    "delta_c","delta_c","delta_c","delta_c","delta_c","delta_c"};
rows=cell(2*numel(rawNames),1); ri=0;
for i=1:numel(rawNames)
    if driverNames{i}=="compound"
        x=d.delta_c.*sqrt(d.theta_c); condition="delta_c*sqrt(theta_c)";
    else
        x=d.(driverNames{i}); condition=driverNames{i};
    end
    raw=d.(rawNames{i}); corrected=d.(correctedNames{i});
    [rRaw,sRaw,bRaw,nRaw]=drift_metrics(x,raw);
    [rCorr,sCorr,bCorr,nCorr]=drift_metrics(x,corrected);
    ri=ri+1; rows{ri}=similarity_rows(d,unit,condition,x,rawNames{i},correctedNames{i}, ...
        "raw",raw,rRaw,sRaw,bRaw,nRaw,bRaw-bCorr);
    ri=ri+1; rows{ri}=similarity_rows(d,unit,condition,x,rawNames{i},correctedNames{i}, ...
        "corrected",corrected,rCorr,sCorr,bCorr,nCorr,bRaw-bCorr);
end
writetable(vertcat(rows{:}),outPath,'Encoding','UTF-8');
end

function t=similarity_rows(d,unit,condition,x,rawName,correctedName,stage,y,r,slope,binSpan,normSpan,reduction)
med=median(y,'omitnan'); rel=100*(y./med-1);
t=table(repmat("DS03",height(d),1),repmat(unit,height(d),1),d.cycle, ...
    repmat(condition,height(d),1),x,repmat(rawName,height(d),1), ...
    repmat(correctedName,height(d),1),repmat(stage,height(d),1),y,rel, ...
    repmat(r,height(d),1),repmat(slope,height(d),1),repmat(binSpan,height(d),1), ...
    repmat(normSpan,height(d),1),repmat(reduction,height(d),1), ...
    'VariableNames',{'subset','unit','cycle','condition','condition_value','raw_channel', ...
    'corrected_channel','stage','value','relative_deviation_pct','correlation', ...
    'standardized_slope','binned_drift_span_pct','normalized_binned_span','drift_reduction_pctpt'});
end

function [r,slope,binSpan,normSpan]=drift_metrics(x,y)
q=isfinite(x)&isfinite(y); x=x(q); y=y(q); r=corr_value(x,y);
zx=(x-mean(x))/max(std(x),eps); zy=(y-mean(y))/max(std(y),eps);
b=[ones(numel(zx),1),zx]\zy; slope=b(2);
rel=100*(y./median(y)-1); edges=quantile(x,linspace(0,1,11));
edges=unique(edges); medians=[];
for k=1:numel(edges)-1
    if k<numel(edges)-1, take=x>=edges(k)&x<edges(k+1); else, take=x>=edges(k)&x<=edges(k+1); end
    if any(take), medians(end+1,1)=median(rel(take)); end %#ok<AGROW>
end
if isempty(medians), binSpan=NaN; else, binSpan=max(medians)-min(medians); end
normSpan=binSpan/max(iqr(rel),eps);
end

function build_baseline_fit(pipe,d,unit,outPath)
X=table2array(d(:,pipe.cfg.OP_COLS)); actual=table2array(d(:,pipe.corrected_cols)); pred=zeros(size(actual));
for j=1:numel(pipe.corrected_cols), pred(:,j)=ncmapss_lib.predict_model(pipe.base{j},X); end
rows=cell(numel(pipe.corrected_cols),1);
for j=1:numel(pipe.corrected_cols)
    a=actual(:,j); p=pred(:,j); r2=1-sum((a-p).^2)/max(sum((a-mean(a)).^2),eps); rs=std(a-p);
    rows{j}=table(repmat("DS03",height(d),1),repmat(unit,height(d),1),d.cycle, ...
        repmat(string(pipe.corrected_cols{j}),height(d),1),a,p,repmat(r2,height(d),1),repmat(rs,height(d),1), ...
        'VariableNames',{'subset','unit','cycle','channel','measured','predicted','R2','residual_std'});
end
writetable(vertcat(rows{:}),outPath,'Encoding','UTF-8');
end

function build_lambda_trajectories(pipe,d,unit,cfg,stream,outPath)
[wins,~]=ncmapss_lib.build_windows(d,unit,'full',1000,cfg,stream);
R=wins.R; truth=wins.TH; T=size(R,1); cycles=sort(unique(d.cycle));
M=ncmapss_lib.build_M(pipe.Hn,T); Cum=ncmapss_lib.build_cum(T,9); lambdas=[.001 31.6 10000];
rows=cell(numel(lambdas)*9,1); ri=0;
for il=1:numel(lambdas)
    hat=ncmapss_lib.solve_D(M,Cum,R,lambdas(il),T,9);
    for j=1:9
        ri=ri+1; scale=100*cfg.THETA_SPAN(j);
        rows{ri}=table(repmat("DS03",T,1),repmat(unit,T,1),cycles, ...
            repmat(lambdas(il),T,1),repmat(string(cfg.THETA9{j}),T,1), ...
            truth(:,j)*scale,hat(:,j)*scale, ...
            'VariableNames',{'subset','unit','cycle','lambda','parameter','theta_true_pct','theta_hat_pct'});
    end
end
writetable(vertcat(rows{:}),outPath,'Encoding','UTF-8');
end

function build_multiunit_error(pipe,cfg,matlabRoot,outPath)
rows=cell(0,1); ri=0;
for iv=1:size(cfg.VALID_FILES,1)
    tag=string(cfg.VALID_FILES{iv,1}); file=cfg.VALID_FILES{iv,2}; faults=cfg.VALID_FILES{iv,3};
    stream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
    if isfield(pipe,'rng_state_json'), ncmapss_lib.restore_rng_state(stream,pipe.rng_state_json); end
    d=ncmapss_lib.load_per_cycle(fullfile(cfg.DATA_DIR,file),'test',cfg,stream,[]);
    [d,~]=ncmapss_lib.add_corrected(d,pipe.ref); d=ncmapss_lib.attach_residuals(pipe,d);
    units=sort(unique(d.unit));
    for u=units.'
        [wins,~]=ncmapss_lib.build_windows(d,u,'full',1000,cfg,stream); R=wins.R; truth=wins.TH; T=size(R,1);
        M=ncmapss_lib.build_M(pipe.Hn,T); Cum=ncmapss_lib.build_cum(T,9);
        hat=ncmapss_lib.solve_D(M,Cum,R,cfg.LAM_MAIN,T,9); nTerminal=max(1,ceil(.1*T));
        for j=1:9
            scale=100*cfg.THETA_SPAN(j); err=(hat(:,j)-truth(:,j))*scale; zerr=-truth(:,j)*scale;
            ri=ri+1; rows{ri}=table(tag,u,string(cfg.THETA9{j}),ismember(cfg.THETA9{j},faults), ...
                sqrt(mean(err.^2)),mean(abs(err(end-nTerminal+1:end))),sqrt(mean(zerr.^2)), ...
                'VariableNames',{'subset','unit','parameter','is_fault','rmse_pct','terminal_abs_error_pct','Z_zero_rmse_pct'});
        end
    end
    clear d
end
writetable(vertcat(rows{:}),outPath,'Encoding','UTF-8');
fprintf('Multiunit component errors saved: %s\n',outPath);
end

function build_extrapolation_trajectories(pipe,d,unit,cfg,matlabRoot,outPath)
seq=unit_sequence_local(d,cfg); T=size(seq.theta,1); rbar=zeros(T,13);
for t=1:T, rbar(t,:)=mean(seq.R_samples{t},1); end
L=16; X=make_seq_local(rbar,L); cur=squeeze(X(:,end,:));
s=load(fullfile(matlabRoot,'outputs','t2_dd','models.mat'),'models','linear');
pred=struct(); pred.E_Lin=[cur,ones(T,1)]*s.linear.E_Lin;
pred.E_MLP=chapter3_models.predict_ffn(s.models.MLP,'mlp',cur);
M=ncmapss_lib.build_M(pipe.Hn,T); Cum=ncmapss_lib.build_cum(T,9);
pred.D=ncmapss_lib.solve_D(M,Cum,rbar,cfg.LAM_MAIN,T,9); pred.Z_zero=zeros(T,9);
cycles=sort(unique(d.cycle)); arms=["truth","D","E_Lin","E_MLP","Z_zero"];
rows=cell(numel(arms)*9,1); ri=0;
for ia=1:numel(arms)
    if arms(ia)=="truth", th=double(seq.theta); else, th=pred.(arms(ia)); end
    for j=1:9
        ri=ri+1; scale=100*cfg.THETA_SPAN(j);
        rows{ri}=table(repmat("DS03",T,1),repmat(unit,T,1),cycles,repmat(arms(ia),T,1), ...
            repmat(string(cfg.THETA9{j}),T,1),repmat(ismember(j,[1 6 7]),T,1),th(:,j)*scale, ...
            'VariableNames',{'subset','unit','cycle','arm','parameter','is_fault','theta_pct'});
    end
end
writetable(vertcat(rows{:}),outPath,'Encoding','UTF-8');
end

function seq=unit_sequence_local(d,cfg)
cycles=sort(unique(d.cycle)); ref=d(ismember(d.cycle,cycles(1:min(cfg.N_REF_CYCLES,numel(cycles)))),:);
b=mean(table2array(ref(:,cfg.RESID_COLS)),1); th0=mean(table2array(ref(:,cfg.THETA9))./cfg.THETA_SPAN,1);
seq.R_samples=cell(numel(cycles),1); seq.theta=zeros(numel(cycles),9);
for t=1:numel(cycles)
    sub=d(d.cycle==cycles(t),:); seq.R_samples{t}=table2array(sub(:,cfg.RESID_COLS))-b;
    seq.theta(t,:)=table2array(sub(1,cfg.THETA9))./cfg.THETA_SPAN-th0;
end
end

function X=make_seq_local(rbar,L)
T=size(rbar,1); padded=[zeros(L-1,13);rbar]; X=zeros(T,L,13,'single');
for t=1:T, X(t,:,:)=single(padded(t:t+L-1,:)); end
end

function r=corr_value(x,y)
q=isfinite(x)&isfinite(y); C=corrcoef(x(q),y(q));
if size(C,1)<2, r=NaN; else, r=C(1,2); end
end
