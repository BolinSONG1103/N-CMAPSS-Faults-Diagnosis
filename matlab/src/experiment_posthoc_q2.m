function verdict=experiment_posthoc_q2(reuseRaw)
%EXPERIMENT_POSTHOC_Q2 A3：Q2s 小 lambda 与 highTRA 阈值事后诊断。
% 这是事后分析，不替代 q=0.75 的正式预注册未通过裁决。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
if nargin<1, reuseRaw=false; end
cfg=ncmapss_lib.config(false);
outDir=fullfile(matlabRoot,'outputs','stage_a','a3_q2');
figDir=fullfile(outDir,'figures');
if ~isfolder(outDir), mkdir(outDir); end
if ~isfolder(figDir), mkdir(figDir); end

qValues=[.75 .85 .90 .95];
lambdaGrid=logspace(-3,3,13);
formalLambda=logspace(-1,3,9);
windows=[1 10 100 1000];
lambdaSmall=lambdaGrid(1);
[~,imain]=min(abs(log(lambdaGrid/cfg.LAM_MAIN))); lambdaMain=lambdaGrid(imain);
formalVerdict=jsondecode(fileread(fullfile(matlabRoot,'outputs','t1_moe','verdict19b.json')));
rawFile=fullfile(outDir,'q2_lambda_raw.csv');
gainFile=fullfile(outDir,'q2_gate_gain_raw.csv');
oldSummaryFile=fullfile(outDir,'q2_quantile_summary.csv');

if reuseRaw && isfile(rawFile) && isfile(gainFile) && isfile(oldSummaryFile)
    fprintf('A3-Q2: reuse existing raw CSV and redraw summaries/figures.\n');
    raw=readtable(rawFile);
    gainRaw=readtable(gainFile);
    oldSummary=readtable(oldSummaryFile);
    fallbackTotal=oldSummary.fallback_rate;
    cycleTotal=ones(size(fallbackTotal));
else
pipeStream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
pipe=ncmapss_lib.build_pipeline(cfg,pipeStream, ...
    fullfile(matlabRoot,'cache','pipeline_cache_exact.mat'),false);
s=load(fullfile(matlabRoot,'outputs','t1_moe','models.mat'),'models','wMu','wSd');
model=s.models.MoE_K4; wMu=double(s.wMu); wSd=double(s.wSd);

loadStream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
data=cell(size(cfg.VALID_FILES,1),1);
for iv=1:size(cfg.VALID_FILES,1)
    d=ncmapss_lib.load_per_cycle(fullfile(cfg.DATA_DIR,cfg.VALID_FILES{iv,2}), ...
        'test',cfg,loadStream,[]);
    [d,~]=ncmapss_lib.add_corrected(d,pipe.ref);
    data{iv}=ncmapss_lib.attach_residuals(pipe,d);
end

rows=cell(0,1); gainRows=cell(0,1); ri=0; gi=0;
fallbackTotal=zeros(numel(qValues),1); cycleTotal=zeros(numel(qValues),1);
for iq=1:numel(qValues)
    qv=qValues(iq); cfgQ=cfg; cfgQ.SEG_Q=qv;
    % 公共初始种子降低不同分位阈值之间的抽样噪声。
    evalStream=ncmapss_lib.make_stream(4275,cfg.RNG_BACKEND);
    fprintf('\nA3-Q2: highTRA quantile=%.2f\n',qv);
    for iv=1:size(cfg.VALID_FILES,1)
        tag=cfg.VALID_FILES{iv,1}; d=data{iv}; units=sort(unique(d.unit));
        for u=units.'
            [wins,fb]=ncmapss_lib.build_windows(d,u,'highTRA',windows,cfgQ,evalStream);
            fallbackTotal(iq)=fallbackTotal(iq)+fb.highTRA;
            cycleTotal(iq)=cycleTotal(iq)+size(wins(1).R,1);
            for iw=1:numel(wins)
                R=wins(iw).R; theta=wins(iw).TH; T=size(R,1);
                Cum=ncmapss_lib.build_cum(T,9);
                H0list=repmat({pipe.Hn},T,1); Hmoe=cell(T,1);
                cycleGain=zeros(T,9);
                for t=1:T
                    op=wins(iw).OPS{t};
                    gains=chapter3_models.moe_gains(model,(double(op)-wMu)./wSd);
                    cycleGain(t,:)=mean(gains,1);
                    Hmoe{t}=pipe.Hn.*cycleGain(t,:);
                end
                for j=1:9
                    gi=gi+1; gainRows{gi,1}=struct('quantile',qv,'subset',tag, ... %#ok<AGROW>
                        'unit',double(u),'N',wins(iw).N,'component',j, ...
                        'mean_gain',mean(cycleGain(:,j)), ...
                        'mean_abs_gain_minus1',mean(abs(cycleGain(:,j)-1)));
                end
                M0=ncmapss_lib.build_M_var(H0list,T);
                Mm=ncmapss_lib.build_M_var(Hmoe,T);
                mseZero=mean(theta.^2,'all');
                for lam=lambdaGrid
                    th0=ncmapss_lib.solve_D_var(M0,Cum,R,lam,T,9);
                    thm=ncmapss_lib.solve_D_var(Mm,Cum,R,lam,T,9);
                    s0=1-mean((th0-theta).^2,'all')/max(mseZero,1e-12);
                    sm=1-mean((thm-theta).^2,'all')/max(mseZero,1e-12);
                    ri=ri+1; rows{ri,1}=struct('quantile',qv,'subset',tag, ... %#ok<AGROW>
                        'unit',double(u),'N',wins(iw).N,'lambda',lam, ...
                        'Z_zero_skill',0,'H0_skill',s0,'MoE_skill',sm,'gap',sm-s0);
                end
            end
            fprintf('  %s unit %d\n',tag,round(u));
        end
    end
end

raw=struct2table(vertcat(rows{:}));
gainRaw=struct2table(vertcat(gainRows{:}));
writetable(raw,rawFile,'Encoding','UTF-8');
writetable(gainRaw,gainFile,'Encoding','UTF-8');
end

gapGrid=groupsummary(raw,{'quantile','subset','N','lambda'},'mean', ...
    {'gap','H0_skill','MoE_skill','Z_zero_skill'});
writetable(gapGrid,fullfile(outDir,'q2_gap_grid.csv'),'Encoding','UTF-8');

summary=table('Size',[numel(qValues),11], ...
    'VariableTypes',repmat({'double'},1,11), ...
    'VariableNames',{'quantile','fallback_rate','formal_rate','formal_median_gap', ...
    'small_lambda','small_rate','small_median_gap','main_lambda','main_rate', ...
    'main_median_gap','median_abs_gain_minus1'});
for iq=1:numel(qValues)
    q=gapGrid.quantile==qValues(iq);
    formalMask=q&ismembertol(gapGrid.lambda,formalLambda,1e-10,'DataScale',1);
    smallMask=q&abs(log(gapGrid.lambda/lambdaSmall))<1e-10;
    mainMask=q&abs(log(gapGrid.lambda/lambdaMain))<1e-10;
    gq=gainRaw.quantile==qValues(iq);
    summary.quantile(iq)=qValues(iq);
    summary.fallback_rate(iq)=fallbackTotal(iq)/max(cycleTotal(iq),1);
    summary.formal_rate(iq)=mean(gapGrid.mean_gap(formalMask)>0);
    summary.formal_median_gap(iq)=median(gapGrid.mean_gap(formalMask));
    summary.small_lambda(iq)=lambdaSmall;
    summary.small_rate(iq)=mean(gapGrid.mean_gap(smallMask)>0);
    summary.small_median_gap(iq)=median(gapGrid.mean_gap(smallMask));
    summary.main_lambda(iq)=lambdaMain;
    summary.main_rate(iq)=mean(gapGrid.mean_gap(mainMask)>0);
    summary.main_median_gap(iq)=median(gapGrid.mean_gap(mainMask));
    summary.median_abs_gain_minus1(iq)=median(gainRaw.mean_abs_gain_minus1(gq));
end
writetable(summary,fullfile(outDir,'q2_quantile_summary.csv'),'Encoding','UTF-8');

[G,qByLambda,lambdaByGroup]=findgroups(gapGrid.quantile,gapGrid.lambda);
lambdaSummary=table(qByLambda,lambdaByGroup, ...
    splitapply(@mean,gapGrid.mean_gap,G), ...
    splitapply(@median,gapGrid.mean_gap,G), ...
    splitapply(@(x)prctile(x,25),gapGrid.mean_gap,G), ...
    splitapply(@(x)prctile(x,75),gapGrid.mean_gap,G), ...
    splitapply(@(x)mean(x>0),gapGrid.mean_gap,G), ...
    'VariableNames',{'quantile','lambda','mean_gap','median_gap','q25_gap','q75_gap','win_rate'});
writetable(lambdaSummary,fullfile(outDir,'q2_lambda_summary.csv'),'Encoding','UTF-8');

q75=gapGrid.quantile==.75;
small=gapGrid(q75&abs(log(gapGrid.lambda/lambdaSmall))<1e-10,{'subset','N','mean_gap'});
main=gapGrid(q75&abs(log(gapGrid.lambda/lambdaMain))<1e-10,{'subset','N','mean_gap'});
small.Properties.VariableNames{3}='small_gap';
main.Properties.VariableNames{3}='main_gap';
paired=innerjoin(small,main,'Keys',{'subset','N'});
paired.delta_small_minus_main=paired.small_gap-paired.main_gap;
writetable(paired,fullfile(outDir,'q2_small_vs_main_paired.csv'),'Encoding','UTF-8');
ci=bootstrap_median_ci(paired.delta_small_minus_main,5000,4312);
supported=median(paired.small_gap)>0 && median(paired.delta_small_minus_main)>0 && ci(1)>0;

plot_lambda_mechanism(figDir,lambdaSummary,lambdaSmall,lambdaMain);
plot_quantile_sensitivity(figDir,summary);
plot_paired_and_gains(figDir,paired,gainRaw,cfg);

verdict=struct('analysis_type','posthoc_not_preregistered', ...
    'formal_Q2s_remains_false',~formalVerdict.Q2s, ...
    'formal_Q2s_dominance_rate',formalVerdict.Q2s_dominance_rate, ...
    'formal_Q2s_median_gap',formalVerdict.Q2s_median_gap,'lambda_small',lambdaSmall, ...
    'lambda_main_nearest',lambdaMain,'q75_small_rate',summary.small_rate(1), ...
    'q75_small_median_gap',summary.small_median_gap(1), ...
    'q75_main_rate',summary.main_rate(1), ...
    'q75_main_median_gap',summary.main_median_gap(1), ...
    'paired_median_improvement',median(paired.delta_small_minus_main), ...
    'paired_median_improvement_ci95',ci, ...
    'shrinkage_explanation_supported',supported, ...
    'median_abs_gain_minus1_q75',summary.median_abs_gain_minus1(1), ...
    'quantile_summary',table2struct(summary));
write_json(fullfile(outDir,'verdict_a3_q2.json'),verdict);
fprintf('\nA3-Q2: small lambda median gap %.5f; main lambda %.5f; paired delta CI [%.5f, %.5f]\n', ...
    verdict.q75_small_median_gap,verdict.q75_main_median_gap,ci(1),ci(2));
end

function ci=bootstrap_median_ci(x,nBoot,seed)
x=x(:); n=numel(x); s=RandStream('mt19937ar','Seed',seed); b=zeros(nBoot,1);
for i=1:nBoot
    b(i)=median(x(randi(s,n,[n 1])));
end
ci=prctile(b,[2.5 97.5]);
end

function plot_lambda_mechanism(figDir,s,lambdaSmall,lambdaMain)
c=thesis_plot.colors(); q=s.quantile==.75; z=s(q,:);
f=thesis_plot.new(17.8,7.8); t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); hold(ax,'on');
set(ax,'XScale','log');
fill(ax,[z.lambda;flipud(z.lambda)],[z.q25_gap;flipud(z.q75_gap)],c(1,:), ...
    'FaceAlpha',.16,'EdgeColor','none');
plot(ax,z.lambda,z.median_gap,'-o','Color',c(1,:),'MarkerFaceColor','white');
yline(ax,0,'Color',[.25 .25 .25],'LineStyle','--');
xline(ax,lambdaSmall,'Color',c(3,:),'LineStyle',':','DisplayName','Small lambda');
xline(ax,lambdaMain,'Color',c(2,:),'LineStyle',':','DisplayName','Main lambda');
xlabel(ax,'Tikhonov lambda'); ylabel(ax,'Median MoE - H0 skill'); grid(ax,'on');
thesis_plot.panel(ax,'a');

ax=nexttile(t); hold(ax,'on'); set(ax,'XScale','log');
plot(ax,z.lambda,100*z.win_rate,'-s','Color',c(2,:),'MarkerFaceColor','white');
yline(ax,95,'Color',[.35 .35 .35],'LineStyle','--');
xline(ax,lambdaSmall,'Color',c(3,:),'LineStyle',':');
xline(ax,lambdaMain,'Color',c(2,:),'LineStyle',':');
ylim(ax,[0 105]); xlabel(ax,'Tikhonov lambda'); ylabel(ax,'MoE dominance rate (%)'); grid(ax,'on');
thesis_plot.panel(ax,'b');
thesis_plot.export(f,figDir,'fig_a3_q2_lambda_mechanism'); close(f);
end

function plot_quantile_sensitivity(figDir,s)
c=thesis_plot.colors(); f=thesis_plot.new(17.8,11.2);
t=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); plot(ax,s.quantile,100*s.formal_rate,'-o','Color',c(1,:),'MarkerFaceColor','white');
hold(ax,'on'); yline(ax,95,'--','Color',[.35 .35 .35]); ylim(ax,[0 105]); grid(ax,'on');
xlabel(ax,'highTRA quantile'); ylabel(ax,'Dominance rate (%)'); thesis_plot.panel(ax,'a');
ax=nexttile(t); plot(ax,s.quantile,s.formal_median_gap,'-o','Color',c(2,:),'MarkerFaceColor','white');
hold(ax,'on'); yline(ax,.01,'--','Color',[.35 .35 .35]); grid(ax,'on');
xlabel(ax,'highTRA quantile'); ylabel(ax,'Median skill gap'); thesis_plot.panel(ax,'b');
ax=nexttile(t); plot(ax,s.quantile,100*s.fallback_rate,'-o','Color',c(3,:),'MarkerFaceColor','white');
grid(ax,'on'); xlabel(ax,'highTRA quantile'); ylabel(ax,'Fallback cycles (%)'); thesis_plot.panel(ax,'c');
ax=nexttile(t); plot(ax,s.quantile,s.median_abs_gain_minus1,'-o','Color',c(4,:),'MarkerFaceColor','white');
grid(ax,'on'); xlabel(ax,'highTRA quantile'); ylabel(ax,'Median |g - 1|'); thesis_plot.panel(ax,'d');
thesis_plot.export(f,figDir,'fig_a3_q2_quantile_sensitivity'); close(f);
end

function plot_paired_and_gains(figDir,paired,gainRaw,cfg)
c=thesis_plot.colors(); f=thesis_plot.new(17.8,8.5);
t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); hold(ax,'on');
for i=1:height(paired)
    ci=1+strcmp(paired.subset{i},'DS03');
    plot(ax,1:2,[paired.main_gap(i) paired.small_gap(i)],'-','Color',[.72 .72 .72]);
    scatter(ax,1,paired.main_gap(i),28,c(ci,:),'filled','Marker',marker_for_n(paired.N(i)));
    scatter(ax,2,paired.small_gap(i),28,c(ci,:),'filled','Marker',marker_for_n(paired.N(i)));
end
yline(ax,0,'--','Color',[.3 .3 .3]); xlim(ax,[.65 2.35]);
set(ax,'XTick',1:2,'XTickLabel',{'Main lambda','lambda = 10^{-3}'});
ylabel(ax,'MoE - H0 skill'); grid(ax,'on'); thesis_plot.panel(ax,'a');

ax=nexttile(t);
g=groupsummary(gainRaw(gainRaw.quantile==.75,:),'component','median','mean_gain');
bar(ax,g.component,g.median_mean_gain,.68,'FaceColor',c(1,:),'EdgeColor','none'); hold(ax,'on');
yline(ax,1,'--','Color',[.3 .3 .3]); xlabel(ax,'Health parameter index'); ylabel(ax,'Median gate gain');
set(ax,'XTick',1:numel(cfg.THETA9)); grid(ax,'on'); thesis_plot.panel(ax,'b');
thesis_plot.export(f,figDir,'fig_a3_q2_paired_gains'); close(f);
end

function m=marker_for_n(N)
switch N
    case 1, m='o';
    case 10, m='s';
    case 100, m='^';
    otherwise, m='d';
end
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
