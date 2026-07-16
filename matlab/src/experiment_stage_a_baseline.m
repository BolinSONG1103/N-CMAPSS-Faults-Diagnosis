function verdict=experiment_stage_a_baseline()
%EXPERIMENT_STAGE_A_BASELINE A1：隔离兼容层并量化纯 MATLAB 基准漂移。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
cfg=ncmapss_lib.config(false);
outDir=fullfile(matlabRoot,'outputs','stage_a','a1_baseline');
figDir=fullfile(outDir,'figures');
if ~isfolder(outDir), mkdir(outDir); end
if ~isfolder(figDir), mkdir(figDir); end

fprintf('\nA1: locked sklearn HistGB vs native MATLAB LSBoost\n');
exactStream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
exactCache=fullfile(matlabRoot,'cache','pipeline_cache_exact.mat');
pipeExact=ncmapss_lib.build_pipeline(cfg,exactStream,exactCache,false);
rawExact=evaluate_ds03(pipeExact,cfg,exactStream,'sklearn_histgb_compat');

cfgNative=cfg;
cfgNative.BASELINE_BACKEND='matlab_lsboost_control';
nativeStream=ncmapss_lib.make_stream(cfgNative.SEED,cfgNative.RNG_BACKEND);
nativeCache=fullfile(matlabRoot,'cache','pipeline_cache_lsboost_control.mat');
pipeNative=ncmapss_lib.build_pipeline(cfgNative,nativeStream,nativeCache,false);
rawNative=evaluate_ds03(pipeNative,cfgNative,nativeStream,'matlab_lsboost_control');

raw=[rawExact;rawNative];
writetable(raw,fullfile(outDir,'baseline_comparison_raw.csv'),'Encoding','UTF-8');

isD=strcmp(raw.method,'D');
skillExact=mean(raw.skill(isD&strcmp(raw.backend,'sklearn_histgb_compat')));
skillNative=mean(raw.skill(isD&strcmp(raw.backend,'matlab_lsboost_control')));
summary=table( ...
    {'sklearn_histgb_compat';'matlab_lsboost_control';'Z_zero'}, ...
    [pipeExact.cond_Hn;pipeNative.cond_Hn;NaN], ...
    [skillExact;skillNative;0], ...
    [13;13;0], ...
    'VariableNames',{'backend','cond_Hn','DS03_skill','n_health_models'});
writetable(summary,fullfile(outDir,'baseline_comparison_summary.csv'),'Encoding','UTF-8');

condAbs=abs(pipeNative.cond_Hn-pipeExact.cond_Hn);
condRel=condAbs/pipeExact.cond_Hn;
skillAbs=abs(skillNative-skillExact);
drift=table({'cond_Hn';'DS03_skill';'Z_zero_skill'}, ...
    [pipeExact.cond_Hn;skillExact;0],[pipeNative.cond_Hn;skillNative;0], ...
    [condAbs;skillAbs;0],[condRel;skillAbs;0],[.01;.01;0], ...
    [condRel<.01;skillAbs<.01;true], ...
    'VariableNames',{'metric','locked_value','native_value','absolute_drift', ...
    'decision_drift','switch_tolerance','within_switch_tolerance'});
writetable(drift,fullfile(outDir,'baseline_drift.csv'),'Encoding','UTF-8');

sensor=(1:numel(pipeExact.corrected_cols)).';
sensorMap=table(sensor,pipeExact.corrected_cols.','VariableNames',{'sensor_index','corrected_channel'});
theta=(1:numel(cfg.THETA9)).';
thetaMap=table(theta,cfg.THETA9.','VariableNames',{'theta_index','parameter'});
writetable(sensorMap,fullfile(outDir,'sensor_axis_mapping.csv'),'Encoding','UTF-8');
writetable(thetaMap,fullfile(outDir,'theta_axis_mapping.csv'),'Encoding','UTF-8');

plot_fingerprint(figDir,raw,pipeExact,pipeNative,skillExact,skillNative,cfg);
plot_matrices(figDir,pipeExact.Hn,pipeNative.Hn);

verdict=struct('analysis_type','stage_A_baseline_sensitivity', ...
    'locked_backend','sklearn_histgb_compat','control_backend','matlab_lsboost_control', ...
    'cond_locked',pipeExact.cond_Hn,'cond_native',pipeNative.cond_Hn, ...
    'cond_relative_drift',condRel,'skill_locked',skillExact,'skill_native',skillNative, ...
    'skill_absolute_drift',skillAbs,'cond_within_one_percent',condRel<.01, ...
    'skill_within_0p01',skillAbs<.01, ...
    'automatic_switch_allowed',condRel<.01 && skillAbs<.01, ...
    'dependency_status','permanent_for_locked_preregistered_results', ...
    'author_decision_required_for_future_native_switch',true);
write_json(fullfile(outDir,'verdict_a1.json'),verdict);
fprintf('  cond drift: %.3f%%; DS03 skill drift: %.6f\n',100*condRel,skillAbs);
end

function raw=evaluate_ds03(pipe,cfg,stream,backend)
path=fullfile(cfg.DATA_DIR,'N-CMAPSS_DS03-012.h5');
d=ncmapss_lib.load_per_cycle(path,'test',cfg,stream,[]);
[d,~]=ncmapss_lib.add_corrected(d,pipe.ref);
d=ncmapss_lib.attach_residuals(pipe,d);
units=sort(unique(d.unit));
idxTrue=find(ismember(cfg.THETA9,{'HPT_eff_mod','LPT_eff_mod','LPT_flow_mod'}));
rows=cell(2*numel(units),1); ri=0;
for i=1:numel(units)
    [wins,~]=ncmapss_lib.build_windows(d,units(i),'full',1000,cfg,stream);
    R=wins.R; theta=wins.TH; T=size(R,1);
    M=ncmapss_lib.build_M(pipe.Hn,T); Cum=ncmapss_lib.build_cum(T,9);
    thetaHat=ncmapss_lib.solve_D(M,Cum,R,cfg.LAM_MAIN,T,9);
    metZero=ncmapss_lib.compute_metrics(zeros(size(theta)),theta,idxTrue);
    metD=ncmapss_lib.compute_metrics(thetaHat,theta,idxTrue);
    ri=ri+1; rows{ri}=make_row(backend,units(i),'Z_zero',metZero,cfg);
    ri=ri+1; rows{ri}=make_row(backend,units(i),'D',metD,cfg);
    fprintf('  %-24s unit %d skill %.6f\n',backend,round(units(i)),metD.skill);
end
raw=struct2table(vertcat(rows{:}));
end

function r=make_row(backend,unit,method,m,cfg)
r=struct('backend',backend,'subset','DS03','unit',double(unit),'N',1000, ...
    'lambda',cfg.LAM_MAIN,'method',method,'skill',m.skill,'rmse',m.rmse, ...
    'rmse_early',m.rmse_early,'false_alarm',m.false_alarm,'detect_corr',m.detect_corr);
end

function plot_fingerprint(figDir,raw,pipeExact,pipeNative,skillExact,skillNative,cfg)
c=thesis_plot.colors(); f=thesis_plot.new(17.8,8.2);
t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

ax=nexttile(t); hold(ax,'on');
bar(ax,1:2,[pipeExact.cond_Hn pipeNative.cond_Hn],.62,'FaceColor','flat', ...
    'CData',[c(1,:);c(2,:)],'EdgeColor','none');
targetBand=cfg.COND_TARGET*[1-cfg.COND_REL_TOL 1+cfg.COND_REL_TOL];
patch(ax,[.55 2.45 2.45 .55],targetBand([1 1 2 2]),[.5 .5 .5], ...
    'FaceAlpha',.12,'EdgeColor','none');
yline(ax,cfg.COND_TARGET,'Color',[.25 .25 .25],'LineStyle','--','LineWidth',1);
set(ax,'XTick',1:2,'XTickLabel',{'Locked HistGB','MATLAB LSBoost'});
ylabel(ax,'cond(H_n)'); ylim(ax,[min(targetBand(1)*.94,1450) max(pipeNative.cond_Hn*1.08,targetBand(2)*1.04)]);
for i=1:2
    vals=[pipeExact.cond_Hn pipeNative.cond_Hn];
    text(ax,i,vals(i)+18,sprintf('%.1f',vals(i)),'HorizontalAlignment','center','FontSize',8);
end
thesis_plot.panel(ax,'a');

ax=nexttile(t); hold(ax,'on');
d=raw(strcmp(raw.method,'D'),:); units=unique(d.unit);
for i=1:numel(units)
    q=d.unit==units(i);
    y=[d.skill(q&strcmp(d.backend,'sklearn_histgb_compat')); ...
        d.skill(q&strcmp(d.backend,'matlab_lsboost_control'))];
    plot(ax,1:2,y,'-','Color',[.72 .72 .72],'LineWidth',.8);
    scatter(ax,1:2,y,24,[c(1,:);c(2,:)],'filled','MarkerEdgeColor','white','LineWidth',.4);
end
plot(ax,1:2,[skillExact skillNative],'-d','Color',c(7,:),'MarkerFaceColor','white', ...
    'MarkerSize',6,'LineWidth',1.5,'DisplayName','Mean');
yline(ax,0,'Color',[.35 .35 .35],'LineStyle',':');
set(ax,'XTick',1:2,'XTickLabel',{'Locked HistGB','MATLAB LSBoost'});
xlim(ax,[.65 2.35]); ylabel(ax,'DS03 skill score'); grid(ax,'on');
legend(ax,'Location','southoutside','Orientation','horizontal');
thesis_plot.panel(ax,'b');

thesis_plot.export(f,figDir,'fig_a1_baseline_fingerprint'); close(f);
end

function plot_matrices(figDir,H0,Hm)
cmap=diverging_map(257); f=thesis_plot.new(17.8,7.8);
t=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
lim=max(abs([H0(:);Hm(:)]));
ax=nexttile(t); imagesc(ax,H0,[-lim lim]); axis(ax,'tight'); colormap(ax,cmap);
xlabel(ax,'Health parameter index'); ylabel(ax,'Sensor index'); title(ax,'Locked HistGB H_n','FontWeight','normal');
colorbar(ax,'Location','southoutside'); thesis_plot.panel(ax,'a');

delta=Hm-H0; dlim=max(abs(delta),[],'all');
ax=nexttile(t); imagesc(ax,delta,[-dlim dlim]); axis(ax,'tight'); colormap(ax,cmap);
xlabel(ax,'Health parameter index'); ylabel(ax,'Sensor index'); title(ax,'LSBoost minus HistGB','FontWeight','normal');
colorbar(ax,'Location','southoutside'); thesis_plot.panel(ax,'b');

ax=nexttile(t); hold(ax,'on');
s0=svd(H0); sm=svd(Hm);
semilogy(ax,1:numel(s0),s0,'-o','Color',[0 114 178]/255,'MarkerFaceColor','white','DisplayName','HistGB');
semilogy(ax,1:numel(sm),sm,'-s','Color',[213 94 0]/255,'MarkerFaceColor','white','DisplayName','LSBoost');
xlabel(ax,'Singular-value index'); ylabel(ax,'Singular value'); grid(ax,'on');
legend(ax,'Location','northeast'); title(ax,'Conditioning spectrum','FontWeight','normal'); thesis_plot.panel(ax,'c');
thesis_plot.export(f,figDir,'fig_a1_hn_structure'); close(f);
end

function map=diverging_map(n)
if nargin<1, n=257; end
lo=[49 54 149]/255; mid=[247 247 247]/255; hi=[165 0 38]/255;
k=floor(n/2);
map=[interp1([1 k+1],[lo;mid],1:k+1);interp1([1 n-k],[mid;hi],2:n-k)];
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
