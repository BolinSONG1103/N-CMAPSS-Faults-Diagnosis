function manifest=build_chapter3_complete_figures(root)
%BUILD_CHAPTER3_COMPLETE_FIGURES 生成第三章完整图集（图3-2至图3-12）。
% 图3-1由作者手画；程序图统一导出 600 dpi PNG 与矢量 PDF。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
resultDir=fullfile(root,'chapter3_results'); figDir=fullfile(resultDir,'figures');
dataDir=fullfile(matlabRoot,'outputs','complete_figures','data');
if ~isfolder(figDir), mkdir(figDir); end
required={fullfile(dataDir,'a1_similarity_correction.csv'), ...
    fullfile(dataDir,'a2_baseline_fit.csv'), ...
    fullfile(dataDir,'b3_lambda_trajectory.csv'), ...
    fullfile(dataDir,'b4_multiunit_component_error.csv'), ...
    fullfile(dataDir,'c3_c4_extrapolation_trajectory.csv'), ...
    fullfile(matlabRoot,'outputs','stage_bc','figure_data','fig3_2_ds03_unit13_trajectory.csv')};
assert(all(cellfun(@isfile,required)),'完整图集中间数据尚未补齐。');
clean_figure_dir(figDir);

rows=cell(12,1);
rows{1}=mrow("图3-1","图3-1_方法总流程图","3.1", ...
    "作者手画的方法总流程图；程序不生成","作者手画，无程序数据", ...
    "方法从数据预处理、影响矩阵辨识到约束反演和外推验证形成闭环。",false,"author");

plot_similarity(dataDir,figDir);
rows{2}=mrow("图3-2","图3-2_相似修正效果","3.2", ...
    "DS03 unit 13 健康早期压力通道相似修正前后工况漂移", ...
    "complete_figures/data/a1_similarity_correction.csv", ...
    "相似修正将代表性压力通道的工况漂移幅度压缩约85%-92%，为健康基准残差化提供稳定输入。",true,"png;pdf");

plot_baseline_fit(dataDir,figDir);
rows{3}=mrow("图3-3","图3-3_健康基准拟合","3.2", ...
    "六个代表修正通道的健康基准预测-实测一致性", ...
    "complete_figures/data/a2_baseline_fit.csv", ...
    "健康基准在正常样本上高精度还原修正通道，其残差可作为部件退化的观测量。",true,"png;pdf");

plot_influence(matlabRoot,figDir);
rows{4}=mrow("图3-4","图3-4_影响矩阵结构与可辨识性","3.3", ...
    "H_n热力图、奇异值谱和部件列向量夹角", ...
    "cache/pipeline_cache_exact.mat", ...
    "影响矩阵虽为病态系统，但部件方向仍可区分，为约束反演定位部件提供几何基础。",true,"png;pdf");

plot_all_trajectories(matlabRoot,figDir);
rows{5}=mrow("图3-5","图3-5_三部件退化轨迹估计","3.4", ...
    "DS03 unit 13 九部件D约束反演轨迹与真值", ...
    "stage_bc/figure_data/fig3_2_ds03_unit13_trajectory.csv", ...
    "约束反演逐循环恢复三部件退化轨迹，并将六个未退化部件限制在零线邻域。",true,"png;pdf");

plot_constraint_value(matlabRoot,figDir);
rows{6}=mrow("图3-6","图3-6_约束抑制未退化部件虚警","3.5", ...
    "同一代表机未退化部件与真退化部件的B/D轨迹对照", ...
    "stage_bc/figure_data/fig3_2_ds03_unit13_trajectory.csv", ...
    "无约束B在未退化部件上产生可见虚警，约束D保持零线并改善真退化轨迹的稳定性。",true,"png;pdf");

plot_lambda_trajectories(dataDir,figDir);
rows{7}=mrow("图3-7","图3-7_收缩强度轨迹效应","3.5", ...
    "DS03 unit 13 的LPT效率在弱、中、强收缩下的D轨迹", ...
    "complete_figures/data/b3_lambda_trajectory.csv", ...
    "收缩过弱导致退化幅度过冲，收缩过强压平轨迹，适中收缩最接近真值。",true,"png;pdf");

plot_multiunit_error(dataDir,figDir);
rows{8}=mrow("图3-8","图3-8_多台发动机逐部件误差","3.5", ...
    "DS02/DS03各测试机真实退化部件D与Z_zero轨迹RMSE配对", ...
    "complete_figures/data/b4_multiunit_component_error.csv", ...
    "约束反演在全部测试发动机的真实退化部件上均获得低于全零哨兵的轨迹误差。",true,"png;pdf");

plot_lambda_mechanism(root,matlabRoot,figDir);
rows{9}=mrow("图3-9","图3-9_经典lambda准则失效","3.5", ...
    "B/D真实技能分路径、经典准则选点与tau_required-sqrt(N)", ...
    "raw_scan.csv; selected_lambda.csv; tau_required_points.csv", ...
    "约束托平D的弱收缩路径，使经典lambda准则偏离最优区；tau_required随sqrt(N)近似线性增长。",true,"png;pdf");

plot_shrinkage(matlabRoot,figDir);
rows{10}=mrow("图3-10","图3-10_工况增益被收缩吸收","3.6", ...
    "MoE-H0技能分差随lambda的full/highTRA路径", ...
    "t1_moe/pointwise_gap.csv; stage_a/a3_q2/q2_lambda_raw.csv", ...
    "工况自适应增益主要出现在弱收缩区，主工作点的约束与收缩将其压缩到接近零。",true,"png;pdf");

plot_extrapolation_fault(dataDir,figDir);
rows{11}=mrow("图3-11","图3-11_外推真退化部件轨迹","3.7", ...
    "未见三故障组合中LPT效率的D/E_Lin/E_MLP/真值轨迹", ...
    "complete_figures/data/c3_c4_extrapolation_trajectory.csv", ...
    "未见故障组合上D最贴近真值，线性模型保留退化趋势，MLP出现轨迹波动与末期过冲。",true,"png;pdf");

plot_extrapolation_false_alarm(matlabRoot,dataDir,figDir);
rows{12}=mrow("图3-12","图3-12_外推未退化部件虚警","3.7", ...
    "同一外推场景中未退化LPC效率轨迹及全OOD虚警比", ...
    "complete_figures/data/c3_c4_extrapolation_trajectory.csv; b2_e4/e4_ood_skill_false_alarm.csv", ...
    "未退化部件上D保持零线，E_Lin出现可见虚警，E_MLP的偏离进一步扩大。",true,"png;pdf");

manifest=vertcat(rows{:});
end

function clean_figure_dir(figDir)
for ext={'.png','.pdf','.svg'}
    old=dir(fullfile(figDir,['图3-*' ext{1}]));
    for i=1:numel(old), delete(fullfile(old(i).folder,old(i).name)); end
end
end

function plot_similarity(dataDir,figDir)
t=readtable(fullfile(dataDir,'a1_similarity_correction.csv'),'TextType','string');
channels=["P50","P21"]; c=thesis_plot.semantic_colors();
f=thesis_plot.new(17.8,11.2); tl=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
letter='a';
for i=1:numel(channels)
    allq=t.raw_channel==channels(i); raw=t(allq&t.stage=="raw",:); cor=t(allq&t.stage=="corrected",:);
    assert(raw.drift_reduction_pctpt(1)>0,'A1代表通道未显示漂移降低。');
    lim=prctile([raw.relative_deviation_pct;cor.relative_deviation_pct],[1 99]);
    pad=.08*max(diff(lim),1); lim=lim+[-pad pad];
    ax=nexttile(tl); plot_condition_cloud(ax,raw,c.secondary,lim);
    title(ax,[char(channels(i)) ' raw'],'FontWeight','normal');
    text(ax,.04,.92,sprintf('binned span = %.1f%%',raw.binned_drift_span_pct(1)), ...
        'Units','normalized','FontSize',8); thesis_plot.panel(ax,letter); letter=letter+1;
    ax=nexttile(tl); plot_condition_cloud(ax,cor,c.D,lim);
    reduction=100*(1-cor.binned_drift_span_pct(1)/raw.binned_drift_span_pct(1));
    title(ax,[char(channels(i)) ' corrected'],'FontWeight','normal');
    text(ax,.04,.92,sprintf('binned span = %.1f%%  (%.1f%% lower)', ...
        cor.binned_drift_span_pct(1),reduction),'Units','normalized','FontSize',8); 
    thesis_plot.panel(ax,letter); letter=letter+1;
end
thesis_plot.export(f,figDir,'图3-2_相似修正效果',false); close(f);
end

function plot_condition_cloud(ax,t,color,limits)
hold(ax,'on'); n=height(t); idx=unique(round(linspace(1,n,min(750,n))));
scatter(ax,t.condition_value(idx),t.relative_deviation_pct(idx),9,color,'filled', ...
    'MarkerFaceAlpha',.16,'MarkerEdgeAlpha',0);
[xc,yc]=binned_median(t.condition_value,t.relative_deviation_pct,12);
plot(ax,xc,yc,'-o','Color',color,'MarkerFaceColor','white','MarkerSize',4,'LineWidth',1.7);
yline(ax,0,':','Color',[.7 .7 .7]); ylim(ax,limits); grid(ax,'on');
xlabel(ax,'Pressure similarity ratio, delta_c'); ylabel(ax,'Relative channel deviation (%)');
end

function [xc,yc]=binned_median(x,y,nBins)
edges=quantile(x,linspace(0,1,nBins+1)); edges=unique(edges); xc=[]; yc=[];
for k=1:numel(edges)-1
    if k<numel(edges)-1, q=x>=edges(k)&x<edges(k+1); else, q=x>=edges(k)&x<=edges(k+1); end
    if any(q), xc(end+1,1)=median(x(q)); yc(end+1,1)=median(y(q)); end %#ok<AGROW>
end
end

function plot_baseline_fit(dataDir,figDir)
t=readtable(fullfile(dataDir,'a2_baseline_fit.csv'),'TextType','string');
channels=["Nf_c","Wf_c","T30_c","T50_c","P40_c","P50_c"]; c=thesis_plot.semantic_colors();
f=thesis_plot.new(17.8,11.5); tl=tiledlayout(f,2,3,'TileSpacing','compact','Padding','compact');
for i=1:numel(channels)
    z=t(t.channel==channels(i),:); ax=nexttile(tl); hold(ax,'on');
    idx=unique(round(linspace(1,height(z),min(650,height(z)))));
    scatter(ax,z.measured(idx),z.predicted(idx),10,c.D,'filled','MarkerFaceAlpha',.20,'MarkerEdgeAlpha',0);
    lo=min([z.measured;z.predicted]); hi=max([z.measured;z.predicted]); pad=.04*max(hi-lo,eps);
    plot(ax,[lo-pad hi+pad],[lo-pad hi+pad],'--','Color',c.truth,'LineWidth',1.2);
    xlim(ax,[lo-pad hi+pad]); ylim(ax,[lo-pad hi+pad]); axis(ax,'square'); grid(ax,'on');
    xlabel(ax,'Measured'); ylabel(ax,'Predicted'); title(ax,pretty_channel(channels(i)),'FontWeight','normal');
    relStd=100*z.residual_std(1)/max(range(z.measured),eps);
    text(ax,.05,.91,sprintf('R^2 = %.5f\nresidual SD/range = %.2f%%',z.R2(1),relStd), ...
        'Units','normalized','FontSize',7.6,'VerticalAlignment','top');
    thesis_plot.panel(ax,char('a'+i-1));
end
thesis_plot.export(f,figDir,'图3-3_健康基准拟合',false); close(f);
end

function plot_influence(matlabRoot,figDir)
s=load(fullfile(matlabRoot,'cache','pipeline_cache_exact.mat'),'pipe'); H=s.pipe.Hn;
cfg=ncmapss_lib.config(false); c=thesis_plot.semantic_colors();
shortSensors=cellfun(@pretty_channel,s.pipe.corrected_cols,'UniformOutput',false);
shortTheta=cellfun(@short_param,cfg.THETA9,'UniformOutput',false);
f=thesis_plot.new(17.8,10.8); tl=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(tl,1,[2 1]); imagesc(ax,H); axis(ax,'tight'); m=max(abs(H),[],'all'); clim(ax,[-m m]);
colormap(ax,thesis_plot.diverging(257)); cb=colorbar(ax); ylabel(cb,'Normalized sensitivity');
set(ax,'XTick',1:9,'XTickLabel',shortTheta,'XTickLabelRotation',45, ...
    'YTick',1:13,'YTickLabel',shortSensors); xlabel(ax,'Health parameter'); ylabel(ax,'Corrected channel');
title(ax,'Normalized influence matrix H_n','FontWeight','normal'); thesis_plot.panel(ax,'a');

ax=nexttile(tl,2); sv=svd(H); semilogy(ax,1:numel(sv),sv,'-o','Color',c.D, ...
    'MarkerFaceColor','white','LineWidth',1.6); grid(ax,'on');
xlabel(ax,'Singular-value index'); ylabel(ax,'Singular value');
text(ax,.05,.10,sprintf('cond(H_n) = %.0f',sv(1)/sv(end)),'Units','normalized','FontSize',8);
title(ax,'Singular-value spectrum','FontWeight','normal'); thesis_plot.panel(ax,'b');

norms=sqrt(sum(H.^2,1)); C=(H.'*H)./(norms.'*norms); C=max(-1,min(1,C)); A=acosd(C);
A(1:10:end)=NaN; [minAngle,idx]=min(A,[],'all','omitnan'); [rr,cc]=ind2sub(size(A),idx);
ax=nexttile(tl,4); imagesc(ax,A,[0 max(A,[],'all','omitnan')]); axis(ax,'square');
colormap(ax,thesis_plot.viridis(256)); cb=colorbar(ax); ylabel(cb,'Angle (deg)');
set(ax,'XTick',1:9,'XTickLabel',shortTheta,'XTickLabelRotation',45,'YTick',1:9,'YTickLabel',shortTheta);
xlabel(ax,'Health parameter'); ylabel(ax,'Health parameter'); title(ax,'Column-vector angle','FontWeight','normal'); hold(ax,'on');
rectangle(ax,'Position',[cc-.5 rr-.5 1 1],'EdgeColor',c.B,'LineWidth',1.7);
text(ax,cc,rr,sprintf(' %.1f deg',minAngle),'Color',c.B,'FontSize',7.5,'FontWeight','bold', ...
    'HorizontalAlignment','left','VerticalAlignment','middle'); thesis_plot.panel(ax,'c');
thesis_plot.export(f,figDir,'图3-4_影响矩阵结构与可辨识性',false); close(f);
end

function plot_all_trajectories(matlabRoot,figDir)
t=readtable(fullfile(matlabRoot,'outputs','stage_bc','figure_data','fig3_2_ds03_unit13_trajectory.csv'),'TextType','string');
faults=["HPT_eff_mod","LPT_eff_mod","LPT_flow_mod"];
others=["fan_eff_mod","fan_flow_mod","HPC_eff_mod","HPC_flow_mod","LPC_eff_mod","LPC_flow_mod"];
params=[faults others]; c=thesis_plot.semantic_colors();
f=thesis_plot.new(17.8,15.2); tl=tiledlayout(f,3,3,'TileSpacing','compact','Padding','compact');
nonLim=[min(t.theta_D_pct(ismember(t.parameter,others)))-.03 .04];
for i=1:numel(params)
    z=t(t.parameter==params(i),:); ax=nexttile(tl); hold(ax,'on');
    if i<=3
        plot(ax,z.cycle,z.theta_true_pct,'--','Color',c.truth,'LineWidth',1.35,'DisplayName','Truth');
        plot(ax,z.cycle,z.theta_D_pct,'-','Color',c.D,'LineWidth',1.75,'DisplayName','D');
    else
        plot(ax,z.cycle,z.theta_D_pct,'-','Color',c.D,'LineWidth',1.55,'DisplayName','D'); ylim(ax,nonLim);
    end
    yline(ax,0,':','Color',c.Z_zero,'HandleVisibility','off'); grid(ax,'on');
    xlabel(ax,'Flight cycle'); ylabel(ax,'Degradation (%)'); title(ax,pretty_param(params(i)),'FontWeight','normal');
    if i==1, legend(ax,'Location','southwest'); end
    thesis_plot.panel(ax,char('a'+i-1));
end
thesis_plot.export(f,figDir,'图3-5_三部件退化轨迹估计',false); close(f);
end

function plot_constraint_value(matlabRoot,figDir)
t=readtable(fullfile(matlabRoot,'outputs','stage_bc','figure_data','fig3_2_ds03_unit13_trajectory.csv'),'TextType','string');
c=thesis_plot.semantic_colors(); f=thesis_plot.new(17.8,7.8); tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
z=t(t.parameter=="LPC_eff_mod",:); ax=nexttile(tl); hold(ax,'on');
plot(ax,z.cycle,z.theta_B_pct,'-','Color',c.B,'LineWidth',1.35,'DisplayName','B unconstrained');
plot(ax,z.cycle,z.theta_D_pct,'-','Color',c.D,'LineWidth',1.8,'DisplayName','D constrained');
yline(ax,0,'--','Color',c.truth,'DisplayName','Truth / Z_{zero}'); grid(ax,'on');
xlabel(ax,'Flight cycle'); ylabel(ax,'Estimated degradation (%)'); title(ax,'Non-fault: LPC efficiency','FontWeight','normal');
legend(ax,'Location','northwest'); thesis_plot.panel(ax,'a');

z=t(t.parameter=="LPT_eff_mod",:); ax=nexttile(tl); hold(ax,'on');
plot(ax,z.cycle,z.theta_true_pct,'--','Color',c.truth,'LineWidth',1.35,'DisplayName','Truth');
plot(ax,z.cycle,z.theta_B_pct,'-','Color',c.B,'LineWidth',1.25,'DisplayName','B unconstrained');
plot(ax,z.cycle,z.theta_D_pct,'-','Color',c.D,'LineWidth',1.75,'DisplayName','D constrained');
yline(ax,0,':','Color',c.Z_zero,'HandleVisibility','off'); grid(ax,'on');
xlabel(ax,'Flight cycle'); ylabel(ax,'Degradation (%)'); title(ax,'Fault: LPT efficiency','FontWeight','normal');
legend(ax,'Location','southwest'); thesis_plot.panel(ax,'b');
thesis_plot.export(f,figDir,'图3-6_约束抑制未退化部件虚警',false); close(f);
end

function plot_lambda_trajectories(dataDir,figDir)
t=readtable(fullfile(dataDir,'b3_lambda_trajectory.csv'),'TextType','string');
t=t(t.parameter=="LPT_eff_mod",:); lambdas=[.001 31.6 10000]; labels={'Weak shrinkage','Main setting','Strong shrinkage'};
c=thesis_plot.semantic_colors(); f=thesis_plot.new(17.8,7.7); tl=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
lim=[min([t.theta_true_pct;t.theta_hat_pct])-.12 .12];
for i=1:numel(lambdas)
    z=t(abs(log10(t.lambda)-log10(lambdas(i)))<1e-8,:); ax=nexttile(tl); hold(ax,'on');
    plot(ax,z.cycle,z.theta_true_pct,'--','Color',c.truth,'LineWidth',1.35,'DisplayName','Truth');
    plot(ax,z.cycle,z.theta_hat_pct,'-','Color',c.D,'LineWidth',1.7,'DisplayName','D estimate');
    yline(ax,0,':','Color',c.Z_zero,'HandleVisibility','off'); ylim(ax,lim); grid(ax,'on');
    rmse=sqrt(mean((z.theta_hat_pct-z.theta_true_pct).^2));
    title(ax,sprintf('%s: \\lambda = %g',labels{i},lambdas(i)),'FontWeight','normal');
    text(ax,.06,.91,sprintf('RMSE = %.3f%%',rmse),'Units','normalized','FontSize',8);
    xlabel(ax,'Flight cycle'); ylabel(ax,'LPT efficiency degradation (%)');
    if i==1, legend(ax,'Location','southwest'); end
    thesis_plot.panel(ax,char('a'+i-1));
end
thesis_plot.export(f,figDir,'图3-7_收缩强度轨迹效应',false); close(f);
end

function plot_multiunit_error(dataDir,figDir)
t=readtable(fullfile(dataDir,'b4_multiunit_component_error.csv'),'TextType','string'); t=t(logical(t.is_fault),:);
params=["HPT_eff_mod","LPT_eff_mod","LPT_flow_mod"]; c=thesis_plot.semantic_colors();
f=thesis_plot.new(17.8,7.8); tl=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact'); ymax=1.08*max(t.Z_zero_rmse_pct);
for ip=1:numel(params)
    z=sortrows(t(t.parameter==params(ip),:),{'subset','unit'}); ax=nexttile(tl); hold(ax,'on');
    jitter=linspace(-.09,.09,height(z)).';
    for i=1:height(z)
        plot(ax,[1+jitter(i) 2+jitter(i)],[z.rmse_pct(i) z.Z_zero_rmse_pct(i)],'-', ...
            'Color',[.78 .78 .78],'LineWidth',.8,'HandleVisibility','off');
    end
    scatter(ax,1+jitter,z.rmse_pct,30,c.D,'o','filled','MarkerEdgeColor','white','DisplayName','D');
    scatter(ax,2+jitter,z.Z_zero_rmse_pct,30,c.Z_zero,'s','filled','MarkerEdgeColor','white','DisplayName','Z_{zero}');
    medD=median(z.rmse_pct); medZ=median(z.Z_zero_rmse_pct);
    plot(ax,[.83 1.17],[medD medD],'-','Color',c.D,'LineWidth',2.2,'HandleVisibility','off');
    plot(ax,[1.83 2.17],[medZ medZ],'-','Color',c.Z_zero,'LineWidth',2.2,'HandleVisibility','off');
    text(ax,1,medD+.045*ymax,sprintf('med %.3f',medD),'HorizontalAlignment','center','FontSize',7);
    text(ax,2,medZ+.045*ymax,sprintf('med %.3f',medZ),'HorizontalAlignment','center','FontSize',7);
    xlim(ax,[.55 2.45]); ylim(ax,[0 ymax]); set(ax,'XTick',[1 2],'XTickLabel',{'D','Z_{zero}'});
    ylabel(ax,'Trajectory RMSE (%)'); title(ax,pretty_param(params(ip)),'FontWeight','normal'); grid(ax,'on');
    if ip==1, legend(ax,'Location','northoutside','Orientation','horizontal'); end
    thesis_plot.panel(ax,char('a'+ip-1));
end
thesis_plot.export(f,figDir,'图3-8_多台发动机逐部件误差',false); close(f);
end

function plot_lambda_mechanism(root,matlabRoot,figDir)
raw=readtable(fullfile(root,'reference','fixed_results','unsupervised','raw_scan.csv'),'TextType','string');
raw.Properties.VariableNames{strcmp(raw.Properties.VariableNames,'lam')}='lambda';
sel=readtable(fullfile(root,'reference','fixed_results','unsupervised','selected_lambda.csv'),'TextType','string');
sel.Properties.VariableNames{strcmp(sel.Properties.VariableNames,'lam')}='lambda';
g=groupsummary(raw(ismember(raw.method,["B","D"]),:),{'subset','N','lambda','method'},'mean','skill');
w=unstack(g,'mean_skill','method'); z=sortrows(w(w.subset=="DS03"&w.N==1000,:),'lambda');
tp=readtable(fullfile(matlabRoot,'outputs','t3_tau','tau_required_points.csv'),'TextType','string');
v=jsondecode(fileread(fullfile(matlabRoot,'outputs','t3_tau','verdict_tau.json')));
c=thesis_plot.semantic_colors(); base=thesis_plot.colors();
f=thesis_plot.new(17.8,7.9); tl=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
plot_skill_path(nexttile(tl),z,'B',c.B,sel,'a',[-2 1.05],true);
plot_skill_path(nexttile(tl),z,'D',c.D,sel,'b',[.25 1.02],false);
ax=nexttile(tl); hold(ax,'on'); subs=["DS02","DS03"];
for i=1:2
    q=tp.subset==subs(i); scatter(ax,tp.sqrtN(q),tp.mean_tau_required(q),38,base(i,:),'filled', ...
        'MarkerEdgeColor','white','DisplayName',subs(i));
end
x=linspace(min(tp.sqrtN),max(tp.sqrtN),100); plot(ax,x,v.intercept+v.slope*x,'-', ...
    'Color',c.D,'LineWidth',1.6,'DisplayName','Linear fit');
xlabel(ax,'sqrt(N)'); ylabel(ax,'tau required'); xlim(ax,[0 34]); ylim(ax,[0 8]); grid(ax,'on');
text(ax,.50,.13,sprintf('R^2 = %.3f',v.R2),'Units','normalized','FontSize',8);
legend(ax,'Location','northwest'); thesis_plot.panel(ax,'c');
thesis_plot.export(f,figDir,'图3-9_经典lambda准则失效',false); close(f);
end

function plot_skill_path(ax,z,method,color,sel,labelName,limits,clipLow)
hold(ax,'on'); set(ax,'XScale','log'); y=z.(method); yp=y;
if clipLow, yp=max(yp,limits(1)); end
plot(ax,z.lambda,yp,'-o','Color',color,'MarkerFaceColor','white','LineWidth',1.55,'DisplayName',method);
if clipLow && any(y<limits(1))
    scatter(ax,z.lambda(y<limits(1)),repmat(limits(1),sum(y<limits(1)),1),34,color,'v','filled', ...
        'MarkerEdgeColor','white','HandleVisibility','off');
    text(ax,.04,.08,sprintf('Values below %.0f clipped; minimum %.1f',limits(1),min(y)), ...
        'Units','normalized','FontSize',7.2);
end
criteria=["L_curve","Morozov_tau1.0","oracle"]; marks={'^','d','p'}; names={'L-curve','Morozov','Oracle'};
for i=1:numel(criteria)
    q=sel.subset=="DS03"&sel.N==1000&sel.method==method&sel.criterion==criteria(i);
    if any(q)
        lx=median(sel.lambda(q)); [~,k]=min(abs(log(z.lambda/lx))); yy=yp(k);
        scatter(ax,z.lambda(k),yy,52,color,'filled','Marker',marks{i},'MarkerEdgeColor','white','DisplayName',names{i});
    end
end
yline(ax,0,'--','Color',[.25 .25 .25],'HandleVisibility','off');
xline(ax,31.6,':','Color',[.45 .45 .45],'LineWidth',1.1,'DisplayName','Main lambda');
xlabel(ax,'Tikhonov lambda'); ylabel(ax,'Skill score'); ylim(ax,limits); grid(ax,'on');
title(ax,[method ' path, DS03 N=1000'],'FontWeight','normal'); legend(ax,'Location','southoutside','NumColumns',2);
thesis_plot.panel(ax,labelName);
end

function plot_shrinkage(matlabRoot,figDir)
x=readtable(fullfile(matlabRoot,'outputs','t1_moe','pointwise_gap.csv'),'TextType','string');
q=readtable(fullfile(matlabRoot,'outputs','stage_a','a3_q2','q2_lambda_raw.csv'),'TextType','string');
base=thesis_plot.colors(); c=thesis_plot.semantic_colors(); f=thesis_plot.new(13.5,7.9); ax=axes(f); hold(ax,'on'); set(ax,'XScale','log');
z=x(x.mode=="full",:); [lam,med,lo,hi]=median_path(z,'gap');
band=fill(ax,[lam;flipud(lam)],[lo;flipud(hi)],base(1,:),'FaceAlpha',.13,'EdgeColor','none'); band.HandleVisibility='off';
plot(ax,lam,med,'-o','Color',base(1,:),'MarkerFaceColor','white','LineWidth',1.6,'DisplayName','Full window');
z=q(q.quantile==0.75,:); [lam,med,lo,hi]=median_path(z,'gap');
band=fill(ax,[lam;flipud(lam)],[lo;flipud(hi)],base(3,:),'FaceAlpha',.13,'EdgeColor','none'); band.HandleVisibility='off';
plot(ax,lam,med,'-s','Color',base(3,:),'MarkerFaceColor','white','LineWidth',1.6,'DisplayName','High-TRA, q=0.75');
yline(ax,0,'--','Color',c.Z_zero,'HandleVisibility','off');
xline(ax,31.6228,':','Color',c.truth,'LineWidth',1.15,'DisplayName','Main lambda = 31.6');
xlabel(ax,'Tikhonov lambda'); ylabel(ax,'Skill-score gap (MoE - H0)'); grid(ax,'on');
legend(ax,'Location','southoutside','NumColumns',3); thesis_plot.panel(ax,'a');
thesis_plot.export(f,figDir,'图3-10_工况增益被收缩吸收',false); close(f);
end

function [lam,med,lo,hi]=median_path(t,field)
lam=sort(unique(t.lambda)); med=zeros(size(lam)); lo=med; hi=med;
for i=1:numel(lam)
    v=t.(field)(t.lambda==lam(i)); med(i)=median(v); lo(i)=prctile(v,25); hi(i)=prctile(v,75);
end
end

function plot_extrapolation_fault(dataDir,figDir)
t=readtable(fullfile(dataDir,'c3_c4_extrapolation_trajectory.csv'),'TextType','string');
t=t(t.parameter=="LPT_eff_mod",:); c=thesis_plot.semantic_colors();
f=thesis_plot.new(13.5,7.9); ax=axes(f); hold(ax,'on');
plot_arm(ax,t,"truth",c.truth,'--',1.45,'Truth');
plot_arm(ax,t,"D",c.D,'-',1.85,'D constrained');
plot_arm(ax,t,"E_Lin",c.E_Lin,'-',1.45,'E_{Lin}');
plot_arm(ax,t,"E_MLP",c.E_MLP,'-',1.35,'E_{MLP}');
yline(ax,0,':','Color',c.Z_zero,'HandleVisibility','off'); ylim(ax,[-3 .2]); grid(ax,'on');
xlabel(ax,'Flight cycle'); ylabel(ax,'LPT efficiency degradation (%)');
legend(ax,'Location','southoutside','NumColumns',4); thesis_plot.panel(ax,'a');
thesis_plot.export(f,figDir,'图3-11_外推真退化部件轨迹',false); close(f);
end

function plot_extrapolation_false_alarm(matlabRoot,dataDir,figDir)
t=readtable(fullfile(dataDir,'c3_c4_extrapolation_trajectory.csv'),'TextType','string');
t=t(t.parameter=="LPC_eff_mod",:); e=readtable(fullfile(matlabRoot,'outputs','stage_bc','b2_e4','e4_ood_skill_false_alarm.csv'),'TextType','string');
c=thesis_plot.semantic_colors(); f=thesis_plot.new(13.5,7.9); ax=axes(f); hold(ax,'on');
yline(ax,0,'--','Color',c.truth,'LineWidth',1.25,'DisplayName','Truth / Z_{zero}');
plot_arm(ax,t,"D",c.D,'-',1.9,'D constrained');
plot_arm(ax,t,"E_Lin",c.E_Lin,'-',1.45,'E_{Lin}');
plot_arm(ax,t,"E_MLP",c.E_MLP,'-',1.35,'E_{MLP}');
grid(ax,'on'); ylim(ax,[-.8 1.8]); xlabel(ax,'Flight cycle'); ylabel(ax,'LPC efficiency estimate (%)');
rLin=e.false_alarm_ratio_to_D(e.arm=="E_Lin"); rMlp=e.false_alarm_ratio_to_D(e.arm=="E_MLP");
text(ax,.04,.92,sprintf('Mean OOD false-alarm ratio to D:  E_{Lin} %.2fx,  E_{MLP} %.2fx',rLin,rMlp), ...
    'Units','normalized','FontSize',7.7,'VerticalAlignment','top');
legend(ax,'Location','southoutside','NumColumns',4); thesis_plot.panel(ax,'a');
thesis_plot.export(f,figDir,'图3-12_外推未退化部件虚警',false); close(f);
end

function plot_arm(ax,t,arm,color,style,width,label)
z=t(t.arm==arm,:); plot(ax,z.cycle,z.theta_pct,style,'Color',color,'LineWidth',width,'DisplayName',label);
end

function row=mrow(id,stem,section,purpose,source,eye,generated,formats)
row=table(string(id),string(stem),string(section),string(purpose),string(source),string(eye),logical(generated),string(formats), ...
    'VariableNames',{'figure_id','file_stem','section','purpose','source_data','figure_eye','generated','formats'});
end

function s=pretty_param(x)
s=strrep(char(x),'_eff_mod',' efficiency'); s=strrep(s,'_flow_mod',' flow'); s=strrep(s,'_mod',''); s=strrep(s,'_',' ');
end

function s=short_param(x)
s=strrep(char(x),'_eff_mod',' eff'); s=strrep(s,'_flow_mod',' flow'); s=strrep(s,'_mod',''); s=strrep(s,'_',' ');
end

function s=pretty_channel(x)
s=strrep(char(x),'_c',' corrected'); s=strrep(s,'_',' ');
end
