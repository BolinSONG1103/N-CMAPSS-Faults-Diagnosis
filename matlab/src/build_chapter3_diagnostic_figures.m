function manifest=build_chapter3_diagnostic_figures()
%BUILD_CHAPTER3_DIAGNOSTIC_FIGURES 诊断闭环的四组必要正文结果图。
% 框架图与算法结构图不在此生成，由作者根据协议手绘。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); root=fileparts(matlabRoot);
addpath(srcDir); ncmapss_lib.set_plot_defaults();
dataDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure');
figDir=fullfile(root,'chapter3_results','diagnostic_figures');
if ~isfolder(figDir), mkdir(figDir); end

cycle=readtable(fullfile(dataDir,'diagnostic_cycle_raw.csv'),'TextType','string');
unit=readtable(fullfile(dataDir,'diagnostic_unit_metrics.csv'),'TextType','string');
component=readtable(fullfile(dataDir,'component_metrics.csv'),'TextType','string');
stage=readtable(fullfile(dataDir,'stage_confusion.csv'),'TextType','string');
detect=readtable(fullfile(dataDir,'detection_confusion.csv'),'TextType','string');
sensitivity=readtable(fullfile(dataDir,'stage_component_sensitivity.csv'),'TextType','string');
unknown=readtable(fullfile(dataDir,'unknown_family_summary.csv'),'TextType','string');
robustDir=fullfile(matlabRoot,'outputs','t5_robustness_ablation');
ablation=readtable(fullfile(robustDir,'ablation_unit_metrics.csv'),'TextType','string');
robustness=readtable(fullfile(robustDir,'robustness_summary.csv'),'TextType','string');

plot_joint_trajectory(cycle,figDir);
plot_detection_isolation(detect,component,figDir);
plot_stage_evidence(stage,sensitivity,figDir);
plot_generalization_unknown(unit,unknown,figDir);
plot_ablation_robustness(ablation,robustness,figDir);

figure_id=["图3-D1";"图3-D2";"图3-D3";"图3-D4";"图3-D5"];
file_stem=["图3-D1_健康参数诊断阶段联合轨迹";"图3-D2_故障检测与多标签隔离"; ...
    "图3-D3_有序退化等级判定";"图3-D4_跨组合泛化与未知故障拒识"; ...
    "图3-D5_约束消融与传感器扰动鲁棒性"];
question=["连续估计如何转化为检测、隔离和阶段输出"; ...
    "闭环是否能够检测故障并隔离多个退化部件"; ...
    "数据定义的有序退化等级是否可重复识别"; ...
    "方法在未见组合和留一故障族上是否仍可靠"; ...
    "性能来自哪些约束且对测量扰动是否稳定"];
source_data=["diagnostic_cycle_raw.csv";"detection_confusion.csv + component_metrics.csv"; ...
    "stage_confusion.csv + stage_component_sensitivity.csv"; ...
    "diagnostic_unit_metrics.csv + unknown_family_summary.csv"; ...
    "ablation_unit_metrics.csv + robustness_summary.csv"];
selection_rule=["DS03 中 unit 编号最小者，运行前固定";"全部测试单元"; ...
    "全部测试单元";"全部测试单元与全部留一故障族折"; ...
    "全部测试单元、全部预注册消融和扰动等级"];
manifest=table(figure_id,file_stem,question,source_data,selection_rule);
writetable(manifest,fullfile(figDir,'diagnostic_figure_manifest.csv'),'Encoding','UTF-8');
end

function plot_joint_trajectory(raw,figDir)
q=raw.subset=="DS03"; assert(any(q),'DS03 结果缺失。');
unit=min(raw.unit(q)); q=q&raw.unit==unit;
names=["HPT_eff_mod","LPT_eff_mod","LPT_flow_mod"];
c=thesis_plot.semantic_colors(); f=thesis_plot.new(17.8,14.2);
t=tiledlayout(f,4,1,'TileSpacing','compact','Padding','compact');
for j=1:3
    z=raw(q&raw.parameter==names(j),:); z=sortrows(z,'cycle'); ax=nexttile(t);
    plot(ax,z.cycle,100*z.theta_true,'-','Color',c.truth,'LineWidth',1.6,'DisplayName','Truth'); hold(ax,'on');
    plot(ax,z.cycle,100*z.theta_hat,'-','Color',c.D,'LineWidth',1.6,'DisplayName','Constrained estimate');
    yline(ax,0,':','Color',[.6 .6 .6]); grid(ax,'on'); ylabel(ax,'Change (%)');
    title(ax,strrep(char(names(j)),'_','\_'),'FontWeight','normal');
    if j==1, legend(ax,'Location','southwest'); end
    thesis_plot.panel(ax,char('a'+j-1));
end
z=raw(q&raw.parameter==names(1),:); z=sortrows(z,'cycle'); ax=nexttile(t);
stairs(ax,z.cycle,z.truth_stage,'-','Color',c.truth,'LineWidth',1.6,'DisplayName','True level'); hold(ax,'on');
stairs(ax,z.cycle,z.pred_stage,'-','Color',c.D,'LineWidth',1.6,'DisplayName','Predicted level');
set(ax,'YTick',0:3,'YTickLabel',{'Healthy','Early','Middle','Severe'}); ylim(ax,[-.15 3.2]);
xlabel(ax,'Flight cycle'); ylabel(ax,'Ordered level'); grid(ax,'on'); legend(ax,'Location','northwest');
title(ax,sprintf('Fixed example: DS03 unit %d',round(unit)),'FontWeight','normal'); thesis_plot.panel(ax,'d');
thesis_plot.export(f,figDir,'图3-D1_健康参数诊断阶段联合轨迹',false); close(f);
end

function plot_detection_isolation(conf,component,figDir)
c=thesis_plot.semantic_colors(); f=thesis_plot.new(17.8,7.7);
t=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
regimes=["test_id","ood_combo"]; titles={'ID fault detection','Unseen-combination detection'};
for k=1:2
    ax=nexttile(t); z=conf(conf.regime==regimes(k),:); C=reshape(z.count,2,2);
    row=sum(C,2); P=C./max(row,1); imagesc(ax,P,[0 1]); colormap(ax,thesis_plot.viridis(256));
    set(ax,'XTick',1:2,'XTickLabel',{'Healthy','Fault'}, ...
        'YTick',1:2,'YTickLabel',{'Healthy','Fault'});
    for i=1:2, for j=1:2, text(ax,j,i,sprintf('%.1f%%\n(n=%d)',100*P(i,j),C(i,j)), ...
            'HorizontalAlignment','center','Color',contrast(P(i,j))); end, end
    xlabel(ax,'Prediction'); ylabel(ax,'Reference'); title(ax,titles{k},'FontWeight','normal');
    thesis_plot.panel(ax,char('a'+k-1));
end

ax=nexttile(t); names=unique(component.parameter,'stable'); regimes=["test_id","ood_combo"];
F=nan(numel(names),2); R=F;
for i=1:numel(names), for j=1:2
    q=component.parameter==names(i)&component.regime==regimes(j);
    if any(q), F(i,j)=component.F1(q); R(i,j)=component.recall(q); end
end, end
b=barh(ax,1:numel(names),F,'grouped'); b(1).FaceColor=c.D; b(2).FaceColor=[0 158 115]/255;
set(ax,'YTick',1:numel(names),'YTickLabel',strrep(cellstr(names),'_','\_'));
xlim(ax,[0 1]); xlabel(ax,'F1 score'); grid(ax,'on'); legend(ax,{'ID test','Unseen combination'},'Location','southoutside');
title(ax,'Multi-label isolation','FontWeight','normal'); thesis_plot.panel(ax,'c');
thesis_plot.export(f,figDir,'图3-D2_故障检测与多标签隔离',false); close(f);
end

function plot_stage_evidence(conf,sensitivity,figDir)
f=thesis_plot.new(17.8,7.6); t=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
labs={'Healthy','Early','Middle','Severe'}; regimes=["test_id","ood_combo"];
titles={'ID ordered levels','Unseen-combination levels'};
for k=1:2
    ax=nexttile(t); z=conf(conf.regime==regimes(k),:); C=reshape(z.count,4,4);
    P=C./max(sum(C,2),1); imagesc(ax,P,[0 1]); colormap(ax,thesis_plot.viridis(256));
    set(ax,'XTick',1:4,'XTickLabel',labs,'YTick',1:4,'YTickLabel',labs,'XTickLabelRotation',35);
    for i=1:4, for j=1:4, text(ax,j,i,sprintf('%.0f%%',100*P(i,j)), ...
            'HorizontalAlignment','center','Color',contrast(P(i,j))); end, end
    xlabel(ax,'Prediction'); ylabel(ax,'Reference'); title(ax,titles{k},'FontWeight','normal');
    thesis_plot.panel(ax,char('a'+k-1));
end

ax=nexttile(t); hold(ax,'on'); regimes=["test_id","ood_combo"]; colors=[0 114 178;0 158 115]/255;
for i=1:2
    q=sensitivity.regime==regimes(i); z=sortrows(sensitivity(q,:),'stage_components');
    plot(ax,z.stage_components,z.weighted_kappa,'-o','Color',colors(i,:), ...
        'MarkerFaceColor',colors(i,:),'DisplayName',strrep(char(regimes(i)),'_',' '));
end
yline(ax,0,':','Color',[.5 .5 .5]); xticks(ax,2:4); xlim(ax,[1.8 4.2]); ylim(ax,[-.05 1]);
xlabel(ax,'Number of post-onset states'); ylabel(ax,'Weighted kappa'); grid(ax,'on');
legend(ax,'Location','southoutside'); title(ax,'Definition sensitivity','FontWeight','normal'); thesis_plot.panel(ax,'c');
thesis_plot.export(f,figDir,'图3-D3_有序退化等级判定',false); close(f);
end

function plot_generalization_unknown(unit,unknown,figDir)
c=thesis_plot.semantic_colors(); f=thesis_plot.new(17.8,7.7);
t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); metrics=["detection_F1","isolation_macroF1","stage_weighted_kappa"];
regimes=["test_id","ood_combo"]; vals=nan(2,3);
for i=1:2, for j=1:3, vals(i,j)=mean(unit.(metrics(j))(unit.regime==regimes(i)),'omitnan'); end, end
b=bar(ax,vals.'); b(1).FaceColor=c.D; b(2).FaceColor=[0 158 115]/255;
set(ax,'XTick',1:3,'XTickLabel',{'Detection F1','Isolation macro-F1','Stage kappa'});
ylim(ax,[0 1]); ylabel(ax,'Unit-balanced mean'); grid(ax,'on');
legend(ax,{'ID test','Unseen combination'},'Location','southoutside');
title(ax,'Closed-set generalization','FontWeight','normal'); thesis_plot.panel(ax,'a');

ax=nexttile(t); y=1:height(unknown); hold(ax,'on');
scatter(ax,unknown.AUROC,y,42,c.D,'filled','DisplayName','AUROC');
scatter(ax,unknown.unknown_reject_rate,y,42,[213 94 0]/255,'s','filled','DisplayName','Unknown rejection');
scatter(ax,unknown.known_accept_rate,y,42,[0 158 115]/255,'^','filled','DisplayName','Known acceptance');
set(ax,'YTick',y,'YTickLabel',cellstr(unknown.family)); xlim(ax,[0 1]); grid(ax,'on');
xlabel(ax,'Rate'); ylabel(ax,'Held-out fault family'); legend(ax,'Location','southoutside');
title(ax,'Leave-one-family-out rejection','FontWeight','normal'); thesis_plot.panel(ax,'b');
thesis_plot.export(f,figDir,'图3-D4_跨组合泛化与未知故障拒识',false); close(f);
end

function plot_ablation_robustness(ablation,robustness,figDir)
c=thesis_plot.semantic_colors(); f=thesis_plot.new(17.8,7.8);
t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); methods=["D_full","sign_only","D_no_shrink","B_unconstrained"];
M=nan(numel(methods),3);
for i=1:numel(methods)
    q=ablation.method==methods(i); M(i,:)=[mean(ablation.inactive_component_error(q)), ...
        mean(ablation.detection_F1(q)),mean(ablation.isolation_macroF1(q))];
end
yyaxis(ax,'left'); b=bar(ax,1:numel(methods),M(:,1)); b.FaceColor=c.D;
ylabel(ax,'Inactive-component error');
yyaxis(ax,'right'); plot(ax,1:numel(methods),M(:,2),'-o','Color',[213 94 0]/255, ...
    'MarkerFaceColor',[213 94 0]/255,'DisplayName','Detection F1'); hold(ax,'on');
plot(ax,1:numel(methods),M(:,3),'-s','Color',[0 158 115]/255, ...
    'MarkerFaceColor',[0 158 115]/255,'DisplayName','Isolation macro-F1');
ylim(ax,[0 1]); ylabel(ax,'Diagnostic score');
set(ax,'XTick',1:numel(methods),'XTickLabel',strrep(cellstr(methods),'_','\_'),'XTickLabelRotation',20);
grid(ax,'on'); legend(ax,'Location','southoutside'); title(ax,'Constraint ablation','FontWeight','normal'); thesis_plot.panel(ax,'a');

ax=nexttile(t); q=robustness.perturbation=="noise"&robustness.regime=="ood_combo";
z=sortrows(robustness(q,:),'amplitude');
plot(ax,z.amplitude,z.mean_detection_F1,'-o','Color',[213 94 0]/255, ...
    'MarkerFaceColor',[213 94 0]/255,'DisplayName','Detection F1'); hold(ax,'on');
plot(ax,z.amplitude,z.mean_isolation_macroF1,'-s','Color',[0 158 115]/255, ...
    'MarkerFaceColor',[0 158 115]/255,'DisplayName','Isolation macro-F1');
plot(ax,z.amplitude,z.mean_stage_kappa,'-^','Color',c.D, ...
    'MarkerFaceColor',c.D,'DisplayName','Stage kappa');
xlabel(ax,'Added noise standard deviation (healthy residual sigma)'); ylabel(ax,'Score');
ylim(ax,[0 1]); grid(ax,'on'); legend(ax,'Location','southoutside');
title(ax,'Unseen-combination noise robustness','FontWeight','normal'); thesis_plot.panel(ax,'b');
thesis_plot.export(f,figDir,'图3-D5_约束消融与传感器扰动鲁棒性',false); close(f);
end

function c=contrast(v)
if v>.52, c=[.08 .08 .08]; else, c='white'; end
end
