function manifest=build_chapter3_figures_nature(root)
%BUILD_CHAPTER3_FIGURES_NATURE 生成正文最终 7 组程序图（图3-2至图3-8）。
% 图3-1为作者手画流程图，仅在 manifest 中保留占位行。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir);
addpath(srcDir);
figDir=fullfile(root,'chapter3_results','figures');
if ~isfolder(figDir), mkdir(figDir); end
clean_figure_dir(figDir);

rows=cell(8,1);
rows{1}=mrow("图3-1","图3-1_方法总流程图","3.1", ...
    "作者手画的方法总流程图；程序不生成","作者手画，无程序数据", ...
    "方法从数据预处理、H辨识到约束反演和外推验证的完整流程。",false,"author");

plot_trajectory(matlabRoot,figDir);
rows{2}=mrow("图3-2","图3-2_三部件退化轨迹估计","3.2-3.4", ...
    "DS03 unit 13 的 D 约束反演轨迹与真值","figure_data/fig3_2_ds03_unit13_trajectory.csv", ...
    "约束反演从含噪残差中逐循环恢复三部件退化轨迹，估计贴合真值，未退化部件保持零线附近。",true,"png;pdf");

plot_influence(matlabRoot,figDir);
rows{3}=mrow("图3-3","图3-3_影响矩阵结构与可辨识性","3.3", ...
    "H_n热力图、奇异值谱和部件列向量夹角","cache/pipeline_cache_exact.mat", ...
    "影响矩阵病态但部件故障方向在传感器空间形成可区分指纹，是约束反演定位部件的几何基础。",true,"png;pdf");

plot_constraint_value(root,figDir);
rows{4}=mrow("图3-4","图3-4_约束独立价值","3.5", ...
    "完整网格占优率与同机未退化部件轨迹","raw_scan.csv; figure_data/fig3_2_ds03_unit13_trajectory.csv", ...
    "约束方法在全部168个网格点降低虚警，D的未退化部件估计贴合零线，而无约束B出现虚警。",true,"png;pdf");

plot_lambda_mechanism(root,matlabRoot,figDir);
rows{5}=mrow("图3-5","图3-5_经典lambda准则失效机制","3.5", ...
    "B/D真实技能分路径与tau_required-sqrt(N)拟合","raw_scan.csv; selected_lambda.csv; tau_required_points.csv", ...
    "约束托平D的弱收缩路径，L-curve和Morozov在D上集体退化，tau_required随sqrt(N)线性增长。",true,"png;pdf");

plot_proxy(root,figDir);
rows{6}=mrow("图3-6","图3-6_无约束路径代理标定","3.5", ...
    "B@L-curve、D@代理lambda和D oracle技能分","tables/表3_代理路径标定.csv", ...
    "无约束L-curve可代理D的收缩强度，DS03达标而DS02以0.009之差未达预注册阈值。",true,"png;pdf");

plot_shrinkage(matlabRoot,figDir);
rows{7}=mrow("图3-7","图3-7_工况增益被收缩吸收","3.6", ...
    "MoE-H0技能分差随lambda的full/highTRA路径","t1_moe/pointwise_gap.csv; stage_a/a3_q2/q2_lambda_raw.csv", ...
    "工况自适应增益仅在弱收缩下显现，主工作点lambda=31.6时约束-收缩吸收了工况幅值失配。",true,"png;pdf");

plot_extrapolation(matlabRoot,figDir);
rows{8}=mrow("图3-8","图3-8_外推技能与虚警分裂","3.7", ...
    "OOD技能-虚警对照与同子集逐配对D减臂技能差","b2_e4/e4_ood_skill_false_alarm.csv; b1_e3prime/e3prime_paired_raw.csv", ...
    "线性臂保有外推能力但虚警约为D的3倍，深层臂崩溃；D兼得高技能与低虚警。",true,"png;pdf");

manifest=vertcat(rows{:});
end

function clean_figure_dir(figDir)
for ext={'.png','.pdf','.svg'}
    old=dir(fullfile(figDir,['图3-*' ext{1}]));
    for i=1:numel(old), delete(fullfile(old(i).folder,old(i).name)); end
end
end

function plot_trajectory(matlabRoot,figDir)
dataDir=fullfile(matlabRoot,'outputs','stage_bc','figure_data');
if ~isfolder(dataDir), mkdir(dataDir); end
csvPath=fullfile(dataDir,'fig3_2_ds03_unit13_trajectory.csv');
if ~isfile(csvPath), generate_trajectory_csv(matlabRoot,csvPath,13); end
t=readtable(csvPath,'TextType','string'); c=thesis_plot.semantic_colors();
faults=["HPT_eff_mod","LPT_eff_mod","LPT_flow_mod"];
others=setdiff(unique(t.parameter,'stable'),faults,'stable');
f=thesis_plot.new(17.8,11.0); tl=tiledlayout(f,2,3,'TileSpacing','compact','Padding','compact');
for i=1:3
    ax=nexttile(tl); q=t.parameter==faults(i); hold(ax,'on');
    plot(ax,t.cycle(q),t.theta_true_pct(q),'--','Color',c.truth,'LineWidth',1.35,'DisplayName','Truth');
    plot(ax,t.cycle(q),t.theta_D_pct(q),'-','Color',c.D,'LineWidth',1.7,'DisplayName','D estimate');
    yline(ax,0,':','Color',c.secondary,'HandleVisibility','off');
    xlabel(ax,'Flight cycle'); ylabel(ax,'Degradation (%)');
    title(ax,pretty_param(faults(i)),'FontWeight','normal'); grid(ax,'on');
    if i==1, legend(ax,'Location','southwest'); end
    thesis_plot.panel(ax,char('a'+i-1));
end
ax=nexttile(tl,4,[1 3]); hold(ax,'on');
base=thesis_plot.colors();
for i=1:numel(others)
    q=t.parameter==others(i);
    plot(ax,t.cycle(q),t.theta_D_pct(q),'-','Color',base(1+mod(i-1,6),:),'LineWidth',1.05, ...
        'DisplayName',short_param(others(i)));
end
yline(ax,0,'--','Color',c.Z_zero,'DisplayName','Z_{zero}=0');
xlabel(ax,'Flight cycle'); ylabel(ax,'Estimated degradation (%)');
title(ax,'Non-fault parameters','FontWeight','normal'); grid(ax,'on');
legend(ax,'Location','southoutside','NumColumns',4); thesis_plot.panel(ax,'d');
thesis_plot.export(f,figDir,'图3-2_三部件退化轨迹估计',false); close(f);
end

function generate_trajectory_csv(matlabRoot,outPath,unit)
cfg=ncmapss_lib.config(false); stream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
cacheFile=fullfile(matlabRoot,'cache','pipeline_cache_exact.mat');
pipe=ncmapss_lib.build_pipeline(cfg,stream,cacheFile,false);
path=fullfile(cfg.DATA_DIR,'N-CMAPSS_DS03-012.h5');
d=ncmapss_lib.load_per_cycle(path,'test',cfg,stream,unit);
[d,~]=ncmapss_lib.add_corrected(d,pipe.ref); d=ncmapss_lib.attach_residuals(pipe,d);
[wins,~]=ncmapss_lib.build_windows(d,unit,'full',1000,cfg,stream);
R=wins.R; T=size(R,1); truth=wins.TH; Cum=ncmapss_lib.build_cum(T,9);
M=ncmapss_lib.build_M(pipe.Hn,T);
hatD=ncmapss_lib.solve_D(M,Cum,R,cfg.LAM_MAIN,T,9);
hatB=ncmapss_lib.solve_B(pipe.Hn,R,cfg.LAM_MAIN);
cycles=sort(unique(d.cycle));
rows=cell(T*9,1); ri=0;
for j=1:9
    scale=100*cfg.THETA_SPAN(j);
    for k=1:T
        ri=ri+1;
        rows{ri}=table("DS03",unit,k,cycles(k),string(cfg.THETA9{j}), ...
            ismember(j,[1 6 7]),truth(k,j)*scale,hatD(k,j)*scale,hatB(k,j)*scale, ...
            'VariableNames',{'subset','unit','cycle_index','cycle','parameter','is_fault', ...
            'theta_true_pct','theta_D_pct','theta_B_pct'});
    end
end
writetable(vertcat(rows{:}),outPath,'Encoding','UTF-8');
end

function plot_influence(matlabRoot,figDir)
s=load(fullfile(matlabRoot,'cache','pipeline_cache_exact.mat'),'pipe'); H=s.pipe.Hn;
cfg=ncmapss_lib.config(false); c=thesis_plot.semantic_colors();
shortSensors=cellfun(@short_param,s.pipe.corrected_cols,'UniformOutput',false);
shortTheta=cellfun(@short_param,cfg.THETA9,'UniformOutput',false);
f=thesis_plot.new(17.8,10.6); tl=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(tl,1,[2 1]); imagesc(ax,H); axis(ax,'tight'); clim(ax,[-max(abs(H),[],'all') max(abs(H),[],'all')]);
colormap(ax,thesis_plot.diverging(257)); colorbar(ax); set(ax,'XTick',1:9,'XTickLabel',shortTheta,'XTickLabelRotation',45, ...
    'YTick',1:13,'YTickLabel',shortSensors); xlabel(ax,'Health parameter'); ylabel(ax,'Corrected channel');
title(ax,'Normalized influence matrix H_n','FontWeight','normal'); thesis_plot.panel(ax,'a');

ax=nexttile(tl,2); sv=svd(H); semilogy(ax,1:numel(sv),sv,'-o','Color',c.D,'MarkerFaceColor','white','LineWidth',1.5); grid(ax,'on');
xlabel(ax,'Singular-value index'); ylabel(ax,'Singular value (log_{10} scale)');
text(ax,.05,.08,sprintf('cond(H_n)=%.0f',sv(1)/sv(end)),'Units','normalized','FontSize',8);
title(ax,'Identifiability spectrum','FontWeight','normal'); thesis_plot.panel(ax,'b');

norms=sqrt(sum(H.^2,1)); C=(H.'*H)./(norms.'*norms); C=max(-1,min(1,C)); A=acosd(C); A(1:10:end)=0;
ax=nexttile(tl,4); imagesc(ax,A,[0 max(90,max(A,[],'all'))]); axis(ax,'square'); colormap(ax,thesis_plot.viridis(256)); colorbar(ax);
set(ax,'XTick',1:9,'XTickLabel',shortTheta,'XTickLabelRotation',45,'YTick',1:9,'YTickLabel',shortTheta);
xlabel(ax,'Health parameter'); ylabel(ax,'Health parameter'); title(ax,'Column angle (deg)','FontWeight','normal'); hold(ax,'on');
rectangle(ax,'Position',[1.5 5.5 1 1],'EdgeColor',c.B,'LineWidth',1.5);
rectangle(ax,'Position',[5.5 1.5 1 1],'EdgeColor',c.B,'LineWidth',1.5);
text(ax,6.1,2,sprintf('%.1f°',A(2,6)),'Color',c.B,'FontSize',7,'FontWeight','bold'); thesis_plot.panel(ax,'c');
thesis_plot.export(f,figDir,'图3-3_影响矩阵结构与可辨识性',false); close(f);
end

function plot_constraint_value(root,figDir)
raw=readtable(fullfile(root,'reference','fixed_results','unsupervised','raw_scan.csv'),'TextType','string');
raw.Properties.VariableNames{strcmp(raw.Properties.VariableNames,'lam')}='lambda';
keys={'subset','N','lambda'};
gs=groupsummary(raw(ismember(raw.method,["B","D"]),:),[keys {'method'}],'mean','skill'); ws=unstack(gs,'mean_skill','method');
gf=groupsummary(raw(ismember(raw.method,["B","D"]),:),[keys {'method'}],'mean','false_alarm'); wf=unstack(gf,'mean_false_alarm','method');
gd=groupsummary(raw(ismember(raw.method,["B","D"]),:),[keys {'method'}],'mean','detect_corr'); wd=unstack(gd,'mean_detect_corr','method');
rates=[mean(ws.D>ws.B),mean(wf.D<wf.B),mean(wd.D>wd.B)];
traj=readtable(fullfile(root,'matlab','outputs','stage_bc','figure_data','fig3_2_ds03_unit13_trajectory.csv'),'TextType','string');
params=unique(traj.parameter,'stable'); nonfault=params(~ismember(params,["HPT_eff_mod","LPT_eff_mod","LPT_flow_mod"])); score=zeros(numel(nonfault),1);
for i=1:numel(nonfault), q=traj.parameter==nonfault(i); score(i)=mean(abs(traj.theta_B_pct(q))); end
[~,ii]=max(score); showParam=nonfault(ii);
f=thesis_plot.new(17.8,7.8); tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact'); c=thesis_plot.semantic_colors();
ax=nexttile(tl); bar(ax,100*rates,.62,'FaceColor',c.D,'EdgeColor','none'); hold(ax,'on');
yline(ax,0,'--','Color',c.Z_zero,'DisplayName','Z_{zero}=0'); ylim(ax,[0 108]);
set(ax,'XTick',1:3,'XTickLabel',{'Skill','Specificity','Detection'}); ylabel(ax,'D dominance rate (%)'); grid(ax,'on');
for i=1:3, text(ax,i,100*rates(i)+3,sprintf('%.1f%%',100*rates(i)),'HorizontalAlignment','center','FontSize',8); end
text(ax,.04,.05,'Z_{zero}=0 reference','Units','normalized','FontSize',7,'Color',c.truth); thesis_plot.panel(ax,'a');
ax=nexttile(tl); q=traj.parameter==showParam; hold(ax,'on');
plot(ax,traj.cycle(q),traj.theta_B_pct(q),'-','Color',c.B,'LineWidth',1.3,'DisplayName','B (unconstrained)');
plot(ax,traj.cycle(q),traj.theta_D_pct(q),'-','Color',c.D,'LineWidth',1.6,'DisplayName','D (constrained)');
yline(ax,0,'--','Color',c.Z_zero,'DisplayName','Z_{zero}=0'); xlabel(ax,'Flight cycle'); ylabel(ax,'Estimated degradation (%)');
title(ax,['Non-fault: ' short_param(showParam)],'FontWeight','normal'); grid(ax,'on'); legend(ax,'Location','northwest'); thesis_plot.panel(ax,'b');
thesis_plot.export(f,figDir,'图3-4_约束独立价值',false); close(f);
end

function plot_lambda_mechanism(root,matlabRoot,figDir)
raw=readtable(fullfile(root,'reference','fixed_results','unsupervised','raw_scan.csv'),'TextType','string'); raw.Properties.VariableNames{strcmp(raw.Properties.VariableNames,'lam')}='lambda';
sel=readtable(fullfile(root,'reference','fixed_results','unsupervised','selected_lambda.csv'),'TextType','string'); sel.Properties.VariableNames{strcmp(sel.Properties.VariableNames,'lam')}='lambda';
g=groupsummary(raw(ismember(raw.method,["B","D"]),:),{'subset','N','lambda','method'},'mean','skill'); w=unstack(g,'mean_skill','method'); z=sortrows(w(w.subset=="DS03"&w.N==1000,:),'lambda');
tp=readtable(fullfile(matlabRoot,'outputs','t3_tau','tau_required_points.csv'),'TextType','string'); v=jsondecode(fileread(fullfile(matlabRoot,'outputs','t3_tau','verdict_tau.json'))); c=thesis_plot.semantic_colors(); base=thesis_plot.colors();
f=thesis_plot.new(17.8,7.8); tl=tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
plot_path(tl,z,'B',c.B,sel,c,'a',[min(z.B)-.05*max(1,range(z.B)) 1.02]);
plot_path(tl,z,'D',c.D,sel,c,'b',[min(z.D)-.05*max(1,range(z.D)) 1.02]);
ax=nexttile(tl); hold(ax,'on'); subs=["DS02","DS03"];
for i=1:2, q=tp.subset==subs(i); scatter(ax,tp.sqrtN(q),tp.mean_tau_required(q),38,base(i,:), ...
        'filled','MarkerEdgeColor','white','DisplayName',subs(i)); end
x=linspace(min(tp.sqrtN),max(tp.sqrtN),100); y=v.intercept+v.slope*x; plot(ax,x,y,'-','Color',c.D,'LineWidth',1.5,'DisplayName','linear fit');
xlabel(ax,'sqrt(N)'); ylabel(ax,'tau required'); xlim(ax,[0 34]); ylim(ax,[0 8]);
text(ax,.55,.12,sprintf('R^2=%.3f',v.R2),'Units','normalized','FontSize',8); grid(ax,'on'); legend(ax,'Location','northwest'); thesis_plot.panel(ax,'c');
thesis_plot.export(f,figDir,'图3-5_经典lambda准则失效机制',false); close(f);
end

function plot_path(tl,z,method,clr,sel,c,labelName,limits)
ax=nexttile(tl); hold(ax,'on'); set(ax,'XScale','log'); y=z.(method);
plot(ax,z.lambda,y,'-o','Color',clr,'MarkerFaceColor','white','LineWidth',1.5,'DisplayName',method);
criteria=["L_curve","Morozov_tau1.0","oracle"]; marks={'^','d','p'}; names={'L-curve','Morozov','oracle'};
for i=1:numel(criteria)
    q=sel.subset=="DS03"&sel.N==1000&sel.method==method&sel.criterion==criteria(i);
    if any(q)
        lx=median(sel.lambda(q)); [~,k]=min(abs(log(z.lambda/lx)));
        scatter(ax,z.lambda(k),y(k),48,c.(method),'filled','Marker',marks{i}, ...
            'MarkerEdgeColor','white','DisplayName',names{i});
    end
end
yline(ax,0,'--','Color',c.Z_zero,'HandleVisibility','off'); xline(ax,31.6,':','Color',c.truth,'HandleVisibility','off');
xlabel(ax,'Tikhonov lambda'); ylabel(ax,'Skill score'); ylim(ax,limits); grid(ax,'on'); title(ax,[method ' path'],'FontWeight','normal');
legend(ax,'Location','southoutside','NumColumns',2); thesis_plot.panel(ax,labelName);
end

function plot_proxy(root,figDir)
t=readtable(fullfile(root,'chapter3_results','tables','表3_代理路径标定.csv'),'TextType','string'); c=thesis_plot.semantic_colors();
f=thesis_plot.new(12.8,7.8); ax=axes(f); vals=[t.mean_B_Lcurve t.mean_D_at_B_Lcurve t.D_oracle]; b=bar(ax,vals,'grouped','EdgeColor','none');
cols=[c.B;c.D;c.E_MixLinear]; for i=1:3, b(i).FaceColor=cols(i,:); end
set(ax,'XTickLabel',t.subset); ylabel(ax,'Skill score'); ylim(ax,[0.82 1.01]); grid(ax,'on');
legend(ax,{'B @ L-curve','D @ proxy lambda','D oracle'},'Location','southoutside','Orientation','horizontal');
for i=1:height(t), text(ax,i,.995,sprintf('loss %.4f',t.loss(i)),'HorizontalAlignment','center','FontSize',8); end
thesis_plot.panel(ax,'a'); thesis_plot.export(f,figDir,'图3-6_无约束路径代理标定',false); close(f);
end

function plot_shrinkage(matlabRoot,figDir)
x=readtable(fullfile(matlabRoot,'outputs','t1_moe','pointwise_gap.csv'),'TextType','string');
q=readtable(fullfile(matlabRoot,'outputs','stage_a','a3_q2','q2_lambda_raw.csv'),'TextType','string');
c=thesis_plot.semantic_colors(); f=thesis_plot.new(12.8,7.8); ax=axes(f); hold(ax,'on'); set(ax,'XScale','log');
z=x(x.mode=="full",:); [lam,med,lo,hi]=median_path(z,'gap'); band=fill(ax,[lam;flipud(lam)],[lo;flipud(hi)],c.D,'FaceAlpha',.14,'EdgeColor','none'); band.HandleVisibility='off';
plot(ax,lam,med,'-o','Color',c.D,'MarkerFaceColor','white','LineWidth',1.5,'DisplayName','full window');
z=q(q.quantile==0.75,:); [lam,med,lo,hi]=median_path(z,'gap'); band=fill(ax,[lam;flipud(lam)],[lo;flipud(hi)],c.E_Lin,'FaceAlpha',.14,'EdgeColor','none'); band.HandleVisibility='off';
plot(ax,lam,med,'-s','Color',c.E_Lin,'MarkerFaceColor','white','LineWidth',1.5,'DisplayName','highTRA q=0.75');
yline(ax,0,'--','Color',c.Z_zero,'HandleVisibility','off'); xline(ax,31.6228,':','Color',c.truth,'LineWidth',1.1,'DisplayName','lambda=31.6');
xlabel(ax,'Tikhonov lambda'); ylabel(ax,'Skill score gap (MoE - H0)'); grid(ax,'on'); legend(ax,'Location','southoutside','NumColumns',3);
text(ax,.04,.94,'weak shrinkage: large gap; main point: near zero','Units','normalized','FontSize',8); thesis_plot.panel(ax,'a');
thesis_plot.export(f,figDir,'图3-7_工况增益被收缩吸收',false); close(f);
end

function [lam,med,lo,hi]=median_path(t,field)
lam=unique(t.lambda); med=zeros(size(lam)); lo=med; hi=med;
for i=1:numel(lam), v=t.(field)(t.lambda==lam(i)); med(i)=median(v); lo(i)=prctile(v,25); hi(i)=prctile(v,75); end
end

function plot_extrapolation(matlabRoot,figDir)
e=readtable(fullfile(matlabRoot,'outputs','stage_bc','b2_e4','e4_ood_skill_false_alarm.csv'),'TextType','string');
p=readtable(fullfile(matlabRoot,'outputs','stage_bc','b1_e3prime','e3prime_paired_raw.csv'),'TextType','string');
c=thesis_plot.semantic_colors(); f=thesis_plot.new(17.8,10.2); tl=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(tl,1,[2 1]); hold(ax,'on'); main=["D","E_Lin","E_MixLinear","Z_zero"];
offset=[.018 0;.018 -.00028;-.018 .00025;.018 .00015]; align={'left','left','right','left'};
for i=1:numel(main)
    q=e.arm==main(i); col=c.(main(i)); scatter(ax,e.ood_skill(q),e.ood_false_alarm(q),62,col,'filled','MarkerEdgeColor','white');
    text(ax,e.ood_skill(q)+offset(i,1),e.ood_false_alarm(q)+offset(i,2),strrep(char(main(i)),'_','\_'), ...
        'FontSize',8,'VerticalAlignment','middle','HorizontalAlignment',align{i});
end
xlim(ax,[0 1.02]); ylim(ax,[0 .0072]); xlabel(ax,'OOD skill score'); ylabel(ax,'OOD false alarm'); grid(ax,'on');
text(ax,.04,.94,'Deep arms: E_{MLP}=0.18, E_{WPMixer}=-7.53','Units','normalized','FontSize',7.5); thesis_plot.panel(ax,'a');
linearVals=p.gap_D_minus_arm(ismember(p.arm,["E_Lin","E_MixLinear"])); linearLim=[min(-.2,min(linearVals)-.03) max(.35,max(linearVals)+.03)];
deepVals=p.gap_D_minus_arm(ismember(p.arm,["E_MLP","E_WPMixer"])); deepLim=[0 max(6,ceil(max(deepVals)*1.05))];
swarm_panel(nexttile(tl,2),p,["E_Lin","E_MixLinear"],c,'Linear arms','b',linearLim);
swarm_panel(nexttile(tl,4),p,["E_MLP","E_WPMixer"],c,'Deep arms','c',deepLim);
thesis_plot.export(f,figDir,'图3-8_外推技能与虚警分裂',false); close(f);
end

function swarm_panel(ax,p,arms,c,titleText,labelName,limits)
hold(ax,'on');
for i=1:numel(arms)
    vals=sort(p.gap_D_minus_arm(p.arm==arms(i))); x=i+linspace(-.18,.18,numel(vals)).';
    scatter(ax,x,vals,16,c.(arms(i)),'filled','MarkerFaceAlpha',.65);
    med=median(vals); scatter(ax,i,med,54,c.(arms(i)),'d','filled','MarkerEdgeColor','white');
    text(ax,i,limits(2)-.06*diff(limits),sprintf('med %.3f',med),'HorizontalAlignment','center','FontSize',7);
end
yline(ax,0,'--','Color',c.Z_zero); xlim(ax,[.5 numel(arms)+.5]); ylim(ax,limits);
set(ax,'XTick',1:numel(arms),'XTickLabel',strrep(cellstr(arms),'_','\_')); ylabel(ax,'D - arm skill'); grid(ax,'on'); title(ax,titleText,'FontWeight','normal'); thesis_plot.panel(ax,labelName);
end

function row=mrow(id,stem,section,purpose,source,eye,generated,formats)
row=table(string(id),string(stem),string(section),string(purpose),string(source),string(eye),logical(generated),string(formats), ...
    'VariableNames',{'figure_id','file_stem','section','purpose','source_data','figure_eye','generated','formats'});
end

function s=pretty_param(x)
s=strrep(char(x),'_mod',''); s=strrep(s,'_',' ');
end

function s=short_param(x)
s=strrep(char(x),'_eff_mod',' eff'); s=strrep(s,'_flow_mod',' flow'); s=strrep(s,'_c',''); s=strrep(s,'_',' ');
end
