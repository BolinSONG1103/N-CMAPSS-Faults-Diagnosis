function build_continuous_figures(repoRoot)
%BUILD_CONTINUOUS_FIGURES 从锁定CSV生成连续估计侧正式图（图3-2至3-9、图3-17）。
% 不依赖原始N-CMAPSS数据或pipeline_cache，不重新拟合、不重新选参。
% 覆盖图3-2至3-9与图3-17（补充鲁棒性）。

if nargin < 1 || strlength(string(repoRoot)) == 0
    here = fileparts(mfilename('fullpath'));
    repoRoot = fileparts(fileparts(fileparts(here)));
end
outDir = fullfile(repoRoot,'thesis','figures','matlab_official');
if ~isfolder(outDir), mkdir(outDir); end

dataDir = fullfile(repoRoot,'matlab','outputs','complete_figures','data');
a1 = readtable(fullfile(dataDir,'a1_similarity_correction.csv'),'TextType','string');
a2 = readtable(fullfile(dataDir,'a2_baseline_fit.csv'),'TextType','string');
b3 = readtable(fullfile(dataDir,'b3_lambda_trajectory.csv'),'TextType','string');
b4 = readtable(fullfile(dataDir,'b4_multiunit_component_error.csv'),'TextType','string');
traj = readtable(fullfile(repoRoot,'matlab','outputs','stage_bc','figure_data', ...
    'fig3_2_ds03_unit13_trajectory.csv'),'TextType','string');
abl = readtable(fullfile(repoRoot,'matlab','outputs','t5_robustness_ablation', ...
    'ablation_unit_metrics.csv'),'TextType','string');
rob = readtable(fullfile(repoRoot,'matlab','outputs','t5_robustness_ablation', ...
    'robustness_summary.csv'),'TextType','string');
t8 = readtable(fullfile(repoRoot,'thesis','tables','表8_外推技能与虚警.csv'),'TextType','string');

draw_similarity(a1,outDir);
draw_baseline(a2,outDir);
Hn = load_Hn(repoRoot);
draw_fingerprint(Hn,outDir);
draw_identifiability(Hn,outDir);
draw_tracking(traj,outDir);
draw_multiunit(b4,outDir);
draw_lambda(b3,outDir);
draw_constraint(abl,t8,outDir);
draw_robustness(rob,outDir);
fprintf('[done] MATLAB正式图已输出到 %s\n',outDir);
end

function draw_similarity(t,outDir)
channels=["Nf","Nc","Wf","T24","T30","T48","T50","P15","P21","P24","Ps30","P40","P50"];
raw=zeros(numel(channels),1); cor=raw;
for i=1:numel(channels)
    q=t.raw_channel==channels(i)&t.stage=="raw"; raw(i)=t.binned_drift_span_pct(find(q,1));
    q=t.raw_channel==channels(i)&t.stage=="corrected"; cor(i)=t.binned_drift_span_pct(find(q,1));
end
f=newfig(18,8); tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(tl); bar(ax,[raw cor],'grouped'); grid(ax,'on');
set(ax,'XTick',1:numel(channels),'XTickLabel',channels,'XTickLabelRotation',35);
ylabel(ax,'分箱漂移跨度 (%)'); title(ax,'(a) 13个气路通道的工况漂移'); legend(ax,{'修正前','相似修正后'});
ax=nexttile(tl); reduction=raw-cor; c=repmat([.18 .55 .34],numel(channels),1); c(reduction<0,:)=repmat([.85 .37 .08],sum(reduction<0),1);
b=barh(ax,reduction); b.FaceColor='flat'; b.CData=c; xline(ax,0,'Color',[.5 .5 .5]); grid(ax,'on');
set(ax,'YTick',1:numel(channels),'YTickLabel',channels,'YDir','reverse'); xlabel(ax,'漂移跨度减少量 (百分点)');
title(ax,'(b) 修正收益及剩余通道偏差');
savepair(f,outDir,'图3-2_相似修正与工况漂移');
end

function draw_baseline(t,outDir)
% 全通道 (1-R^2) 排序 + 最难拟合通道的诚实散点
[uch,ia]=unique(t.channel,'stable'); err=(1-t.R2(ia))*1e6; [err,ix]=sort(err); uch=uch(ix);
worst=uch(end);
f=newfig(18,8); tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
% (a) 最难拟合通道散点
z=t(t.channel==worst,:); idx=unique(round(linspace(1,height(z),min(1200,height(z)))));
ax=nexttile(tl); scatter(ax,z.measured(idx),z.predicted(idx),8,[.12 .47 .71],'filled','MarkerFaceAlpha',.25); hold(ax,'on');
lo=min([z.measured;z.predicted]); hi=max([z.measured;z.predicted]); plot(ax,[lo hi],[lo hi],'--','Color',[.2 .2 .2],'DisplayName','理想 y=x');
respct=100*std(z.measured-z.predicted)/mean(z.measured);
axis(ax,'square'); grid(ax,'on'); xlabel(ax,'实测值'); ylabel(ax,'健康基准预测值');
title(ax,'(a) 最难通道的基准拟合');
text(ax,.05,.9,sprintf('最差拟合通道 %s\nR^2=%.4f\n残差 std=%.3f%% 读数',worst,z.R2(1),respct),'Units','normalized','VerticalAlignment','top');
% (b) 全通道拟合误差排序（涡轮出口温压通道橙色高亮）
ax=nexttile(tl); hot=ismember(uch,["T50_c","P50_c","T48_c"]);
c=repmat([.42 .68 .84],numel(uch),1); c(hot,:)=repmat([.85 .37 .08],sum(hot),1);
b=barh(ax,err); b.FaceColor='flat'; b.CData=c; set(ax,'YTick',1:numel(uch),'YTickLabel',uch,'YDir','reverse');
xlabel(ax,'(1-R^2)×10^6（越小越好）'); title(ax,'(b) 全通道拟合误差排序'); grid(ax,'on');
savepair(f,outDir,'图3-3_健康基准拟合与残差质量');
end

function Hn=load_Hn(repoRoot)
t=readtable(fullfile(repoRoot,'reference','fixed_results','influence_matrix','H_9cols.csv'),'TextType','string');
H=table2array(t(:,2:end)); span=[0.018668 0.223446 0.121209 0.025540 0.070658 0.036194 0.032908 0.118767 0.046187];
Hn=H.*span;
end

function draw_fingerprint(Hn,outDir)
params={'HPT效率','Fan效率','Fan流量','HPC效率','HPC流量','LPT效率','LPT流量','LPC效率','LPC流量'};
sensors={'Nf','Nc','Wf','T24','T30','T48','T50','P15','P21','P24','Ps30','P40','P50'};
Hdir=Hn./vecnorm(Hn); f=newfig(15,10); ax=axes(f); imagesc(ax,Hdir); colormap(ax,redblue(257)); colorbar(ax);
set(ax,'XTick',1:9,'XTickLabel',params,'XTickLabelRotation',35,'YTick',1:13,'YTickLabel',sensors);
xlabel(ax,'健康参数'); ylabel(ax,'标准化气路残差通道'); title(ax,'量程归一化影响矩阵的部件故障指纹');
savepair(f,outDir,'图3-4_影响矩阵故障指纹');
end

function draw_identifiability(Hn,outDir)
% (b) 四检查单元层可分、涡轮内单参数层不可分——统一支撑四部件合并
unitIdx={[2 3],[4 5],[8 9],[1 6 7]}; unitName={'风扇','高压压气机','低压压气机','涡轮'};
tpIdx={1,6,7}; tpName={'HPT效率','LPT效率','LPT流量'};
unitAng=zeros(1,4); for i=1:4,other=setdiff(1:9,unitIdx{i}); s=svd(orth(Hn(:,unitIdx{i}))'*orth(Hn(:,other))); unitAng(i)=acosd(min(1,max(s)));end
tpAng=zeros(1,3); for i=1:3,other=setdiff(1:9,tpIdx{i}); s=svd(orth(Hn(:,tpIdx{i}))'*orth(Hn(:,other))); tpAng(i)=acosd(min(1,max(s)));end
sv=svd(Hn); f=newfig(18,8); tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(tl); semilogy(ax,1:9,sv,'-o','Color',[.84 .15 .16],'MarkerFaceColor','white'); grid(ax,'on');
xlabel(ax,'奇异值序号'); ylabel(ax,'奇异值'); title(ax,'(a) 病态性：奇异值谱');
text(ax,.95,.92,sprintf('cond(H_n)=%.1f',sv(1)/sv(end)),'Units','normalized','HorizontalAlignment','right');
ax=nexttile(tl); hold(ax,'on'); x1=1:4; x2=6:8;
b1=bar(ax,x1,unitAng,.6,'FaceColor',[.17 .63 .17],'DisplayName','四检查单元（可分）');
b2=bar(ax,x2,tpAng,.6,'FaceColor',[.84 .15 .16],'DisplayName','涡轮内单参数（不可分）');
yline(ax,10,'--','Color',[.47 .47 .47]); grid(ax,'on');
for i=1:4,text(ax,x1(i),unitAng(i)+.6,sprintf('%.1f°',unitAng(i)),'HorizontalAlignment','center');end
for i=1:3,text(ax,x2(i),tpAng(i)+.6,sprintf('%.2f°',tpAng(i)),'HorizontalAlignment','center');end
set(ax,'XTick',[x1 x2],'XTickLabel',[unitName tpName],'XTickLabelRotation',25);
ylabel(ax,'到其余部件子空间的最小主夹角 (°)'); title(ax,'(b) 检查单元可分、涡轮内不可分'); legend(ax,[b1 b2]);
savepair(f,outDir,'图3-5_病态性与子空间可辨识性');
end

function draw_tracking(t,outDir)
active=["HPT_eff_mod","LPT_eff_mod","LPT_flow_mod"]; names=["HPT效率","LPT效率","LPT流量"];
f=newfig(18,11); tl=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
for i=1:3
    z=t(t.parameter==active(i),:); ax=nexttile(tl); hold(ax,'on');
    plot(ax,z.cycle_index,z.theta_true_pct,'-','Color',[.2 .2 .2],'LineWidth',1.8);
    plot(ax,z.cycle_index,z.theta_B_pct,'--','Color',[.12 .47 .71]);
    plot(ax,z.cycle_index,z.theta_D_pct,'-','Color',[.84 .15 .16],'LineWidth',1.5);
    grid(ax,'on'); xlabel(ax,'飞行循环'); ylabel(ax,'健康参数变化 (%)'); title(ax,sprintf('(%c) %s',char('a'+i-1),names(i)));
    if i==1,legend(ax,{'真值','无约束B','约束D'});end
end
inactive=setdiff(unique(t.parameter,'stable'),active,'stable'); cycles=unique(t.cycle_index); envB=zeros(size(cycles)); envD=envB;
for k=1:numel(cycles)
    q=t.cycle_index==cycles(k)&ismember(t.parameter,inactive);
    envB(k)=max(abs(t.theta_B_pct(q))); envD(k)=max(abs(t.theta_D_pct(q)));
end
ax=nexttile(tl); hold(ax,'on'); plot(ax,cycles,envB,'--','Color',[.12 .47 .71]); plot(ax,cycles,envD,'-','Color',[.84 .15 .16]);
grid(ax,'on'); xlabel(ax,'飞行循环'); ylabel(ax,'max|theta| (%)'); title(ax,'(d) 未退化部件最大估计幅值'); legend(ax,{'无约束B','约束D'});
savepair(f,outDir,'图3-6_代表发动机连续退化轨迹');
end

function draw_multiunit(t,outDir)
a=t(logical(t.is_fault),:); units=a.subset+"-U"+string(a.unit); f=newfig(18,8); tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(tl); hold(ax,'on'); x=(1:height(a))';
for i=1:height(a),plot(ax,[x(i) x(i)],[a.rmse_pct(i) a.Z_zero_rmse_pct(i)],'-','Color',[.8 .8 .8]);end
scatter(ax,x,a.Z_zero_rmse_pct,36,[.65 .65 .65],'s','filled'); scatter(ax,x,a.rmse_pct,40,[.84 .15 .16],'o','filled');
set(ax,'XTick',x,'XTickLabel',units,'XTickLabelRotation',50); ylabel(ax,'活动部件轨迹RMSE (%)'); title(ax,'(a) 跨发动机活动部件误差'); grid(ax,'on');
ax=nexttile(tl); vals=[t.rmse_pct(t.is_fault==1);t.rmse_pct(t.is_fault==0);t.terminal_abs_error_pct(t.is_fault==1)];
grp=[repmat("活动部件轨迹RMSE",sum(t.is_fault==1),1);repmat("未活动部件轨迹RMSE",sum(t.is_fault==0),1);repmat("活动部件终点误差",sum(t.is_fault==1),1)];
boxchart(ax,categorical(grp),vals); ylabel(ax,'误差 (%)'); title(ax,'(b) 多口径误差分布'); grid(ax,'on');
savepair(f,outDir,'图3-7_多发动机多部件估计误差');
end

function draw_lambda(t,outDir)
active=["HPT_eff_mod","LPT_eff_mod","LPT_flow_mod"]; lam=unique(t.lambda); rmse=zeros(size(lam));
for i=1:numel(lam),z=t(t.lambda==lam(i)&ismember(t.parameter,active),:);rmse(i)=sqrt(mean((z.theta_hat_pct-z.theta_true_pct).^2));end
f=newfig(18,8); tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact'); ax=nexttile(tl);
semilogx(ax,lam,rmse,'-o','Color',[.84 .15 .16],'MarkerFaceColor','white'); grid(ax,'on'); set(ax,'XTick',lam); xlabel(ax,'Tikhonov参数lambda'); ylabel(ax,'活动部件RMSE (%)'); title(ax,'(a) 聚合精度敏感性');
ax=nexttile(tl); hold(ax,'on'); cols=[.12 .47 .71;.84 .15 .16;.95 .5 .08];
for i=1:numel(lam),z=t(t.lambda==lam(i)&t.parameter=="LPT_eff_mod",:);plot(ax,z.cycle,z.theta_hat_pct,'Color',cols(i,:),'DisplayName',sprintf('lambda=%g',lam(i)));end
z=t(t.lambda==lam(1)&t.parameter=="LPT_eff_mod",:);plot(ax,z.cycle,z.theta_true_pct,'--','Color',[.2 .2 .2],'LineWidth',1.5,'DisplayName','真值');
grid(ax,'on'); xlabel(ax,'飞行循环'); ylabel(ax,'LPT效率变化 (%)'); title(ax,'(b) 欠收缩—适中—过收缩'); legend(ax);
savepair(f,outDir,'图3-8_正则化参数敏感性');
end

function draw_constraint(abl,t8,outDir)
q=ismember(abl.method,["D_full","B_unconstrained"]); z=abl(q,:); methods=["D_full","B_unconstrained"];
vals=zeros(2,2);for i=1:2,w=z.method==methods(i);vals(i,:)=[mean(z.RMSE(w)) mean(z.inactive_component_error(w))];end
f=newfig(18,8); tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact'); ax=nexttile(tl);
bar(ax,vals');set(ax,'XTickLabel',{'估计RMSE','未退化部件虚警'});ylabel(ax,'量程归一化误差');title(ax,'(a) 硬约束的独立价值');legend(ax,{'约束D','无约束B'});grid(ax,'on');
ax=nexttile(tl); t8=t8(t8.arm~="Z_zero",:); barh(ax,t8.ood_skill); set(ax,'YTick',1:height(t8),'YTickLabel',t8.arm); xline(ax,0); xlabel(ax,'未见组合技能分'); title(ax,'(b) 与数据驱动对照'); grid(ax,'on');
for i=1:height(t8),text(ax,max(t8.ood_skill(i),-.58)+.02,i,sprintf('虚警×%.0f',t8.false_alarm_ratio_to_D(i)));end
savepair(f,outDir,'图3-9_约束反演的独立价值');
end

function draw_robustness(t,outDir)
t=t(t.regime=="test_id",:); n=sortrows(t(t.perturbation=="noise",:),'amplitude'); b=sortrows(t(t.perturbation=="bias",:),'amplitude');
miss=t(t.perturbation=="missing_channel",:);
f=newfig(18,8); tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact'); ax=nexttile(tl); hold(ax,'on');
plot(ax,n.amplitude,n.mean_RMSE/n.mean_RMSE(1),'-o','DisplayName','RMSE相对倍数');plot(ax,n.amplitude,n.mean_detection_F1,'-s','DisplayName','检测F1');plot(ax,n.amplitude,n.mean_isolation_macroF1,'-^','DisplayName','隔离macro-F1');
grid(ax,'on');xlabel(ax,'附加噪声幅值（健康残差标准差倍数）');ylabel(ax,'相对误差/指标值');title(ax,'(a) 加性噪声鲁棒性');legend(ax);
ax=nexttile(tl);hold(ax,'on');plot(ax,b.amplitude,b.mean_detection_F1,'-o','DisplayName','检测F1');plot(ax,b.amplitude,b.mean_isolation_macroF1,'-s','DisplayName','隔离macro-F1');
if height(miss)>0
    xm=max(b.amplitude)+.18;
    scatter(ax,xm,miss.mean_detection_F1(1),65,[.46 .42 .70],'d','filled','DisplayName','单通道缺失：检测F1');
    scatter(ax,xm,miss.mean_isolation_macroF1(1),70,[.19 .64 .33],'p','filled','DisplayName','单通道缺失：隔离macro-F1');
end
grid(ax,'on');xlabel(ax,'固定偏置幅值 / 缺失通道情形');ylabel(ax,'指标值');ylim(ax,[0 1.02]);title(ax,'(b) 偏置与通道缺失');legend(ax,'Location','best');
savepair(f,outDir,'图3-17_物理约束管线鲁棒性');
end

function f=newfig(w,h)
f=figure('Color','white','Units','centimeters','Position',[2 2 w h]);set(groot,'defaultAxesFontName','Microsoft YaHei','defaultTextFontName','Microsoft YaHei');
end

function savepair(f,outDir,stem)
exportgraphics(f,fullfile(outDir,[stem '.pdf']),'ContentType','vector');exportgraphics(f,fullfile(outDir,[stem '.png']),'Resolution',600);close(f);
end

function cmap=redblue(n)
if nargin<1,n=257;end;x=linspace(0,1,n)';cmap=[interp1([0 .5 1],[.13 .97 .70],x),interp1([0 .5 1],[.40 .97 .09],x),interp1([0 .5 1],[.67 .97 .10],x)];
end
