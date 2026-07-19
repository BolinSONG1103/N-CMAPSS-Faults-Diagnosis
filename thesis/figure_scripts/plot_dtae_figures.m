function plot_dtae_figures()
%PLOT_DTAE_FIGURES 从 improve/figdata 的 CSV 绘制 DTAE 诊断的王昆式正式图。
% 用法：在本文件所在目录运行 plot_dtae_figures，或先 cd 到本目录再调用。
% 依赖：MATLAB（含中文字体，如 SimHei / Microsoft YaHei）。
% 产出：thesis/figures/ 下 图3-6/3-7/3-8/3-9 的 .png(600dpi) 与 .pdf(矢量)。
%
% 数据由 Python 落盘（cd improve && python3 dtae_export.py），本脚本只负责美观出图。

here = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, '..', '..', 'improve', 'figdata');
outDir  = fullfile(here, '..', 'figures');
if ~isfolder(outDir), mkdir(outDir); end

% ---- 全局审美（对齐参考论文：干净留白、序贯蓝、无上右边框）----
set(groot, 'defaultAxesFontName', 'Microsoft YaHei');   % 若无则改 'SimHei'
set(groot, 'defaultTextFontName', 'Microsoft YaHei');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultAxesBox', 'off');
set(groot, 'defaultAxesTickDir', 'out');
blueMap = flipud(gray(0));                               % 占位，下方用自定义
blues = [linspace(0.97,0.03,256)', linspace(0.98,0.19,256)', linspace(1.0,0.42,256)'];

famColors = [0.84 0.15 0.16;   % HPT/涡轮相关 红
             0.17 0.63 0.17;   % Fan 绿
             0.12 0.47 0.71;   % HPC 蓝
             1.00 0.50 0.05;   % LPT 橙
             0.58 0.40 0.74];  % LPC 紫

%% ===== 图3-6 故障检测 =====
M = readmatrix(fullfile(dataDir,'detection_confusion.csv'), 'Range', 2, ...
               'NumHeaderLines', 1);
M = M(:, end-1:end);                                     % 去掉行名列
Mn = M ./ sum(M,2);
f = figure('Color','w','Position',[100 100 520 440]);
imagesc(Mn); colormap(blues); caxis([0 1]); axis square;
set(gca,'XTick',1:2,'XTickLabel',{'正常','故障'},'YTick',1:2,'YTickLabel',{'正常','故障'});
xlabel('预测'); ylabel('真值'); title('故障检测混淆矩阵（逐循环）');
for i=1:2, for j=1:2
    c = 'k'; if Mn(i,j)>0.55, c='w'; end
    text(j,i,sprintf('%.0f%%\n(%d)',Mn(i,j)*100,M(i,j)),'HorizontalAlignment','center','Color',c);
end, end
colorbar; box off;
exportgraphics(f, fullfile(outDir,'图3-6_故障检测.png'), 'Resolution',600);
exportgraphics(f, fullfile(outDir,'图3-6_故障检测.pdf'), 'ContentType','vector'); close(f);

%% ===== 图3-7 四部件隔离（逐类召回混淆 + 逐部件 P/R/F1）=====
T = readtable(fullfile(dataDir,'confusion_part.csv'), 'ReadRowNames', true, ...
              'Encoding','UTF-8');
names = T.Properties.VariableNames; C = table2array(T);
Mt = readtable(fullfile(dataDir,'dtae_metrics_part.csv'), 'Encoding','UTF-8');
f = figure('Color','w','Position',[100 100 1100 440]);
% (a) 逐类召回混淆
subplot(1,2,1);
imagesc(C); colormap(blues); caxis([0 1]); axis square;
set(gca,'XTick',1:numel(names),'XTickLabel',names,'YTick',1:numel(names),'YTickLabel',names);
xtickangle(30); xlabel('预测部件'); ylabel('真值部件'); title('(a) 逐类召回混淆矩阵');
for i=1:size(C,1), for j=1:size(C,2)
    if C(i,j)>=0.005
        c='k'; if C(i,j)>0.55, c='w'; end
        text(j,i,sprintf('%.2f',C(i,j)),'HorizontalAlignment','center','Color',c,'FontSize',10);
    end
end, end
colorbar; box off;
% (b) 逐部件 P/R/F1
subplot(1,2,2);
vals = [Mt.precision, Mt.recall, Mt.F1];
b = bar(vals, 'grouped'); ylim([0 1.05]); box off;
set(gca,'XTickLabel', Mt.name); xtickangle(20);
ylabel('指标值'); title('(b) 逐部件精确率/召回率/F1');
legend({'精确率','召回率','F1'}, 'Location','southoutside','Orientation','horizontal');
sgtitle(sprintf('四部件故障隔离（macro-F1 = %.3f）', mean(Mt.F1)));
exportgraphics(f, fullfile(outDir,'图3-7_四部件故障隔离.png'), 'Resolution',600);
exportgraphics(f, fullfile(outDir,'图3-7_四部件故障隔离.pdf'), 'ContentType','vector'); close(f);

%% ===== 图3-8 DTAE 潜空间 t-SNE =====
L = readtable(fullfile(dataDir,'latent_tsne.csv'), 'Encoding','UTF-8');
cats = unique(L.label, 'stable');
palette = [0.6 0.6 0.6; famColors];
f = figure('Color','w','Position',[100 100 620 520]); hold on;
for k=1:numel(cats)
    m = strcmp(L.label, cats{k});
    scatter(L.tsne1(m), L.tsne2(m), 14, palette(min(k,size(palette,1)),:), 'filled', ...
            'MarkerFaceAlpha',0.6);
end
xlabel('潜维度 1'); ylabel('潜维度 2'); title('DTAE 潜空间 t-SNE 可视化');
legend(cats, 'Location','bestoutside'); box off;
exportgraphics(f, fullfile(outDir,'图3-8_DTAE潜空间.png'), 'Resolution',600);
exportgraphics(f, fullfile(outDir,'图3-8_DTAE潜空间.pdf'), 'ContentType','vector'); close(f);

%% ===== 图3-9 诊断时间线 =====
TL = readtable(fullfile(dataDir,'timeline.csv'), 'Encoding','UTF-8');
f = figure('Color','w','Position',[100 100 900 520]);
subplot(2,1,1);
plot(TL.cycle, TL.severity_true, '-', 'Color',[0.23 0.23 0.23],'LineWidth',2); hold on;
plot(TL.cycle, TL.severity_est, '-', 'Color',[0.84 0.15 0.16],'LineWidth',1.8);
onset = TL.cycle(find(TL.detected==1,1));
if ~isempty(onset), xline(onset,'--r','检测'); end
ylabel('退化程度'); legend({'真值','估计'},'Location','northwest'); box off;
title('(a) 退化程度与检测');
subplot(2,1,2);
det = double(TL.detected);
area(TL.cycle, det, 'FaceColor',[0.12 0.47 0.71],'FaceAlpha',0.4,'EdgeColor','none');
ylim([0 1.2]); set(gca,'YTick',[0 1],'YTickLabel',{'正常','故障'});
xlabel('飞行循环'); title('(b) 部件级诊断输出'); box off;
sgtitle('代表发动机诊断时间线');
exportgraphics(f, fullfile(outDir,'图3-9_诊断时间线.png'), 'Resolution',600);
exportgraphics(f, fullfile(outDir,'图3-9_诊断时间线.pdf'), 'ContentType','vector'); close(f);

fprintf('完成：图3-6/3-7/3-8/3-9 已输出到 %s\n', outDir);
end
