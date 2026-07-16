function verdict=experiment_stage_a_t0_raw()
%EXPERIMENT_STAGE_A_T0_RAW A2：补齐 raw_scan 与 MATLAB T0 的原始对照数字。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); projectRoot=fileparts(matlabRoot);
addpath(srcDir); cfg=ncmapss_lib.config(false);
outDir=fullfile(matlabRoot,'outputs','stage_a','a2_t0'); figDir=fullfile(outDir,'figures');
if ~isfolder(outDir), mkdir(outDir); end
if ~isfolder(figDir), mkdir(figDir); end

ref=readtable(fullfile(projectRoot,'reference','fixed_results','unsupervised','raw_scan.csv'), ...
    'TextType','string');
q=ref.subset=="DS03"&ref.N==1000&ref.method=="D"&isfinite(ref.lam);
available=unique(ref.lam(q)); [~,iLam]=min(abs(log(available/cfg.LAM_MAIN)));
rawLambda=available(iLam);
refRows=ref(q&ref.lam==rawLambda,:);
X=mean(refRows.skill);

mat=readtable(fullfile(matlabRoot,'outputs','t0','t0_ds03_raw.csv'),'TextType','string');
Y=mean(mat.skill); absDiff=abs(X-Y); pass=absDiff<=cfg.T0_SKILL_ABS_TOL;

comparison=table( ...
    ["reference_raw_scan";"matlab_recompute";"Z_zero"], ...
    ["DS03";"DS03";"DS03"],[1000;1000;1000], ...
    [rawLambda;cfg.LAM_MAIN;cfg.LAM_MAIN],["D";"D";"Z_zero"], ...
    [height(refRows);height(mat);height(mat)],[X;Y;0],[0;absDiff;NaN], ...
    'VariableNames',{'source','subset','N','lambda','method','n_units','skill','abs_diff_from_raw'});
writetable(comparison,fullfile(outDir,'t0_raw_comparison.csv'),'Encoding','UTF-8');

refUnit=refRows(:,{'unit','skill'}); refUnit.Properties.VariableNames{2}='raw_scan_skill';
matUnit=mat(:,{'unit','skill'}); matUnit.Properties.VariableNames{2}='matlab_skill';
pairs=innerjoin(refUnit,matUnit,'Keys','unit'); pairs.difference=pairs.matlab_skill-pairs.raw_scan_skill;
writetable(pairs,fullfile(outDir,'t0_unit_pairs.csv'),'Encoding','UTF-8');
plot_unit_agreement(figDir,pairs,X,Y,cfg);

verdict=struct('stage','A2','subset','DS03','N',1000, ...
    'raw_scan_lambda',rawLambda,'matlab_lambda',cfg.LAM_MAIN, ...
    'raw_scan_aggregate_X',X,'matlab_recompute_Y',Y,'absolute_difference',absDiff, ...
    'tolerance',cfg.T0_SKILL_ABS_TOL,'pass',pass, ...
    'formal_T0_judgement_unchanged',pass);
write_json(fullfile(outDir,'verdict_a2.json'),verdict);
fprintf('A2: raw X=%.12f; MATLAB Y=%.12f; |X-Y|=%.12f; pass=%d\n',X,Y,absDiff,pass);
end

function plot_unit_agreement(figDir,p,X,Y,cfg)
c=thesis_plot.colors(); f=thesis_plot.new(17.8,7.8);
t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); scatter(ax,p.raw_scan_skill,p.matlab_skill,38,c(1,:),'filled', ...
    'MarkerEdgeColor','white'); hold(ax,'on');
lims=[min([p.raw_scan_skill;p.matlab_skill]) max([p.raw_scan_skill;p.matlab_skill])];
pad=.04*range(lims); lims=lims+[-pad pad]; plot(ax,lims,lims,'--','Color',[.3 .3 .3]);
xlim(ax,lims); ylim(ax,lims); axis(ax,'square'); grid(ax,'on');
xlabel(ax,'raw\_scan skill'); ylabel(ax,'MATLAB skill'); thesis_plot.panel(ax,'a');

ax=nexttile(t); stem(ax,p.unit,p.difference,'filled','Color',c(2,:), ...
    'MarkerFaceColor',c(2,:),'LineWidth',1.2); hold(ax,'on');
yline(ax,0,'Color',[.3 .3 .3]); yline(ax,cfg.T0_SKILL_ABS_TOL,'--','Color',[.55 .55 .55]);
yline(ax,-cfg.T0_SKILL_ABS_TOL,'--','Color',[.55 .55 .55]); grid(ax,'on');
xlabel(ax,'DS03 unit'); ylabel(ax,'MATLAB - raw\_scan skill');
text(ax,.03,.96,sprintf('mean: %.9f vs %.9f',X,Y),'Units','normalized', ...
    'VerticalAlignment','top','FontSize',7.5); thesis_plot.panel(ax,'b');
thesis_plot.export(f,figDir,'fig_a2_t0_agreement'); close(f);
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
