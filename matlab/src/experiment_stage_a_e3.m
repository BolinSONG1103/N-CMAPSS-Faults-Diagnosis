function verdict=experiment_stage_a_e3()
%EXPERIMENT_STAGE_A_E3 A3：退化严重度证据与同子集配对替代指标。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
cfg=ncmapss_lib.config(false);
outDir=fullfile(matlabRoot,'outputs','stage_a','a3_e3');
figDir=fullfile(outDir,'figures');
if ~isfolder(outDir), mkdir(outDir); end
if ~isfolder(figDir), mkdir(figDir); end

stream=ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
severityRows=cell(0,1); componentRows=cell(0,1); si=0; ci=0;

% 分布内：严格复用 T2 的 DS05/DS07 最后两台留出单元定义。
idIndex=[3 4];
for ii=idIndex
    file=cfg.IDENT_FILES{ii,1}; faults=cfg.IDENT_FILES{ii,2};
    tag=strrep(extractBefore(file,'.h5'),'N-CMAPSS_',''); tag=[char(tag) '_id'];
    d=ncmapss_lib.load_per_cycle(fullfile(cfg.DATA_DIR,file),'dev',cfg,stream,[]);
    units=sort(unique(d.unit)); held=units(max(1,numel(units)-1):end);
    [severityRows,componentRows,si,ci]=append_units(severityRows,componentRows, ...
        si,ci,d,held,'in_dist',tag,faults,cfg);
end

% 组合外推：DS02/DS03 全部 test 单元。
for iv=1:size(cfg.VALID_FILES,1)
    tag=cfg.VALID_FILES{iv,1}; file=cfg.VALID_FILES{iv,2}; faults=cfg.VALID_FILES{iv,3};
    d=ncmapss_lib.load_per_cycle(fullfile(cfg.DATA_DIR,file),'test',cfg,stream,[]);
    units=sort(unique(d.unit));
    [severityRows,componentRows,si,ci]=append_units(severityRows,componentRows, ...
        si,ci,d,units,'ood_combo',tag,faults,cfg);
end

severity=struct2table(vertcat(severityRows{:}));
component=struct2table(vertcat(componentRows{:}));
writetable(severity,fullfile(outDir,'severity_per_unit.csv'),'Encoding','UTF-8');
writetable(component,fullfile(outDir,'severity_per_component.csv'),'Encoding','UTF-8');

sevSummary=summarize_severity(severity);
writetable(sevSummary,fullfile(outDir,'severity_summary.csv'),'Encoding','UTF-8');
truth=severity(strcmp(severity.arm,'truth'),:);
inVals=truth.terminal_median(strcmp(truth.regime,'in_dist'));
oodVals=truth.terminal_median(strcmp(truth.regime,'ood_combo'));
severityCi=bootstrap_two_sample_median(oodVals,inVals,5000,4421);
severityDelta=median(oodVals)-median(inVals);
severitySupported=severityDelta>0 && severityCi(1)>0;
metricComparison=compare_severity_metrics(truth);
writetable(metricComparison,fullfile(outDir,'severity_metric_comparison.csv'),'Encoding','UTF-8');

dd=readtable(fullfile(matlabRoot,'outputs','t2_dd','dd_raw.csv'),'TextType','string');
paired=build_pairs(dd);
writetable(paired,fullfile(outDir,'same_subset_paired_raw.csv'),'Encoding','UTF-8');
pairSummary=summarize_pairs(paired);
writetable(pairSummary,fullfile(outDir,'same_subset_paired_summary.csv'),'Encoding','UTF-8');

plot_severity_distribution(figDir,severity);
plot_severity_metrics(figDir,metricComparison);
plot_component_heatmap(figDir,component,cfg);
plot_skill_vs_severity(figDir,dd,severity);
plot_pair_small_multiples(figDir,paired);
plot_pair_summary(figDir,pairSummary);

verdict=struct('analysis_type','stage_A_posthoc_diagnostic', ...
    'formal_E3_remains_false',true, ...
    'in_dist_terminal_median',median(inVals), ...
    'ood_terminal_median',median(oodVals), ...
    'ood_minus_in_dist_median',severityDelta, ...
    'ood_minus_in_dist_ci95',severityCi, ...
    'severity_amplitude_explanation_supported',severitySupported, ...
    'all_reported_severity_metrics_ood_greater', ...
        all(metricComparison.ood_greater(strcmp(metricComparison.arm,'truth'))), ...
    'severity_metric_comparison',table2struct(metricComparison), ...
    'specific_claim_ood_degradation_is_larger_rejected',~severitySupported, ...
    'cross_regime_gap_remains_noncausal_across_different_fault_compositions',true, ...
    'replacement_metric_status','diagnostic_only_until_stage_B_confirmation', ...
    'same_subset_pair_summary',table2struct(pairSummary));
write_json(fullfile(outDir,'verdict_a3_e3.json'),verdict);
fprintf('\nA3-E3: terminal severity OOD-in = %.5f, CI [%.5f, %.5f], supported=%d\n', ...
    severityDelta,severityCi(1),severityCi(2),severitySupported);
end

function [severityRows,componentRows,si,ci]=append_units(severityRows,componentRows, ...
        si,ci,d,units,regime,subset,faults,cfg)
idxTrue=find(ismember(cfg.THETA9,faults));
for u=units.'
    du=d(d.unit==u,:); cycles=sort(unique(du.cycle));
    ref=du(ismember(du.cycle,cycles(1:min(cfg.N_REF_CYCLES,numel(cycles)))),:);
    th0=mean(table2array(ref(:,cfg.THETA9))./cfg.THETA_SPAN,1);
    theta=zeros(numel(cycles),9);
    for t=1:numel(cycles)
        q=find(du.cycle==cycles(t),1);
        theta(t,:)=table2array(du(q,cfg.THETA9))./cfg.THETA_SPAN-th0;
    end
    nTerminal=max(1,ceil(.1*numel(cycles)));
    terminal=abs(theta(end-nTerminal+1:end,idxTrue)); values=terminal(:);
    vals=struct('terminal_median',median(values),'terminal_q25',prctile(values,25), ...
        'terminal_q75',prctile(values,75),'terminal_q90',prctile(values,90), ...
        'terminal_max',max(values),'trajectory_rms',sqrt(mean(theta(:,idxTrue).^2,'all')));
    si=si+1; severityRows{si,1}=severity_row(regime,subset,u,'truth',numel(cycles),vals);
    z=struct('terminal_median',0,'terminal_q25',0,'terminal_q75',0, ...
        'terminal_q90',0,'terminal_max',0,'trajectory_rms',0);
    si=si+1; severityRows{si,1}=severity_row(regime,subset,u,'Z_zero',numel(cycles),z);
    for j=1:9
        if ismember(j,idxTrue)
            v=median(abs(theta(end-nTerminal+1:end,j)));
        else
            v=NaN;
        end
        ci=ci+1; componentRows{ci,1}=component_row(regime,subset,u,'truth',j,cfg.THETA9{j},v,ismember(j,idxTrue));
        ci=ci+1; componentRows{ci,1}=component_row(regime,subset,u,'Z_zero',j,cfg.THETA9{j},0,ismember(j,idxTrue));
    end
    fprintf('  %-9s %-7s unit %d terminal median %.5f\n',regime,subset,round(u),vals.terminal_median);
end
end

function r=severity_row(regime,subset,unit,arm,nCycles,v)
r=struct('regime',regime,'subset',subset,'unit',double(unit),'arm',arm, ...
    'n_cycles',nCycles,'terminal_median',v.terminal_median,'terminal_q25',v.terminal_q25, ...
    'terminal_q75',v.terminal_q75,'terminal_q90',v.terminal_q90, ...
    'terminal_max',v.terminal_max,'trajectory_rms',v.trajectory_rms);
end

function r=component_row(regime,subset,unit,arm,j,name,value,isFault)
r=struct('regime',regime,'subset',subset,'unit',double(unit),'arm',arm, ...
    'component',j,'parameter',name,'terminal_abs',value,'is_fault',logical(isFault));
end

function out=summarize_severity(severity)
groups={'in_dist','DS05_id';'in_dist','DS07_id';'ood_combo','DS02'; ...
    'ood_combo','DS03';'in_dist','ALL_IN';'ood_combo','ALL_OOD'};
rows=cell(0,1); ri=0;
for ig=1:size(groups,1)
    regime=groups{ig,1}; subset=groups{ig,2};
    if startsWith(subset,'ALL_')
        q=strcmp(severity.regime,regime)&strcmp(severity.arm,'truth');
    else
        q=strcmp(severity.regime,regime)&strcmp(severity.subset,subset)&strcmp(severity.arm,'truth');
    end
    x=severity.terminal_median(q);
    ri=ri+1; rows{ri}=struct('regime',regime,'subset',subset,'arm','truth', ... %#ok<AGROW>
        'n_units',numel(x),'median',median(x),'q25',prctile(x,25), ...
        'q75',prctile(x,75),'q90',prctile(x,90),'mean',mean(x));
    ri=ri+1; rows{ri}=struct('regime',regime,'subset',subset,'arm','Z_zero', ... %#ok<AGROW>
        'n_units',numel(x),'median',0,'q25',0,'q75',0,'q90',0,'mean',0);
end
out=struct2table(vertcat(rows{:}));
end

function out=compare_severity_metrics(truth)
metrics={'terminal_median','terminal_q75','terminal_q90','terminal_max','trajectory_rms'};
labels={'Terminal median','Terminal Q75','Terminal Q90','Terminal maximum','Trajectory RMS'};
rows=cell(numel(metrics)+1,1);
for i=1:numel(metrics)
    x=truth.(metrics{i})(strcmp(truth.regime,'in_dist'));
    y=truth.(metrics{i})(strcmp(truth.regime,'ood_combo'));
    ci=bootstrap_two_sample_median(y,x,5000,4600+i);
    delta=median(y)-median(x);
    rows{i}=struct('metric',metrics{i},'label',labels{i},'arm','truth', ...
        'in_dist_median',median(x),'ood_median',median(y),'ood_to_in_ratio',median(y)/median(x), ...
        'ood_minus_in',delta,'ci_low',ci(1),'ci_high',ci(2), ...
        'ood_greater',delta>0&&ci(1)>0);
end
rows{end}=struct('metric','Z_zero','label','Z-zero sentinel','arm','Z_zero', ...
    'in_dist_median',0,'ood_median',0,'ood_to_in_ratio',NaN,'ood_minus_in',0, ...
    'ci_low',0,'ci_high',0,'ood_greater',false);
out=struct2table(vertcat(rows{:}));
end

function paired=build_pairs(dd)
dd=dd(dd.regime=="ood_combo",:);
d=dd(dd.arm=="D",{'subset','unit','N','skill'}); d.Properties.VariableNames{4}='D_skill';
a=dd(dd.arm~="D",{'subset','unit','N','arm','skill'}); a.Properties.VariableNames{5}='arm_skill';
paired=innerjoin(a,d,'Keys',{'subset','unit','N'});
paired.gap_D_minus_arm=paired.D_skill-paired.arm_skill;
paired.D_win=paired.gap_D_minus_arm>0;
end

function out=summarize_pairs(paired)
arms=unique(paired.arm,'stable'); subsets=["DS02";"DS03";"ALL_OOD"];
rows=cell(0,1); ri=0;
for is=1:numel(subsets)
    for ia=1:numel(arms)
        if subsets(is)=="ALL_OOD"
            q=paired.arm==arms(ia);
        else
            q=paired.subset==subsets(is)&paired.arm==arms(ia);
        end
        x=paired.gap_D_minus_arm(q); ci=bootstrap_median_ci(x,3000,4500+100*is+ia);
        ri=ri+1; rows{ri}=struct('subset',char(subsets(is)),'arm',char(arms(ia)), ... %#ok<AGROW>
            'n_pairs',numel(x),'D_win_rate',mean(x>0),'median_gap',median(x), ...
            'q25_gap',prctile(x,25),'q75_gap',prctile(x,75),'mean_gap',mean(x), ...
            'median_ci_low',ci(1),'median_ci_high',ci(2));
    end
end
out=struct2table(vertcat(rows{:}));
end

function ci=bootstrap_two_sample_median(x,y,nBoot,seed)
x=x(:); y=y(:); s=RandStream('mt19937ar','Seed',seed); b=zeros(nBoot,1);
for i=1:nBoot
    bx=x(randi(s,numel(x),[numel(x) 1])); by=y(randi(s,numel(y),[numel(y) 1]));
    b(i)=median(bx)-median(by);
end
ci=prctile(b,[2.5 97.5]);
end

function ci=bootstrap_median_ci(x,nBoot,seed)
x=x(:); s=RandStream('mt19937ar','Seed',seed); b=zeros(nBoot,1);
for i=1:nBoot, b(i)=median(x(randi(s,numel(x),[numel(x) 1]))); end
ci=prctile(b,[2.5 97.5]);
end

function plot_severity_distribution(figDir,severity)
c=thesis_plot.colors(); truth=severity(strcmp(severity.arm,'truth'),:);
f=thesis_plot.new(17.8,8.2); t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); hold(ax,'on');
plot_group_points(ax,truth.terminal_median(strcmp(truth.regime,'in_dist')),1,c(1,:));
plot_group_points(ax,truth.terminal_median(strcmp(truth.regime,'ood_combo')),2,c(2,:));
set(ax,'XTick',1:2,'XTickLabel',{'In-distribution','OOD combination'}); xlim(ax,[.55 2.45]);
ylabel(ax,'Terminal |theta| median'); grid(ax,'on'); thesis_plot.panel(ax,'a');

ax=nexttile(t); hold(ax,'on'); subsets={'DS05_id','DS07_id','DS02','DS03'};
for i=1:numel(subsets)
    plot_group_points(ax,truth.terminal_median(strcmp(truth.subset,subsets{i})),i,c(1+mod(i-1,6),:));
end
set(ax,'XTick',1:4,'XTickLabel',subsets); xlim(ax,[.55 4.45]);
ylabel(ax,'Terminal |theta| median'); grid(ax,'on'); thesis_plot.panel(ax,'b');
thesis_plot.export(f,figDir,'fig_a3_e3_severity_distribution'); close(f);
end

function plot_severity_metrics(figDir,s)
c=thesis_plot.colors(); s=s(strcmp(s.arm,'truth'),:); n=height(s);
f=thesis_plot.new(12.5,8.2); ax=axes(f); hold(ax,'on');
for i=1:n
    plot(ax,[s.ci_low(i) s.ci_high(i)],[i i],'-','Color',c(1,:),'LineWidth',1.7);
    scatter(ax,s.ood_minus_in(i),i,42,c(2,:),'filled','MarkerEdgeColor','white');
end
xline(ax,0,'--','Color',[.3 .3 .3]);
set(ax,'YTick',1:n,'YTickLabel',s.label,'YDir','reverse');
xlabel(ax,'OOD minus in-distribution median'); grid(ax,'on');
thesis_plot.panel(ax,'a'); thesis_plot.export(f,figDir,'fig_a3_e3_severity_metrics'); close(f);
end

function plot_group_points(ax,x,x0,color)
n=numel(x); jitter=linspace(-.09,.09,max(n,2)); jitter=jitter(1:n);
scatter(ax,x0+jitter,x,28,repmat(color,n,1),'filled','MarkerFaceAlpha',.78, ...
    'MarkerEdgeColor','white','LineWidth',.4);
q=prctile(x,[25 50 75]); plot(ax,[x0 x0],[q(1) q(3)],'-','Color',[.15 .15 .15],'LineWidth',1.6);
plot(ax,[x0-.11 x0+.11],[q(2) q(2)],'-','Color',[.15 .15 .15],'LineWidth',2.2);
end

function plot_component_heatmap(figDir,component,cfg)
q=strcmp(component.arm,'truth'); x=component(q,:);
[~,ord]=sortrows(table(categorical(x.regime),categorical(x.subset),x.unit)); x=x(ord,:);
keys=unique(strcat(string(x.subset),"-",string(x.unit)),'stable');
M=nan(numel(keys),9);
for i=1:numel(keys)
    qi=strcat(string(x.subset),"-",string(x.unit))==keys(i);
    for j=1:9
        v=x.terminal_abs(qi&x.component==j); if ~isempty(v), M(i,j)=v(1); end
    end
end
f=thesis_plot.new(17.8,9.2); ax=axes(f);
h=imagesc(ax,M); h.AlphaData=~isnan(M); colormap(ax,viridis_map(256)); colorbar(ax);
set(ax,'Color',[.94 .94 .94],'XTick',1:9,'XTickLabel',compose('P%d',1:9), ...
    'YTick',1:numel(keys),'YTickLabel',keys);
xlabel(ax,'Health parameter index'); ylabel(ax,'Held-out engine');
title(ax,'Terminal normalized degradation amplitude','FontWeight','normal');
thesis_plot.panel(ax,'a');
thesis_plot.export(f,figDir,'fig_a3_e3_component_heatmap'); close(f);
map=table((1:9).',cfg.THETA9.','VariableNames',{'index','parameter'});
writetable(map,fullfile(fileparts(figDir),'component_axis_mapping.csv'),'Encoding','UTF-8');
end

function plot_skill_vs_severity(figDir,dd,severity)
d=dd(dd.arm=="D",:);
g=groupsummary(d,{'regime','subset','unit'},'mean','skill');
s=severity(strcmp(severity.arm,'truth'),{'regime','subset','unit','terminal_median'});
s.regime=string(s.regime); s.subset=string(s.subset);
j=innerjoin(g,s,'Keys',{'regime','subset','unit'});
c=thesis_plot.colors(); f=thesis_plot.new(9,8); ax=axes(f); hold(ax,'on');
qin=j.regime=="in_dist"; qout=j.regime=="ood_combo";
scatter(ax,j.terminal_median(qin),j.mean_skill(qin),42,c(1,:),'o','filled', ...
    'MarkerEdgeColor','white','DisplayName','In-distribution');
scatter(ax,j.terminal_median(qout),j.mean_skill(qout),42,c(2,:),'s','filled', ...
    'MarkerEdgeColor','white','DisplayName','OOD combination');
xlabel(ax,'Terminal |theta| median'); ylabel(ax,'Mean D skill score'); grid(ax,'on');
legend(ax,'Location','best'); thesis_plot.panel(ax,'a');
thesis_plot.export(f,figDir,'fig_a3_e3_skill_vs_severity'); close(f);
end

function plot_pair_small_multiples(figDir,paired)
c=thesis_plot.colors(); arms=["E_Lin","E_MLP","E_MixLinear","E_WPMixer"];
f=thesis_plot.new(17.8,11.5); t=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
for ia=1:numel(arms)
    ax=nexttile(t); hold(ax,'on'); qa=paired.arm==arms(ia);
    for is=1:2
        subset=["DS02","DS03"]; x=paired.gap_D_minus_arm(qa&paired.subset==subset(is));
        jitter=linspace(-.12,.12,max(numel(x),2)); jitter=jitter(1:numel(x));
        scatter(ax,is+jitter,x,24,c(is,:),'filled','MarkerEdgeColor','white','LineWidth',.3);
        med=median(x); plot(ax,[is-.18 is+.18],[med med],'-','Color',[.12 .12 .12],'LineWidth',2);
    end
    yline(ax,0,'--','Color',[.3 .3 .3]); set(ax,'XTick',1:2,'XTickLabel',{'DS02','DS03'});
    xlim(ax,[.55 2.45]); ylabel(ax,'D - arm skill'); grid(ax,'on');
    title(ax,strrep(char(arms(ia)),'_','\_'),'FontWeight','normal'); thesis_plot.panel(ax,char('a'+ia-1));
end
thesis_plot.export(f,figDir,'fig_a3_e3_paired_gaps'); close(f);
end

function plot_pair_summary(figDir,s)
subsets=["DS02","DS03"]; arms=["E_Lin","E_MLP","E_MixLinear","E_WPMixer"];
W=nan(2,4); G=nan(2,4);
for i=1:2
    for j=1:4
        q=string(s.subset)==subsets(i)&string(s.arm)==arms(j);
        W(i,j)=s.D_win_rate(q); G(i,j)=s.median_gap(q);
    end
end
f=thesis_plot.new(17.8,7.5); t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(t); imagesc(ax,100*W,[0 100]); colormap(ax,viridis_map(256)); colorbar(ax);
set(ax,'XTick',1:4,'XTickLabel',strrep(cellstr(arms),'_','\_'),'YTick',1:2,'YTickLabel',cellstr(subsets));
for i=1:2
    for j=1:4
        text(ax,j,i,sprintf('%.0f%%',100*W(i,j)), ...
            'HorizontalAlignment','center','Color',contrast_color(W(i,j)));
    end
end
title(ax,'D dominance rate','FontWeight','normal'); thesis_plot.panel(ax,'a');
ax=nexttile(t); imagesc(ax,G); colormap(ax,viridis_map(256)); colorbar(ax);
set(ax,'XTick',1:4,'XTickLabel',strrep(cellstr(arms),'_','\_'),'YTick',1:2,'YTickLabel',cellstr(subsets));
for i=1:2
    for j=1:4
        text(ax,j,i,sprintf('%.3f',G(i,j)), ...
            'HorizontalAlignment','center','Color',contrast_color_rescaled(G(i,j),G));
    end
end
title(ax,'Median D - arm skill','FontWeight','normal'); thesis_plot.panel(ax,'b');
thesis_plot.export(f,figDir,'fig_a3_e3_pair_summary'); close(f);
end

function c=contrast_color(v)
if v>.55, c=[.1 .1 .1]; else, c='white'; end
end

function c=contrast_color_rescaled(v,M)
z=(v-min(M,[],'all'))/max(max(M,[],'all')-min(M,[],'all'),eps);
if z>.55, c=[.1 .1 .1]; else, c='white'; end
end

function map=viridis_map(n)
anchors=[68 1 84;59 82 139;33 145 140;94 201 98;253 231 37]/255;
x=linspace(0,1,size(anchors,1)); xi=linspace(0,1,n);
map=interp1(x,anchors,xi,'pchip'); map=min(max(map,0),1);
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
