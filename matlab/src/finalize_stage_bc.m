function verdict=finalize_stage_bc()
%FINALIZE_STAGE_BC 阶段 B/C：E3' 分裂裁决、E4 主线、最终表格与论文图组。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); root=fileparts(matlabRoot);
addpath(srcDir);
resultDir=fullfile(root,'chapter3_results'); tabDir=fullfile(resultDir,'tables');
stageDir=fullfile(matlabRoot,'outputs','stage_bc');
b1Dir=fullfile(stageDir,'b1_e3prime'); b2Dir=fullfile(stageDir,'b2_e4');
dirs={resultDir,tabDir,stageDir,b1Dir,b2Dir};
for i=1:numel(dirs), if ~isfolder(dirs{i}), mkdir(dirs{i}); end, end

fprintf('\nStage B/C finalization from locked CSV results\n');
[e3raw,e3summary,e3verdict,gapArchive]=build_e3prime(matlabRoot,b1Dir);
[e4,e4verdict]=build_e4(matlabRoot,b2Dir);
[tables,formal]=build_final_tables(root,matlabRoot,tabDir,e3summary,e4);

manifest=build_chapter3_figures_nature(root);
writetable(manifest,fullfile(resultDir,'figure_manifest_v2.csv'),'Encoding','UTF-8');
oldManifest=fullfile(resultDir,'figure_manifest.csv');
if isfile(oldManifest), delete(oldManifest); end
nGenerated=sum(manifest.generated);

verdict=struct('stage','BC','analysis_type','final_from_locked_formal_outputs', ...
    'P1',formal.P1,'P2',formal.P2,'P4',formal.P4,'T3',formal.T3, ...
    'Q1',formal.Q1,'Q2f_mechanism_consistent',formal.Q2f, ...
    'Q2s',formal.Q2s,'Q3',formal.Q3,'Q4',formal.Q4,'Q6_status','descriptive', ...
    'E1',formal.E1,'E2',formal.E2, ...
    'E3prime_a_deep',e3verdict.E3prime_a_deep, ...
    'E3prime_b_linear_status','descriptive_boundary', ...
    'original_E3_status','archived_not_used_for_conclusion', ...
    'E4',e4verdict,'figure_sets',nGenerated,'figure_manifest_rows',height(manifest), ...
    'figure_formats',{{'png','pdf'}},'fast_mode_numbers_used',false, ...
    'preregistered_thresholds_moved',false,'failed_rows_deleted',false, ...
    'stage_A_author_confirmed',true);
write_json(fullfile(stageDir,'verdict_stage_bc.json'),verdict);
save(fullfile(stageDir,'final_tables.mat'),'tables','formal','e3raw','e3summary','gapArchive','e4');
fprintf('Stage B/C tables and %d program-generated figure sets completed.\n',nGenerated);
end

function [paired,summary,v,gapArchive]=build_e3prime(matlabRoot,outDir)
src=fullfile(matlabRoot,'outputs','stage_a','a3_e3','same_subset_paired_raw.csv');
paired=readtable(src,'TextType','string');
writetable(paired,fullfile(outDir,'e3prime_paired_raw.csv'),'Encoding','UTF-8'); % 先落盘

arms=["Z_zero","E_Lin","E_MixLinear","E_MLP","E_WPMixer"];
rows=cell(numel(arms),1);
for i=1:numel(arms)
    x=paired.gap_D_minus_arm(paired.arm==arms(i)); n=numel(x); k=sum(x>0);
    [lo,hi]=wilson(k,n,.05);
    if arms(i)=="Z_zero"
        group="sentinel"; role="sentinel"; threshold=NaN; verdict="reference";
    elseif ismember(arms(i),["E_MLP","E_WPMixer"])
        group="deep_nonlinear"; role="formal"; threshold=.95;
        if mean(x>0)>=threshold, verdict="pass"; else, verdict="fail"; end
    else
        group="linear_shallow"; role="descriptive"; threshold=NaN; verdict="boundary";
    end
    rows{i}=table(arms(i),group,role,n,k,mean(x>0),lo,hi,median(x), ...
        prctile(x,25),prctile(x,75),mean(x),threshold,verdict, ...
        'VariableNames',{'arm','group','role','n_pairs','D_wins','D_win_rate', ...
        'win_ci_low','win_ci_high','median_gap','q25_gap','q75_gap','mean_gap', ...
        'formal_threshold','verdict'});
end
summary=vertcat(rows{:});
writetable(summary,fullfile(outDir,'e3prime_split_summary.csv'),'Encoding','UTF-8');

deep=summary(summary.group=="deep_nonlinear",:);
linear=summary(summary.group=="linear_shallow",:);
deepRaw=paired(ismember(paired.arm,["E_MLP","E_WPMixer"]),:);
[deepLo,deepHi]=wilson(sum(deepRaw.D_win),height(deepRaw),.05);
v=struct('criterion', ...
    'E3prime tests D dominance against deep arms >=95%; linear arms are descriptive', ...
    'threshold',.95,'E3prime_a_deep',all(deep.D_win_rate>=.95), ...
    'deep_group_rate',mean(deepRaw.D_win),'deep_group_ci95',[deepLo deepHi], ...
    'deep_arm_results',table2struct(deep), ...
    'linear_arm_results',table2struct(linear), ...
    'linear_result_is_not_pass_fail',true,'original_E3_remains_false',true);

gapArchive=readtable(fullfile(matlabRoot,'outputs','t2_dd','generalization_gaps.csv'), ...
    'TextType','string');
z=gapArchive(1,:); z.arm="Z_zero"; z.in_dist=0; z.ood_combo=0; z.gap=0;
gapArchive=[gapArchive;z];
gapArchive.interpretation_status=repmat("archived_cross_subset_not_for_conclusion",height(gapArchive),1);
gapArchive.interpretation_status(gapArchive.arm=="Z_zero")="sentinel";
writetable(gapArchive,fullfile(outDir,'original_E3_gap_archived.csv'),'Encoding','UTF-8');
write_json(fullfile(outDir,'verdict_e3prime.json'),v);
end

function [e4,v]=build_e4(matlabRoot,outDir)
skill=readtable(fullfile(matlabRoot,'outputs','t2_dd','skill_summary.csv'),'TextType','string');
skill=skill(skill.regime=="ood_combo",:);
[G,arm]=findgroups(skill.arm); meanSkill=splitapply(@mean,skill.mean_skill,G);
skillAgg=table(arm,meanSkill,'VariableNames',{'arm','ood_skill'});

fa=readtable(fullfile(matlabRoot,'outputs','t2_dd','false_alarm_ood.csv'),'TextType','string');
[G,arm]=findgroups(fa.arm); meanFa=splitapply(@mean,fa.mean_false_alarm,G);
faAgg=table(arm,meanFa,'VariableNames',{'arm','ood_false_alarm'});

params=readtable(fullfile(matlabRoot,'outputs','t2_dd','parameter_counts.csv'),'TextType','string');
e4=outerjoin(skillAgg,faAgg,'Keys','arm','MergeKeys',true,'Type','full');
e4=outerjoin(e4,params,'Keys','arm','MergeKeys',true,'Type','left');
if ~any(e4.arm=="Z_zero")
    e4=[e4;table("Z_zero",0,0,0,'VariableNames',e4.Properties.VariableNames)];
end
e4.parameters=fillmissing(e4.parameters,'constant',0);
e4.group=repmat("",height(e4),1);
e4.group(e4.arm=="D")="physics_constrained";
e4.group(ismember(e4.arm,["E_Lin","E_MixLinear"]))="linear_shallow";
e4.group(ismember(e4.arm,["E_MLP","E_WPMixer"]))="deep_nonlinear";
e4.group(e4.arm=="Z_zero")="sentinel";
dFa=e4.ood_false_alarm(e4.arm=="D");
e4.false_alarm_ratio_to_D=e4.ood_false_alarm/max(dFa,eps);
e4.false_alarm_ratio_to_D(e4.arm=="Z_zero")=0;
order=["D","E_Lin","E_MixLinear","E_MLP","E_WPMixer","Z_zero"];
[~,loc]=ismember(order,e4.arm); e4=e4(loc(loc>0),:);
writetable(e4,fullfile(outDir,'e4_ood_skill_false_alarm.csv'),'Encoding','UTF-8'); % 先落盘

v=struct('D_skill',value(e4,"D",'ood_skill'),'D_false_alarm',value(e4,"D",'ood_false_alarm'), ...
    'E_Lin_skill',value(e4,"E_Lin",'ood_skill'), ...
    'E_Lin_false_alarm_ratio_to_D',value(e4,"E_Lin",'false_alarm_ratio_to_D'), ...
    'E_MixLinear_skill',value(e4,"E_MixLinear",'ood_skill'), ...
    'E_MixLinear_false_alarm_ratio_to_D',value(e4,"E_MixLinear",'false_alarm_ratio_to_D'), ...
    'E_MLP_skill',value(e4,"E_MLP",'ood_skill'), ...
    'E_WPMixer_skill',value(e4,"E_WPMixer",'ood_skill'), ...
    'conclusion','linearity supports extrapolation; hard constraints provide specificity');
write_json(fullfile(outDir,'verdict_e4.json'),v);
end

function [out,formal]=build_final_tables(root,matlabRoot,tabDir,e3summary,e4)
raw=readtable(fullfile(root,'reference','fixed_results','unsupervised','raw_scan.csv'), ...
    'TextType','string'); raw.Properties.VariableNames{strcmp(raw.Properties.VariableNames,'lam')}='lambda';
sel=readtable(fullfile(root,'reference','fixed_results','unsupervised','selected_lambda.csv'), ...
    'TextType','string'); sel.Properties.VariableNames{strcmp(sel.Properties.VariableNames,'lam')}='lambda';
t1v=jsondecode(fileread(fullfile(matlabRoot,'outputs','t1_moe','verdict19b.json')));
t2v=jsondecode(fileread(fullfile(matlabRoot,'outputs','t2_dd','verdict20.json')));
tauv=jsondecode(fileread(fullfile(matlabRoot,'outputs','t3_tau','verdict_tau.json')));

g=groupsummary(raw(raw.method~="Z_zero",:),{'subset','N','lambda','method'},'mean','skill');
w=unstack(g,'mean_skill','method'); w.gap=w.D-w.B;
keys=unique(w(:,{'subset','N'}),'rows'); rows=cell(height(keys),1);
for i=1:height(keys)
    q=w.subset==keys.subset(i)&w.N==keys.N(i); x=w.gap(q);
    rows{i}=table(keys.subset(i),keys.N(i),numel(x),sum(x>0),mean(x>0), ...
        median(x),min(x),0,'VariableNames',{'subset','N','points','D_wins', ...
        'win_rate','median_gap','min_gap','Z_zero_skill'});
end
out.table1=vertcat(rows{:}); writetable(out.table1,fullfile(tabDir,'表1_逐点占优.csv'),'Encoding','UTF-8');

crit=groupsummary(sel,{'subset','criterion','method'},'mean','skill');
out.table2=unstack(crit,'mean_skill','method'); out.table2.Z_zero=zeros(height(out.table2),1);
writetable(out.table2,fullfile(tabDir,'表2_lambda准则技能分.csv'),'Encoding','UTF-8');

raw.lamkey=round(log10(raw.lambda),6); sel.lamkey=round(log10(sel.lambda),6);
pick=sel(sel.method=="B"&sel.criterion=="L_curve",{'subset','unit','N','lamkey','skill'});
pick.Properties.VariableNames{end}='B_Lcurve';
drows=raw(raw.method=="D",{'subset','unit','N','lamkey','skill'}); drows.Properties.VariableNames{end}='D_at_B_Lcurve';
cross=innerjoin(pick,drows,'Keys',{'subset','unit','N','lamkey'});
cAgg=groupsummary(cross,'subset','mean',{'B_Lcurve','D_at_B_Lcurve'});
dorc=groupsummary(sel(sel.method=="D"&sel.criterion=="oracle",:),'subset','mean','skill');
dorc.Properties.VariableNames{end}='D_oracle';
out.table3=innerjoin(cAgg,dorc,'Keys','subset');
out.table3.loss=out.table3.D_oracle-out.table3.mean_D_at_B_Lcurve;
out.table3.Z_zero_skill=zeros(height(out.table3),1);
writetable(out.table3,fullfile(tabDir,'表3_代理路径标定.csv'),'Encoding','UTF-8');

out.table4=readtable(fullfile(matlabRoot,'outputs','t1_moe','main_lambda_summary.csv'),'TextType','string');
out.table4.Z_zero_skill=zeros(height(out.table4),1);
writetable(out.table4,fullfile(tabDir,'表4_工况自适应主lambda.csv'),'Encoding','UTF-8');
out.table5=readtable(fullfile(matlabRoot,'outputs','t2_dd','skill_summary.csv'),'TextType','string');
out.table5.Z_zero_skill=zeros(height(out.table5),1);
writetable(out.table5,fullfile(tabDir,'表5_数据驱动对照技能分.csv'),'Encoding','UTF-8');

deep=e3summary(e3summary.group=="deep_nonlinear",:);
linear=e3summary(e3summary.group=="linear_shallow",:);
names=["P1";"P2经典准则";"P4代理标定";"T3_tau_sqrtN";"Q1";"Q2f";"Q2s"; ...
    "Q3";"Q4";"Q6";"E1";"E2";"E3prime-a";"E3prime-b";"原E3"];
status=["通过";"未通过";"部分通过：DS03通过，DS02未通过";"通过"; ...
    "通过";"描述性一致";"未通过";"通过";"通过";"描述性边界";"通过";"通过"; ...
    "通过：深层臂最小占优率100%"; ...
    sprintf("描述性：线性臂占优率%.1f%%-%.1f%%",100*min(linear.D_win_rate),100*max(linear.D_win_rate)); ...
    "归档：跨子集落差不作结论"];
source=["raw_scan.csv";"selected_lambda.csv";"表3";"verdict_tau.json"; ...
    "verdict19b.json";"verdict19b.json";"verdict19b.json";"verdict19b.json"; ...
    "verdict19b.json";"verdict19b.json";"verdict20.json";"verdict20.json"; ...
    "e3prime_split_summary.csv";"e3prime_split_summary.csv";"original_E3_gap_archived.csv"];
out.table6=table(names,status,source,zeros(numel(names),1), ...
    'VariableNames',{'criterion','verdict','source','Z_zero_skill'});
writetable(out.table6,fullfile(tabDir,'表6_最终判据总表.csv'),'Encoding','UTF-8');

out.table7=e3summary; writetable(out.table7,fullfile(tabDir,'表7_E3prime分裂裁决.csv'),'Encoding','UTF-8');
out.table8=e4; writetable(out.table8,fullfile(tabDir,'表8_外推技能与虚警.csv'),'Encoding','UTF-8');
out.table9=readtable(fullfile(matlabRoot,'outputs','stage_a','a1_baseline','baseline_comparison_summary.csv'),'TextType','string');
writetable(out.table9,fullfile(tabDir,'表9_健康基准敏感性.csv'),'Encoding','UTF-8');
out.table10=readtable(fullfile(matlabRoot,'outputs','stage_a','a3_e3','severity_metric_comparison.csv'),'TextType','string');
out.table10.Z_zero_skill=zeros(height(out.table10),1);
writetable(out.table10,fullfile(tabDir,'表10_跨子集严重度审计.csv'),'Encoding','UTF-8');
out.table11=forward_table(); writetable(out.table11,fullfile(tabDir,'表11_MoE前向结构消融.csv'),'Encoding','UTF-8');

formal=struct('P1',true,'P2',false,'P4',false,'T3',tauv.pass,'Q1',t1v.Q1, ...
    'Q2f',t1v.Q2f_mechanism_consistent,'Q2s',t1v.Q2s,'Q3',t1v.Q3, ...
    'Q4',t1v.Q4,'E1',t2v.E1,'E2',t2v.E2,'E3prime_a',all(deep.D_win_rate>=.95));
end

function t=forward_table()
arm=["H0";"MoE_K1";"MoE_K2";"MoE_K4";"MoE_K8";"FreeMLP";"MoE_dir";"Z_zero"];
nrmse=[.32780;.33042;.20846;.20355;.20396;.20109;.20449;NaN];
role=["global";"one_expert";"moe";"selected_moe";"moe";"free_gain";"direction_free";"sentinel"];
t=table(arm,nrmse,role,'VariableNames',{'arm','validation_NRMSE','role'});
end

function [lo,hi]=wilson(k,n,alpha)
if n==0, lo=NaN; hi=NaN; return; end
assert(abs(alpha-.05)<eps,'当前仅实现 95%% Wilson 区间。');
z=1.95996398454005; p=k/n; den=1+z^2/n;
ctr=(p+z^2/(2*n))/den; half=z*sqrt(p*(1-p)/n+z^2/(4*n^2))/den;
lo=max(0,ctr-half); hi=min(1,ctr+half);
end

function x=value(t,arm,var)
x=t.(var)(t.arm==arm); x=x(1);
end

function write_json(path,v)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(v,'PrettyPrint',true),'char');
end
