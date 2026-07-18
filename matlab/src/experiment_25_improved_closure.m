function verdict=experiment_25_improved_closure(fastMode)
%EXPERIMENT_25_IMPROVED_CLOSURE 决策层改进对比实验（不覆盖 v2）。
%
% 复用 v2 锁定的 closure_sequences（残差 R、真值、健康标签）、split 缓存中的
% Hn 与锁定 lambda，重新反演后在同一 theta_hat 上对比 v2 基线决策层与 v3
% 改进决策层（单元自适应阈值 + 因果时序平滑）。另附 lambda 的 1-SE 稳定化
% 选择与部件族可辨识性诊断。全部参数只在 calibration 上选择，测试真值仅用于
% 计算指标。输出写入 outputs/t7_improved_closure/。

if nargin<1, fastMode=true; end
srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); addpath(srcDir);
cfg=ncmapss_lib.config(fastMode);
dcfg2=chapter3_diagnostic_lib_v3.config(cfg);
dcfg1=chapter3_diagnostic_lib.config(cfg);

baseDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure');
if fastMode, baseDir=fullfile(matlabRoot,'outputs','t4_diagnostic_closure_smoke'); end
outDir=fullfile(matlabRoot,'outputs','t7_improved_closure');
if fastMode, outDir=fullfile(matlabRoot,'outputs','t7_improved_closure_smoke'); end
if ~isfolder(outDir), mkdir(outDir); end

S=load(fullfile(baseDir,'closure_sequences.mat'),'calRaw','idRaw','oodRaw');
D=load(fullfile(baseDir,'diagnostic_model.mat'),'lambdaSelected');
P=load(fullfile(matlabRoot,'cache','pipeline_cache_split_v2.mat'),'pipe'); pipe=P.pipe;
if isempty(pipe.Hn), error('chapter3:MissingHn','split pipeline 缓存缺少 Hn。'); end
Hn=pipe.Hn; lambda=D.lambdaSelected;

% ---- 同一反演，供 v2 / v3 共用 ----
cal=solve_sequences(S.calRaw,Hn,lambda);
test=solve_sequences([S.idRaw,S.oodRaw],Hn,lambda);

% ---- 两套决策层 ----
model2=chapter3_diagnostic_lib.fit(cal,dcfg1,Hn);
model3=chapter3_diagnostic_lib_v3.fit(cal,dcfg2,Hn);
fprintf('v3 选中：参考窗口 w=%d，尺度裕度 m=%g，平滑窗 L=%d，持续 K=%d。\n', ...
    model3.ref_window,model3.scale_margin,model3.smooth_window,model3.persistence);

famMap=[1 2 2 3 3 4 4 5 5];   % 9 参数 -> 5 部件族
rows=cell(0,1); ri=0;
for i=1:numel(test)
    s=test{i};
    p2=chapter3_diagnostic_lib.predict(s.theta_hat,s.R,Hn,model2);
    p3=chapter3_diagnostic_lib_v3.predict(s.theta_hat,s.R,Hn,model3);
    y=chapter3_diagnostic_lib.truth(s.theta_true,s.hs,model2.stage,model2.truth_active_eps,s.fault_idx);
    ri=ri+1; rows{ri}=compare_row('v2_baseline',s,p2,y,famMap); %#ok<AGROW>
    ri=ri+1; rows{ri}=compare_row('v3_improved',s,p3,y,famMap); %#ok<AGROW>
end
comparison=vertcat(rows{:});
writetable(comparison,fullfile(outDir,'decision_layer_comparison.csv'),'Encoding','UTF-8');
summary=summarize(comparison);
writetable(summary,fullfile(outDir,'decision_layer_summary.csv'),'Encoding','UTF-8');
disp(summary);

% ---- lambda 1-SE 稳定化 ----
lambdaTable=lambda_one_se(S.calRaw,Hn,cfg.LAMBDA_GRID);
writetable(lambdaTable,fullfile(outDir,'lambda_1se_stabilization.csv'),'Encoding','UTF-8');
lamMin=lambdaTable.lambda(lambdaTable.min_mse_selected==1);
lam1se=lambdaTable.lambda(lambdaTable.one_se_selected==1);
fprintf('lambda：最小-MSE=%.6g，1-SE 稳定化=%.6g。\n',lamMin,lam1se);

% ---- 部件族可辨识性诊断（解释 HPT 拒识边界）----
ident=family_identifiability(Hn,famMap);
writetable(ident,fullfile(outDir,'family_identifiability.csv'),'Encoding','UTF-8');
disp(ident);

verdict=struct('analysis_type','decision_layer_improvement_vs_v2', ...
    'fast_mode',fastMode,'lambda_used_for_inversion',lambda, ...
    'v3_ref_window',model3.ref_window,'v3_scale_margin',model3.scale_margin, ...
    'v3_smooth_window',model3.smooth_window,'v3_persistence',model3.persistence, ...
    'lambda_min_mse',lamMin,'lambda_one_se',lam1se, ...
    'params_selected_on_calibration_only',true);
write_json(fullfile(outDir,'verdict25.json'),verdict);
end

% ------------------------------------------------------------------ 反演
function out=solve_sequences(raw,Hn,lambda)
out=raw;
for i=1:numel(raw)
    s=raw{i}; T=size(s.R,1); n=size(Hn,2);
    s.theta_hat=ncmapss_lib.solve_D(ncmapss_lib.build_M(Hn,T), ...
        ncmapss_lib.build_cum(T,n),s.R,lambda,T,n);
    out{i}=s;
end
end

% ------------------------------------------------------------------ 逐单元对比行
function row=compare_row(method,s,p,y,famMap)
m=chapter3_diagnostic_lib.sequence_metrics(p,y,s.cycles);
famPred=aggregate_family(p.component_label,famMap);
famTruth=aggregate_family(y.component_label,famMap);
famMacro=chapter3_diagnostic_lib.macro_f1(famPred,famTruth);
row=table(string(method),string(s.regime),string(s.subset),s.unit, ...
    m.detection_FAR,m.detection_DR,m.detection_F1, ...
    m.isolation_macroF1,famMacro,m.stage_weighted_kappa,m.stage_MAE, ...
    'VariableNames',{'method','regime','subset','unit','detection_FAR', ...
    'detection_DR','detection_F1','isolation_macroF1_9param', ...
    'isolation_macroF1_family','stage_kappa','stage_MAE'});
end

function fam=aggregate_family(label9,famMap)
fam=false(size(label9,1),max(famMap));
for g=1:max(famMap)
    cols=find(famMap==g);
    fam(:,g)=any(label9(:,cols),2);
end
end

function out=summarize(t)
[G,method,regime]=findgroups(t.method,t.regime);
out=table(method,regime,splitapply(@numel,t.unit,G), ...
    splitapply(@mean,t.detection_FAR,G),splitapply(@mean,t.detection_DR,G), ...
    splitapply(@mean,t.detection_F1,G),splitapply(@mean,t.isolation_macroF1_9param,G), ...
    splitapply(@mean,t.isolation_macroF1_family,G), ...
    splitapply(@(x)mean(x,'omitnan'),t.stage_kappa,G), ...
    'VariableNames',{'method','regime','n_units','FAR','DR','detF1', ...
    'iso_macroF1_9param','iso_macroF1_family','stage_kappa'});
end

% ------------------------------------------------------------------ lambda 1-SE
function out=lambda_one_se(calSeqs,Hn,grid)
grid=grid(:); nL=numel(grid); n=size(Hn,2);
perUnit=zeros(numel(calSeqs),nL);
for i=1:numel(calSeqs)
    s=calSeqs{i}; T=size(s.R,1); M=ncmapss_lib.build_M(Hn,T); Cum=ncmapss_lib.build_cum(T,n);
    for il=1:nL
        th=ncmapss_lib.solve_D(M,Cum,s.R,grid(il),T,n);
        perUnit(i,il)=mean((th-s.theta_true).^2,'all');   % unit-balanced MSE
    end
end
meanMSE=mean(perUnit,1).'; seMSE=(std(perUnit,0,1)/sqrt(size(perUnit,1))).';
[~,iMin]=min(meanMSE); thr=meanMSE(iMin)+seMSE(iMin);
% 1-SE 规则：在 MSE<=最小+1SE 的候选中选最大的 lambda（更强、更稳定的正则）
elig=find(meanMSE<=thr); [~,jj]=max(grid(elig)); iOne=elig(jj);
out=table(grid,meanMSE,seMSE,(1:nL).'==iMin,(1:nL).'==iOne, ...
    'VariableNames',{'lambda','mean_MSE','se_MSE','min_mse_selected','one_se_selected'});
end

% ------------------------------------------------------------------ 可辨识性
function out=family_identifiability(Hn,famMap)
g=max(famMap); names=["HPT";"Fan";"HPC";"LPT";"LPC"];
minAng=zeros(g,1);
for k=1:g
    cols=find(famMap==k); others=setdiff(1:size(Hn,2),cols);
    B=Hn(:,others); [Q,~]=qr(B,0);
    a=zeros(numel(cols),1);
    for c=1:numel(cols)
        v=Hn(:,cols(c)); v=v/norm(v); r=v-Q*(Q.'*v);
        a(c)=asind(min(norm(r),1));
    end
    minAng(k)=min(a);
end
out=table(names,minAng,minAng<3, ...
    'VariableNames',{'family','min_angle_to_other_families_deg','near_collinear_reject_hard'});
end

% ------------------------------------------------------------------ json
function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end
