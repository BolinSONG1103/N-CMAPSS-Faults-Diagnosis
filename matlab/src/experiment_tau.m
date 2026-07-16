function verdict=experiment_tau()
% T3：用正式 raw_scan 反解使 Morozov 选中 lambda=31.6 所需的 tau。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); root=fileparts(matlabRoot); addpath(srcDir); ncmapss_lib.set_plot_defaults();
outDir=fullfile(matlabRoot,'outputs','t3_tau'); if ~isfolder(outDir), mkdir(outDir); end
src=fullfile(root,'reference','fixed_results','unsupervised','raw_scan.csv');
assert(isfile(src),'缺少脚本18正式 raw_scan.csv。');
raw=readtable(src,'VariableNamingRule','preserve');
raw.Properties.VariableNames{strcmp(raw.Properties.VariableNames,'lam')}='lambda';
d=raw(strcmp(raw.method,'D'),:);
lamValues=unique(d.lambda); lamValues=lamValues(isfinite(lamValues));
[~,ii]=min(abs(log(lamValues/31.6))); lamPick=lamValues(ii);
d=d(abs(d.lambda-lamPick)<1e-10,:);
d.tau_required=d.res_norm./d.delta;
point=groupsummary(d,{'subset','N'},'mean',{'tau_required','delta','res_norm'});
point.sqrtN=sqrt(point.N);
writetable(d,fullfile(outDir,'tau_required_per_unit.csv'),'Encoding','UTF-8');
writetable(point,fullfile(outDir,'tau_required_points.csv'),'Encoding','UTF-8');

mdl=fitlm(point.sqrtN,point.mean_tau_required);
r2=mdl.Rsquared.Ordinary; coef=mdl.Coefficients.Estimate;
cNoise=mean(point.mean_delta.*point.sqrtN);
modelErrorEstimate=coef(2)*cNoise;
perSubset=struct(); subsets=unique(point.subset);
for i=1:numel(subsets)
    q=strcmp(point.subset,subsets{i}); m=fitlm(point.sqrtN(q),point.mean_tau_required(q));
    perSubset.(subsets{i})=struct('R2',m.Rsquared.Ordinary, ...
        'intercept',m.Coefficients.Estimate(1),'slope',m.Coefficients.Estimate(2));
end
verdict=struct('criterion','R2>=0.9','lambda',lamPick,'R2',r2, ...
    'intercept',coef(1),'slope',coef(2),'pass',r2>=.9, ...
    'noise_constant_estimate',cNoise,'model_error_norm_estimate',modelErrorEstimate, ...
    'per_subset',perSubset);
write_json(fullfile(outDir,'verdict_tau.json'),verdict);

f=figure('Visible','off','Position',[100 100 820 560]); hold on;
colors=lines(numel(subsets));
for i=1:numel(subsets)
    q=strcmp(point.subset,subsets{i}); scatter(point.sqrtN(q),point.mean_tau_required(q),65,colors(i,:),'filled','DisplayName',subsets{i});
end
x=linspace(min(point.sqrtN),max(point.sqrtN),200); plot(x,coef(1)+coef(2)*x,'k-','LineWidth',2,'DisplayName',sprintf('线性拟合 R^2=%.3f',r2));
xlabel('sqrt(N)'); ylabel('tau required'); title('使 Morozov 选中 lambda=31.6 所需的 tau'); grid on; legend('Location','best');
exportgraphics(f,fullfile(outDir,'fig_tau_sqrtN.png'),'Resolution',180); close(f);
fprintf('T3: tau=%.4f%+.4f*sqrt(N), R2=%.4f -> %s\n',coef(1),coef(2),r2,pass_text(verdict.pass));
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); c=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end

function s=pass_text(v)
if v, s='PASS'; else, s='FAIL'; end
end
