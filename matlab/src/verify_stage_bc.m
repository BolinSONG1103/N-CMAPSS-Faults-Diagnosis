function report=verify_stage_bc()
%VERIFY_STAGE_BC 核查阶段 B/C 裁决与正文图件最终版。

srcDir=fileparts(mfilename('fullpath')); matlabRoot=fileparts(srcDir); root=fileparts(matlabRoot);
resultDir=fullfile(root,'chapter3_results'); figDir=fullfile(resultDir,'figures');
tableDir=fullfile(resultDir,'tables'); stageDir=fullfile(matlabRoot,'outputs','stage_bc');

v=jsondecode(fileread(fullfile(stageDir,'verdict_stage_bc.json')));
assert(v.P1 && ~v.P2 && ~v.P4 && v.T3 && v.Q1 && v.Q2f_mechanism_consistent);
assert(~v.Q2s && v.Q3 && v.Q4 && v.E1 && v.E2 && v.E3prime_a_deep);
assert(strcmp(v.E3prime_b_linear_status,'descriptive_boundary'));
assert(strcmp(v.original_E3_status,'archived_not_used_for_conclusion'));
assert(v.figure_sets==7 && v.figure_manifest_rows==8 && ~v.fast_mode_numbers_used);
assert(~v.preregistered_thresholds_moved && ~v.failed_rows_deleted);

e3=jsondecode(fileread(fullfile(stageDir,'b1_e3prime','verdict_e3prime.json')));
assert(e3.E3prime_a_deep && e3.deep_group_rate==1 && e3.linear_result_is_not_pass_fail);
t1=jsondecode(fileread(fullfile(matlabRoot,'outputs','t1_moe','verdict19b.json')));
assert(~t1.FAST_MODE && t1.Q1 && ~t1.Q2s && t1.Q3 && t1.Q4);

manifest=readtable(fullfile(resultDir,'figure_manifest_v2.csv'),'TextType','string');
assert(height(manifest)==8 && numel(unique(manifest.figure_id))==8);
generatedMask=logical(manifest.generated);
assert(nnz(~generatedMask)==1 && manifest.figure_id(~generatedMask)=="图3-1");
generated=manifest(generatedMask,:); assert(height(generated)==7);
dpi=zeros(height(generated),1);
for i=1:height(generated)
    png=fullfile(figDir,generated.file_stem(i)+".png");
    pdf=fullfile(figDir,generated.file_stem(i)+".pdf");
    assert(isfile(png) && isfile(pdf),'缺少正文图：%s',generated.file_stem(i));
    info=imfinfo(png); dpi(i)=png_dpi(info(1));
    assert(dpi(i)>=590,'PNG 分辨率不足：%s (%.1f dpi)',generated.file_stem(i),dpi(i));
    assert(strlength(generated.source_data(i))>0 && strlength(generated.figure_eye(i))>0);
end
assert(numel(dir(fullfile(figDir,'*.png')))==7 && numel(dir(fullfile(figDir,'*.pdf')))==7);
assert(isempty(dir(fullfile(figDir,'*.svg'))),'最终正文图目录不应保留 SVG。');

trajectory=fullfile(stageDir,'figure_data','fig3_2_ds03_unit13_trajectory.csv');
assert(isfile(trajectory)); tr=readtable(trajectory,'TextType','string');
assert(all(ismember({'theta_true_pct','theta_D_pct','theta_B_pct'},tr.Properties.VariableNames)));
assert(numel(unique(tr.parameter))==9 && any(tr.is_fault) && any(~tr.is_fault));

tables=dir(fullfile(tableDir,'表*.csv')); assert(numel(tables)==11);
for i=1:numel(tables)
    t=readtable(fullfile(tables(i).folder,tables(i).name),'TextType','string');
    hasColumn=any(contains(string(t.Properties.VariableNames),'Z_zero')); hasRow=false;
    for j=1:width(t)
        try
            hasRow=hasRow || any(string(t{:,j})=="Z_zero",'all');
        catch
        end
    end
    assert(hasColumn || hasRow,'%s 缺少 Z_zero。',tables(i).name);
end

draft=fileread(fullfile(resultDir,'第3章完整稿.md'));
assert(contains(draft,'图3-1（作者手画，待插入）'));
for i=1:height(generated)
    assert(contains(draft,char(generated.file_stem(i)+".png")),'完整稿未引用：%s',generated.file_stem(i));
    caption="!["+generated.figure_id(i)+" "+generated.figure_eye(i)+"]";
    assert(contains(string(draft),caption),'图眼未作为图注首句：%s',generated.figure_id(i));
end
assert(~contains(draft,'figures/图3-9_') && ~contains(draft,'figures/图3-23_'));

scanFiles={fullfile(srcDir,'build_chapter3_figures_nature.m'), ...
    fullfile(resultDir,'figure_manifest_v2.csv'),fullfile(resultDir,'第3章完整稿.md'), ...
    fullfile(resultDir,'第3章实验结果初稿.md')};
for i=1:numel(scanFiles)
    txt=fileread(scanFiles{i}); low=lower(string(txt));
    assert(~contains(txt,char(8722)),'检测到 U+2212：%s',scanFiles{i});
    assert(~contains(txt,'显著') && ~contains(txt,'证明了') && ~contains(txt,'始终'), ...
        '检测到禁用表述：%s',scanFiles{i});
    assert(~contains(low,'signed-log') && ~contains(low,'symlog'), ...
        '检测到禁用坐标轴：%s',scanFiles{i});
end

fastFiles=dir(fullfile(matlabRoot,'outputs','**','*FAST_MODE*')); assert(isempty(fastFiles));
report=struct('stage','BC_figure_v2','pass',true,'manifest_rows',height(manifest), ...
    'program_figure_sets',height(generated),'figure_files_per_format',7, ...
    'formats',{{'png','pdf'}},'png_min_dpi',min(dpi),'png_max_dpi',max(dpi), ...
    'final_tables',numel(tables),'all_tables_have_Z_zero',true, ...
    'signed_log_absent',true,'figure_eye_captions_verified',true, ...
    'fast_mode_numbers_used',false,'verdict_consistent',true);
write_json(fullfile(stageDir,'verification_stage_bc.json'),report);
fprintf('Stage B/C figure-v2 verification passed: 7 PNG/PDF sets, min %.1f dpi.\n',min(dpi));
end

function dpi=png_dpi(info)
x=info.XResolution; unit=lower(string(info.ResolutionUnit));
if contains(unit,'inch'), dpi=x;
elseif contains(unit,'centimeter'), dpi=x*2.54;
elseif contains(unit,'meter'), dpi=x/39.3700787401575;
elseif x>2000, dpi=x/39.3700787401575;
else, dpi=x;
end
end

function write_json(path,v)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); cleanup=onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(v,'PrettyPrint',true),'char'); clear cleanup
end
