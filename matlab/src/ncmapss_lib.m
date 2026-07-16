classdef ncmapss_lib
    % N-CMAPSS 第3章 MATLAB 共享库。
    %
    % 设计原则：物理进入 H 的结构；非正与单调约束通过
    % theta = -Cum*d, d>=0 恒等满足。所有实验共用的数据管线只保留此处一份。

    methods (Static)
        function cfg = config(fastMode)
            if nargin < 1
                fastMode = true;
            end
            cfg = struct();
            cfg.SEED = 42;
            cfg.N_HEALTHY_ID = 60000;
            cfg.N_DEGRADED_ID = 150000;
            cfg.MAX_PER_CYCLE = 1000;
            cfg.N_REF_CYCLES = 3;
            cfg.OP_COLS = {'TRA','Mach','theta_c','delta_c'};
            cfg.THETA9 = {'HPT_eff_mod','fan_eff_mod','fan_flow_mod', ...
                'HPC_eff_mod','HPC_flow_mod','LPT_eff_mod','LPT_flow_mod', ...
                'LPC_eff_mod','LPC_flow_mod'};
            cfg.THETA_SPAN = [0.018668,0.223446,0.121209,0.025540, ...
                0.070658,0.036194,0.032908,0.118767,0.046187];
            cfg.LAM_MAIN = 31.6;
            cfg.DATA_DIR = 'D:/N-CMPASS/data_set';
            cfg.FAST_MODE = logical(fastMode);
            cfg.SEG_Q = 0.75;
            cfg.SEG_MIN = 5;
            cfg.COND_TARGET = 1562.8;
            cfg.COND_REL_TOL = 0.01;
            cfg.T0_SKILL_TARGET = 0.957064895267337;
            cfg.T0_SKILL_ABS_TOL = 0.005;
            % MATLAB 没有逐位等价的 HistGradientBoostingRegressor。
            % 默认窄兼容层只用 MATLAB 调用已安装的 sklearn；native 可用于敏感性对照。
            cfg.BASELINE_BACKEND = 'sklearn_histgb_compat';
            cfg.RNG_BACKEND = 'numpy_pcg64';
            cfg.IDENT_FILES = { ...
                'N-CMAPSS_DS01-005.h5', {'HPT_eff_mod'}; ...
                'N-CMAPSS_DS04.h5', {'fan_eff_mod','fan_flow_mod'}; ...
                'N-CMAPSS_DS05.h5', {'HPC_eff_mod','HPC_flow_mod'}; ...
                'N-CMAPSS_DS07.h5', {'LPT_eff_mod','LPT_flow_mod'}; ...
                'N-CMAPSS_DS06.h5', {'LPC_eff_mod','LPC_flow_mod', ...
                    'HPC_eff_mod','HPC_flow_mod'}};
            cfg.VALID_FILES = { ...
                'DS02','N-CMAPSS_DS02-006.h5', ...
                    {'HPT_eff_mod','LPT_eff_mod','LPT_flow_mod'}; ...
                'DS03','N-CMAPSS_DS03-012.h5', ...
                    {'HPT_eff_mod','LPT_eff_mod','LPT_flow_mod'}};
            cfg.CORRECTED_COLS = {'Nf_c','Nc_c','Wf_c','T24_c','T30_c', ...
                'T48_c','T50_c','P15_c','P21_c','P24_c','Ps30_c','P40_c','P50_c'};
            cfg.RESID_COLS = strcat('r_',cfg.CORRECTED_COLS);
        end

        function stream = make_stream(seed,backend)
            if nargin < 1
                seed = 42;
            end
            if nargin < 2
                backend = 'matlab';
            end
            if strcmpi(backend,'numpy_pcg64')
                stream = struct('backend','numpy_pcg64', ...
                    'object',health_baseline_py('make_rng',seed));
            else
                stream = RandStream('mt19937ar','Seed',seed);
            end
        end

        function idx = randperm_stream(stream,n,k)
            if isstruct(stream) && strcmp(stream.backend,'numpy_pcg64')
                tmp = health_baseline_py('randperm',stream.object,n,k);
                idx = tmp(:)+1;
            else
                idx = randperm(stream,n,k).';
            end
        end

        function stateJson = capture_rng_state(stream)
            stateJson = '';
            if isstruct(stream) && strcmp(stream.backend,'numpy_pcg64')
                stateJson = health_baseline_py('capture_rng_state',stream.object);
            end
        end

        function restore_rng_state(stream,stateJson)
            if isstruct(stream) && strcmp(stream.backend,'numpy_pcg64') && ...
                    ~isempty(stateJson)
                health_baseline_py('restore_rng_state',stream.object,stateJson);
            end
        end

        function print_preregistered(stage,cfg)
            fprintf('\n%s\n',repmat('=',1,96));
            fprintf('预注册判据：%s（运行后不移动阈值）\n',stage);
            fprintf('%s\n',repmat('=',1,96));
            if strcmpi(stage,'T0')
                fprintf('  T0-a cond(H_n): %.1f ± %.1f%%\n', ...
                    cfg.COND_TARGET,100*cfg.COND_REL_TOL);
                fprintf('  T0-b DS03/test, N=1000, lambda=%.1f, D技能分: %.7f ± %.3f\n', ...
                    cfg.LAM_MAIN,cfg.T0_SKILL_TARGET,cfg.T0_SKILL_ABS_TOL);
            end
        end

        function names = get_names(path,key)
            raw = string(h5read(path,['/' char(key)]));
            % 固定宽度 HDF5 字符串的填充是 NUL，不是普通空格；先去 NUL 再 strip。
            raw = erase(raw,char(0));
            names = cellstr(strip(raw(:))).';
        end

        function X = h5read_rows(path,key,rowIdx)
            % HDF5 点选择：只读指定行，避免把数 GB 的 W/X_s/T 整体载入内存。
            rowIdx = double(rowIdx(:));
            info = h5info(path,['/' char(key)]);
            nCols = double(info.Dataspace.Size(1));
            X = zeros(numel(rowIdx),nCols);
            if isempty(rowIdx)
                return;
            end

            fileId = H5F.open(path,'H5F_ACC_RDONLY','H5P_DEFAULT');
            cleanerFile = onCleanup(@() H5F.close(fileId)); %#ok<NASGU>
            dsetId = H5D.open(fileId,['/' char(key)]);
            cleanerDset = onCleanup(@() H5D.close(dsetId)); %#ok<NASGU>

            % 一次选择过多坐标会占用大量临时内存；分块不改变读取顺序。
            blockRows = 20000;
            for first = 1:blockRows:numel(rowIdx)
                last = min(first+blockRows-1,numel(rowIdx));
                rr = (rowIdx(first:last)-1).'; % HDF5 使用 0 基坐标
                nr = numel(rr);
                rows = repelem(rr,nCols);
                cols = repmat(0:nCols-1,1,nr);
                coords = [rows;cols];

                fileSpace = H5D.get_space(dsetId);
                H5S.select_elements(fileSpace,'H5S_SELECT_SET',coords);
                memSpace = H5S.create_simple(2,[nr nCols],[]);
                chunk = H5D.read(dsetId,'H5ML_DEFAULT',memSpace,fileSpace,'H5P_DEFAULT');
                H5S.close(memSpace);
                H5S.close(fileSpace);
                X(first:last,:) = double(chunk).';
            end
        end

        function tbl = load_for_ident(path,cfg,stream)
            aNames = ncmapss_lib.get_names(path,'A_var');
            wNames = ncmapss_lib.get_names(path,'W_var');
            xNames = ncmapss_lib.get_names(path,'X_s_var');
            tNames = ncmapss_lib.get_names(path,'T_var');

            A = double(h5read(path,'/A_dev')).';
            hsCol = find(strcmp(aNames,'hs'),1);
            healthy = find(A(:,hsCol)==1);
            degraded = find(A(:,hsCol)==0);
            kh = min(cfg.N_HEALTHY_ID,numel(healthy));
            kd = min(cfg.N_DEGRADED_ID,numel(degraded));
            takeH = healthy(ncmapss_lib.randperm_stream(stream,numel(healthy),kh));
            takeD = degraded(ncmapss_lib.randperm_stream(stream,numel(degraded),kd));
            take = sort([takeH;takeD]);

            As = A(take,:);
            clear A
            W = ncmapss_lib.h5read_rows(path,'W_dev',take);
            Xs = ncmapss_lib.h5read_rows(path,'X_s_dev',take);
            T = ncmapss_lib.h5read_rows(path,'T_dev',take);
            tbl = array2table([As,W,Xs,T], ...
                'VariableNames',[aNames,wNames,xNames,tNames]);
        end

        function tbl = load_per_cycle(path,split,cfg,stream,onlyUnits)
            if nargin < 5
                onlyUnits = [];
            end
            aNames = ncmapss_lib.get_names(path,'A_var');
            wNames = ncmapss_lib.get_names(path,'W_var');
            xNames = ncmapss_lib.get_names(path,'X_s_var');
            tNames = ncmapss_lib.get_names(path,'T_var');

            keyA = ['/A_' char(split)];
            A = double(h5read(path,keyA)).';
            unitCol = find(strcmp(aNames,'unit'),1);
            cycleCol = find(strcmp(aNames,'cycle'),1);
            [groupId,unitKey,~] = findgroups(A(:,unitCol),A(:,cycleCol));
            groups = accumarray(groupId,(1:size(A,1)).',[],@(v){v});
            selected = cell(numel(groups),1);
            keepCount = 0;
            for g = 1:numel(groups)
                if ~isempty(onlyUnits) && ~ismember(unitKey(g),onlyUnits)
                    continue;
                end
                idx = groups{g};
                if numel(idx)>cfg.MAX_PER_CYCLE
                    idx = idx(ncmapss_lib.randperm_stream(stream,numel(idx),cfg.MAX_PER_CYCLE));
                end
                keepCount = keepCount+1;
                selected{keepCount} = idx;
            end
            if keepCount==0
                error('ncmapss:NoRows','没有找到满足条件的 unit/cycle。');
            end
            take = sort(vertcat(selected{1:keepCount}));
            As = A(take,:);
            clear A groups selected groupId

            W = ncmapss_lib.h5read_rows(path,['W_' char(split)],take);
            Xs = ncmapss_lib.h5read_rows(path,['X_s_' char(split)],take);
            T = ncmapss_lib.h5read_rows(path,['T_' char(split)],take);
            tbl = array2table([As,W,Xs,T], ...
                'VariableNames',[aNames,wNames,xNames,tNames]);
        end

        function [tbl,cols] = add_corrected(tbl,ref)
            % Walsh & Fletcher 相似修正；13 通道次序是反演矩阵的行定义。
            tbl.theta_c = tbl.T2/ref.T_ref;
            tbl.delta_c = tbl.P2/ref.P_ref;
            sq = sqrt(tbl.theta_c);
            tbl.Nf_c = tbl.Nf./sq;
            tbl.Nc_c = tbl.Nc./sq;
            tbl.Wf_c = tbl.Wf./(tbl.delta_c.*sq);
            tbl.T24_c = tbl.T24./tbl.T2;
            tbl.T30_c = tbl.T30./tbl.T2;
            tbl.T48_c = tbl.T48./tbl.T2;
            tbl.T50_c = tbl.T50./tbl.T2;
            tbl.P15_c = tbl.P15./tbl.P2;
            tbl.P21_c = tbl.P21./tbl.P2;
            tbl.P24_c = tbl.P24./tbl.P2;
            tbl.Ps30_c = tbl.Ps30./tbl.P2;
            tbl.P40_c = tbl.P40./tbl.P2;
            tbl.P50_c = tbl.P50./tbl.P2;
            cols = {'Nf_c','Nc_c','Wf_c','T24_c','T30_c','T48_c','T50_c', ...
                'P15_c','P21_c','P24_c','Ps30_c','P40_c','P50_c'};
        end

        function pipe = build_pipeline(cfg,stream,cacheFile,rebuild)
            if nargin < 2 || isempty(stream)
                stream = ncmapss_lib.make_stream(cfg.SEED,cfg.RNG_BACKEND);
            end
            if nargin < 3
                cacheFile = '';
            end
            if nargin < 4
                rebuild = false;
            end
            if ~rebuild && ~isempty(cacheFile) && isfile(cacheFile)
                saved = load(cacheFile,'pipe');
                pipe = saved.pipe;
                if isfield(pipe,'base_paths') && ~isempty(pipe.base_paths)
                    pipe.base = ncmapss_lib.load_python_models(pipe.base_paths);
                end
                if isfield(pipe,'rng_state_json')
                    ncmapss_lib.restore_rng_state(stream,pipe.rng_state_json);
                end
                fprintf('  使用 MATLAB 管线缓存: %s\n',cacheFile);
                return;
            end

            nFiles = size(cfg.IDENT_FILES,1);
            data = cell(nFiles,1);
            fprintf('\n[1/3] 选择性读取 5 个辨识子集 ...\n');
            for i = 1:nFiles
                fileName = cfg.IDENT_FILES{i,1};
                path = fullfile(cfg.DATA_DIR,fileName);
                if ~isfile(path)
                    error('ncmapss:MissingData','找不到数据文件: %s',path);
                end
                data{i} = ncmapss_lib.load_for_ident(path,cfg,stream);
                fprintf('  %-25s %7d 行\n',fileName,height(data{i}));
            end

            healthy = cell(nFiles,1);
            for i = 1:nFiles
                healthy{i} = data{i}(data{i}.hs==1,:);
            end
            pool = vertcat(healthy{:});
            ref = struct('T_ref',median(pool.T2),'P_ref',median(pool.P2));
            fprintf('  全局参考: T_ref=%.6f, P_ref=%.6f\n',ref.T_ref,ref.P_ref);
            for i = 1:nFiles
                [data{i},correctedCols] = ncmapss_lib.add_corrected(data{i},ref);
            end
            [pool,~] = ncmapss_lib.add_corrected(pool,ref);
            residCols = strcat('r_',correctedCols);

            fprintf('\n[2/3] 拟合 13 个健康基准（后端=%s）...\n',cfg.BASELINE_BACKEND);
            Xpool = table2array(pool(:,cfg.OP_COLS));
            base = cell(1,numel(correctedCols));
            basePaths = {};
            if strcmpi(cfg.BASELINE_BACKEND,'sklearn_histgb_compat')
                if isempty(cacheFile)
                    modelDir = fullfile(pwd,'sklearn_models');
                else
                    modelDir = fullfile(fileparts(cacheFile),'sklearn_models');
                end
                if ~isfolder(modelDir), mkdir(modelDir); end
                for j = 1:numel(correctedCols)
                    y = pool.(correctedCols{j});
                    base{j} = ncmapss_lib.fit_python_histgb(Xpool,y,cfg.SEED);
                    modelPath = fullfile(modelDir,sprintf('histgb_%02d.joblib',j));
                    health_baseline_py('save_model',base{j},modelPath);
                    basePaths{j} = modelPath; %#ok<AGROW>
                    fprintf('  [%02d/13] %s -> %s\n',j,correctedCols{j},modelPath);
                end
            else
                tree = templateTree('MaxNumSplits',30,'MinLeafSize',20,'Surrogate','off');
                for j = 1:numel(correctedCols)
                    y = pool.(correctedCols{j});
                    base{j} = fitrensemble(Xpool,y,'Method','LSBoost', ...
                        'NumLearningCycles',150,'LearnRate',0.1,'Learners',tree, ...
                        'NumBins',255);
                    fprintf('  [%02d/13] %s\n',j,correctedCols{j});
                end
            end
            Rpool = zeros(height(pool),numel(correctedCols));
            for j = 1:numel(correctedCols)
                Rpool(:,j) = pool.(correctedCols{j})- ...
                    ncmapss_lib.predict_model(base{j},Xpool);
            end
            residStd = std(Rpool,0,1);
            if any(~isfinite(residStd) | residStd<=0)
                error('ncmapss:BadResidualScale','健康残差标准差存在非正或非有限值。');
            end
            clear Xpool Rpool healthy

            fprintf('\n[3/3] 辨识 H 并做量程归一化 ...\n');
            H = nan(numel(correctedCols),numel(cfg.THETA9));
            primary = 1:4; % DS01/04/05/07；DS06 只补 LPC 两列
            for ii = primary
                params = cfg.IDENT_FILES{ii,2};
                df = data{ii};
                R = ncmapss_lib.residual(base,correctedCols,cfg.OP_COLS,df)./residStd;
                B = [ones(height(df),1),table2array(df(:,params))]\R;
                for k = 1:numel(params)
                    col = find(strcmp(cfg.THETA9,params{k}),1);
                    H(:,col) = B(k+1,:).';
                end
            end
            df6 = data{5};
            df6 = df6(df6.hs==0,:);
            params6 = cfg.IDENT_FILES{5,2};
            R6 = ncmapss_lib.residual(base,correctedCols,cfg.OP_COLS,df6)./residStd;
            B6 = [ones(height(df6),1),table2array(df6(:,params6))]\R6;
            for k = 1:numel(params6)
                if startsWith(params6{k},'LPC')
                    col = find(strcmp(cfg.THETA9,params6{k}),1);
                    H(:,col) = B6(k+1,:).';
                end
            end
            if any(isnan(H),'all')
                error('ncmapss:IncompleteH','H 的 9 个部件列未全部辨识。');
            end
            Hn = H.*cfg.THETA_SPAN;
            condHn = cond(Hn);
            fprintf('  cond(H_n) = %.6f\n',condHn);

            rngStateJson = ncmapss_lib.capture_rng_state(stream);
            pipe = struct('base',{base},'base_paths',{basePaths},'ref',ref, ...
                'corrected_cols',{correctedCols}, ...
                'resid_cols',{residCols},'resid_std',residStd,'H',H,'Hn',Hn, ...
                'cond_Hn',condHn,'span',cfg.THETA_SPAN,'cfg',cfg, ...
                'baseline_backend',cfg.BASELINE_BACKEND, ...
                'rng_state_json',rngStateJson);
            if ~isempty(cacheFile)
                cacheDir = fileparts(cacheFile);
                if ~isempty(cacheDir) && ~isfolder(cacheDir)
                    mkdir(cacheDir);
                end
                pipeToSave = pipe;
                if strcmpi(cfg.BASELINE_BACKEND,'sklearn_histgb_compat') && ...
                        isfield(pipeToSave,'base')
                    pipeToSave.base = {};
                end
                pipe = pipeToSave; %#ok<NASGU>
                save(cacheFile,'pipe','-v7.3');
                pipe = pipeToSave;
                pipe.base = base;
                fprintf('  管线缓存已保存: %s\n',cacheFile);
            end
        end

        function R = residual(base,correctedCols,opCols,tbl)
            X = table2array(tbl(:,opCols));
            R = zeros(height(tbl),numel(correctedCols));
            for j = 1:numel(correctedCols)
                R(:,j) = tbl.(correctedCols{j})- ...
                    ncmapss_lib.predict_model(base{j},X);
            end
        end

        function model = fit_python_histgb(X,y,seed)
            model = health_baseline_py('fit_histgb',X,y,seed);
        end

        function models = load_python_models(paths)
            models = health_baseline_py('load_models',paths);
        end

        function yhat = predict_model(model,X)
            if health_baseline_py('is_python_model',model)
                yhat = health_baseline_py('predict',model,X);
            else
                yhat = predict(model,X);
                yhat = double(yhat(:));
            end
        end

        function tbl = attach_residuals(pipe,tbl)
            R = ncmapss_lib.residual(pipe.base,pipe.corrected_cols, ...
                pipe.cfg.OP_COLS,tbl)./pipe.resid_std;
            for j = 1:numel(pipe.resid_cols)
                tbl.(pipe.resid_cols{j}) = R(:,j);
            end
        end

        function [wins,fallbackCount] = build_windows(tbl,unitValue,modes,windows,cfg,stream)
            if ischar(modes) || isstring(modes)
                modes = cellstr(modes);
            end
            du = tbl(tbl.unit==unitValue,:);
            cycles = sort(unique(du.cycle));
            wins = struct('mode',{},'N',{},'R',{},'TH',{},'OPS',{});
            fallbackCount = struct('full',0,'highTRA',0);
            outIdx = 0;
            for im = 1:numel(modes)
                mode = char(modes{im});
                segmentRows = cell(numel(cycles),1);
                for t = 1:numel(cycles)
                    idx = find(du.cycle==cycles(t));
                    if strcmp(mode,'highTRA')
                        q = prctile(du.TRA(idx),100*cfg.SEG_Q);
                        seg = idx(du.TRA(idx)>=q);
                        if numel(seg)<cfg.SEG_MIN
                            seg = idx;
                            fallbackCount.highTRA = fallbackCount.highTRA+1;
                        end
                        segmentRows{t} = seg;
                    else
                        segmentRows{t} = idx;
                    end
                end

                refRows = vertcat(segmentRows{1:min(cfg.N_REF_CYCLES,numel(cycles))});
                bUnit = mean(table2array(du(refRows,pipe_cols(cfg,'resid'))),1);
                th0 = mean(table2array(du(refRows,cfg.THETA9))./cfg.THETA_SPAN,1);

                for iw = 1:numel(windows)
                    N = windows(iw);
                    R = zeros(numel(cycles),numel(cfg.RESID_COLS));
                    TH = zeros(numel(cycles),numel(cfg.THETA9));
                    OPS = cell(numel(cycles),1);
                    for t = 1:numel(cycles)
                        idx = segmentRows{t};
                        k = min(N,numel(idx));
                        pick = idx(ncmapss_lib.randperm_stream(stream,numel(idx),k));
                        R(t,:) = mean(table2array(du(pick,cfg.RESID_COLS)),1)-bUnit;
                        TH(t,:) = table2array(du(idx(1),cfg.THETA9))./cfg.THETA_SPAN-th0;
                        OPS{t} = table2array(du(pick,cfg.OP_COLS));
                    end
                    outIdx = outIdx+1;
                    wins(outIdx) = struct('mode',mode,'N',N,'R',R,'TH',TH,'OPS',{OPS});
                end
            end

            function cols = pipe_cols(localCfg,kind)
                if strcmp(kind,'resid')
                    cols = localCfg.RESID_COLS;
                else
                    cols = {};
                end
            end
        end

        function Cum = build_cum(T,n)
            Cum = kron(tril(ones(T)),eye(n));
        end

        function M = build_M(Hn,T)
            M = kron(tril(ones(T)),-Hn);
        end

        function M = build_M_var(HnList,T)
            [m,n] = size(HnList{1});
            M = zeros(T*m,T*n);
            for t = 1:T
                rows = (t-1)*m+(1:m);
                for i = 1:t
                    cols = (i-1)*n+(1:n);
                    M(rows,cols) = -HnList{t}; % 下标必须是 t，不是 i
                end
            end
        end

        function theta = solve_B(Hn,R,lambda)
            n = size(Hn,2);
            Binv = (Hn.'*Hn+lambda*eye(n))\Hn.';
            theta = R*Binv.';
        end

        function theta = solve_D(M,Cum,R,lambda,T,n)
            Maug = [M;sqrt(lambda)*Cum];
            yaug = [reshape(R.',[],1);zeros(T*n,1)];
            opts = optimset('Display','off','MaxIter',50*T*n);
            d = lsqnonneg(Maug,yaug,opts);
            theta = -cumsum(reshape(d,n,T).',1);
        end

        function theta = solve_D_var(M,Cum,R,lambda,T,n)
            theta = ncmapss_lib.solve_D(M,Cum,R,lambda,T,n);
        end

        function metrics = compute_metrics(thetaHat,thetaTrue,trueFaultIdx)
            mseZero = mean(thetaTrue.^2,'all');
            err = thetaHat-thetaTrue;
            mse = mean(err.^2,'all');
            T = size(thetaTrue,1);
            nEarly = max(1,floor(0.3*T));
            other = setdiff(1:size(thetaTrue,2),trueFaultIdx);
            corrs = [];
            for i = trueFaultIdx(:).'
                if std(thetaTrue(:,i))>1e-9 && std(thetaHat(:,i))>1e-9
                    C = corrcoef(thetaHat(:,i),thetaTrue(:,i));
                    corrs(end+1) = C(1,2); %#ok<AGROW>
                end
            end
            if isempty(corrs)
                detect = 0;
            else
                detect = mean(corrs);
            end
            metrics = struct('skill',1-mse/max(mseZero,1e-12), ...
                'rmse',sqrt(mse),'rmse_early',sqrt(mean(err(1:nEarly,:).^2,'all')), ...
                'false_alarm',mean(abs(thetaHat(:,other)),'all'), ...
                'detect_corr',detect,'mse_zero',mseZero);
        end

        function set_plot_defaults()
            set(groot,'defaultAxesFontName','Microsoft YaHei');
            set(groot,'defaultTextFontName','Microsoft YaHei');
            set(groot,'defaultAxesTickLabelInterpreter','none');
            set(groot,'defaultTextInterpreter','none');
            set(groot,'defaultLegendInterpreter','none');
        end
    end
end
