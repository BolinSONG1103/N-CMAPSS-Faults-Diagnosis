classdef chapter3_diagnostic_lib_v3
    %CHAPTER3_DIAGNOSTIC_LIB_V3 改进版诊断决策层（不覆盖 v2）。
    %
    % 相对 v2 的三项改进，均只作用于决策层、不改变约束反演，参数只在
    % calibration 单元选择，测试真值仅用于算指标：
    %   1) 单元自适应阈值：用每台发动机自身前 w 个（健康）循环估计逐部件
    %      基线偏移与稳健尺度，把检测阈值抬到该机健康噪声之上，吸收分布外
    %      健康期漂移导致的虚警（不放大分布内召回的损失）。阈值以 v2 全局
    %      标定阈值为下限。
    %   2) 因果时序中值平滑：严重度经只用过去信息的中值滤波后再判阈与定级，
    %      抑制逐循环反演抖动，提升噪声条件下的隔离与阶段鲁棒性。
    %   3) 参考窗口 w、尺度裕度 m、平滑窗 L 与持续周期 K 联合在 calibration
    %      上选择（先满足 FAR<=5%，再最大化检测+隔离综合分）。
    %
    % 工具方法（阈值锁存、GMM、指标等）复用 chapter3_diagnostic_lib，保证
    % 与 v2 口径一致，只替换阈值与平滑逻辑。

    methods (Static)
        function cfg = config(baseCfg)
            cfg = chapter3_diagnostic_lib.config(baseCfg);
            cfg.ref_window_grid = [3,5,8];      % 每机健康参考窗口候选
            cfg.scale_margin_grid = [0,2,4];    % 稳健尺度裕度候选
            cfg.smooth_grid = [1,3,5];          % 因果中值平滑窗候选（1=不平滑）
        end

        function model = fit(calibrationSeqs, cfg, Hn)
            assert(~isempty(calibrationSeqs),'calibration sequences 不能为空。');
            n = size(calibrationSeqs{1}.theta_hat,2);
            healthy = zeros(0,n); trueSeverity = zeros(0,1); qKnown = zeros(0,1);
            for i = 1:numel(calibrationSeqs)
                s = calibrationSeqs{i};
                sev = max(0,-s.theta_hat);
                h = sev(logical(s.hs),:);
                if size(h,1)>cfg.threshold_samples_per_unit
                    pick = round(linspace(1,size(h,1),cfg.threshold_samples_per_unit)); h=h(pick,:);
                end
                healthy = [healthy;h]; %#ok<AGROW>
                active = max(0,-s.theta_true); values = active(active>cfg.truth_active_eps);
                if numel(values)>cfg.stage_samples_per_unit
                    pick = round(linspace(1,numel(values),cfg.stage_samples_per_unit)); values=values(pick);
                end
                trueSeverity = [trueSeverity;values(:)]; %#ok<AGROW>
                qKnown = [qKnown;chapter3_diagnostic_lib.reconstruction_score(s.R,s.theta_hat,Hn)]; %#ok<AGROW>
            end
            assert(~isempty(healthy),'calibration 中没有健康周期。');
            assert(~isempty(trueSeverity),'calibration 中没有活动退化真值。');

            tauGlobal = zeros(1,n);
            for j = 1:n
                tauGlobal(j) = prctile(healthy(:,j),100*(1-cfg.alpha));
            end
            stage = chapter3_diagnostic_lib.fit_stage_gmm( ...
                trueSeverity,cfg.stage_components,cfg.gmm_replicates,cfg.seed);
            unknownTau = prctile(qKnown,100*(1-cfg.unknown_alpha));

            sel = chapter3_diagnostic_lib_v3.select_hyperparams(calibrationSeqs,tauGlobal,cfg);
            model = struct('component_threshold',tauGlobal,'persistence',sel.K, ...
                'ref_window',sel.w,'scale_margin',sel.m,'smooth_window',sel.L, ...
                'selection',sel.table,'stage',stage,'unknown_threshold',unknownTau, ...
                'truth_active_eps',cfg.truth_active_eps,'alpha',cfg.alpha, ...
                'unknown_alpha',cfg.unknown_alpha,'fit_units',numel(calibrationSeqs), ...
                'variant','v3_adaptive_smoothed');
        end

        function sel = select_hyperparams(seqs, tauGlobal, cfg)
            % 联合选择 (w,m,L,K)：先 FAR<=max_far，再最大化 0.5*detF1+0.5*macroF1。
            rows = cell(0,1); ri = 0;
            for w = cfg.ref_window_grid
                for m = cfg.scale_margin_grid
                    for L = cfg.smooth_grid
                        for K = cfg.persistence_grid(:).'
                            fars = zeros(numel(seqs),1); detF1 = fars; macroF1 = fars;
                            for i = 1:numel(seqs)
                                s = seqs{i};
                                lab = chapter3_diagnostic_lib_v3.adaptive_labels( ...
                                    s.theta_hat,tauGlobal,w,m,L,K);
                                truth = false(size(s.theta_true));
                                truth(~logical(s.hs),s.fault_idx) = true;
                                hs = logical(s.hs);
                                if any(hs), fars(i) = mean(any(lab(hs,:),2)); end
                                fault = any(lab,2);
                                bm = chapter3_diagnostic_lib.binary_metrics(fault,~hs);
                                detF1(i) = bm.F1;
                                macroF1(i) = chapter3_diagnostic_lib.macro_f1(lab,truth);
                            end
                            ri = ri+1;
                            rows{ri} = [w,m,L,K,mean(fars),mean(detF1),mean(macroF1)]; %#ok<AGROW>
                        end
                    end
                end
            end
            G = vertcat(rows{:});
            far = G(:,5); score = 0.5*G(:,6)+0.5*G(:,7);
            feasible = far<=cfg.max_calibration_far;
            if any(feasible), idxPool = find(feasible); else, idxPool = find(far==min(far)); end
            [~,ii] = max(score(idxPool)); pick = idxPool(ii);
            sel = struct('w',G(pick,1),'m',G(pick,2),'L',G(pick,3),'K',G(pick,4), ...
                'table',array2table(G,'VariableNames', ...
                {'w','m','L','K','FAR','detF1','macroF1'}));
        end

        function tau = adaptive_tau(sev, tauGlobal, w, m)
            % 逐部件自适应阈值：max(全局阈值, 早期健康基线 + m*稳健尺度)。
            T = size(sev,1); ww = min(w,T); early = sev(1:ww,:);
            b = median(early,1);
            sig = 1.4826*median(abs(early-b),1);
            tau = max(tauGlobal, b + m*sig);
        end

        function sevs = causal_smooth(sev, L)
            if L<=1, sevs = sev; return; end
            sevs = movmedian(sev,[L-1 0],1);   % 只用过去 L-1 个循环 + 当前
        end

        function labels = adaptive_labels(thetaHat, tauGlobal, w, m, L, K)
            sev = max(0,-thetaHat);
            tau = chapter3_diagnostic_lib_v3.adaptive_tau(sev,tauGlobal,w,m);
            sevs = chapter3_diagnostic_lib_v3.causal_smooth(sev,L);
            labels = chapter3_diagnostic_lib.threshold_labels(sevs,tau,K);
        end

        function out = predict(thetaHat, R, Hn, model)
            sev = max(0,-thetaHat);
            sevs = chapter3_diagnostic_lib_v3.causal_smooth(sev,model.smooth_window);
            tau = chapter3_diagnostic_lib_v3.adaptive_tau( ...
                sev,model.component_threshold,model.ref_window,model.scale_margin);
            labels = chapter3_diagnostic_lib.threshold_labels(sevs,tau,model.persistence);
            componentStage = zeros(size(sev));
            for j = 1:size(sev,2)
                active = labels(:,j);
                if any(active)
                    componentStage(active,j) = chapter3_diagnostic_lib.stage_predict( ...
                        sevs(active,j),model.stage);
                    componentStage(:,j) = cummax(componentStage(:,j));
                end
            end
            q = chapter3_diagnostic_lib.reconstruction_score(R,thetaHat,Hn);
            out = struct('severity',sev,'component_label',labels, ...
                'fault',any(labels,2),'component_stage',componentStage, ...
                'stage',max(componentStage,[],2),'reconstruction_score',q, ...
                'unknown',q>model.unknown_threshold);
        end
    end
end
