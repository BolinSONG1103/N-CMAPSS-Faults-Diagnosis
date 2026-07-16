classdef chapter3_diagnostic_lib
    %CHAPTER3_DIAGNOSTIC_LIB 连续健康参数后的诊断闭环。
    % 阈值、持续周期数、退化等级与拒识阈值只允许在 calibration 单元拟合。

    methods (Static)
        function cfg = config(baseCfg)
            cfg = struct();
            cfg.alpha = baseCfg.DECISION_ALPHA;
            cfg.unknown_alpha = baseCfg.UNKNOWN_ALPHA;
            cfg.persistence_grid = baseCfg.PERSISTENCE_GRID;
            cfg.max_calibration_far = 0.05;
            cfg.truth_active_eps = 1e-4;
            cfg.stage_components = baseCfg.STAGE_COMPONENTS;
            cfg.stage_component_sensitivity = [2,3,4];
            cfg.stage_samples_per_unit = 200;
            cfg.threshold_samples_per_unit = 200;
            cfg.gmm_replicates = 20;
            cfg.seed = baseCfg.SEED+2100;
        end

        function model = fit(calibrationSeqs,cfg,Hn)
            assert(~isempty(calibrationSeqs),'calibration sequences 不能为空。');
            n = size(calibrationSeqs{1}.theta_hat,2);
            healthy = zeros(0,n); trueSeverity = zeros(0,1);
            qKnown = zeros(0,1);
            for i = 1:numel(calibrationSeqs)
                s = calibrationSeqs{i};
                sev = max(0,-s.theta_hat);
                h=sev(logical(s.hs),:);
                if size(h,1)>cfg.threshold_samples_per_unit
                    pick=round(linspace(1,size(h,1),cfg.threshold_samples_per_unit)); h=h(pick,:);
                end
                healthy = [healthy;h]; %#ok<AGROW>
                active = max(0,-s.theta_true); values=active(active>cfg.truth_active_eps);
                if numel(values)>cfg.stage_samples_per_unit
                    pick=round(linspace(1,numel(values),cfg.stage_samples_per_unit));
                    values=values(pick);
                end
                trueSeverity = [trueSeverity;values(:)]; %#ok<AGROW>
                qKnown = [qKnown;chapter3_diagnostic_lib.reconstruction_score( ...
                    s.R,s.theta_hat,Hn)]; %#ok<AGROW>
            end
            assert(~isempty(healthy),'calibration 中没有健康周期。');
            assert(~isempty(trueSeverity),'calibration 中没有活动退化真值。');

            tau = zeros(1,n);
            for j = 1:n
                tau(j) = prctile(healthy(:,j),100*(1-cfg.alpha));
            end
            [bestK,selection] = chapter3_diagnostic_lib.select_persistence( ...
                calibrationSeqs,tau,cfg);
            stage = chapter3_diagnostic_lib.fit_stage_gmm( ...
                trueSeverity,cfg.stage_components,cfg.gmm_replicates,cfg.seed);
            unknownTau = prctile(qKnown,100*(1-cfg.unknown_alpha));
            model = struct('component_threshold',tau,'persistence',bestK, ...
                'persistence_selection',selection,'stage',stage, ...
                'unknown_threshold',unknownTau,'truth_active_eps',cfg.truth_active_eps, ...
                'alpha',cfg.alpha,'unknown_alpha',cfg.unknown_alpha, ...
                'fit_units',numel(calibrationSeqs));
        end

        function [bestK,selection] = select_persistence(seqs,tau,cfg)
            Kgrid = cfg.persistence_grid(:);
            far = zeros(numel(Kgrid),1); macroF1 = far;
            for ik = 1:numel(Kgrid)
                fars = zeros(numel(seqs),1); f1s = fars;
                for i = 1:numel(seqs)
                    s = seqs{i};
                    labels = chapter3_diagnostic_lib.threshold_labels( ...
                        max(0,-s.theta_hat),tau,Kgrid(ik));
                    truth = false(size(s.theta_true));
                    truth(~logical(s.hs),s.fault_idx)=true;
                    healthy = logical(s.hs);
                    if any(healthy)
                        fars(i) = mean(any(labels(healthy,:),2));
                    end
                    f1s(i) = chapter3_diagnostic_lib.macro_f1(labels,truth);
                end
                far(ik) = mean(fars); macroF1(ik) = mean(f1s);
            end
            feasible = far<=cfg.max_calibration_far;
            if any(feasible)
                candidates = find(feasible);
                target = max(macroF1(candidates));
                candidates = candidates(abs(macroF1(candidates)-target)<1e-12);
            else
                target = min(far);
                candidates = find(abs(far-target)<1e-12);
                target = max(macroF1(candidates));
                candidates = candidates(abs(macroF1(candidates)-target)<1e-12);
            end
            [~,ii] = max(Kgrid(candidates));
            pick = candidates(ii); bestK = Kgrid(pick);
            selection = table(Kgrid,far,macroF1,Kgrid==bestK, ...
                'VariableNames',{'K','unit_balanced_FAR','unit_balanced_macroF1','selected'});
        end

        function labels = threshold_labels(severity,tau,K)
            above = severity>tau;
            labels = false(size(above));
            for j = 1:size(above,2)
                count = 0; latched = false;
                for t = 1:size(above,1)
                    if above(t,j), count=count+1; else, count=0; end
                    if count>=K, latched=true; end
                    labels(t,j)=latched;
                end
            end
        end

        function out = predict(thetaHat,R,Hn,model)
            severity = max(0,-thetaHat);
            labels = chapter3_diagnostic_lib.threshold_labels( ...
                severity,model.component_threshold,model.persistence);
            componentStage = zeros(size(severity));
            for j = 1:size(severity,2)
                active = labels(:,j);
                if any(active)
                    componentStage(active,j) = chapter3_diagnostic_lib.stage_predict( ...
                        severity(active,j),model.stage);
                    componentStage(:,j)=cummax(componentStage(:,j));
                end
            end
            q = chapter3_diagnostic_lib.reconstruction_score(R,thetaHat,Hn);
            out = struct('severity',severity,'component_label',labels, ...
                'fault',any(labels,2),'component_stage',componentStage, ...
                'stage',max(componentStage,[],2),'reconstruction_score',q, ...
                'unknown',q>model.unknown_threshold);
        end

        function out = truth(thetaTrue,hs,stageModel,epsActive,faultIdx)
            severity = max(0,-thetaTrue);
            if nargin<5 || isempty(faultIdx)
                labels = severity>epsActive;
            else
                labels=false(size(thetaTrue)); labels(~logical(hs),faultIdx)=true;
            end
            componentStage = zeros(size(severity));
            for j = 1:size(severity,2)
                active=labels(:,j);
                if any(active)
                    componentStage(active,j)=chapter3_diagnostic_lib.stage_predict( ...
                        severity(active,j),stageModel);
                    componentStage(:,j)=cummax(componentStage(:,j));
                end
            end
            out = struct('severity',severity,'component_label',labels, ...
                'fault',~logical(hs),'component_stage',componentStage, ...
                'stage',max(componentStage,[],2));
        end

        function stage = fit_stage_gmm(x,K,replicates,seed)
            x = double(x(:)); x = x(isfinite(x) & x>0);
            if numel(x)<max(30,10*K)
                error('chapter3:TooFewStageSamples','退化等级 GMM 样本不足。');
            end
            old = rng; cleanup = onCleanup(@() rng(old)); %#ok<NASGU>
            rng(seed,'twister');
            opts = statset('MaxIter',1000,'Display','off');
            gm = fitgmdist(x,K,'RegularizationValue',1e-8, ...
                'Replicates',replicates,'Options',opts);
            mu = gm.mu(:); sigma = sqrt(reshape(gm.Sigma,[],1));
            weight = gm.ComponentProportion(:);
            [mu,order] = sort(mu);
            stage = struct('K',K,'mu',mu,'sigma',sigma(order), ...
                'weight',weight(order),'AIC',gm.AIC,'BIC',gm.BIC, ...
                'n_fit',numel(x),'definition','ordered_1d_GMM_on_calibration_truth');
        end

        function labels = stage_predict(x,stage)
            shape = size(x); z = double(x(:));
            p = zeros(numel(z),stage.K);
            for k = 1:stage.K
                sd = max(stage.sigma(k),sqrt(eps));
                p(:,k) = stage.weight(k)./sd.*exp(-0.5*((z-stage.mu(k))/sd).^2);
            end
            [~,labels] = max(p,[],2);
            labels = reshape(labels,shape);
        end

        function q = reconstruction_score(R,thetaHat,Hn)
            mismatch = R-thetaHat*Hn.';
            q = sqrt(mean(mismatch.^2,2));
        end

        function m = sequence_metrics(pred,truth,cycles)
            p = pred.component_label; y = truth.component_label;
            tp = sum(p & y,'all'); fp = sum(p & ~y,'all'); fn = sum(~p & y,'all');
            precision = tp/max(tp+fp,1); recall = tp/max(tp+fn,1);
            microF1 = 2*precision*recall/max(precision+recall,eps);
            detection = chapter3_diagnostic_lib.binary_metrics(pred.fault,truth.fault);
            delay = NaN; tTrue = find(truth.fault,1); tPred = find(pred.fault,1);
            if ~isempty(tTrue) && ~isempty(tPred), delay=cycles(tPred)-cycles(tTrue); end
            [kappa,stageMae] = chapter3_diagnostic_lib.ordinal_metrics(pred.stage,truth.stage);
            m = struct('detection_FAR',detection.FAR,'detection_DR',detection.DR, ...
                'detection_precision',detection.precision,'detection_F1',detection.F1, ...
                'detection_delay',delay,'isolation_microF1',microF1, ...
                'isolation_macroF1',chapter3_diagnostic_lib.macro_f1(p,y), ...
                'hamming_loss',mean(p~=y,'all'),'stage_weighted_kappa',kappa, ...
                'stage_MAE',stageMae,'stage_reversals',sum(diff(pred.stage)<0), ...
                'unknown_rate',mean(pred.unknown));
        end

        function m = binary_metrics(pred,truth)
            pred=logical(pred(:)); truth=logical(truth(:));
            tp=sum(pred&truth); tn=sum(~pred&~truth); fp=sum(pred&~truth); fn=sum(~pred&truth);
            precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1);
            m=struct('FAR',fp/max(fp+tn,1),'DR',recall,'precision',precision, ...
                'F1',2*precision*recall/max(precision+recall,eps), ...
                'tp',tp,'tn',tn,'fp',fp,'fn',fn);
        end

        function f = macro_f1(pred,truth)
            include = any(pred,1)|any(truth,1);
            if ~any(include), f=1; return; end
            idx = find(include); vals = zeros(1,numel(idx));
            for k = 1:numel(idx)
                j=idx(k);
                m = chapter3_diagnostic_lib.binary_metrics(pred(:,j),truth(:,j));
                vals(k)=m.F1;
            end
            f = mean(vals);
        end

        function [kappa,mae] = ordinal_metrics(pred,truth)
            pred=double(pred(:)); truth=double(truth(:)); mae=mean(abs(pred-truth));
            K=max([pred;truth])+1; C=zeros(K,K);
            for i=1:numel(pred), C(truth(i)+1,pred(i)+1)=C(truth(i)+1,pred(i)+1)+1; end
            n=sum(C,'all'); if n==0, kappa=NaN; return; end
            W=zeros(K,K);
            for i=1:K, for j=1:K, W(i,j)=((i-j)/max(K-1,1))^2; end, end
            O=C/n; E=sum(C,2)*sum(C,1)/n^2;
            kappa=1-sum(W.*O,'all')/max(sum(W.*E,'all'),eps);
        end

        function [aucRoc,aucPr] = binary_auc(score,label)
            score=double(score(:)); label=logical(label(:));
            [score,order]=sort(score,'descend'); label=label(order); %#ok<ASGLU>
            P=sum(label); N=sum(~label);
            tp=cumsum(label); fp=cumsum(~label);
            tpr=[0;tp/max(P,1);1]; fpr=[0;fp/max(N,1);1];
            aucRoc=trapz(fpr,tpr);
            precision=tp./max(tp+fp,1); recall=tp/max(P,1);
            aucPr=trapz([0;recall],[1;precision]);
        end
    end
end
