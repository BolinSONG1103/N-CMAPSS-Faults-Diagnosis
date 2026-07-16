classdef chapter3_family_lib
    %CHAPTER3_FAMILY_LIB 将九维连续健康参数映射为五个物理部件族诊断。

    methods (Static)
        function names=names()
            names=["HPT","Fan","HPC","LPT","LPC"];
        end

        function groups=groups()
            groups={1,[2 3],[4 5],[6 7],[8 9]};
        end

        function severity=severity(theta)
            z=max(0,-double(theta)); groups=chapter3_family_lib.groups();
            assert(size(z,2)==9,'部件族映射要求九维健康参数。');
            severity=zeros(size(z,1),numel(groups));
            for g=1:numel(groups)
                severity(:,g)=sqrt(mean(z(:,groups{g}).^2,2));
            end
        end

        function labels=truth_labels(hs,faultIdx,nCycles)
            groups=chapter3_family_lib.groups(); labels=false(nCycles,numel(groups));
            degraded=~logical(hs(:));
            for g=1:numel(groups)
                if ~isempty(intersect(faultIdx,groups{g}))
                    labels(degraded,g)=true;
                end
            end
        end

        function model=fit(seqs,cfg)
            healthy=zeros(0,5);
            for i=1:numel(seqs)
                s=seqs{i}; z=chapter3_family_lib.severity(s.theta_hat);
                h=z(logical(s.hs),:);
                if size(h,1)>cfg.threshold_samples_per_unit
                    pick=round(linspace(1,size(h,1),cfg.threshold_samples_per_unit));
                    h=h(pick,:);
                end
                healthy=[healthy;h]; %#ok<AGROW>
            end
            assert(~isempty(healthy),'calibration 中没有健康周期。');
            tau=zeros(1,5);
            for g=1:5, tau(g)=prctile(healthy(:,g),100*(1-cfg.alpha)); end
            [bestK,selection]=chapter3_family_lib.select_persistence(seqs,tau,cfg);
            model=struct('family_names',chapter3_family_lib.names(), ...
                'family_groups',{chapter3_family_lib.groups()}, ...
                'family_threshold',tau,'persistence',bestK, ...
                'persistence_selection',selection,'alpha',cfg.alpha, ...
                'fit_units',numel(seqs),'severity_definition','RMS_of_normalized_parameters');
        end

        function [bestK,selection]=select_persistence(seqs,tau,cfg)
            grid=cfg.persistence_grid(:); far=zeros(numel(grid),1); macroF1=far;
            for ik=1:numel(grid)
                unitFar=zeros(numel(seqs),1); unitF1=unitFar;
                for i=1:numel(seqs)
                    s=seqs{i}; p=chapter3_diagnostic_lib.threshold_labels( ...
                        chapter3_family_lib.severity(s.theta_hat),tau,grid(ik));
                    y=chapter3_family_lib.truth_labels(s.hs,s.fault_idx,size(p,1));
                    h=logical(s.hs);
                    if any(h), unitFar(i)=mean(any(p(h,:),2)); end
                    unitF1(i)=chapter3_diagnostic_lib.macro_f1(p,y);
                end
                far(ik)=mean(unitFar); macroF1(ik)=mean(unitF1);
            end
            feasible=far<=cfg.max_calibration_far;
            if any(feasible)
                candidates=find(feasible); target=max(macroF1(candidates));
                candidates=candidates(abs(macroF1(candidates)-target)<1e-12);
            else
                target=min(far); candidates=find(abs(far-target)<1e-12);
                target=max(macroF1(candidates));
                candidates=candidates(abs(macroF1(candidates)-target)<1e-12);
            end
            [~,ii]=max(grid(candidates)); pick=candidates(ii); bestK=grid(pick);
            selection=table(grid,far,macroF1,grid==bestK,'VariableNames', ...
                {'K','unit_balanced_FAR','unit_balanced_macroF1','selected'});
        end

        function out=predict(thetaHat,model)
            severity=chapter3_family_lib.severity(thetaHat);
            labels=chapter3_diagnostic_lib.threshold_labels( ...
                severity,model.family_threshold,model.persistence);
            out=struct('severity',severity,'family_label',labels,'fault',any(labels,2));
        end

        function m=sequence_metrics(pred,truth,cycles)
            p=pred.family_label; y=truth.family_label;
            tp=sum(p&y,'all'); fp=sum(p&~y,'all'); fn=sum(~p&y,'all');
            precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1);
            detection=chapter3_diagnostic_lib.binary_metrics(pred.fault,truth.fault);
            tTrue=find(truth.fault,1); tPred=find(pred.fault,1); delay=NaN;
            if ~isempty(tTrue)&&~isempty(tPred), delay=cycles(tPred)-cycles(tTrue); end
            healthy=~truth.fault; falseIdx=find(pred.fault&healthy,1);
            falseAlarmEvent=~isempty(falseIdx); preOnsetLead=NaN;
            if falseAlarmEvent&&~isempty(tTrue), preOnsetLead=cycles(tTrue)-cycles(falseIdx); end
            m=struct('detection_FAR',detection.FAR,'detection_DR',detection.DR, ...
                'detection_precision',detection.precision,'detection_F1',detection.F1, ...
                'detection_delay',delay,'unit_false_alarm',double(falseAlarmEvent), ...
                'pre_onset_lead',preOnsetLead,'family_microF1', ...
                2*precision*recall/max(precision+recall,eps), ...
                'family_macroF1',chapter3_diagnostic_lib.macro_f1(p,y), ...
                'family_hamming_loss',mean(p~=y,'all'));
        end
    end
end
