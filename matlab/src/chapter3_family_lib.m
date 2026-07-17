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

        function labels=aggregate_labels(parameterLabels)
            groups=chapter3_family_lib.groups();
            assert(size(parameterLabels,2)==9,'标签聚合要求九维参数标签。');
            labels=false(size(parameterLabels,1),numel(groups));
            for g=1:numel(groups)
                labels(:,g)=any(logical(parameterLabels(:,groups{g})),2);
            end
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
