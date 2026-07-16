function tests=test_chapter3_diagnostic_lib
tests=functiontests(localfunctions);
end

function testThresholdPersistenceAndStages(testCase)
rng(7); n=9; H=randn(13,n); seqs=cell(1,4);
for i=1:4
    T=60; theta=zeros(T,n); theta(16:end,1)=-linspace(.001,.8,T-15).';
    theta(26:end,6)=-linspace(.001,.6,T-25).';
    thetaHat=theta+.002*randn(T,n); thetaHat=min(thetaHat,0);
    thetaHat=-cummax(max(0,-thetaHat),1);
    hs=true(T,1); hs(16:end)=false; R=thetaHat*H.'+.01*randn(T,13);
    seqs{i}=struct('theta_hat',thetaHat,'theta_true',theta,'hs',hs,'R',R, ...
        'fault_idx',[1 6]);
end
base=struct('DECISION_ALPHA',.01,'UNKNOWN_ALPHA',.01, ...
    'PERSISTENCE_GRID',[1 2 3 5],'STAGE_COMPONENTS',3,'SEED',42);
cfg=chapter3_diagnostic_lib.config(base); model=chapter3_diagnostic_lib.fit(seqs,cfg,H);
p=chapter3_diagnostic_lib.predict(seqs{1}.theta_hat,seqs{1}.R,H,model);
y=chapter3_diagnostic_lib.truth(seqs{1}.theta_true,seqs{1}.hs,model.stage, ...
    model.truth_active_eps,seqs{1}.fault_idx);
verifySize(testCase,p.component_label,[60 9]);
verifyTrue(testCase,all(diff(p.stage)>=0));
m=chapter3_diagnostic_lib.sequence_metrics(p,y,(1:60).');
verifyGreaterThan(testCase,m.detection_DR,.8);
end

function testBinaryAuc(testCase)
[a,b]=chapter3_diagnostic_lib.binary_auc([.1 .2 .8 .9],[0 0 1 1]);
verifyEqual(testCase,a,1,'AbsTol',1e-12); verifyGreaterThan(testCase,b,.9);
end
