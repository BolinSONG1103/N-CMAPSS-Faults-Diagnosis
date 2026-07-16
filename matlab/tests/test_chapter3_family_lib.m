function tests=test_chapter3_family_lib
tests=functiontests(localfunctions);
end

function testFrozenFamilyMapping(testCase)
theta=zeros(4,9); theta(:,2)=-[0 .1 .2 .3].'; theta(:,3)=-[0 .2 .4 .6].';
z=chapter3_family_lib.severity(theta);
verifySize(testCase,z,[4 5]);
verifyEqual(testCase,z(:,1),zeros(4,1),'AbsTol',1e-12);
verifyEqual(testCase,z(:,2),sqrt(mean(max(0,-theta(:,2:3)).^2,2)),'AbsTol',1e-12);

y=chapter3_family_lib.truth_labels([1;0;0;0],[2 3],4);
verifyEqual(testCase,y(:,2),logical([0;1;1;1]));
verifyFalse(testCase,any(y(:,[1 3 4 5]),'all'));
end

function testCalibrationOnlyFamilyDecision(testCase)
T=50; seqs=cell(1,4);
for i=1:4
    theta=zeros(T,9); theta(16:end,2)=-linspace(.002,.5,T-15).';
    theta(16:end,3)=-linspace(.001,.3,T-15).';
    hs=true(T,1); hs(16:end)=false;
    seqs{i}=struct('theta_hat',theta,'theta_true',theta,'hs',hs,'fault_idx',[2 3]);
end
base=struct('DECISION_ALPHA',.01,'UNKNOWN_ALPHA',.01, ...
    'PERSISTENCE_GRID',[1 2 3 5],'STAGE_COMPONENTS',3,'SEED',42);
cfg=chapter3_diagnostic_lib.config(base); model=chapter3_family_lib.fit(seqs,cfg);
p=chapter3_family_lib.predict(seqs{1}.theta_hat,model);
y=struct('family_label',chapter3_family_lib.truth_labels( ...
    seqs{1}.hs,seqs{1}.fault_idx,T),'fault',~seqs{1}.hs);
m=chapter3_family_lib.sequence_metrics(p,y,(1:T).');
verifyGreaterThan(testCase,m.family_macroF1,.9);
verifyEqual(testCase,nnz(model.persistence_selection.selected),1);
verifyEqual(testCase,m.unit_false_alarm,0);
end
