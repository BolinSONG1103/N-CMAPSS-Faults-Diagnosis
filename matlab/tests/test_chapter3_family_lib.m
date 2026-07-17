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

function testLockedLabelAggregationPreservesDetection(testCase)
parameterLabels=false(6,9);
parameterLabels(3:end,2)=true; parameterLabels(5:end,7)=true;
familyLabels=chapter3_family_lib.aggregate_labels(parameterLabels);
verifyEqual(testCase,familyLabels(:,2),parameterLabels(:,2));
verifyEqual(testCase,familyLabels(:,4),parameterLabels(:,7));
verifyEqual(testCase,any(familyLabels,2),any(parameterLabels,2));
verifyFalse(testCase,any(familyLabels(:,[1 3 5]),'all'));
end
