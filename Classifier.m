function [ posGroup, negGroup, posGroupLabel, negGroupLabel, accuracy , error] = Classifier( te_feats, te_label, pos, neg , weight)
%CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here

pos_and_neg = [pos neg];

temp_feats = [];
temp_label = [];

for i = 1 : size(pos_and_neg,2)
    temp_feats = [temp_feats ; te_feats(find( te_label == pos_and_neg(i)) , : )];
    temp_label = [temp_label ; te_label(find( te_label == pos_and_neg(i)) , : )];
end

%AccuracyY
accuracyY = zeros( size( temp_feats, 1 ), 1 );

%Positive Group's answer is 1
positiveCounts = 0;
for i=1:size(pos, 2)
    positiveCounts = positiveCounts + size( find( te_label == pos(i) ) , 1);        
end
accuracyY(1:positiveCounts) = 1;
%Negative Group's answer is -1
negativeCounts = 0;
for i=1:size(neg, 2)
    negativeCounts = negativeCounts + size( find( te_label == neg(i) ) , 1);        
end
accuracyY( positiveCounts+1 : positiveCounts + negativeCounts ) = -1;


%Use the training weights to computer our answer
Result = [temp_feats ones( size(accuracyY) )] * weight;
Y = sign(Result);
Y(Y==0) = -1;

error = sum(Y ~= accuracyY);

accuracy = ( size(temp_feats,1)-error )/size(temp_feats,1) ;

posGroup(1:size( find( Y== 1 ), 1),:) = temp_feats( find( Y== 1 ), :);
negGroup(1:size( find( Y==-1 ), 1),:) = temp_feats( find( Y==-1 ), :);
posGroupLabel(1:size(find(Y== 1), 1), 1) = temp_label( find(Y== 1), 1);
negGroupLabel(1:size(find(Y==-1), 1), 1) = temp_label( find(Y==-1), 1);
end

