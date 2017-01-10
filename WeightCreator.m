function [ weight ] = WeightCreator( tr_feats, tr_label, pos, neg )
%WEIGHTCREATOR Summary of this function goes here
%   Detailed explanation goes here

pos_and_neg = [pos neg];

temp_feats = [];
temp_label = [];

for i = 1 : size(pos_and_neg,2)
    temp_feats = [temp_feats ; tr_feats(find( tr_label == pos_and_neg(i)) , : )];
    temp_label = [temp_label ; tr_label(find( tr_label == pos_and_neg(i)) , : )];
end

for i = 1 : size(pos,2)
    temp_label(temp_label==pos(i)) = 1;
end

for i = 1 : size(neg,2)
    temp_label(temp_label==neg(i)) = -1;
end

weight = FH_perceptron(temp_feats , temp_label);

end

