clear

%% read Data
te_label_file = 't10k-labels.idx1-ubyte';
tr_label_file = 'train-labels.idx1-ubyte';
te_data_file  = 't10k-images.idx3-ubyte';
tr_data_file  = 'train-images.idx3-ubyte';

% read training labels
fid = fopen(tr_label_file,'r');
B = fread(fid); fclose(fid);
%60000 training labels are the last 60000 bytes
tr_label = B(end-60000+1:end);

fid = fopen(te_label_file,'r');
B = fread(fid); fclose(fid);
te_label = B(end-10000+1:end);

% read the training data
fid = fopen(tr_data_file,'r');
B = fread(fid); fclose(fid);
%ignore the headers 4x4 bytes
B = B(17:end); 
tr_feats = reshape(B,28*28,60000)';

fid = fopen(te_data_file,'r');
B = fread(fid); fclose(fid);
B = B(17:end);
te_feats = reshape(B,28*28,10000)';

%% Create Sub DataSet
tr_feats_sub = zeros(10000, 784);
tr_label_sub = zeros(10000, 1);
te_feats_sub = zeros(1000, 784);
te_label_sub = zeros(1000, 1);

for  i = 1 : 10000
    tr_feats_sub(i,:) = tr_feats(i,:);
    tr_label_sub(i) = tr_label(i);
    if i < 1001
        te_feats_sub(i,:) = te_feats(i,:);
        te_label_sub(i) = te_label(i);
    end
end

save('sub_feats', 'te_feats_sub' , 'tr_feats_sub' , 'te_label_sub' , 'tr_label_sub');
clear
load('sub_feats');

%% Task1  2 & 5 perceptron

weights = WeightCreator(tr_feats_sub,tr_label_sub,2 ,5);
[~,~,~,~,accuracy,~] = Classifier(tr_feats_sub, tr_label_sub, 2, 5, weights);

fprintf('Accuracy for testing Data : %f\n', accuracy);

[~,~,~,~,accuracy,~] = Classifier(te_feats_sub, te_label_sub, 2, 5, weights);

fprintf('Accuracy for testing Data : %f\n', accuracy);

clear
load ('sub_feats');
%% task 2 two layer perceptron
fprintf('class 17 & 25 perceptron\n');

weight_1257 = WeightCreator(tr_feats_sub,tr_label_sub, [1 7] , [2 5] );
[posGroup,negGroup,posGroupLabel,negGroupLabel,~,error1257] = Classifier(te_feats_sub, te_label_sub, [1 7] , [2 5] , weight_1257);

weight_17 = WeightCreator(tr_feats_sub,tr_label_sub, 1 , 7 );
[~,~,~,~,~,error17] = Classifier(posGroup, posGroupLabel, 1 , 7 , weight_17);

weight_25 = WeightCreator(tr_feats_sub,tr_label_sub, 2 , 5 );
[~,~,~,~,~,error25] = Classifier(negGroup, negGroupLabel, 2 , 5 , weight_25);

accuracy1257 = (size(negGroup,1) + size(posGroup,1) - error17 -error17 -error1257) / (size(negGroup,1) + size(posGroup,1));

fprintf('Accuracy for testing Data : %f\n', accuracy1257);

clear
load('sub_feats');
%% task3
fprintf('task3\n');
error = [];
weight_0146789_235 = WeightCreator(tr_feats_sub,tr_label_sub, [0 1 4 6 7 8 9] , [2 3 5] );
[posGroup0146789, negGroup235, posGroupLabel0146789, negGroupLabel235,~,error(1)] = Classifier(te_feats_sub, te_label_sub, [0 1 4 7 8 9] , [2 3 5 6] , weight_0146789_235);

weight_23_5 = WeightCreator(tr_feats_sub,tr_label_sub, [2 3] , 5);

[posGroup23 , ~ , posGroupLabel23 , ~ , ~ , error(2)] = Classifier(negGroup235,negGroupLabel235,[2 3],[5],weight_23_5); 

weight_2_3 = WeightCreator(tr_feats_sub,tr_label_sub, 2 , 3);

[~,~,~,~,~,error(3)] = Classifier(posGroup23,posGroupLabel23,2,3,weight_2_3);

weight_179_0468 = WeightCreator(tr_feats_sub,tr_label_sub,[1 7 9],[0 4 6 8]);

[posGroup179,negGroup0468,posGroupLabel179,negGroupLabel0468,~,error(4)] = Classifier(posGroup0146789,posGroupLabel0146789,[1 7 9],[0 4 6 8],weight_179_0468);

weight_1_79 = WeightCreator(tr_feats_sub,tr_label_sub,[1],[7 9]);

[~,negGroup79,~,negGroupLabel79,~,error(5)] = Classifier(posGroup179,posGroupLabel179,[1],[7 9],weight_1_79);

weight_7_9 = WeightCreator(tr_feats_sub,tr_label_sub,[7],[9]);

[~,~,~,~,~,error(6)] = Classifier(negGroup79,negGroupLabel79,[7],[9],weight_7_9);

weight_06_48 = WeightCreator(tr_feats_sub,tr_label_sub,[0 6],[4 8]);

[posGroup06,negGroup48,posGroupLabel06,negGroupLabel48,~,error(7)] = Classifier(negGroup0468,negGroupLabel0468,[0 6],[4 8],weight_06_48);

weight_0_6 = WeightCreator(tr_feats_sub,tr_label_sub,0,6);

[~,~,~,~,~,error(8)] = Classifier(posGroup06,posGroupLabel06,0,6,weight_0_6);

weight_4_8 = WeightCreator(tr_feats_sub, tr_label_sub,4,8);

[~,~,~,~,~,error(9)] = Classifier(negGroup48,negGroupLabel48,4,8,weight_4_8);

Accuracy = (size(tr_feats_sub,1) - sum(error))/size(tr_feats_sub,1);

fprintf('Accuracy for testing Data : %f\n', Accuracy);



