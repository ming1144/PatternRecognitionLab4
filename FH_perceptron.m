function weights=FH_perceptron(feats,label,eta)
%% feats: num_of_samples * num_of_dimension
%% eta: learning rate

if nargin<3 %Number of function input arguments
    eta=0.1;
end

max_iter=1000;  % maximum of iteration number, in case of non-linear data

assert(sum(abs(label)==1)==size(label,1));  % assert that label contains either 1 or -1
%Generate an error when a condition is violated

[nums,ndims]=size(feats);

rand('state',0);    % initialize the random seed, in order to make the rand function generate the same numbers

% initialization
weights = rand(ndims+1,1)*2-1;    % initialize weights to random numbers ranging from -1 to 1

feats = [feats ones(nums,1)];     % augment feats with another column filled with 1

delta_weights = ones(ndims+1,1);

iter=1;
while abs(sum(delta_weights))>0.1 && iter<max_iter
    R = feats*weights;
    Y = sign(R); %if the element greater than zero return 1, and elseif the element smaller than zero return -1, else 0 if Y==0
    Y(Y==0)=-1;
    delta_weights = eta * feats' * (label-Y);
    weights = weights + delta_weights;
%     error(iter)=abs(sum(delta_weights));
    iter=iter+1;
end

end