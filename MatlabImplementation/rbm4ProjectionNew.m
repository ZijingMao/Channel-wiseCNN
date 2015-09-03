function [Az_raw, Az_rbm, HiddenLayer] = rbm4ProjectionNew...
    (alpha, numhid, x, y, sigmoidFlag, classificationFlag, Fold)

% A = max(max(max(data)));
% B = min(min(min(data)));
% range = [min(B);max(A)];
% data	= (data - range(1))/(-range(1)+range(2));

%% create data
data = x;
[a,b,c] = size(data);
data = reshape(data, [a*b, c]);

%% initialization
numepochs = 3;
opts.numepochs = numepochs;
opts.batchsize = 60;
rbm.alpha = alpha;
weightPenalty = 0.01;
disp(['Numepochs:' num2str(numepochs) ', lr:' num2str(alpha), ', hid:' num2str(numhid)]);

[numcases, numdims]=size(data);

rbm.W  = 0.1*randn(numhid, numdims);
rbm.vW = zeros(numhid, numdims);

rbm.b  = zeros(numdims, 1);
rbm.vb = zeros(numdims, 1);

rbm.c  = zeros(numhid, 1);
rbm.vc = zeros(numhid, 1);

m = size(data, 1);
numbatches = m / opts.batchsize;

% assert(rem(numbatches, 1) == 0, 'numbatches not integer');
% for loop
Az = zeros(1, opts.numepochs);
Er = zeros(1, opts.numepochs);

for i = 1 : opts.numepochs
    if mod(i, 2) == 0
        rbm.alpha = rbm.alpha/3;
    end
    if i>2
        rbm.momentum=0.9;	% if epoch is larger than 5
    else
        rbm.momentum=0.5;
    end;
    kk = randperm(m);
    err = 0;
    for l = 1 : numbatches
        batch = data(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
        
        v1 = batch;
        h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');
        if sigmoidFlag
            v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);
        else
        	v2 = normrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W, 1);
        end
        h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');
        
        c1 = h1' * v1;
        c2 = h2' * v2;
        
        %rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize;
        rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * ...
								((c1 - c2) / opts.batchsize - weightPenalty * rbm.W);
        rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize;
        rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;
        
        rbm.W = rbm.W + rbm.vW;
        rbm.b = rbm.b + rbm.vb;
        rbm.c = rbm.c + rbm.vc;
        
        err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;
    end
    
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ...
        '. Average reconstruction error is: ' num2str(err / numbatches)]);
    Er(i) = err / numbatches;
    if isnan(Er(i))
        Az_rbm(i) = NaN;
        Az_raw(i) = NaN;
        HiddenLayer = NaN;
    else
        [ Az_raw(i), Az_rbm(i), HiddenLayer ] = rbm4ward( rbm, numcases, numhid, x, y, classificationFlag, Fold );
    end    
    disp(['Az RBM: ' num2str(Az_rbm(i)) ' | Az Raw: ' num2str(Az_raw(i))]);
end


