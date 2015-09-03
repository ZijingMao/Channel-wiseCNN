function [ Az_Norm_20Epoch, Er_Norm_20Epoch, Az_Sigm_20Epoch, Er_Sigm_20Epoch ]...
    = dbn4ProjectionNew( x, y )

%% random selection
alpha = logspace(0,-4, 10);
hidden = 2.^(1:8);

%% train first layer
[~, ~, OutputNorm] = rbm4ProjectionNew(...
        alpha(4), hidden(7), 1, data, y, false, true);
OutputNorm = reshape(OutputNorm, [10940*30, 128]);
    
%% generate random sequence
randomRun = 36;
randomIdx = randi([0, 10*8-1], randomRun, 1);
Az_Norm_20Epoch = cell(1, randomRun);
Er_Norm_20Epoch = cell(1, randomRun);

Az_Sigm_20Epoch = cell(1, randomRun);
Er_Sigm_20Epoch = cell(1, randomRun);

%% train second layer, random selection, norm sampling
parfor i = 1:randomRun
    idx = randomIdx(i);
    alphaIdx = mod(idx, 10)+1;
    hiddenIdx = floor(idx/10)+1;
    [Az_Norm_20Epoch{i}, Er_Norm_20Epoch{i}, ~] = rbm4ProjectionNew(...
        alpha(alphaIdx), hidden(hiddenIdx), 1, OutputNorm, y, false, true);
end

%% train second layer, random selection, sigmoid sampling
alpha = logspace(0,-4, 10);
hidden = 2.^(9:11);

randomRun = 20;
randomIdx = randi([0, 10*3-1], randomRun, 1);
Az_Sigm_20Epoch = cell(1, randomRun);
Er_Sigm_20Epoch = cell(1, randomRun);

parfor i = 1:randomRun
    idx = randomIdx(i);
    alphaIdx = mod(idx, 10)+1;
    hiddenIdx = floor(idx/10)+1;
    [Az_Sigm_20Epoch{i}, Er_Sigm_20Epoch{i}, ~] = rbm4ProjectionNew(...
        alpha(alphaIdx), hidden(hiddenIdx), 1, OutputNorm, y, true, true);
end

end

