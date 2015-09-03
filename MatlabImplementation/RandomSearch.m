function [ HidNormLayer, FinalAzRBM, bestHid, bestAlp ] = RandomSearch(x, y, layer)
%% for the first layer
disp(['Start parsing ' num2str(layer)])
% random selection
hidden_exp = 8;
alpha = logspace(0,-4, 10);
hidden = 2.^(1:hidden_exp);

randomRun = 36;
randomIdx = randi([0, 10*hidden_exp-1], randomRun, 1);
Az_RAW_Epoch = cell(1, randomRun);
Az_RBM_Epoch = cell(1, randomRun);
parfor i = 1:randomRun
    idx = randomIdx(i);
    alphaIdx = mod(idx, 10)+1;
    hiddenIdx = floor(idx/10)+1;
    [Az_RAW_Epoch{i}, Az_RBM_Epoch{i}, ~] = rbm4ProjectionNew(...
        alpha(alphaIdx), hidden(hiddenIdx), x   , y    , false  , true    , 5);
        % learn rate   , hidden unit      , data, label, sigmoid, baggging, fold
end

% parse data
FitboxAzRaw = zeros(10, hidden_exp);
FitboxAzRBM = zeros(10, hidden_exp);
for i = 1:randomRun
    idx = randomIdx(i);
    alphaIdx = mod(idx, 10)+1;
    hiddenIdx = floor(idx/10)+1;
    FitboxAzRaw(alphaIdx, hiddenIdx) = Az_RAW_Epoch{i}(end);
    FitboxAzRBM(alphaIdx, hiddenIdx) = Az_RBM_Epoch{i}(end);
end
h = figure;
subplot(1, 2, 1);
pcolor(FitboxAzRaw);
subplot(1, 2, 2);
pcolor(FitboxAzRBM);
savefig(h, ['fitbox' num2str(layer)]);
save(['fitbox' num2str(layer) '.mat'], 'FitboxAzRaw', 'FitboxAzRBM');

[row, col] = find(FitboxAzRBM == max(max(FitboxAzRBM)));
disp(['End parsing ' num2str(layer)])
disp(['Best combination: (hidden ' num2str(hidden(col)), ' alpha ' num2str(alpha(row)) ')'])
disp(['Best performance: ' num2str(max(max(FitboxAzRBM)))])
disp(['Compare with Bag: ' num2str(FitboxAzRaw(row, col))])

bestHid = hidden(col);
bestAlp = alpha(row);

[~, ~, HidNormLayer] = rbm4ProjectionNew(...
         alpha(row), hidden(col), x, y, false, false, 5);
FinalAzRBM = max(max(FitboxAzRBM));
disp('%%%%%%%%%%%%%%%%%%%%%%')