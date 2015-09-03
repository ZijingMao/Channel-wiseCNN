function GreedyLearning(x, y)

%% create data
max_len = 6;
x = permute(x, [3, 1, 2]);
bestHid = zeros(max_len, 1);
bestAlp = zeros(max_len, 1);

AzRBM = 0;
HidNormLayer = x;

for layer = 1:max_len
    [ HidNormLayer, FinalAzRBM, bestHid(layer), bestAlp(layer) ] = RandomSearch(HidNormLayer, y, layer);
    if FinalAzRBM > AzRBM
        AzRBM = FinalAzRBM;
    else
        disp(['The best layer is: ' num2str(layer-1)]);
        disp('Parsing finished.');
        return;
    end
end

end

