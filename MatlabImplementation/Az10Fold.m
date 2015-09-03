function [ Az_training, Az_testing ] = Az10Fold

dbn_for_fully_connected = [];
load('dbn_for_fully_connected.mat');
Az_training = cell(10, 1);
Az_testing = cell(10, 1);

parfor fold = 1:10
    
    [ Az_training{fold}, Az_testing{fold} ] = Az1Fold( fold,  dbn_for_fully_connected );
    
end

end

