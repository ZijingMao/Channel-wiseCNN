function [ Az_raw_avg, Az_rbm_avg, hid ] = rbm4ward( rbm, numcases, numhid, x, y, classificationFlag, Fold )

%% create data
data = x;
[a,b,c] = size(data);
data = reshape(data, [a*b, c]);

%% generate data
hid = sigm(repmat(rbm.c', numcases, 1) + data * rbm.W');

hid = reshape(hid, [a, b, numhid]);
% plot
% scatter(hid(y==0, 1, 1), hid(y==0, 1, 2), 'Marker','.');
% hold on
% scatter(hid(y==1, 1, 1), hid(y==1, 1, 2), 'Marker','.');

hidnew = reshape(hid, [a, b*numhid]);

raw = reshape(x, [a, b*c]);

Az_raw_avg = 0;
Az_rbm_avg = 0;
%% classification
if classificationFlag
    fold_size = round(a/Fold);
    % use fold test
    for idx = 1:Fold
        % EEG data
        test_x = raw(fold_size*(idx-1)+1:fold_size*idx, :);
        test_y = y(fold_size*(idx-1)+1:fold_size*idx, :);
        train_x = raw;
        train_y = y;
        train_x(fold_size*(idx-1)+1:fold_size*idx, :) = [];
        train_y(fold_size*(idx-1)+1:fold_size*idx, :) = [];
        ens = fitensemble(train_x,train_y,'Bag',100,'Tree','Type','Classification');
        [~,A]=predict(ens,test_x);
        [~, ~, ~, Az_raw(idx) ] = perfcurve(test_y, A(:, 2), 1);
        
        % RBM data
        test_x = hidnew(fold_size*(idx-1)+1:fold_size*idx, :);
        test_y = y(fold_size*(idx-1)+1:fold_size*idx, :);
        train_x = hidnew;
        train_y = y;
        train_x(fold_size*(idx-1)+1:fold_size*idx, :) = [];
        train_y(fold_size*(idx-1)+1:fold_size*idx, :) = [];
        ens = fitensemble(train_x,train_y,'Bag',100,'Tree','Type','Classification');
        [~,A]=predict(ens,test_x);
        [~, ~, ~, Az_rbm(idx) ] = perfcurve(test_y, A(:, 2), 1);
    end
    Az_raw_avg = mean(Az_raw);
    Az_rbm_avg = mean(Az_rbm);
else
    Az_raw = -1;
    Az_rbm = -1;
end

end

