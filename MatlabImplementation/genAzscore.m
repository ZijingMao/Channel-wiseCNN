function [ Az_train, Az_test ] = genAzscore( nn, nn_full_c, train_x, train_y, test_x, test_y )

[nn_final,nn_full_c_final,keep_final,keep_full_c_final] = nnff(nn,nn_full_c, train_x, train_y);
predict1=nn_full_c_final.a{1,3};
% [Az_train,~,~,~]=rocarea(predict1(:,1),train_y);
[~, ~, ~, Az_train ] = perfcurve(train_y, predict1(:, 1), 1);

[dummy, i] = max(nn_full_c_final.a{1,3},[],2);
    labels = i;
    A=nn_full_c_final.a{1,3};

[dummy, expected] = max([train_y,~train_y],[],2);
    bad = find(labels ~= expected);    
    er_train = numel(bad) / size(train_y, 1);

%% test as test
[nn_final,nn_full_c_final,keep_final,keep_full_c_final] = nnff(nn,nn_full_c,  test_x, test_y);
predict1=nn_full_c_final.a{1,3};
% [Az_test,~,~,~]=rocarea(predict1(:,1),test_y);
[~, ~, ~, Az_test ] = perfcurve(test_y, predict1(:, 1), 1);


[dummy, i] = max(nn_full_c_final.a{1,3},[],2);
    labels = i;
    A=nn_full_c_final.a{1,3};

[dummy, expected] = max([test_y,~test_y],[],2);
    bad = find(labels ~= expected);    
    er_test = numel(bad) / size(test_y, 1);
    
end

