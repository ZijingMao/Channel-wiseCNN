clear all;

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
addpath(genpath(pwd))
rand('state',0)
load('dbn_lay1_60_for_conv_lay_init_tenpercent_1.mat');%load('dbn_new_1.mat');
load('dbn_for_fully_connected_10percent1_lay1_60.mat');

nn = dbnunfoldtonn(dbn_new);

%%%%% fully connected layer
nn_full_c = dbnunfoldtonn_fully_connected(dbn_for_fully_connected_10percent1_lay1_60);


%train nn
opts.numepochs = 53;%100;
opts.batchsize =  9846;%1094, 1641, 3282, 4923, 9846

load('x_for_C_DBN_train_1.mat');
load('y_for_C_DBN_train_1.mat')
load('x_for_C_DBN_test_1.mat')
load('y_for_C_DBN_test_1.mat')
x_for_C_DBN=x_for_C_DBN_train;
    

train_x=x_for_C_DBN;
train_y=y_for_C_DBN_train;
test_x=x_for_C_DBN_test;
test_y=y_for_C_DBN_test;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% train as test
[nn,nn_full_c,L, exch, Az_train, Az_test] = ...
    nntrain(nn,nn_full_c, train_x, train_y, test_x, test_y, opts, 50, 2);
% [er, bad] = nntest(nn, test_x, test_y);
[nn_final,nn_full_c_final,~,~] = nnff(nn,nn_full_c, train_x, train_y);
predict1=nn_full_c_final.a{1,3};
[Az_train,~,~,~]=rocarea(predict1(:,1),train_y);
%[~, ~, ~, Az ] = perfcurve(train_y, predict1(:, 1), 1);

[dummy, i] = max(nn_full_c_final.a{1,3},[],2);
    labels = i;
    A=nn_full_c_final.a{1,3};

[dummy, expected] = max([train_y,~train_y],[],2);
    bad = find(labels ~= expected);    
    er_train = numel(bad) / size(train_y, 1);



%% test as test
[nn_final,nn_full_c_final,~,~] = nnff(nn,nn_full_c,  test_x, test_y);
predict1=nn_full_c_final.a{1,3};
[Az_test,~,~,~]=rocarea(predict1(:,1),test_y);
%[~, ~, ~, Az ] = perfcurve(test_y, predict1(:, 1), 1);


[dummy, i] = max(nn_full_c_final.a{1,3},[],2);
    labels = i;
    A=nn_full_c_final.a{1,3};

[dummy, expected] = max([test_y,~test_y],[],2);
    bad = find(labels ~= expected);    
    er_test = numel(bad) / size(test_y, 1);
