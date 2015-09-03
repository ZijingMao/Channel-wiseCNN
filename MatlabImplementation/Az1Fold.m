function [ Az_train, Az_test ] = Az1Fold( fold,  dbn_for_fully_connected )

    rand('state',0)
    dbn_new = [];
    load(['dbn_lay1_60_for_conv_lay_init_tenpercent_' num2str(fold) '.mat']);
    
    nn = dbnunfoldtonn(dbn_new);
    
    %%%%% fully connected layer
    nn_full_c = dbnunfoldtonn_fully_connected(dbn_for_fully_connected(fold));
    
    
    %train nn
    opts.numepochs = 100;%100;
    opts.batchsize =  9846;%1094, 1641, 3282, 4923, 9846
    
    x_for_C_DBN_train = [];
    y_for_C_DBN_train = [];
    x_for_C_DBN_test = [];
    y_for_C_DBN_test = [];
    
    load(['x_for_C_DBN_train_' num2str(fold) '.mat']);
    load(['y_for_C_DBN_train_' num2str(fold) '.mat']);
    load(['x_for_C_DBN_test_' num2str(fold) '.mat']);
    load(['y_for_C_DBN_test_' num2str(fold) '.mat']);    
    
    train_x=x_for_C_DBN_train;
    train_y=y_for_C_DBN_train;
    test_x=x_for_C_DBN_test;
    test_y=y_for_C_DBN_test;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [~,~,~, ~, Az_train, Az_test] = ...
        nntrain(nn,nn_full_c, train_x, train_y, test_x, test_y, opts, 50, 5);

end

