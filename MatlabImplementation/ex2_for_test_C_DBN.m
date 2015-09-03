clear all;

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
load('dbn_lay1_60_for_conv_lay_init.mat');%load('dbn_new_1.mat');
load('dbn_for_fully_connected_layer_lay1_60.mat');

nn = dbnunfoldtonn(dbn_new);

%%%%% fully connected layer
nn_full_c = dbnunfoldtonn_fully_connected(dbn_for_fully_connected_layer_lay1_60);


%train nn
opts.numepochs =  2;%100;
opts.batchsize =  1564;%10948;% 1564, 2737, 5474, 10948
load('drive_data_corrected_1s_80.mat');

x_for_C_DBN=x;

%%%% normalize the data
s=x_for_C_DBN;

 
keep_mat_for_80sbj_C_DBN_train1=zeros((30*10948),250);
u=1;
v=30;

for j=1:10948
    j
s2= s(:,:,j); 
   

keep_mat_for_80sbj_C_DBN_train1(u:v,:)=s2;

u=u+30;
v=v+30;
  
end

[keep_mat_for_80sbj_C_DBN_train1, ~, ~] = zscore(keep_mat_for_80sbj_C_DBN_train1);

u1=1;
v1=30;

for j=1:10948

s1(:,:,j)=keep_mat_for_80sbj_C_DBN_train1(u1:v1,:);

u1=u1+30;
v1=v1+30;

end
x_for_C_DBN=s1;

%%%%%%%%%%%%%%%%%%%%%%%%




train_x=x_for_C_DBN;

train_y=y;
nn = nntrain(nn,nn_full_c, train_x, train_y, opts);
% [er, bad] = nntest(nn, test_x, test_y);
[nn_final,nn_full_c_final,keep_final,keep_full_c_final] = nnff(nn,nn_full_c, train_x, train_y)
predict=nn_full_c_final.a{1,3};
[Az,~,~,~]=rocarea(predict(:,1),train_y);

% [nn_final,nn_full_c_final,keep_final,keep_full_c_final] = nnff(nn,nn_full_c, x_for_C_DBN_test, y_for_C_DBN_test)
% predict=nn_full_c_final.a{1,3};
% [Az,~,~,~]=rocarea(predict(:,1),y_for_C_DBN_test);
% 
