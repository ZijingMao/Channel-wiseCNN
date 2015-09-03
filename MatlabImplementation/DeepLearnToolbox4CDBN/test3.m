clear all;

% load('n_feat_w_drive_row_wise_lay2_230_5.mat');
% load('drive_data_1s.mat');

load('w_drive_f80_row_wise_lay1_50.mat');
load('drive_data_1s_80.mat');

cdata=w(1:10000,:);%
grp=y(1:10000,:);%(1:2796);% 
[m,n]=size(cdata);
parti=floor(m/10);
aa=1;
bb=parti;
sum=0;
sum2=0;
jj=0;


for ii=1:10
jj=jj+1

[cd,gr] = shuffle( cdata, grp);

train_x =cd(parti+1:end,:);
train_y = [gr(parti+1:end,:),~gr(parti+1:end,:)];

test_x =cd(aa:bb,:);
test_y = [gr(aa:bb,:),~gr(aa:bb,:)];

% load mnist_uint8;
% 
% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [100 50];% [100 50]
opts.numepochs =   10;%50
opts.batchsize = 5;
opts.momentum  =   0.5;
opts.alpha     =   0.001;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);% 2 is output size
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  10;%50
opts.batchsize = 5;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad, A] = nntest(nn, test_x, test_y);

% assert(er < 0.10, 'Too big error');

[Az_score,~,~,~]=rocarea(A,gr(aa:bb,:));
 Az_score
% aa=aa+parti;
% bb=bb+parti;
sum=sum+ Az_score  


end