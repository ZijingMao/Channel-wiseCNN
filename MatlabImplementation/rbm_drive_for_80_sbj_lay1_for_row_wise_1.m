%% Add fixed folders to the path
addpath(genpath(['/home/mehdi.hajinoroozi/Research/DeepLearnToolbox-master']));
% 
% load('drive_data_corrected_1s_80_ten_leave_out_7th_grp.mat')
load('merged_ica6.mat')
% y_train=hdf5read('drive_data_ten_leav_out_7th_grp_y_train.hdf5', '/dataset1');

mmmm=size(y_train,1);

rbm_feature11_new=keep_mat_for_80sbj_ten_leave_out;

mmm=size(keep_mat_for_80sbj_ten_leave_out,1);
%considering 0 and 1 event, 1s before event in rsvp
% rbm_feature11_new=keep_staticall;

%
% rbm_feature=feature_rbm_event_rsvp;%feature_for_rbm_per_event_no1;%feature_for_rbm_per_event1
% [rbm_feature11_new, mu, sigma] = zscore(rbm_feature11_new);
%rbm_feature1=(rbm_feature_zscore-min(min(rbm_feature_zscore)))/(max(max(rbm_feature_zscore))-min(min(rbm_feature_zscore)));
rbm_feature11_new=rbm_feature11_new(1:mmm,:);% rbm_feature1(1:4200,:);
dbn_new=[];
i=1;
test11_drive_lay1=cell(1,1);

for k=80:80:80%k=120:20:260

rand('state',0)
%train dbn
dbn_new.sizes = [k];
opts.numepochs =10;%10
opts.batchsize = 10;
opts.momentum  =  0.5;%0.1
opts.alpha     =   0.003; %0.000001
dbn_new = dbnsetup(dbn_new, rbm_feature11_new, opts);%dbnsetup(dbn, train_x, opts);
dbn_new = dbntrain(dbn_new, rbm_feature11_new, opts);

% test11_rsvp1_staticall_lay1_700=sigm(repmat(dbn_new.rbm{1,1}.c', size(rbm_feature11_new, 1), 1)  + rbm_feature11_new * dbn_new.rbm{1,1}.W');
test11_drive_lay1{1,i}=sigm(repmat(dbn_new.rbm{1,1}.c', size(rbm_feature11_new, 1), 1)  + rbm_feature11_new * dbn_new.rbm{1,1}.W');
k
i=i+1
% dbn_new=[];
end

% 
save('dbn_lay1_80_ten_leave_out_for_conv_lay_init.mat','dbn_new');

%%% new feature for classification
k=80;
for j=1:1
    
test11=test11_drive_lay1{1,j};

[n,m]=size(test11);

w=zeros(mmmm,(30*k));
w1=[]
w1=test11(1:30,:);%%??
w1=w1';
w(1,:)=reshape(w1,1,[]);
u=1+30;%%??
v=30+30;%%??



for i=1:(mmmm-1)% 10948-1=10947
w1=test11(u:v,:);
w1=w1';
w1=reshape(w1,1,[]);
w((i+1),:)=w1;
w1=[];
  u=u+30;%%?? 
  v=v+30;%%??  
  i;
end

  fileroot = sprintf('w_drive_f80_row_wise_ten_leave_out_lay1_%d', k);
  filename = [fileroot '.mat'];

save(filename,'w');

k=k+10;
end