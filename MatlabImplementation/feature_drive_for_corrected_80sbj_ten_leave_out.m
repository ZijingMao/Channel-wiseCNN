clear all;
load('merged_ica6.mat')
s=x_train;

mm=size(s,3);

keep_mat_for_80sbj_ten_leave_out=zeros((30*mm),250);
u=1;
v=30;

for j=1:mm
    j;
s2= s(:,:,j); 
   

keep_mat_for_80sbj_ten_leave_out(u:v,:)=s2;

u=u+30;
v=v+30;
  
end

clear j;
clear mm;
clear s;
clear s2;
clear u;
clear v;
clear x_test;
clear y_test;
clear x_train;
clear y_train;



% save('feature_row_wise_1s_for_corrected_80sbj.mat','keep_mat_for_80sbj_10_percent');