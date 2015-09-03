function [cdata_shuffle,grp_shuffle ] = shuffle( cdata, grp)

w_suffle=[cdata grp];
w_suffle=w_suffle';
w_suffle=w_suffle(:,randperm(size(w_suffle,2)));
w_suffle=w_suffle';
cdata_shuffle=w_suffle(:,1:end-1);
grp_shuffle=w_suffle(:,end);


end

