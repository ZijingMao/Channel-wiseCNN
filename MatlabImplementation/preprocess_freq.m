clear;
num =1;
load(['grp_freq' num2str(num) '.mat']);

size_train  =size(x_train, 3);
size_test  =size(x_test, 3);
[ PSD_x_train ] = RunPreprocessSteps_for_ten_leave( x_train);
[ PSD_x_test ] = RunPreprocessSteps_for_ten_leave( x_test);

x_train = reshape(PSD_x_train, [30, 11*250, size_train]);
x_test = reshape(PSD_x_test, [30, 11*250, size_test]);

save(['grp_merged' num2str(num) 'freq.mat'], 'x_test', 'x_train', 'y_test', 'y_train', '-v7.3');


% hdf5write(['grp' num2str(num) 'freq.hdf5'], '/dataset/x_train', x_train, ...
%     '/dataset/x_test', x_test,'/dataset/y_train', y_train, '/dataset/y_test', y_test);