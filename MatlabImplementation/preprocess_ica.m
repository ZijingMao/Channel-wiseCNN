function preprocess_ica(num)

x_train = [];
x_test = [];
y_train = [];
y_test = [];
load(['merged' num2str(num) 'mat.mat']);

train_size = size(x_train, 3);
test_size = size(x_test, 3);
x_train = reshape(x_train, [30, train_size*250]);
x_test = reshape(x_test, [30, test_size*250]);

[weight, sphere] = runica(x_train);

x_train = weight*(sphere*bsxfun(@minus,x_train,mean(x_train, 2)));
x_test = weight*(sphere*bsxfun(@minus,x_test,mean(x_test, 2)));

x_train = reshape(x_train, [30, 250, train_size]);
x_test = reshape(x_test, [30, 250, test_size]);

save(['merged_ica' num2str(num) '.mat'], 'x_test', 'x_train', 'y_test', 'y_train', 'sphere', 'weight', '-v7.3');
