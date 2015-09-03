function [Az] = preprocess_dnn(num)

x_train = [];
y_train = [];
x_test = [];
y_test = [];
load(['grp_pca' num2str(num) '.mat']);

[ Az ] = test_example_DBN(x_train, y_train, x_test, y_test);

end