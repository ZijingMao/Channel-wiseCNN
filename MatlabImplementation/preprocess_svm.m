function [Az] = preprocess_svm(num)

x_train = [];
y_train = [];
x_test = [];
y_test = [];
load(['grp_pca' num2str(num) '.mat']);

svmMod = fitcsvm(x_train, y_train);
[~,A]=predict(svmMod,x_test);
[~, ~, ~, Az ] = perfcurve(y_test, A(:, 2), 1);

end