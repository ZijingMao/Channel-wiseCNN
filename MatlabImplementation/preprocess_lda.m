function [Az] = preprocess_lda(num)

x_train = [];
y_train = [];
x_test = [];
y_test = [];
load(['grp_pca' num2str(num) '.mat']);

ldaMod = fitcdiscr(x_train, y_train);
[~,A]=predict(ldaMod,x_test);
[~, ~, ~, Az ] = perfcurve(y_test, A(:, 2), 1);

end