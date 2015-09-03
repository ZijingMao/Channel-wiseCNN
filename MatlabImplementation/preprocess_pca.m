function preprocess_pca(num)
% clear;
% num = 1;
x_train = [];
y_train = [];
x_test = [];
y_test = [];
load(['grp_merged' num2str(num) '.mat']);

x_train = reshape(x_train, [30*250, size(x_train, 3)]);
x_test = reshape(x_test, [30*250, size(x_test, 3)]);
x_train = x_train';
x_test = x_test';
disp('Parsing PCA...');
[coeff, score, latent] = pca(x_train);
disp('PCA Finished.');
latent = latent/sum(latent);

l = 0;
comp = 1;
for i = 1:7500
    l= l+latent(i);
    if l > 0.999
        comp = i;
        break;
    end
end
x_train = score(:, 1:comp);
test_x = bsxfun(@minus,x_test, mean(x_test, 2))*coeff;
x_test = test_x(:, 1:comp);

save(['grp_pca' num2str(num) '.mat'], 'x_test', 'x_train', 'y_test', 'y_train', 'coeff', 'comp', '-v7.3');
