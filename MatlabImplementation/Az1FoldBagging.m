function Az = Az1FoldBagging(data, algorithm)

x_test = [];
y_test = [];
x_train = [];
y_train = [];

load(data);

train_size = size(x_train, 3);
test_size = size(x_test, 3);
x_train = reshape(x_train, [30*250, train_size])';
x_test = reshape(x_test, [30*250, test_size])';

% y_train = y_train';
% y_test = y_test';

if strcmp(algorithm, 'bagging')
    ens = fitensemble(x_train,y_train,'Bag',100,'Tree','Type','Classification');
else
    ens = fitcdiscr(x_train,y_train);
end
[~,A]=predict(ens,x_test);
[~, ~, ~, Az ] = perfcurve(y_test, A(:, 2), 1);

end