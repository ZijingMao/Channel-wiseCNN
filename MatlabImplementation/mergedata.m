function mergedata(fold, data, label, fold_size)


range = fold_size*(fold-1)+1:fold_size*fold;
x_test = data(:, :, range);
y_test = label(range, :);
x_train = data;
y_train = label;
x_train(:, :, range) = [];
y_train(range, :) = [];
save(['merged' num2str(fold) 'mat'], 'x_test', 'y_test', 'x_train', 'y_train', '-v7.3');

end
