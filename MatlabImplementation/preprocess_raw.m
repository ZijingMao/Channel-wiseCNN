function preprocess_raw(x, y, num)

x_test = x(:, :, 1000*(num-1)+1:1000*num);
y_test = y(1000*(num-1)+1:1000*num);
x_train = x;
y_train = y;
x_train(:, :, 1000*(num-1)+1:1000*num) = [];
y_train(1000*(num-1)+1:1000*num) = [];
save(['grp_merged' num2str(num) '.mat'], 'x_test', 'x_train', 'y_test', 'y_train', '-v7.3');

end