function Az7Fold

%% test on cnn
parfor i = 1:7
    Az_CNN_Bagging(i) = Az1FoldBagging(['grp' num2str(i) 'cnn_ip2.mat'], 'bagging');
end

parfor i = 1:7
    Az_CNN_LDA(i) = Az1FoldBagging(['grp' num2str(i) 'cnn_ip2.mat'], 'lda');
end

%% test on ccnn
parfor i = 1:7
    Az_CCNN_Bagging(i) = Az1FoldBagging(['grp' num2str(i) 'ccnn_ip2.mat'], 'bagging');
end

parfor i = 1:7
    Az_CCNN_LDA(i) = Az1FoldBagging(['grp' num2str(i) 'ccnn_ip2.mat'], 'lda');
end

%% test on dnn
parfor i = 1:7
    Az_DNN_Bagging(i) = Az1FoldBagging(['grp' num2str(i) 'dnn_ip2.mat'], 'bagging');
end

parfor i = 1:7
    Az_DNN_LDA(i) = Az1FoldBagging(['grp' num2str(i) 'dnn_ip2.mat'], 'lda');
end

save('DNNFeaturesPerformanceICA', 'Az_CNN_Bagging', 'Az_CNN_LDA',...
    'Az_CCNN_Bagging', 'Az_CCNN_LDA', 'Az_DNN_Bagging', 'Az_DNN_LDA');

end

