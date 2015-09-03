function Visualization

%% concat all data
load('y_for_C_DBN_train_1.mat')
load('x_for_C_DBN_train_1.mat')
load('y_for_C_DBN_test_1.mat')
load('x_for_C_DBN_test_1.mat')
x = cat(3, x_test, x_train);
y = cat(1, y_test, y_train);

%% or load data here
addpath(genpath(pwd))
load('Driving.mat')
target = x(:, :, y ==1);
nontarget = x(:, :, y==0);

%% averaging epochs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% channel 1 what is the result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
channelIDX = 1;
channel1Data = squeeze(x(channelIDX, :, :));
channel1Lbl = y';
% do target and nontarge separation
channel1Target = channel1Data(:, channel1Lbl==1);
channel1NonTarget = channel1Data(:, channel1Lbl==0);
% plot it out
subplot(3, 1, 1)
plot(channel1NonTarget)
title(['Nontarget (1) vs target (2) on channel' num2str(channelIDX)]);
subplot(3, 1, 2)
plot(channel1Target)
% have average view of what is target and nontarget
subplot(3, 1, 3)
plot(mean(channel1NonTarget, 2))
hold on 
plot(mean(channel1Target, 2))
title(['Average nontarget vs target on channel' num2str(channelIDX)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 30 channels what is the result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:6
    for j = 1:5
        % plot the average of them
        subplot(6, 5, 5*(i-1)+j)
        channelData = squeeze(x(5*(i-1)+j, :, :));
        channelLbl = y';
        avgChannelTarget = mean(channelData(:, channelLbl==1), 2);
        avgChannelNonTarget = mean(channelData(:, channelLbl==0), 2);
        plot(avgChannelTarget)
        hold on 
        plot(avgChannelNonTarget)
        set(gca, 'XTickLabel',{'0','0.2','0.4','0.6','0.8','1'},'FontSize',8);
        xlabel('Time');
        ylabel('\muVotage');
        h_legend = legend('Alert','Drowsy','Location','NorthEast');
        set(h_legend,'FontSize',5);
        hold off
    end
end

% also plot the average of whole 30 channels
avgAllChannelTarget = mean(mean(x(:, :, y==1)), 3);
avgAllChannelNonTarget = mean(mean(x(:, :, y==0)), 3);
plot(avgAllChannelTarget, 'LineWidth',4)
hold on 
plot(avgAllChannelNonTarget, 'LineWidth',4)
%plot(mean(x(:, :, y==1), 3)', 'LineStyle','-')
%plot(mean(x(:, :, y==0), 3)', 'LineStyle',':')
set(gca, 'XTickLabel',{'0','0.2','0.4','0.6','0.8','1'},'FontSize',12);
xlabel('Time');
ylabel('\muVotage');
legend('Alert','Drowsy','Location','NorthEast');
title(['ERP of Alert&Drowsy Across Channels']);

hold on;
for i = 1:6
    for j = 1:5
        % plot the average of them
        channelData = squeeze(x(5*(i-1)+j, :, :));
        channelLbl = y';
        avgChannelTarget = mean(channelData(:, channelLbl==1), 2);
        avgChannelNonTarget = mean(channelData(:, channelLbl==0), 2);
        r = rand(1);
        g = rand(1);
        b = rand(1);
        color = [r, g, b];
        plot(avgChannelTarget, 'LineWidth',1, 'LineStyle', '-', 'Color',color);
        plot(avgChannelNonTarget, 'LineWidth',1, 'LineStyle', ':', 'Color',color);
    end
end

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualizing ERP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% this code must execute separately

% plot ERP of all channels
figure; pop_timtopo(EEG, [8 992.1875], [NaN], 'ERP and scalp maps for drowsy');

% plot spectopo of all channels
figure; pop_spectopo(EEG, 1, [0 992.1875], ...
    'EEG' , 'freq', [5 7.5 10.5], 'freqrange',[0.1 12],'electrodes','off');

% script to reset EEG set structure
NTsize = size(target, 3);
EEG.trials = NTsize;
EEG.data = target;

for i = 155:NTsize
    EEG.epoch(i) = EEG.epoch(1);
end
EEG = eeg_checkset(EEG);
EEG.epoch(NTsize:end) = [];

% reference to 0
EEG.data = nontarget;
EEG.data = target;
for i = 1:30
    EEG.data(i, :, :) = squeeze(EEG.data(i, :, :)) -...
        repmat(mean(squeeze(EEG.data(i, :, :)), 1), 250, 1);
end

% plot the ERP of a selected channel 16
figure; pop_erpimage(EEG,1, [16],[[]],'Pz',256,128,{},[],'' ...
    ,'yerplabel','\muV','erp','on','cbar','on','phasesort',[0 0 10] );

% plot the ERP and the ITC of channel 16
for channelIDX = 1:30
    figure; pop_erpimage(EEG,1, [channelIDX],[[]],...
    'Cz',256,128,{},[],'' ,'yerplabel','\muV',...
    'erp','on','cbar','on','phasesort',[0 0 9 11] ,'coher',[9 11 0.01] ...
    ,'topo', { [channelIDX] EEG.chanlocs EEG.chaninfo } );
    savefig(['TargetChannel' num2str(channelIDX) '.fig']);
end

%% time-freq analysis
% Morlet wavelet
figure; pop_newtimef( EEG, 1, 16, [0  996], [3 0.5] , 'topovec', 16, ...
    'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, 'caption', 'Cz', ...
    'baseline',[0], 'plotphase', 'off', 'padratio', 16, 'winsize', 128);

% plot coherence between channels
figure; pop_newcrossf( EEG, 1, 1, 16, [0  996], [3 0.5] ,'type', ...
    'phasecoher', 'topovec', [1  16], 'elocs', EEG.chanlocs, 'chaninfo', ...
    EEG.chaninfo, 'title','Channel Fp2-F9 Phase Coherence','padratio', 16);

%% project 2 RBM space (obsolete)
% normalization, not required
mu=mean(x, 3);	
sigma=max(std(x, 0, 3),eps);
xx=bsxfun(@minus,x,mu);
xx=bsxfun(@rdivide,x,sigma);
% scale to 0-1
A = max(max(max(x)));
B = min(min(min(x)));
range = [min(B);max(A)];
xx	= (x - range(1))/(-range(1)+range(2));

[err2, batchposhidprobs2, batchposvisprob2] = rbm4Projection(xx, 2, 1 );
scatter(batchposhidprobs2(1, 1, y==0), batchposhidprobs2(1, 2, y==0), 'Marker','.');
hold on
scatter(batchposhidprobs2(1, 1, y==1), batchposhidprobs2(1, 2, y==1), 'Marker','.');

% parse using 100 hidden units
A = batchposhidprobs100(:, :, y==0);
B = batchposhidprobs100(:, :, y==1);
D = squeeze(B(:, :, 1))';
for i = 1:10
    for j = 1:10
        subplot(10, 10, (i-1)*10+j)
        imagesc(pdist2(squeeze(B(:, :, (i-1)*10+j)), D'));
    end
end
for i = 1:10
    for j = 1:10
        subplot(10, 10, (i-1)*10+j)
        imagesc(pdist2(squeeze(A(:, :, (i-1)*10+j)), D'));
    end
end

%% project 2 RBM space (new)

% random selection
hidden_exp = 8;
alpha = logspace(0,-4, 10);
hidden = 2.^(1:hidden_exp);

randomRun = 36;
randomIdx = randi([0, 10*hidden_exp-1], randomRun, 1);
Az_RAW_Epoch = cell(1, randomRun);
Az_RBM_Epoch = cell(1, randomRun);
parfor i = 1:randomRun
    idx = randomIdx(i);
    alphaIdx = mod(idx, 10)+1;
    hiddenIdx = floor(idx/10)+1;
    [Az_RAW_Epoch{i}, Az_RBM_Epoch{i}, ~] = rbm4ProjectionNew(...
        alpha(alphaIdx), hidden(hiddenIdx), 1        , x   , y    , false  , true    , 5);
        % learn rate   , hidden unit      , numepochs, data, label, sigmoid, baggging, fold
end

% parse data
FitboxAzRaw = zeros(10, hidden_exp);
FitboxAzRBM = zeros(10, hidden_exp);
for i = 1:randomRun
    idx = randomIdx(i);
    alphaIdx = mod(idx, 10)+1;
    hiddenIdx = floor(idx/10)+1;
    FitboxAzRaw(alphaIdx, hiddenIdx) = Az_RAW_Epoch{i};
    FitboxAzRBM(alphaIdx, hiddenIdx) = Az_RBM_Epoch{i};
end
subplot(2, 1, 2);
pcolor(FitboxAzRaw);
subplot(2, 2, 2);
pcolor(FitboxAzRBM);

% [~, ~, HidNormLayer1] = rbm4ProjectionNew(...
%         alpha(4), hidden(7), 1, data, y, false, true);

%% project 2 DBN space
[ ~, ~, Az_Sigm_20Epoch, Er_Sigm_20Epoch ] = dbn4ProjectionNew( x, y );

% parse data
FitboxAzRaw = zeros(10, 3);
FitboxAzRBM = zeros(10, 3);
for i = 1:randomRun
    idx = randomIdx(i);
    alphaIdx = mod(idx, 10)+1;
    hiddenIdx = floor(idx/10)+1;
    FitboxAzRaw(alphaIdx, hiddenIdx) = Az_Sigm_20Epoch{i};
    FitboxAzRBM(alphaIdx, hiddenIdx) = Er_Sigm_20Epoch{i};
end

end

