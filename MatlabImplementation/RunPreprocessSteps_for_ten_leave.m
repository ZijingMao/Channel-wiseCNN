function [ PSD ] = RunPreprocessSteps_for_ten_leave( PSD_EEG)
%SELECTTRAININGSET Summary of this function goes here
% Inputs:
%   PSD_target_EEG    - select target EEG for event record from target EEG set
%   PSD_nontarget_EEG    - select nontarget EEG for event record from nontarget EEG set
%
% Output:
%   PSD_target    - select features from target epoch data set
%   PSD_nontarget   - select features from non-target epoch data set
f_no=11;
frequencies = logspace(log10(2),...
    log10(12), f_no);

points = 250;%
%data = eeg_getdatact(PSD_EEG, 'component', [1:size(PSD_EEG.icaweights,1)]); %PSD_EEG.data;

m_size=PSD_EEG;
mmm=size(m_size,3);

data = zeros(30, 250, mmm);

for i = 1 : mmm
    data(:, :, i) =PSD_EEG(:, :, i); %weights*sphere*bsxfun(@minus,PSD_EEG.data(:, :, i),avg);
end

trials = mmm;

PSD=zeros(30,f_no,...
   250 ,mmm);

% wavelet transform
[wavelet,~,~,~] = dftfilt3(frequencies, [3 6],...
    250);

for c=1:30
    disp(['Calculate PSD: C',num2str(c)]);
    for w = 1:f_no
        for epoch =1:size(data,3)
            singleEpochPower = squeeze(abs(conv(data(c,:, epoch), wavelet{w}, 'same')) .^2);
            ds=singleEpochPower(1:1:points);
            PSD(c,w, :, epoch) = ds;
        end
    end
end

% log PSD
% switch Parameters.log_PSD
%     case 'Y'
%         PSD=log(PSD);
%     otherwise
% end

return

