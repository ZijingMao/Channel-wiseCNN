function [err_curve, batchposhidprobs, batchposvisprobs] = ...
    rbm4Projection(batchdata, numhid, maxepoch)

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning

epsilonw      = 0.0006;   % Learning rate for weights
epsilonvb     = 0.0006;   % Learning rate for biases of visible units
epsilonhb     = 0.0006;   % Learning rate for biases of hidden units
weightcost  = 0.05;
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases, numdims, numbatches]=size(batchdata);

epoch=1;

% Initializing symmetric weights and biases.
vishid     = 0.1*randn(numdims, numhid);
hidbiases  = zeros(1,numhid);
visbiases  = zeros(1,numdims);
% hidbiases = 0.1*randn(1, numhid);
% visbiases = 0.1*randn(1, numdims);

poshidprobs = zeros(numcases,numhid);
neghidprobs = zeros(numcases,numhid);
posprods    = zeros(numdims,numhid);
negprods    = zeros(numdims,numhid);
vishidinc  = zeros(numdims,numhid);
hidbiasinc = zeros(1,numhid);
visbiasinc = zeros(1,numdims);
batchposhidprobs=zeros(numcases,numhid,numbatches);
batchposvisprobs=zeros(numcases,numdims,numbatches);

err_curve = zeros(1, maxepoch);

for epoch = epoch:maxepoch,
    fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    for batch = 1:numbatches,
        if mod(batch, 2000) == 0
            fprintf(1,'epoch %d batch %d\r',epoch,batch);
        end
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
        batchposvisprobs(:, :, batch) = negdata;
        % negdata = normrnd(-poshidstates*vishid' - repmat(visbiases,numcases,1), 0.01);
        neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;
        
        if epoch>5,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = momentum*vishidinc + ...
            epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
    err_curve(epoch) = errsum;
    
    if mod(epoch, 10) == 0
        epsilonw      = epsilonw/5;   % Learning rate for weights
        epsilonvb     = epsilonvb/5;   % Learning rate for biases of visible units
        epsilonhb     = epsilonhb/5;   % Learning rate for biases of hidden units
    end
end;
