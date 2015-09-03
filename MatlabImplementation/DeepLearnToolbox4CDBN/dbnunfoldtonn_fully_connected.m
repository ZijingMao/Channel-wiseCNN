function nn = dbnunfoldtonn_fully_connected(dbn, outputsize)
%DBNUNFOLDTONN Unfolds a DBN to a NN
%   dbnunfoldtonn(dbn, outputsize ) returns the unfolded dbn with a final
%   layer of size outputsize added.
%     if(exist('outputsize','var'))
        

size = [(dbn.sizes(end))*30 outputsize];% 30 is number of channels
%     else
%         size = [dbn.sizes];
%     end
    nn = nnsetup(size);
%     for i = 1 : numel(dbn.rbm)
%         nn.W{i} = [dbn.rbm{i}.c dbn.rbm{i}.W];
%     end
end
