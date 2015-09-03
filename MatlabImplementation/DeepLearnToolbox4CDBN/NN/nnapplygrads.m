function [nn,nn_full_c] = nnapplygrads(nn,nn_full_c)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
%         if(nn.weightPenaltyL2>0)
%             dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
%         else
            dW = nn.dW_sum{1,i};
            
%         end
        
        dW = nn.learningRate * dW;
        
%         if(nn.momentum>0)
%             nn.vW{i} = nn.momentum*nn.vW{i} + dW;
%             dW = nn.vW{i};
%         end
            
        nn.W{1,i} = nn.W{1,i} - dW;
    end
    
    %%%% fully connected layer
    for i = 1 : (nn_full_c.n - 1)
%         if(nn.weightPenaltyL2>0)
%             dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
%         else
            dW_f = nn_full_c.dW{1,i};
            
%         end
        
        dW_f = nn_full_c.learningRate * dW_f;
        
%         if(nn.momentum>0)
%             nn.vW{i} = nn.momentum*nn.vW{i} + dW;
%             dW = nn.vW{i};
%         end
            
       nn_full_c.W{1,i} =nn_full_c.W{1,i} - dW_f;
    end
    
    
    
    
    
end
