function [nn,nn_full_c] = nnbp(nn,nn_full_c,keep,keep_full_c)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
%     sparsityError = 0;
%     switch nn.output
%         case 'sigm'
%             d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
%         case {'softmax','linear'}
%             d{n} = - nn.e;
%     end
%     
    %%%%%%%%%%%%%%%%%%%%%%% back propagation in fully connected layer
    n_f=nn_full_c.n;
  
    if n_f==2
        
        d_f{2} =  - nn_full_c.e .* (nn_full_c.a{1,2} .* (1 - nn_full_c.a{1,2}));

        nn_full_c.dW{1,1} = (d_f{2}' * nn_full_c.a{1,1}) / size(d_f{2}, 1);
        
    end
    
    %%%%%%%%%%%%%%%%%%%%% back propagation in convolutional layer
     n = nn.n;
     
     if n==3
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       tt=nn_full_c.size(1)/30;
       u=2;
       v=tt+1;
       for j=1:30
           
       d{3} = (d_f{2} *nn_full_c.W{1,1}(:,u:v)) .* keep{1,j}{1,3} .* (1 - keep{1,j}{1,3});   
       d{2} = (d{3} *nn.W{1,2}) .* keep{1,j}{1,2} .* (1 - keep{1,j}{1,2});   
       
       nn.dW{j,2} = (d{3}' * keep{1,j}{1,2}) / size(d{3}, 1);
       nn.dW{j,1} = (d{2}(:,2:end)' * keep{1,j}{1,1}) / size(d{2}, 1);
       u=u+tt;
       v=v+tt;
       end
       
       nn.dW_sum{1,1}=zeros(size(nn.dW{1,1}));
       for j=1:30
       nn.dW_sum{1,1}=nn.dW_sum{1,1}+nn.dW{j,1};
       end
        nn.dW_sum{1,1}= nn.dW_sum{1,1}/30;
        
       nn.dW_sum{1,2}=zeros(size(nn.dW{1,2}));
       for j=1:30
       nn.dW_sum{1,2}=nn.dW_sum{1,2}+nn.dW{j,2};
       end
       nn.dW_sum{1,2}=nn.dW_sum{1,2}/30;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
     end
    
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
     
     
     if n==2
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       tt=nn_full_c.size(1)/30;
       u=2;
       v=tt+1;
       for j=1:30
           
       d{2} = (d_f{2} *nn_full_c.W{1,1}(:,u:v)) .* keep{1,j}{1,2} .* (1 - keep{1,j}{1,2});   
       
       
       nn.dW{j,1} = (d{2}' * keep{1,j}{1,1}) / size(d{2}, 1);
      
       u=u+tt;
       v=v+tt;
       end
       
       nn.dW_sum{1,1}=zeros(size(nn.dW{1,1}));
       for j=1:30
       nn.dW_sum{1,1}=nn.dW_sum{1,1}+nn.dW{j,1};
       end
       
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
     end
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     for i = (n - 1) : -1 : 2
%         % Derivative of the activation function
%         switch nn.activation_function 
%             case 'sigm'
%                 d_act = nn.a{i} .* (1 - nn.a{i});
%             case 'tanh_opt'
%                 d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
%         end
%         
%         if(nn.nonSparsityPenalty>0)
%             pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
%             sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
%         end
%         
%         % Backpropagate first derivatives
%         if i+1==n % in this case in d{n} there is not the bias term to be removed             
%             d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
%         else % in this case in d{i} the bias term has to be removed
%             d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
%         end
%         
%         if(nn.dropoutFraction>0)
%             d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
%         end
% 
%     end
% 
%     for i = 1 : (n - 1)
%         if i+1==n
%             nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
%         else
%             nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
%         end
%     end
end
