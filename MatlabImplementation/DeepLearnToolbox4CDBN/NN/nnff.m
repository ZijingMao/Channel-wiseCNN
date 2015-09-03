function [nn,nn_full_c,keep,keep_full_c] = nnff(nn,nn_full_c, batch_x, batch_y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)
    y=batch_y;
    x=batch_x;
    n = nn.n;
    m = size(x, 3);%m = size(x, 1);
    
    %% 
    for j=1:30
    x_1=squeeze(batch_x(j,:,:))';
    x_1 = [ones(m,1) x_1];
    nn1=nn;
    nn1.a{1} = x_1;

    %feedforward pass
    for i = 2 : n-1
        switch nn1.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn1.a{i} = sigm(nn1.a{i - 1} * nn1.W{i - 1}');
            case 'tanh_opt'
                nn1.a{i} = tanh_opt(nn1.a{i - 1} * nn1.W{i - 1}');
        end
        

        nn1.a{i} = [ones(m,1) nn1.a{i}];
    end
    
    nn1.a{n} = sigm(nn1.a{n - 1} * nn1.W{n - 1}');% output of convolution
    
    keep{1,j}=nn1.a;
    
    clear nn1;
    end
   
    %% concatinate the activation of the output of convolution for input of fully connected layer
     if nn.n==3
    in_of_ful_con_lay=[keep{1,1}{1,3},keep{1,2}{1,3}, keep{1,3}{1,3}, keep{1,4}{1,3},...
        keep{1,5}{1,3}, keep{1,6}{1,3}, keep{1,7}{1,3}, keep{1,8}{1,3}, keep{1,9}{1,3}, keep{1,10}{1,3},...
        keep{1,11}{1,3}, keep{1,12}{1,3}, keep{1,13}{1,3}, keep{1,14}{1,3},...
        keep{1,15}{1,3}, keep{1,16}{1,3}, keep{1,17}{1,3}, keep{1,18}{1,3}, keep{1,19}{1,3}, keep{1,20}{1,3},...
        keep{1,21}{1,3}, keep{1,22}{1,3}, keep{1,23}{1,3}, keep{1,24}{1,3},...
        keep{1,25}{1,3}, keep{1,26}{1,3}, keep{1,27}{1,3}, keep{1,28}{1,3}, keep{1,29}{1,3}, keep{1,30}{1,3}];
     elseif nn.n==2
       in_of_ful_con_lay=[keep{1,1}{1,2},keep{1,2}{1,2}, keep{1,3}{1,2}, keep{1,4}{1,2},...
        keep{1,5}{1,2}, keep{1,6}{1,2}, keep{1,7}{1,2}, keep{1,8}{1,2}, keep{1,9}{1,2}, keep{1,10}{1,2},...
        keep{1,11}{1,2}, keep{1,12}{1,2}, keep{1,13}{1,2}, keep{1,14}{1,2},...
        keep{1,15}{1,2}, keep{1,16}{1,2}, keep{1,17}{1,2}, keep{1,18}{1,2}, keep{1,19}{1,2}, keep{1,20}{1,2},...
        keep{1,21}{1,2}, keep{1,22}{1,2}, keep{1,23}{1,2}, keep{1,24}{1,2},...
        keep{1,25}{1,2}, keep{1,26}{1,2}, keep{1,27}{1,2}, keep{1,28}{1,2}, keep{1,29}{1,2}, keep{1,30}{1,2}];  
     end
    in_of_ful_con_lay = [ones(m,1) in_of_ful_con_lay];
    %% feed forward of fully connected layer
     
    nn1=nn_full_c;
    nn1.a{1} = in_of_ful_con_lay;
     n = nn1.n;
    %feedforward pass
    if n>=2
    for i = 2 : n-1
        switch nn1.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn1.a{i} = sigm(nn1.a{i - 1} * nn1.W{i - 1}');
            case 'tanh_opt'
                nn1.a{i} = tanh_opt(nn1.a{i - 1} * nn1.W{i - 1}');
        end
       

        nn1.a{i} = [ones(m,1) nn1.a{i}];
    end
    
    nn1.a{n} = sigm(nn1.a{n - 1} * nn1.W{n - 1}');% output of fully connected layer
    
    else
        
        nn1.a{2} = sigm(nn1.a{2 - 1} * nn1.W{2 - 1}');% output of fully connected layer
    
    
    end 
    
    
    keep_full_c=nn1.a;
    nn_full_c.a=nn1.a;
    clear nn1;
    
    %%
    %error and loss
    y_real=[y ~y];
    nn_full_c.e = y_real - nn_full_c.a{n};
    nn_full_c.L = 1/2 * sum(sum(nn_full_c.e .^ 2)) / m; 
%     switch nn.output
%         case {'sigm', 'linear'}
%             nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
%         case 'softmax'
%             nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
%     end
end
