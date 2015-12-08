function [min, opt_point, iter] = gradient_descent(mode, N, X, Y, sfunc, lambda, step_size, threshold)
    iter = 1;
    diff = Inf;
        
    D = size(X,2);
    K = size(Y,2);
    NIS = size(Y,1);
    
    X = [X ones(size(X,1),1)];

    
    w{1} = (rand(D+1, N)-0.5);
    w{2} = (rand(N + 1, K)-0.5);
    
    [~, ~, ~, yk] = fwd_prop( D, N, K, w, X, sfunc);
    
    w1 = w{1};
    w1=w1(1:end,:);
    w2 = w{2};
    w2=w2(1:end,:);
    
    f_0 = 1.0/NIS*sum(sum( -Y.*log(yk) - (1-Y).*log(1-yk))) +...
                lambda*(dot(w1(:),w1(:)) + dot(w2(:),w2(:)));
    
    if strcmp(mode, 'stochastic')
        threshold = -Inf;
        k = 0.8;
        t0 = 100;
    end
                
    while iter < 3000 && (strcmp(mode, 'stochastic') || diff > threshold)
        
        if strcmp(mode, 'stochastic')
            i = randi(size(Y,1));
            [DE1, DE2, yk] = bcwd_prop(D, N, K, w, X(i,:), Y(i,:), sfunc);
            step_size = (t0 + iter)^-k;
        else
            [DE1, DE2] = bcwd_prop(D, N, K, w, X, Y, sfunc);
        end
        
        w{1} = w{1} - step_size*(DE1 + 2*lambda*abs(w{1}));
        w{2} = w{2} - step_size*(DE2+ 2*lambda*abs(w{2}));
        
        % Comptue the value of the new function
        [~, ~, ~, yk] = fwd_prop( D, N, K, w, X, sfunc);      
        w1 = w{1};
        w1=w1(1:end,:);
        w2 = w{2};
        w2=w2(1:end,:);

        f_1 = 1.0/NIS*sum(sum( -Y.*log(yk) - (1-Y).*log(1-yk))) +...
                    lambda*(dot(w1(:),w1(:)) + dot(w2(:),w2(:)));

        
        % Compute the values between consecutive optimizations
        diff = f_0 - f_1;
        
        % Upadte the values of the variable for next iteration
        f_0 = f_1;       

        iter = iter + 1;
        
        if mod(iter,100) == 0
            fprintf('Training ANN [%d] Objective: %2.2f\n', iter, f_0);
        end
    end
    
    min = f_1;
    opt_point = w;
end

