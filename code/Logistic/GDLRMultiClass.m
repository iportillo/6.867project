function [min, opt_point, f_calls, iter] = GDLRMultiClass(X, Y, Yt, initial_point, step_size, threshold)
   
    iter = 1;
    f_0 = LRMultiClassCost(X,Y,initial_point);
    diff = Inf;
    f_calls = 1;
    point = initial_point;
    while diff > threshold && iter < 300
        
        g = LRMultiClassErr(X,Y,point);
        new_point = point - step_size*g*1.0;
        
        % Comptue the value of the new function
        f_1 = LRMultiClassCost(X,Y,new_point);
        
        if isnan(f_1) || mod(iter, 25)==0
            disp(iter);
            Y2 = predictLRMultiClass(X, point);
            sum(Yt==Y2)/length(Yt)
        end
        % Compute the values between consecutive optimizations
        diff = f_0 - f_1;
        
        % Upadte the values of the variable for next iteration
        f_0 = f_1;
        point = new_point;
        
        % Compute the number of function calls
        f_calls = f_calls + 2; 
        iter = iter + 1;
    end
    
    min = f_1;
    opt_point = point;
end
