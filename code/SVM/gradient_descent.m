function [min, opt_point, f_calls, iter] = gradient_descent(f, step_size, initial_point, threshold, mode,  varargin)
    plot_graph = false;
    iter = 1;
    
    if isempty(varargin) == false
        ax = varargin{1};      
        plot_graph = true;
    end
    
    if strcmp(mode, 'analytic')
        jac.function = jacobian(f.function, f.variables);
        jac.function = matlabFunction(jac.function);
        jac.variables = f.variables;
        
    else
        jac = f;
    end
    
    f_mat = matlabFunction(f.function);
        
    diff = Inf;
    f_0 = evaluate_function(f_mat, initial_point, 'analytic');
    f_calls = 1;
    point = initial_point;
    while diff > threshold
        
        points(iter, :) = point;
        f_vals(iter, :) = f_0;
        
        g = compute_gradient(jac, point, mode);
        new_point = point - step_size*g*1.0;
        
        % Comptue the value of the new function
        f_1 = evaluate_function(f_mat, new_point, 'analytic');
        
        % Compute the values between consecutive optimizations
        diff = f_0 - f_1;
        
        % Upadte the values of the variable for next iteration
        f_0 = f_1;
        point = new_point;
        
        % Compute the number of function calls
        if strcmp(mode, 'analytic')
            f_calls = f_calls + 2; 
        elseif strcmp(mode, 'approximation') 
            f_calls = f_calls + 3; 
        end
        iter = iter + 1;
    end
    
    min = f_1;
    opt_point = point;
    
    if plot_graph
        h = plot(ax, double(points(:,1)), double(points(:,2)), '-');
        it = 1;
        for p_i = 1:size(points,1)
            p = points(p_i,:);
            col = get(h, 'Color');
            plot(ax, p(1), p(2), 'o', 'MarkerSize', round(10/(it)^0.3), 'Color', col);
            it = it + 1;
        end
    end
end

