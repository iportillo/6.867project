function ret = compute_gradient(f, point, mode)
    if strcmp(mode, 'analytic')
        ret = evaluate_function(f.function, point, mode);
        return;
    elseif strcmp(mode, 'approximation') 
        epsilon = 1e-3;
        N_vars = length(f.variables);
        ret = zeros(1, N_vars);
        
        f_mat = matlabFunction(f.function);
        for i = 1:N_vars
            new_point2 = point;
            new_point1 = point;
            new_point1(i) = new_point1(i) - 0.5*epsilon;
            new_point2(i) = new_point2(i) + 0.5*epsilon;
            f_0 = evaluate_function(f_mat, new_point1, 'analytic');
            f_1 = evaluate_function(f_mat, new_point2, 'analytic');
            ret(i) = (f_1 - f_0)/(epsilon);
        end
        return
    end
end