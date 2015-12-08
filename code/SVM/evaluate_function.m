function ret = evaluate_function(f, point, mode)
    if strcmp(mode, 'analytic')
        p = num2cell(point);
        ret = f(p{:});
        return;
    elseif strcmp(mode, 'approximation') 
        f(point)
    end
end