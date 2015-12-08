function Y_est = predictSVMforX(sol, X, Y, w0, fun, x )
    for j = 1:size(x,1)
        v = x(j,:);
        pred = w0;
        for i = 1:size(X,1)
            pred = pred + sol(i).*Y(i)*fun(v, X(i,:));
        end
        Y_est(j) = pred;
    end
end