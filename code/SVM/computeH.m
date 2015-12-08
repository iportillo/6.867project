function H =computeH(X, fun)
    N = size(X,1);
    H = zeros(N);
    for i = 1:N
        H(i,:) = bsxfun(fun, X', repmat(X(i,:),[size(X,1),1])');
        if mod(i,100)==0
            fprintf('%d\n',i);
        end
    end
end