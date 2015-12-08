function err = LRMultiClassErr(X,Y,W)
    X = [X ones(size(X,1),1)];
    tmp = X*W;
    tmp = tmp - repmat(max(tmp,[],2),[1,size(tmp,2)]);
    a = exp(tmp);
    mu_y = (a./repmat((sum(exp(tmp),2)),[1,size(tmp,2)]) - Y);
    mu_y(a == 0) = 0;
    err = zeros(size(X,2),size(Y,2));
    for c = 1:size(Y,2)
        mu = mu_y(:,c);
        err(:,c) = sum(bsxfun(@times,mu,X),1);
    end
end