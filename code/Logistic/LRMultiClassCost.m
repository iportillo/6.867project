function C = LRMultiClassCost(X,Y,W)
    W = reshape(W, [size(X,2)+1,size(Y,2)]);
    tmp = [X ones(size(X,1),1)]*W;
    tmp = tmp - repmat(max(tmp,[],2),[1,size(tmp,2)]);
    k = sum(Y.*tmp,2) - log(sum(exp(tmp),2));
    C = - sum(k);
end