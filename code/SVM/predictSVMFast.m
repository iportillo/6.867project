function [Y,p] = predictSVMFast(X, W, w0)
    res = [X ones(size(X,1),1)]*[W; w0];
    res = res - repmat(max(res,[],2),[1,size(res,2)]);
    p = exp(res)./repmat((sum(exp(res),2)),[1,size(res,2)]);
    [~,Y] = max(res,[],2);
end