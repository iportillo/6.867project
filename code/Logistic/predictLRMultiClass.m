function Y = predictLRMultiClass(X, w)
    res = [];
    X = [X ones(size(X,1),1)];
    for d = 1:size(w,2)
        res(:,d) = (X*w(:,d));
    end
    [~,Y] = max(res,[],2);
end