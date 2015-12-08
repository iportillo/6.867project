function Y = predictSVMFast(X, w, w0)
    res = [];
    for d = 1:size(w,2)
        res(:,d) = (X*w(:,d)+w0(d));
    end
    [~,Y] = max(res,[],2);
end