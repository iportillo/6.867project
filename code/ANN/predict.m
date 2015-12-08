function [Y, YC] = predict(X, Y, N, w, sfunc)
    D = size(X,2);
    K = size(Y,2);
    X = [X ones(size(X,1),1)];

    [~, ~, ~,Y] = fwd_prop( D, N, K, w, X, sfunc);
    [~, YC] = max(Y,[],2);
end