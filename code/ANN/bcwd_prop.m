function [DE1, DE2, yk] = bcwd_prop(D, N, K, w, X, Y, sfunc)
% Dimensions of w:
% w{1}{2} (D+1)xN
% w{2}{2} (N+1)xK
% yk = ISxK 
% y = ISxK
% x = ISxD
% a2 = ISxK
% d2 = ISxK
% d1 = ISxN
% a1 = ISxN

DE1 = zeros(D+1 ,N );
DE2 = zeros(N+1, K);
[a1, z1, a2, yk] =fwd_prop(D, N, K, w, X, sfunc);

w2 = w{2};
d2 = (yk - Y)./(yk.*(1-yk)).*sfunc(a2).*(1-sfunc(a2));
d1 = [(w2(1:end-1,:)*d2') .* (sfunc(a1).*(1-sfunc(a1)))']';
    
DE1 = X'*d1;
DE2 = z1'*d2;

end