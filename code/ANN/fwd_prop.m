function [a1, z1, a2, yk] =fwd_prop( D, N, K, w, x, sfunc)
% Dimensions of w:
% w{1} (D + 1)xN
% w{2} (N + 1)xK

% Add a column of ones to the input for the bias coefficient
w1 = w{1};
a1 = x*w1;
z1 = sfunc(a1);

% Add a column of ones to the intermediate varibales for the bias coefficient
z1 = [ z1 ones(size(z1,1),1)];
w2 = w{2};
a2 = z1*w2;
yk = sfunc(a2);
end