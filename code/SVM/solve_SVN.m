function [sol, w0, w, n_w, s_idx, m_idx, fun] = solve_SVN(X, Y, H, fun, C)
N = size(X,1);

[Hy, f, A, b, Aeq, beq] = built_quad_prog(Y, C);

H2 = -H.*Hy;

optim_ver = ver('optim');
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex', 'MaxIter', 30, 'Display', 'off');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end
% Maximize the function is equivalent to minimize -func 
[sol, val] = quadprog(-H2, -f, A, b, Aeq, beq, [], [], [], opts);
sol= sol';

threshold = 1e-7;
m_idx = sol > threshold & sol < C-threshold;
s_idx = sol > threshold;

m_idx2 = find(sol > threshold & sol < C-threshold);
s_idx2 = find(sol > threshold);

% Compute the values of the vectors
tmp = repmat((sol(s_idx2)'.*Y(s_idx2)),[1,size(X,2)]).*X(s_idx2,:);
w = sum(tmp, 1);

w0 = 0;
if ~isempty(m_idx2) && ~isempty(s_idx2)
    tmp = repmat(sol(s_idx2).*Y(s_idx2)',[length(m_idx2),1]).*H(m_idx2, s_idx2);
    w0 = (Y(m_idx2) - sum(tmp,2));
    w0 = sum(w0(:));
end

if ~isempty(m_idx2)
    w0 = 1/length(m_idx2)*w0;
else 
    w0 = 0;
end

n_w = 0;
