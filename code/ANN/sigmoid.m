function s = sigmoid(x)
    s = 1./(1+exp(-x));
    s = min(s, 1-1e-8);
    s = max(s, 1e-8);
end