function s = tanh(x)
    s = (exp(x)-exp(-x))./(exp(x)+exp(-x));
    s = min(s, 1-1e-8);
    s = max(s, 1e-8);
end