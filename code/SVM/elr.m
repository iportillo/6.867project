function f = elr(lambda, x, y)
    if iscolumn(y)==0
        y = y';
    end
    
    f.name = ['ELR (' num2str(lambda) ')'];
    mv = sym(zeros((size(x,2) + 1),1));
    
    mv(1) = sym('w0', 'real');
    for k=1:size(x,2)
        mv(k+1) = sym(sprintf('w%d', k), 'real');
    end
    tmp = log(1 + exp(-y.*(x*mv(2:end) + mv(1))));
    f.function = sum(tmp) + lambda*(mv'*mv);
    f.variables = mv;
end