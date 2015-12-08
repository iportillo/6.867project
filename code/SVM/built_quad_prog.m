function [Hy, f, A, b, Aeq, beq] = built_quad_prog(Y, C)
    % X vector of features
    % Y vector of classifications labels (+,-,1)

    N = size(Y,1);

    A = zeros(2*N, N);
    
    Aeq = zeros(1,N);
    beq = 0;

    f = ones(N,1);
    b = zeros(2*N,1);
    %Build H and A
    for i = 1:N
        A(i,i) = -1;
        A(N + i,i) = 1;
        b(N + i) = C;
        Aeq(i) = Y(i);
    end
    Hy = Y*Y';