function [y, p] = predictQDA(X,g,b,mu,S, type)
    p = zeros(size(X,1), size(mu,1));
    warning('off','all')
    if ~isempty(strfind(type, 'LDA'))
        p = (X*b' + repmat(g,[1,size(X,1)])');
    else
        for i = 1:size(mu,1)
            m = repmat(mu(i,:), [size(X,1),1]);
            p(:,i) = -bsxfun(@dot,((X-m)*inv(S(:,:,i)))',(X-m)'); 
        end
    end
    [~, y] = max(p, [], 2);
end