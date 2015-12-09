function Ypred = kNN(Xclass, Xtrain, Ytrain, k, dist, varargin)

    % Determine the distance function
    if strcmp(dist, 'euclidean')
        fun = @(x,y) (x-y)*(x-y)';
    elseif strcmp(dist, 'cosine')
        fun = @(x,y) 1 - x*y'/sqrt((x*x')*(y*y'));
    elseif strcmp(dist, 'mahalanobis')
        CMLE = cov(Xtrain);
        C =  CMLE + 1e-10 *diag(diag(CMLE));
        E = inv(C);
        fun = @(x,y) (x-y)*E*(x-y)';
    elseif strcmp(dist, 'minkovski')
        p = varargin{1};
        fun = @(x,y) (sum(abs(x-y).^p))^(1.0/p);
    elseif strcmp(dist, 'chebychev')
        fun = @(x,y) max(abs(x-y));
    end

    % Classify each data point
    Ypred = zeros(size(Xclass,1),1);
    for i = 1:size(Xclass, 1)    
        % Compute the vector of distances among the points
        funw = @(x) fun(Xclass(i,:),x);
        D = arrayfun(@(j) funw(Xtrain(j,:)), 1:size(Xtrain,1));

        % Sort distances and choose the most representative label
        [~, I] = sort(D,2, 'ascend');
        candidateLabels = Ytrain(I(1:k));
        u = unique(candidateLabels);

        % If the unique value existing is 0, hist does not work, so this is
        % a quick and dirty hack.
        if u~=0
            [N,candidateLabels] = hist(candidateLabels,u);
            tmp = candidateLabels(N==max(N));
            Ypred(i) = tmp(1);
        else
            Ypred(i) = 0;
        end
    end

end