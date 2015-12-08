function [g,b,mu,S] = GDA(X,Y, type, varargin)

    % Determine size of input data
    [n, m] = size(X);

    % Discover and count unique class labels
    Y_u = unique(Y);
    k = length(Y_u);

    % Initialize
    Nc     = NaN(k,1);            % Group counts
    mu         = NaN(k,m);        % Group sample means
    S          = zeros(m,m,k);    % Pooled covariance
    g          = NaN(k,1);        % model coefficients
    b          = NaN(k,m);        % model coefficients

    % Loop over classes to perform intermediate calculations
    for i = 1:k
        % Establish location and size of each class
        idx_g      = (Y == Y_u(i));
        Nc(i)  = sum(double(idx_g));

        % Calculate group mean vectors
        mu(i,:) = mean(X(idx_g,:));

        % Accumulate pooled covariance information
        S(:,:,i) = (1 / Nc(i) ).* cov(X(idx_g,:));
    end

    % Assign prior probabilities
    Prior = Nc / n;

    % Modify the covariance matrix depending on the type of Discriminant
    % Analysis classification
    if ~isempty(strfind(type, 'LDA'))
        E = sum(S,3);

        % In case we have a regularization LDA
        if strcmp(type, 'RLDA')
            l = varargin{1};
            E = l*diag(diag(E)') + (1-l)*E ;
        end

        % In case we have a diagonal LDA
        if strcmp(type, 'DLDA')
            E = diag(diag(E)');
        end

        % Assign to every sigma matrix 
        for i = 1:k 
            S(:,:,i) = E;
        end
    else
         % In case we have a regularization LDA
        if strcmp(type, 'RQDA')
            l = varargin{1};
            for i = 1:k 
                S(:,:,i) = l*diag(diag(S(:,:,i))') + (1-l)* S(:,:,i) ;
            end
        elseif strcmp(type, 'DQDA')
            for i = 1:k 
                S(:,:,i) = diag(diag(S(:,:,i))');
            end
        elseif strcmp(type, 'QDA')
            l = 1e-10;
            for i = 1:k 
                S(:,:,i) = l*diag(diag(S(:,:,i))') + (1-l)* S(:,:,i) ;
            end
        end        
    end

    % Loop over classes to calculate linear discriminant coefficients
    for i = 1:k,
        % Intermediate calculation for efficiency
        % This replaces:  GroupMean(g,:) * inv(PooledCov)
        tmp = mu(i,:) / S(:,:,i);

        % Constant
        g(i) = -0.5 * tmp * mu(i,:)' + log(Prior(i));

        % Linear
        b(i,1:end) = tmp;
    end
    
end