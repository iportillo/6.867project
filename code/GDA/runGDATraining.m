clear all
close all

load('./train_images/data.mat')

HOG(Y==0,:) = [];
YC(Y==0,:) = [];
Y(Y==0,:) = [];

L = [0.1 0.3 0.5 0.7 0.8 0.85 0.9 0.95];
for k = L
    disp(k)
    %% Hold Out validation and test datasets
    N = size(HOG,1);
    N_val_test = round(N*k);

    idx_train = true(N,1);
    idx_test = false(N,1);
    idx_val = false(N,1);

    idx = randperm(N);
    idx = idx(1:N_val_test);
    idx_train(idx) = 0;

    idx_val(idx(1:round(N_val_test*0.4))) = 1; 
    idx_test(idx(round(N_val_test*0.6) + 1:end)) = 1; 

    Xtrain = HOG(idx_train,:); 
    Ytrain = Y(idx_train);

    Xval = HOG(idx_val,:);
    Yval = Y(idx_val);

    Xtest = HOG(idx_test,:);
    Ytest = Y(idx_test);

    %% Train GDA
    methods = {'LDA', 'DLDA', 'QDA', 'DQDA'};

    for i = 1:length(methods)
        %disp(['Training ' methods{i}])
        [g,b,mu,S] = GDA(Xtrain, Ytrain, methods{i});
        [ytrain,p] = predictQDA(Xtrain,g,b,mu,S, methods{i});
        [yval,p]   = predictQDA(Xval,g,b,mu,S, methods{i});
        [ytest,p]  = predictQDA(Xtest,g,b,mu,S, methods{i});

        fprintf('Accuracy %s: %2.3f \n', methods{i}, sum(Ytest == ytest)/length(Ytest)*100);
    end

    %% Train Regularized GDA
    methods = {'RLDA', 'RQDA'};
    lambda = linspace(0,1,21);
    for i = 1:length(methods)

        for j = 1:length(lambda)
            l = lambda(j);
            [g,b,mu,S] = GDA(Xtrain, Ytrain, methods{i}, l);
            [yval,p]   = predictQDA(Xval,g,b,mu,S, methods{i});
            e_val(j) = sum(Yval == yval -1);
        end

        % Select lambda based on maximum accuracy in validation dataset
        idx = find(e_val==max(e_val));
        l = lambda(idx(1));
        [g,b,mu,S] = GDA(Xtrain, Ytrain, methods{i}, l);
        [ytest,p]  = predictQDA(Xtest,g,b,mu,S, methods{i});
        fprintf('Accuracy %s: %2.3f lambda: %1.2f \n', methods{i}, sum(Ytest == ytest)/length(Ytest)*100, l);
    end
end