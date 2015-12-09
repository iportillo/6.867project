clear all
close all

load('./train_images/data.mat')

HOG(Y==0,:) = [];
YC(Y==0,:) = [];
Y(Y==0,:) = [];

%% Hold Out validation and test datasets
N = size(HOG,1);
N_val_test = round(N*0.3);

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
    er(i,l) = sum(Ytest == ytest)/length(Ytest)*100;

    fprintf('Accuracy %s: %2.3f \n', methods{i}, er(i,l));
end

%% Train Regularized GDA
methods = {'RLDA', 'RQDA'};
lambda = linspace(0,1,21);
for i2 = 1:length(methods)

    for j = 1:length(lambda)
        k = lambda(j);
        [g,b,mu,S] = GDA(Xtrain, Ytrain, methods{i2}, k);
        [yval,p]   = predictQDA(Xval,g,b,mu,S, methods{i2});
        e_val(j) = sum(Yval == yval);
    end

    % Select lambda based on maximum accuracy in validation dataset
    idx = find(e_val==max(e_val));
    k = lambda(idx(1));
    [g,b,mu,S] = GDA(Xtrain, Ytrain, methods{i2}, k);
    [ytest,p]  = predictQDA(Xtest,g,b,mu,S, methods{i2});
    fprintf('Accuracy %s: %2.3f lambda: %1.2f \n', methods{i2}, sum(Ytest == ytest)/length(Ytest)*100, k);
    i = i + 1;
    er(i,l) = sum(Ytest == ytest)/length(Ytest)*100;
end
save('GDA_results', 'Ytrain', 'Xtrain', 'Xtest', 'Ytest', 'L','er');