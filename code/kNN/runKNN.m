clear all
close all

load('./train_images/data.mat')

%% Hold Out validation and test datasets
N = size(HOG,1);
N_val_test = round(N*0.3);

idx_train = true(N,1);
idx_test = false(N,1);
idx_val = false(N,1);

idx = randi(N, N_val_test, 1); 
idx_train(idx) = 0;

idx_val(idx(1:round(N_val_test*0.4))) = 1; 
idx_test(idx(round(N_val_test*0.6) + 1:end)) = 1; 

Xtrain = HOG(idx_train,:); 
Ytrain = Y(idx_train);
YCtrain = 2*YC(idx_train,:) - 1;

Xval = HOG(idx_val,:);
Yval = Y(idx_val);
YCval = 2*YC(idx_val,:) - 1;

Xtest = HOG(idx_test,:);
Ytest = Y(idx_test);
YCtest = 2*YC(idx_test,:) - 1;

%% Train different k-NN classifiers with different parameters
K = [1,3,5,9,11,15,21,35,50];
dists = {'euclidean','cosine','chebychev','minkovski'};
for l = 1:length(K)
    k = K(l);
    for i = 1:length(dists)
        tic;
        Ypred(:,i,l) = kNN(Xtest, Xtrain, Ytrain, k, dists{i}, 1);
        time(i,l) = toc;
        fprintf('Accuracy %d-NN with %s distance: %2.3f \n', k, dists{i}, sum(Ytest == Ypred(:,i,l))/length(Ytest)*100);
    end
end
save('kNN_results', 'Ytrain', 'Xtrain', 'Ypred', 'Ytest', 'K', 'dists', 'time');