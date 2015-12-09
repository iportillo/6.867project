clear all
close all
clc
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

idx = randi(N, N_val_test, 1); 
idx_train(idx) = 0;

idx_val(idx(1:round(N_val_test*0.4))) = 1; 
idx_test(idx(round(N_val_test*0.6) + 1:end)) = 1; 

Xtrain = HOG(idx_train,:); 
Ytrain = Y(idx_train);
Xtest = HOG(idx_test,:);
Ytest = Y(idx_test);
YCtest = 2*YC(idx_test,:) - 1;
YCtrain = 2*YC(idx_train,:) - 1;

Xval = HOG(idx_val,:);
Yval = Y(idx_val);
YCval = 2*YC(idx_val,:) - 1;

%% Separe N samples from every training set
N = 5;
for i = 1:43
    k = i-1;
    idx = find(Ytrain == i);
    idx = datasample(idx,N);
    Xtrain2(k*N+1:k*N+5,:) = Xtrain(idx,:);
    Ytrain2(k*N+1:k*N+5) = Ytrain(idx); 
    YCtrain2(k*N+1:k*N+5,:) = YCtrain(idx,:);
end
Xtrain = Xtrain2;
Ytrain = Ytrain2';
YCtrain= YCtrain2;