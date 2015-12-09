clear all
close all
clc
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

%% Fit the logistic model
W = rand(size(Xtrain,2)+1,size(YCtrain,2))/10;
[NLL, W, f_calls, iter] = GDLRMultiClass(Xtrain,YCtrain,Ytrain, W, 0.001, 1e-6);
Y = predictLRMultiClass(Xtrain, W);
er = sum(Ytrain==Y)/length(Ytrain);
save('LR_results', 'Ytrain', 'Xtrain', 'Xtest', 'Ytest', 'W','er');
confusion_matrix(Ytrain, Y, [])