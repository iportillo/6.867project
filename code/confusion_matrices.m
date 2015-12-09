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
YCtrain = 2*YC(idx_train,:) - 1;

Xval = HOG(idx_val,:);
Yval = Y(idx_val);
YCval = 2*YC(idx_val,:) - 1;

Xtest = HOG(idx_test,:);
Ytest = Y(idx_test);
YCtest = 2*YC(idx_test,:) - 1;

%% Compute confusion matrices for different models
% LR
W = rand(size(Xtrain,2)+1,size(YCtrain,2))/10;
[NLL, W, f_calls, iter] = GDLRMultiClass(Xtrain,YCtrain,Ytrain, W, 0.001, 1e-6);
Y = predictLRMultiClass(Xtest, W);
im = confusion_matrix(Ytest, Y, [], 0);
print('.\report\figures\confusionLR','-dpng')

% kNN
Y = kNN(Xtest, Xtrain, Ytrain, 1, 'minkovski', 1);
im = confusion_matrix(Ytest, Y, [], 0);
print('.\report\figures\confusionkNN','-dpng')

%SVM
fun = getFun('linear');
H = computeH(Xtrain, fun);
for d = 1:size(YCtrain, 2)
    [sol, w0, w, n_w, s_idx, m_idx, fun]  = solve_SVN(Xtrain(1:5:end,:), YCtrain(1:5:end,d), H(1:5:end,1:5:end), fun, 100 );
end
Y = predictSVMFast(Xtest, w, w0);
im = confusion_matrix(Ytest, Y, [], 0);
print('.\report\figures\confusionSVM','-dpng')

% GDA
[g,b,mu,S] = GDA(Xtrain, Ytrain, 'RLDA', 0.1);
[Y,p]   = predictQDA(Xtest,g,b,mu,S, 'RLDA');
im = confusion_matrix(Ytest, Y-1, [], 0);
print( '.\report\figures\confusionGDA','-dpng');

% ANN
sfunc = @(x) sigmoid(x);
[min, w, iter] = gradient_descent('batch', 60, Xtrain, YCtrain, sfunc, 1e-4, 0.0001, 1e-6);
scoreFun = @(x) predict(x, Ytrain, n, w, sfunc);
[Y, YCtest_pred] = scoreFun(Xtest);
im = confusion_matrix(Ytest, Y-1, [], 0);
print( '.\report\figures\confusionANN','-dpng');