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

%% Run SVN
fun = getFun('linear');
H = computeH(Xtrain, fun);

C = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, Inf];
C = Inf;
for i = 1:length(C)
    c = C(i);
    for d = 1:size(YCtrain, 2)
        [sol(i,:,d), w0(i,d), w(i,:,d), n_w, s_idx(i,:,d), m_idx(i,:,d), fun]  = solve_SVN(Xtrain(1:2:end,:), YCtrain(1:2:end,d), H(1:2:end,1:2:end), fun, c );
        predictSVM = @(x) predictSVMforX(sol(i,:,d), Xtrain, YCtrain(:,d), w0(i,d), fun, x );
        %fprintf('Trained SVN for class %d with C %1.2f\n', d, c);
    end
    
    % Predict 
    Yt(i,:) = predictSVMFast(Xtest, squeeze(w(i,:,:)), w0(i,:));
    er(i) = sum(Yt==Ytest)/length(Ytest)*100;
    fprintf('Accuracy for SVM with C %1.1e : %2.2f\n', c, er(i));
end