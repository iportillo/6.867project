clear all;
close all;
clc;
%% Load the data
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
YCtrain = YC(idx_train,:);

Xval = HOG(idx_val,:);
Yval = Y(idx_val);
YCval = YC(idx_val,:);

Xtest = HOG(idx_test,:);
Ytest = Y(idx_test);
YCtest = YC(idx_test,:);

delete(gcp('nocreate'))
parpool(2);
%% Train and plot results
i = 1; j = 1;
N = [20 30 40 60 70 80];
l = 1e-4;
type = 'batch';
parfor i = 1:length(N)
    tic;
    n = N(i);
    sfunc = @(x) sigmoid(x);
    [min, w, iter] = gradient_descent(type, n, Xtrain, YCtrain, sfunc, 1e-4, 0.0001, 1e-6);

    scoreFun = @(x) predict(x, Ytrain, n, w, sfunc);

    [Yk, YCtrain_pred] = scoreFun(Xtrain);
    [Yk2, YCvalid_pred] = scoreFun(Xval);
    [Yk3, YCtest_pred] = scoreFun(Xtest);

    et(i) = sum(YCtrain_pred ~= Ytrain)*100/size(Ytrain,1);
    ev(i) = sum(YCvalid_pred ~= Yval)*100/size(Yval,1);
    etest(i) = sum(YCtest_pred ~= Ytest)*100/size(Ytest,1);
    time(i) = toc;
    weights{i} = w;

    tmp = struct();
    tmp.num_hidden = n;
    tmp.lambda = l;
    tmp.step = 0.001;
    tmp.max_iter = 3000;
    tmp.threshold = 1e-6;
    parameters{i} = tmp;
    toc;
    fprintf('N %d l %1.0e Ev %1.2f Etest %1.2f\n', n, l, ev(i), etest(i) );
end

save('./code/ANN/results/ANN_N.mat','parameters', 'weights', 'time', 'ev', 'et', 'etest');

%% Plot results
figure;
ax = subplot(2,1,1);
hold(ax, 'on');
plot(ax, N, ev, 'LineWidth',2);
grid(ax, 'on');
ylabel(ax, 'Validation Error (%)');
xlabel(ax, 'Number hidden nodes');
set(ax, 'FontSize', 20);
box on;

ax = subplot(2,1,2);
hold(ax, 'on');
plot(ax, N, time, 'LineWidth',2);
grid(ax, 'on');
ylabel(ax, 'Training Time (s)');
xlabel(ax, 'Number hidden nodes');
set(ax, 'FontSize', 20);
box on;
print(['.\report\figures\ANN_error'], '-dpng');