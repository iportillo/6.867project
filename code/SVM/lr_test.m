function [w, Nt, Nv] = lr_test(name, lambda)
disp('======Training======');
% load data from csv files

data = importdata(strcat('data_',name,'_train.csv'));
CM = flipud(redbluecmap(10));

X = data(:,1:2);
Y = data(:,3);

% Carry out training.
%%% TODO %%%
figure();
ax = axes();
cols = linspecer(length(lambda));
i = 1;

for i = 1:length(lambda)
    l = lambda(i);
    f = elr(l, X, Y);
    options = optimoptions(@fminunc,'Algorithm','quasi-newton');
    f_mat = (@(x) convert_function(matlabFunction(f.function),x));
    [w(i,:), val(i)] = fminunc(f_mat, zeros(1,3) , options);


    predictLR = @(x) (1/(1+exp(-(x*w(i,2:3)') + w(i,1))) - 0.5);
    hold on;
    axis square;
    % plot training results
    plotDecisionBoundary(ax, X, Y, predictLR, [0, 0], ['LR Train for ' name], cols(i,:), 0.5);
    
    % Do one by one because of numerical precision errors
    for k = 1:size(X,1) 
        Y_est(k,i) = 2*(predictLR(X(k,:)) > 0)-1;
    end
    
    Nt(i) = sum(Y_est(:,i) ~= Y);
    i = i + 1;
end

%legend(arrayfun(@(x)['\lambda = ' num2str(x)], lambda, 'UniformOutput', false));

%Plot the training points
scatter(ax, X(Y==1,1),X(Y==1,2),50,CM(end,:),'filled','o','MarkerEdgeColor', 'k');
scatter(ax, X(-Y==1,1),X(-Y==1,2),50,CM(3,:),'filled','d','MarkerEdgeColor', 'k');

% Plot the missclassified datapoints
if size(Y_est,2) == 1 
    scatter(ax, X(Y==1 & Y_est ~= Y,1),X(Y==1 & Y_est ~= Y,2),50,CM(end-3,:),'filled','o','MarkerEdgeColor', 'k');
    scatter(ax, X(-Y==1 & Y_est ~= Y,1),X(-Y==1 & Y_est ~= Y,2),50,CM(5,:),'filled','d','MarkerEdgeColor', 'k');
end


grid on
xlabel('x_1')
ylabel('x_2')
set(gca, 'FontSize', 24)
print(['.\figures\train' name],'-dpng')

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

figure()
axv = axes();
i = 1;
for i = 1:length(lambda)
    l = lambda(i);
    predictLR = @(x) (1/(1+exp(-(x*w(i,2:3)') + w(i,1))) - 0.5);

    % plot training results
    plotDecisionBoundary(axv, X, Y, predictLR, [0, 0], ['LR Validate for ' name], cols(i,:), 0.5);
    
    % Do one by one because of numerical precision errors
    for k = 1:size(X,1) 
        Y_est(k,i) = 2*(predictLR(X(k,:)) > 0)-1;
    end
    
    Nv(i) = sum(Y_est(:,i) ~= Y);
    i = i + 1;
end

%Plot the training points
scatter(axv, X(Y==1,1),X(Y==1,2),50,CM(end,:),'filled','o','MarkerEdgeColor', 'k');
scatter(axv, X(-Y==1,1),X(-Y==1,2),50,CM(3,:),'filled','d','MarkerEdgeColor', 'k');
axis square;
% Plot the missclassified datapoints
if size(Y_est,2) == 1 
    scatter(axv, X(Y==1 & Y_est ~= Y,1),X(Y==1 & Y_est ~= Y,2),50,CM(end-3,:),'filled','o','MarkerEdgeColor', 'k');
    scatter(axv, X(-Y==1 & Y_est ~= Y,1),X(-Y==1 & Y_est ~= Y,2),50,CM(5,:),'filled','d','MarkerEdgeColor', 'k');
end

grid on
xlabel(axv, 'x_1')
ylabel(axv,'x_2')
set(axv, 'FontSize', 24)
print(['.\figures\validate' name],'-dpng')

fprintf('Results for dataset %s\n', name)
for i = 1:length(lambda)
    fprintf('%1.2f, | %1.0f, %1.0f, | %s | %f\n', lambda(i), Nt(i), Nv(i), sprintf('%1.3f ',w(i,:)), val(i))
end
