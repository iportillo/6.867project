function [w, Nt, Nv, n_w, s_idx] = svm_test(name, C, kernel, varargin)
s = [];
bplot = true;
pref = 'P2';
if ~isempty(varargin)
    bplot = varargin{1};
    s = varargin{2};
    pref = varargin{3};
end

CM = flipud(redbluecmap(10));


disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data_',name,'_train.csv'));
X = data(:,1:2);
Y = data(:,3);

% Carry out training, primal and/or dual
%%% TODO %%%
clear sol w0 w1 s_idx m_idx fun
for i = 1:length(C)
    c = C(i);
    [sol(i,:), w0(i), w(i,:), n_w(i), s_idx(i,:), m_idx(i,:), fun] = solve_SVN(X, Y, c, kernel, s);

    % Define the predictSVM(x) function, which uses trained parameters
    %%% TODO %%%
    if strcmp(kernel, 'linear')
        predictSVM = @(x) x*w' + w0;
    else
        predictSVM = @(x) predictSVMforX(sol(i,:), X, Y, w0(i),fun, x );
    end

    for k = 1:size(X,1) 
        Y_est(k,i) =sign(predictSVM(X(k,:)));
    end

    Nt(i) = sum(Y_est(:,i) ~= Y);
    
    if bplot
        hold on;
        figure();
        ax = axes();

        % plot training results
        if length(s) > 0
            title = ['C = ' num2str(c) ' \sigma = ' num2str(s)];
        else
            title = ['SVM Train for ' name ' and C = ' num2str(c)];
        end
    
        [~, ~, xx,yy,zz] = plotDecisionBoundary(ax, X, Y, predictSVM, [-1, -1, 0, 1, 1], title , [0,0,0], 0);

        %Plot the training points
        scatter(ax, X(Y==1,1),X(Y==1,2),50,CM(end,:),'filled','o','MarkerEdgeColor', 'k');
        scatter(ax, X(-Y==1,1),X(-Y==1,2),50,CM(3,:),'filled','d','MarkerEdgeColor', 'k');

        %Plot MV
        scatter(ax, X(m_idx(i,Y(m_idx(i,:))==1) ,1),X(m_idx(i,Y(m_idx(i,:))==1) ,2),80,CM(end,:),'filled','o','MarkerEdgeColor', 'k', 'LineWidth',4);
        scatter(ax, X(m_idx(i,-Y(m_idx(i,:))==1) ,1),X(m_idx(i,-Y(m_idx(i,:))==1) ,2),80,CM(3,:),'filled','d','MarkerEdgeColor', 'k', 'LineWidth',4);

        %Plot SV
        scatter(ax, X(s_idx(i,Y(s_idx(i,:))==1) ,1),X(s_idx(i,Y(s_idx(i,:))==1) ,2),80,CM(end,:),'filled','o','MarkerEdgeColor', 'r', 'LineWidth',2);
        scatter(ax, X(s_idx(i,-Y(s_idx(i,:))==1) ,1),X(s_idx(i,-Y(s_idx(i,:))==1) ,2),80,CM(3,:),'filled','d','MarkerEdgeColor', 'r', 'LineWidth',2);
 
        % Make a square grid
        axis square

        grid on
        xlabel(ax, 'x_1')
        ylabel(ax,'x_2')
        set(ax, 'FontSize', 24)
        
        print(['.\figures\' pref strrep(num2str(c),'.','') '-' strrep(num2str(s),'.','') 'SVMtrain' name],'-dpng')

    end
end

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X2 = validate(:,1:2);
Y2 = validate(:,3);

% Pot validation results
for i = 1:length(C)
    c = C(i);

    if strcmp(kernel, 'linear')
        predictSVM = @(x) x*w' + w0;
    else
        predictSVM = @(x) predictSVMforX(sol(i,:), X, Y, w0(i),fun, x );
    end
    
    for k = 1:size(X2,1) 
        Y_est(k,i) =sign(predictSVM(X2(k,:)));

    end

    Nv(i) = sum(Y_est(:,i) ~= Y);
    
    % plot training results
    if bplot
        hold on;
        figure();
        ax = axes();

        if length(s) > 0
            title = ['C = ' num2str(c) ' \sigma = ' num2str(s)];
        else
            title = ['SVM Validation for ' name ' and C = ' num2str(c)];
        end
        plotDecisionBoundary(ax, X, Y, predictSVM, [-1, 0, 1], title, [0,0,0], 0);

        %Plot the training points
        scatter(ax, X(Y==1,1),X(Y==1,2),50,CM(end,:),'filled','o','MarkerEdgeColor', 'k');
        scatter(ax, X(-Y==1,1),X(-Y==1,2),50,CM(3,:),'filled','d','MarkerEdgeColor', 'k');
        
        % Make a square grid
        axis square
        grid on
        xlabel(ax, 'x_1')
        ylabel(ax,'x_2')
        set(ax, 'FontSize', 24)
        print(['.\figures\' pref strrep(num2str(c),'.','') '-' strrep(num2str(s),'.','') 'SVMvalidate' name],'-dpng')

    end
end

fprintf('Results for dataset %s\n', name)
for i = 1:length(C)
    fprintf('%1.2f, | %1.0f, %1.0f, | %s | %s\n', C(i), Nt(i), Nv(i), sprintf('%1.3f ',w(i,:)), sprintf('%1.3f ',sol(i,:)))
end