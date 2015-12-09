ax = axes();
hold on;
L = [0.1 0.3 0.5 0.7 0.8 0.85 0.9 0.95];
methods = {'LDA', 'DLDA', 'QDA', 'DQDA', 'RLDA'};

for i = 1:length(methods)
    plot(ax, (1-L)*8600, 100-er(i,:), 'LineWidth', 2)
end
xlabel('Classification Samples');
ylabel('Test Error (%)');
set(ax, 'FontSize', 16);
grid(ax, 'on')
legend(ax, methods, 'Location', 'Best');