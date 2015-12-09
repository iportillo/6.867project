ax = axes();
hold on;
dists = {'euclidean','chebychev','minkovski'};
K = [1,3,5,9,11,15,21,35,50];
for i = 1:length(dists)
    for l = 1:length(K)
        ev(i,l) = sum(Ytest == Ypred(:,i+1,l),1)./length(Ytest);
    end
    plot(ax, K, ev(i,:), 'LineWidth', 2)
end
xlabel('K (number of neighbors)');
ylabel('Accuracy (%)');
set(ax, 'FontSize', 16);
grid(ax, 'on')
legend(ax, dists, 'Location', 'Best');

figure;
ax = axes();
dists = {'euclidean','cosine','chebychev','minkovski'};
K = [1,3,5,9,11,15,21,35,50];
for i = 1:length(dists)
    plot(ax, K, time/969, 'LineWidth', 2)
end
xlabel('K (number of neighbors)');
ylabel('Time (s)');
set(ax, 'FontSize', 16);
grid(ax, 'on')
legend(ax, dists, 'Location', 'Best');
