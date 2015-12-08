Ypred = Yk3(idx_fail,:);
YCtest(idx_fail)
for i = 1:size(ims,1)
    im = reshape(ims(i,:), 28,28);
    figure;
    ax = subplot(1,2,1);
    imshow(abs(fliplr(rot90(im,3))-1));
    ax = subplot(1,2,2);
    barh(1:6, Ypred(i,:), 'b');
    axis square
    grid(ax, 'on')  
    xlabel('Probability value')
    ylabel('Predicted value')
    set(ax, 'FontSize', 16);
    print(['.\figures\MNIST_fail' num2str(i)], '-dpng');
end
close all

plotconfusion(Ytest',Yk3')