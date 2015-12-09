 function [ax, Cmat, DA]= confusion_matrix(predicted,labels,classes_names, text)
%This function allows to compute and plot confusion matrix after
%classification process
%Inputs: 
%labels: N by 1 vector of known labels, should be numeric
%classes:  N by 1 vector of pridicted labels
%classes_names: cell array cotaining names of classes Example: {'AWA','S1','S2','SWS','Rem'}
%Ouptputs:
%DA= Decoding accuracy value ie mean diagonal of the confusion matrix
%Cmat: confusion matrix values
C=confusionmat(labels,predicted);

if sum(0==unique(labels))
    C = C(2:end, 2:end);
end

L=length(unique(labels));
Cmat = zeros(L);
Cmat2 = zeros(L);
for i=1:L
    Cmat(i,:)=C(i,:)./sum(C(i,:));
    Cmat2(i,:)=C(i,:);
end
Cmat2 = Cmat2>1;
if isempty(classes_names)
    classes_names = arrayfun(@(x) num2str(x), unique(labels), 'UniformOutput', false)';
end
figure('visible','on');
ax = axes();
im = imagesc(Cmat);colormap([[1 1 1];repmat([1 0 0],10,1);repmat([0 1 0],90,1)]);caxis([0,1]);
im.AlphaData = .5;
points = [8,12,16,24,39];
for p = points
    line([(p+0.5) (p + 0.5)],get(ax,'YLim'),'Color',[0.3 0.3 0.3]);
    line(get(ax,'XLim'),[(p+0.5) (p + 0.5)],'Color',[0.3 0.3 0.3]);
end
axis square
set(gca,'XTick',1:L,'XTickLabel',classes_names,'YTick',1:L,'YTickLabel',classes_names,'TickLength',[0,0],'FontSize',7,'FontName','Times New Roman');


if text
    textstr=num2str(Cmat(:),'%0.2f');
    textstr=strtrim(cellstr(textstr));
    textstr = reshape(textstr, [L L]);
    [x,y]=meshgrid(1:L);
    hstrg=text(x(:),y(:),textstr(:),'HorizontalAlignment','center','FontSize',7,'FontName','Times New Roman');
    midvalue=mean(get(gca,'Clim'));
    textColors=repmat(Cmat(:)>midvalue,1,3);
    set(hstrg,{'color'},num2cell(textColors,2));
end
DA=mean(diag(Cmat))*100;
end