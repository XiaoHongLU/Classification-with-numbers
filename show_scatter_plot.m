function show_scatter_plot( )
%scatter plot of two dimensions of each data point with color 
%input: X=train_features
%       train_classes=train_classes
addpath('..');
load('svhn.mat');
digits(6); 
[~,~,~,~,~,Xpca] = apply_pca_function(train_features);
data_names = [{'1'},{'2'},{'3'},{'4'},{'5'},{'6'},{'7'},{'8'},{'9'},{'0'}];
figure;
hold on;
x = Xpca(:,1);
y = Xpca(:,2);
y1=scatter(x(train_classes==1),y(train_classes==1),36,'y','filled');%yellow
y2=scatter(x(train_classes==2),y(train_classes==2),36,'m','filled');%magenta
y3=scatter(x(train_classes==3),y(train_classes==3),36,'c','filled');%cyan
y4=scatter(x(train_classes==4),y(train_classes==4),36,'r','filled');%red
y5=scatter(x(train_classes==5),y(train_classes==5),36,'g','filled');%green
y6=scatter(x(train_classes==6),y(train_classes==6),36,'b','filled');%blue
y7=scatter(x(train_classes==7),y(train_classes==7),36,'k','filled');%black
y8=scatter(x(train_classes==8),y(train_classes==8),36,[1,0.4,0.6],'filled');%pink
y9=scatter(x(train_classes==9),y(train_classes==9),36,[0.5,0,0],'filled');%marron
y0=scatter(x(train_classes==10),y(train_classes==10),36,[0,0.5,0.5],'filled');%teal
legend([y1,y2,y3,y4,y5,y6,y7,y8,y9,y0],data_names,'Location','best');
xlabel('Xpca(x)');
ylabel('Xpca(y)');
title('Xpca Data Graph');
end


