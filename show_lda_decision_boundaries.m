function show_lda_decision_boundaries()
%Write code to visulise decision boundaries of the full covaruabce model
%and the LDA model from last section .Show the plots with both the colored
%decision boundaries and the scatter plot of the test set 
%Input param:label_names need loading 

digits(6);
addpath('../Task1');
addpath('../Task2');
addpath('..');
load('svhn.mat');
load('label_names.mat');
train_2=train_features(:,1:2);
test_2=test_features(:,1:2);%1000*2

plotDecisionBoundaries(test_2, test_classes, label_names,train_2,train_classes);
end

function plotDecisionBoundaries(X, true_labels, label_names,train_features,train_classes)
%Inputs:
%    X : your data to be visualized, N x 2, where N is the number of datapoints you are plotting.
%    true_labels : N sized vector of the true classes of the N data points 
%    label_names : Cell array with the name of the labels, i.e. {'1', '2', '3', ..., '0'}
%    classifier : This is the function that you've programmed to classify. It can be k-NN, or any of the two Gaussian classifiers. Note that you can add arguments to plotDecisionBoundaries if you classifier function needs more arguments or adapt it as you want.
%    
    
%   We define here the colormap we will use to colour each of the 10 classes.
    cmap = [0.80369089,  0.61814689,  0.46674357;
        0.81411766,  0.58274512,  0.54901962;
        0.58339103,  0.62000771,  0.79337179;
        0.83529413,  0.5584314 ,  0.77098041;
        0.77493273,  0.69831605,  0.54108421;
        0.72078433,  0.84784315,  0.30039217;
        0.96988851,  0.85064207,  0.19683199;
        0.93882353,  0.80156864,  0.4219608 ;
        0.83652442,  0.74771243,  0.61853136;
        0.7019608 ,  0.7019608 ,  0.7019608];

    % Stepsize defines how fine-grained we want our grid. The small the
    % value, the more resolution the visualization will have, at the
    % expense of computational cost (we would need to classify more
    % data-points since the grid would be denser).
    
%     stepSize = 0.05;
%     This is the original step, u can set 0.05->0.1 to
%     reduce the computational work or set 0.05->0.01 to get more accuracy

    stepSize = 0.1;
    
    x1range = (max(X(:,1)) - min(X(:,1)));
    x2range = (max(X(:,2)) - min(X(:,2)));
    
    x1plot = linspace(min(X(:,1)), max(X(:,1)), x1range/stepSize)';
    x2plot = linspace(min(X(:,2)), max(X(:,2)), x2range/stepSize)';
    
    % We obtain the grid vectors for the two dimensions.    
    [X1, X2] = meshgrid(x1plot, x2plot);
    
    % Concatenate them such that we can feed 'gridX' to your classifier.
    gridX = [X1(:), X2(:)];
    
    % Call here your classification method using the function you've coded to obtain the labels for each point
    % in the grid. Adapt this to you code!:
    
    [grid_labels,~,~,~] = gaussian_lda_function(train_features,gridX,train_classes,true_labels,2);

    % Now we create the figure to visualize:
    figure;
    
    % This function will draw the decision boundaries
    [C,h] = contourf(x1plot(:), x2plot(:), reshape(grid_labels, length(x2plot),length(x1plot)));
    set(h,'LineColor','none')

    % Important calls to properly define the color map:
    colormap(cmap);

    % Range of our class labels for the color mapping.
    caxis([1 10]);
    
    hold on;
    
    % Plot the scatter plots grouped by their classes, with black border.
    scatters = gscatter(X(:,1),X(:,2),true_labels, [0,0,0], 'o', 10);
    
    % Fill in the color of each point according to the class labels.
    for n = 1:length(scatters)
      set(scatters(n), 'MarkerFaceColor', cmap(n,:));
    end
    
    legend(scatters,label_names);    
    title('lda\_2d');
    hold off;
end