function show_cumulative_variance()
%Show plot of the cumulative variance
addpath('..');
load('svhn.mat');
[~,Evals]=compute_pca(train_features);

%part1:plot cummulative variabnce 
Yk=cumsum(Evals)/(sum(Evals));
plot(1:1:size(Evals,1),Yk);
xlabel('k');
ylabel('Yk');
title('Cumulative Variance');
end

%part2:calculate k components to explain at least 90% of total variance of
%the data
%function [k]=show_cumulative_variabnce(input) 
%[Evecs,Evals]=compute_pca(input);
%Yk=0;
%for i=1:size(Evals,1)
%Yi=Evals(i)/sum(Evals); Yk=Yk+Yi;
%if Yk>0.9
%k=i;
%break 
%end

    