addpath('..');
load('svhn.mat');
[r1,r2,e1,e2,E2,X_pca] = apply_pca_function(train_features);