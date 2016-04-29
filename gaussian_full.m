addpath('..');
load('svhn.mat');
[prediction_full,determinant_full,confusion_matrix_full,accuracy_full] = gaussian_full_function(train_features,test_features,train_classes,test_classes,1);