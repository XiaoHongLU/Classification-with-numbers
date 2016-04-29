addpath('..');
load('svhn.mat');
[prediction_knn,confusion_matrix_knn,accuracy_knn] = knn_function(1,train_features,test_features,train_classes,test_classes,1);