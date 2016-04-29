addpath('..');
load('svhn.mat');
 [confusion_matrix_knn_2d,accuracy_knn_2d,confusion_matrix_full_2d,accuracy_full_2d,...
    confusion_matrix_lda_2d,accuracy_lda_2d]=classify_with2d_function(train_features,train_classes,test_features,test_classes,1);