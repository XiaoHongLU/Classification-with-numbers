addpath('..');
load('svhn.mat');
[prediction_lda,determinant_lda,confusion_matrix_lda,accuracy_lda] = gaussian_lda_function(train_features,test_features,train_classes,test_classes,1);