function [confusion_matrix_knn_2d,accuracy_knn_2d,confusion_matrix_full_2d,accuracy_full_2d,...
    confusion_matrix_lda_2d,accuracy_lda_2d]=classify_with2d_function(train_features,train_classes,test_features,test_classes,k)
%show a table with the classification results for the classificantion
%method
%Input param: train_features,train_classes,test_features,test_classes
%             k for knn ,classlist for displaying cfmatrix
addpath('../Task1');
addpath('../Task2');
digits(6);
% reduce dimension
train_2=train_features(:,1:2);
test_2=test_features(:,1:2);

% knn
[~,confusion_matrix_knn_2d,accuracy_knn_2d]=knn_function(k,train_2,test_2,train_classes,test_classes,1);
 

%gaussian_full
[~,~,confusion_matrix_full_2d,accuracy_full_2d] = gaussian_full_function(train_2,test_2,train_classes,test_classes,1);

%gaussian_lda
[~,~,confusion_matrix_lda_2d,accuracy_lda_2d]=gaussian_lda_function(train_2,test_2,train_classes,test_classes,1);

end





