function [prediction,determinant,confusion_matrix,accuracy] = gaussian_full_function(train_features,test_features,train_classes,test_classes,type)
%  type = 1(run for task 2) or 2(run for task 3)
digits(6);
[row,~] = size(test_features);
[row_train,~] = size(train_features);
determinant = zeros(1,10);
test_result = zeros(row,10);
accuracy = zeros(row,1);
confusion_matrix = zeros(10,10);
for i = 1:10
    class = train_features(train_classes==i,:);
    [N,~] = size(class);
    mean = sum(class,1) / N;
    means = repmat(mean,N,1);
    different = class - means;
    covariance = different' * different / N;
    invco = inv(covariance);
    determinant(1,i) = det(covariance);
    for j = 1:row
        test_different = test_features(j,:) - mean;
        test_result(j,i) = test_different*invco*test_different'*(-0.5);
        test_result(j,i) = test_result(j,i)-0.5*log(determinant(1,i));
        test_result(j,i) = test_result(j,i)+log(N/row_train);
    end
end
[~,idx] = sort(test_result,2,'descend');
prediction = idx(:,1);
if type==1
    for i = 1:row
        confusion_matrix(test_classes(1,i),idx(i,1))...
        = confusion_matrix(test_classes(1,i),idx(i,1))+1;
    end
    b = diag(confusion_matrix);
    c = sum(confusion_matrix,2);
    accuracy = b/c;
    accuracy = sum(accuracy(:,1))/10;
end
end

