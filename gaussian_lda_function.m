function [ prediction,determinant,confusion_matrix,accuracy ] = gaussian_lda_function(train_features,test_features,train_classes,test_classes,type)
%  type = 1(running for task 2) or 2(running for task 3)
digits(6);
[N_train,~] = size(train_features);
[N_test,~] = size(test_features);
test_result = zeros(N_test,10);
confusion_matrix = zeros(10,10);
accuracy = zeros(N_test,1);
mean = sum(train_features,1) / N_train;
means = repmat(mean,N_train,1);
different = train_features - means;
covar = different' * different / N_train;
determinant = det(covar);
invco = inv(covar);
for i = 1:10
    class = train_features(train_classes==i,:);
    [N,~] = size(class);
    mean = sum(class,1) / N;
    Wk_T = mean * invco;
    Wk_0 = -0.5*Wk_T*mean' + log(N/N_train);
    for j = 1:N_test
        test_result(j,i) = Wk_T*test_features(j,:)'+Wk_0;
    end
end
[~,idx] = sort(test_result,2,'descend');
prediction = idx(:,1);
if type==1
    for i = 1:N_test
        confusion_matrix(test_classes(1,i),idx(i,1))...
        = confusion_matrix(test_classes(1,i),idx(i,1))+1;
    end
    b = diag(confusion_matrix);
    c = sum(confusion_matrix,2);
    accuracy = b/c;
    accuracy = sum(accuracy(:,1))/10;
end
end

