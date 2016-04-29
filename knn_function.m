function [prediction,confusion_matrix,accuracy] = knn_function(k,train_features,test_features,train_classes,test_classes,type)
%UNTITLED Summary of this function goes here
%    k = the number of neighbours
%    train_features = the train_features
%    test_features = the test_features
%    train_classes = the train_classes
%    test_classes = the test_classes
%    type = 1(running for task 2) or 2(running for task 3)
digits(6);
[t_row,~] = size(test_features);
confusion_matrix = zeros(10,10);
accuracy = zeros(t_row,1);
prediction = zeros(1,t_row);
for  i = 1:t_row
    [~,neighbors]=top_k_neighbors(train_features,test_features(i,:),k);
    k_class=train_classes(neighbors);
    a1=tabulate(k_class);
    a2=a1(:,1);
    [temp,m]=sort(a1(:,2),'descend');
    a3=a2(m);

    if size(a1,1)>1
        if temp(1)==temp(2)
            [~,neighbor]=top_k_neighbors(train_features,test_features(i,:),1);
            result=train_classes(neighbor);
        else
            result=a3(1);
        end
    else
        result=a1(1);
    end
    prediction(1,i) = result;
end
if type==1
    for i = 1:t_row
        confusion_matrix(test_classes(1,i),prediction(1,i))...
        = confusion_matrix(test_classes(1,i),prediction(1,i))+1;
    end
    b = diag(confusion_matrix);
    c = sum(confusion_matrix,2);
    accuracy = b/c;
    accuracy = sum(accuracy(:,1))/10;
end
end

function [dist,neighbors] = top_k_neighbors(train_features,test_feature,k)
%Input  param:  train_features test_features k 
%Output param:  dist:distances of k neighbors in ascending order
%               neighbors:index of neihbors

[size_x,~] = size(train_features); %size_x=10000 
test_mat = repmat(test_feature,size_x,1);
dist_mat = (train_features-double(test_mat)).^2;
dist_array = sum(dist_mat');
[dists,neighbors] = sort(dist_array);
dist = dists(1:k);
neighbors = neighbors(1:k);
end

