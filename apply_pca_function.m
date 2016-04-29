function [r1,r2,e1,e2,E2,X_pca]=apply_pca_function(train_features)
%r1:largest eigenvalue
%r2:second largest eigenvalue
%e1:first five rows of E2
%e2:first five rows of X-pca
%E2:D-by-2matrix,holds the two eigenvectors for the largest two eigenvalues
%X:input data set,here X=train_features
digits(6); 
[vecs,vals]=compute_pca(train_features);
E2=vecs(:,1:2); 
X_pca=train_features*E2;
r1=vals(1);
r2=vals(2);
e1=E2(1:5,:);
e2=X_pca(1:5,:);
fprintf('The two eigenvalues are %i and %i \n',r1,r2);
fprintf('The first five rows of the E2 are\n');
fprintf([repmat('%f\t', 1, size(E2, 2)) '\n'], e1(1:5,:)');
fprintf('The first five rows of Xpca are\n');
fprintf([repmat('%f\t', 1, size(X_pca, 2)) '\n'], e2(1:5,:)');
end


