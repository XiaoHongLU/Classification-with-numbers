function [Evecs,Evals]=compute_pca(X)
% For calculating the principal components of data set.
% X:N-by-D matrix,N=number of data points,D=dimension of data pint. i.e. train_features 10000*100 
% EVecs:D-by-D matrix:the eigen vectors of covariance matrix of X
% EVals:D-by-1 matrix:eigen valus of the covariance matrix of X. 
% Note:the returned eigenvectors should all have its first element positive
% and eigenvalues in desending order.

%significant figures 
digits(6);
%get row N and column D
[N,D]=size(X);

%--------------calculate the cov of input data set---------------
%Center the matrix of input data 
X=X-repmat(MyMean(X),N,1);%subtract the mean of the data 
X_cov=(X'*X)./(size(X,1)-1);%compute the cov matrix 

%--------------calculate the eig of the cov matric ---------------
[vecs,vals]=eig(X_cov);

%make the first element positive
for i=1:D
    vec_i=vecs(:,i);
    if vec_i(1)<0
        vec_i=-vec_i;  
        vecs(:,i)=vec_i;
    end
end 
% make valus in descending order
[Evals,i]=sort(diag(vals),'descend');
%vecs correspond  vals 
Evecs=vecs(:,i);
end


function [X_mean]=MyMean(X)
%For calculating the mean of input matrix
%We only to know the mean of each column(dimension) in this situation 
X_mean=sum(X)./(size(X,1));

%--------------Four:calculate ratio -----------
%ratio=0;
%for k=1:n
%    r=Evals(k)/sum(Evals);
%    ratio=ratio+r;
%    if(ratio>=0.9)
%       break;
%    end
%end
end


