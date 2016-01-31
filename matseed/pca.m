function [U, S] = pca(X)

 [r, c] = size(X);

 %Sanity checking assignation
 U = zeros(size(c));
 S = zeros(size(c));

 %Running PCA on the dataset X
 Sigma = 1/r * (X' * X); %covariance matrix
 [U, S, V] = svd(Sigma); %finding eigenvectors and eigenvalues

end
