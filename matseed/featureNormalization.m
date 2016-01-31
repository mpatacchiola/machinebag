function [X_norm, mu, sigma] = featureNormalization(X)

 mu = mean(X); %dataset mean
 X_norm = bsxfun(@minus, X, mu);

 sigma = std(X_norm); %sigma
 X_norm = bsxfun(@rdivide, X_norm, sigma);

end
