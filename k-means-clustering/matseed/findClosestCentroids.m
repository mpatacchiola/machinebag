function idx = findClosestCentroids(X, centroids)

 K = size(centroids, 1);
 idx = zeros(size(X,1), 1);

 for i=1:size(X,1)
  x = X(i,:);
  distance = zeros(K,1);

  %computing the distance between the two vectors
  for j=1:K
   distance(j) = sum((x - centroids(j,:)) .^ 2);
  end

  [U, I] = min(distance);
  idx(i) = I; %assigning the index
 end

end

