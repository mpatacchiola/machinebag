function centroids = computeCentroids(X, idx, K)

[r c] = size(X); %row and columns of X

centroids = zeros(K, c);

%computing the means of the data points assigned to each centroid.
%loop over the centroids
for i=1:K
 counter = 0;

 %for each centroid we check how many points
 %have been assigned to it in idx. 
 for j=1:size(idx,1)

  %if the value in idx is equal to the current
  %centroid then its corresponding vector in X
  %is counted for the mean
  if idx(j) == i
   centroids(i,:) = centroids(i,:) + X(j,:);
   counter = counter + 1;
  end

 end

 %normalizing the current centroid mean
 centroids(i,:) = centroids(i,:) / counter;
end

end

