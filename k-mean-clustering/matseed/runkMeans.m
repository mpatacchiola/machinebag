function [centroids, idx] = runkMeans(X, initial_centroids, max_iters)

% Initialization
[m n] = size(X);
K = size(initial_centroids, 1);
centroids = initial_centroids;
previous_centroids = centroids;
idx = zeros(m, 1);

fprintf('\nStarting K-Means for %d iterations \n', max_iters); 

for i=1:max_iters
    
    fprintf('K-Means iteration %d \n', i);

    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
       
    idx = findClosestCentroids(X, centroids); % assigning each example to the closest centroid           
    centroids = computeCentroids(X, idx, K); % computing new centroids
end

end

