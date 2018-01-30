%======================== PART 1 ============================
%The dataset is loaded and shuffled. The dataset is then
%projected on a 3D space and visualized.

close all; close all; clc
% Load the seed dataset
%The name of the variable is X
load('seed.mat');

%shuffle the rows of X
X =  X(randperm(end),:); 

%The dataset is made of 8 scalars
%the last one represents the
%seed category.
y = X(:,end);
X = X(:,1:end-1); %cutting off the last element

%Normalising the X matrix
[X_norm, mu, sigma] = featureNormalization(X);

%Applying PCA on normalized data
[U, S] = pca(X_norm);

% Projection on D=3 dimensions
K = 3;
Z = X * U(:, 1:K); %taking the top K eigenvectors in U (first K columns)

%Visualizing the data in 3D
figure;
scatter3(Z(:, 1), Z(:, 2), Z(:, 3), 10, y);
title('original labels');


%======================== PART 2 ============================
%Here we apply the k-means algorithm for clustering the data
% in three groups. After that we show the 3D plot of the data.

fprintf('\n* Now we will apply the k-mean algorithm. Press enter to continue.\n');
pause;

% Initialize the centroids picking them randomly from X
K = 3; 
initial_centroids = zeros(K, size(X, 2));%security checking
randidx = randperm(size(X, 1));
initial_centroids = X(randidx(1:K), :);

%Start the k-means algorithm
max_iterations = 15;
[centroids, idx] = runkMeans(X, initial_centroids, max_iterations);

%  Setup Color Palette
palette = eye(K);
colors = palette(idx, :);

%  Visualize the data in 3D
figure;
scatter3(Z(:, 1), Z(:, 2), Z(:, 3), 10, colors);
title('k-means clustering');


%======================== PART 3 ============================
%The data are projected on a 2D surface to better visualize
%the results of the k-mean clustering.

fprintf('\n* We will project the results in a plane. Press enter to continue.\n');
pause;

% Project the data in 2D
K = 2;
Z = X * U(:, 1:K); %taking the top K eigenvectors in U (first K columns)

% Plot in 2D
figure;

% Plot the original labels
subplot(1,2,1);
scatter(Z(:,1), Z(:,2), 15, y);
axis('square');
title('original labels')

% Plot the k-mean results
palette = eye(D + 1);
colors = palette(idx, :);
subplot(1,2,2);
scatter(Z(:,1), Z(:,2), 15, colors);
axis('square');
title('k-means clustering');

