%% Jackie's modification to ex7 to compare result of kmean image compressing.

%% Initialization
clear ; close all; clc

fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Load an image of a bird
A = double(imread('222.JPG'));

A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels.
% Each row will contain the Red, Green and Blue pixel values
% This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

% Run your K-Means algorithm on this data
% You should try different values of K and max_iters here
K = 16; 
max_iters = 10;

for compression_round = 1:20

	% initialize the centroids randomly. 
	initial_centroids = kMeansInitCentroids(X, K);

	% Run K-Means
	[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

	% Find closest cluster members
	idx = findClosestCentroids(X, centroids);

	% We can now recover the image from the indices (idx) by mapping each pixel
	% (specified by it's index in idx) to the centroid value
	X_recovered = centroids(idx,:);

	% Reshape the recovered image into proper dimensions
	X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

	% Display the original image 
	subplot(4, 5, compression_round);
	imagesc(X_recovered)
	title(sprintf('Compressed, with %d colors.', K));
end

fprintf('Program paused. Press enter to continue.\n');

