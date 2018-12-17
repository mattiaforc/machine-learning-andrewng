# Exercise 7
*NOTE: This file illustrates the main solutions to the problems specified in ex7.pdf, referred to Andrew Ng's Coursera Machine Learning course. It shows only snippets of code modified from the start kit.*

## Graphical Results

![](/Pictures/Ex-7-1.png)

![](/Pictures/Ex-7-2.png)

![](/Pictures/Ex-7-3.png)

![](/Pictures/Ex-7-4.png)

![](/Pictures/Ex-7-5.png)

![](/Pictures/Ex-7-6.png)

![](/Pictures/Ex-7-7.png)

![](/Pictures/Ex-7-8.png)

![](/Pictures/Ex-7-9.png)

![](/Pictures/Ex-7-10.png)

## pca.m
PCA Run principal component analysis on the dataset X
[U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X.
Returns the eigenvectors U, the eigenvalues (on diagonal) in S

```matlab
Sigma = (1/m) * X' * X;
[U, S, V] = svd(Sigma);
```

## projectData.m
Z = projectData(X, U, K) computes the projection of the normalized inputs X into the reduced dimensional space spanned by the first K columns of U. It returns the projected examples in Z.

```matlab
for i = 1:size(X, 1)
    x = X(i, :)';
    Z(i, :) = x' * U(:, 1:K);
end
```

## recoverData.m
RECOVERDATA(Z, U, K) recovers an approximation the original data that has been reduced to K dimensions. It returns the approximate reconstruction in X_rec.

```matlab
for i = 1 : size(Z, 1)
    v = Z(i, :)';
    for j=1: size(U, 1)
        X_rec(i, j) = v' * U(j, 1:K)';
    end
end
```

## findClosestCentroids.m
FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids in idx for a dataset X where each row is a single example. idx = m x 1 vector of centroid assignments (i.e. each entry in range [1..K])

```matlab
for i=1:size(X,1)
  temp = zeros(1, K);
  for j=1:K
    temp(j) = sum(abs(X(i, :) - centroids(j, :)).^2);
  end
  [value, idx(i)] = min(temp);
end
```

## computeCentroids.m
COMPUTECENTROIDS(X, idx, K) returns the new centroids by computing the means of the data points assigned to each centroid. It is given a dataset X where each row is a single data point, a vector idx of centroid assignments (i.e. each entry in range [1..K]) for each example, and K, the number of centroids. You should return a matrix centroids, where each row of centroids is the mean of the data points assigned to it.

```matlab
num = zeros(K,1);
sum = zeros(K,n);
for i = 1:size(idx,1)
	z = idx(i);
	num(z) += 1;
	sum(z,:) += X(i,:);
end

centroids = sum./num;
```

## kMeansInitCentroids.m
KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X

```matlab
% Randomly reorder the indices of examples
randidx = randperm(size(X, 1));
% Take the first K examples as centroids
centroids = X(randidx(1:K), :);
```