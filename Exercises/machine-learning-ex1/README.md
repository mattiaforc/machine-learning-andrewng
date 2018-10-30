# Exercise 1
*NOTE: This file illustrates the main solutions to the problems specified in ex1.pdf, referred to Andrew Ng's Coursera Machine Learning course. It shows only snippets of code modified from the start kit.*

## Graphical results:

![](/Pictures/Ex-1-1.png)

![](/Pictures/Ex-1-2.png)

![](/Pictures/Ex-1-3.png)
## computeCost.m
Compute cost for linear regression.
COMPUTECOST(X, y, theta) computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

```matlab
for i = 1:m
  J = J+(theta' * X(i, :)' - y(i, :))^2; 
end
J = J/(2*m)
```

## computeCostMulti.m
Compute cost for linear regression with multiple variables.
COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

```matlab
J = ((X*theta - y)' * (X*theta - y))/(2*m);
```

## featureNormalize.m
Normalizes the features in X.
FEATURENORMALIZE(X) returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when working with learning algorithms.

```matlab
mu = mean(X);
sigma = std(X);

for i = 1:size(X,2)
    X_norm(:,i) = (X(:,i) - mu(i)) / sigma(i);
end
```

## gradientDescent.m
Performs gradient descent to learn theta.
GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by taking num_iters gradient steps with learning rate alpha.

```matlab
% Temporary values
t1 = 0;
t2 = 0;

for i = 1:m
  t1 = t1 + (theta' * X(i, :)' - y(i));
  t2 = t2 + (theta' * X(i, :)' - y(i))*X(i, 2);
end 
theta(1) = theta(1) - (alpha/m) * t1;
theta(2) = theta(2) - (alpha/m) * t2;
computeCost(X, y, theta) 
```

## gradientDescentMulti.m
Performs gradient descent to learn theta.
GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by taking num_iters gradient steps with learning rate alpha.

```matlab
h = X * theta;
temp = zeros(3, 1);
for row = 1:size(theta, 1)
    temp(row) = theta(row) - alpha / m * sum((h - y) .* X(:,row));
end
theta = temp;
```
## normalEqn.m
Computes the closed-form solution to linear regression.
NORMALEQN(X,y) computes the closed-form solution to linear regression using the normal equations.

```matlab
theta = pinv(X'*X) *X'*y;
```