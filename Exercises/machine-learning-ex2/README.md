# Exercise 2
*NOTE: This file illustrates the main solutions to the problems specified in ex2.pdf, referred to Andrew Ng's Coursera Machine Learning course. It shows only snippets of code modified from the start kit.*

## Graphical results:

![](/Pictures/Ex-2-1.png)

![](/Pictures/Ex-2-2.png)

![](/Pictures/Ex-2-3.png)

![](/Pictures/Ex-2-4.png)

## CostFunction.m 
Compute cost and gradient for logistic regression.
COSTFUNCTION(theta, X, y) computes the cost of using theta as the parameter for logistic regression and the gradient of the cost w.r.t. to the parameters.

```matlab
m = length(y); % number of training examples
n = length(theta);

grad = zeros(size(theta));
h = zeros(size(X), 1);

% h_theta(x) = g(theta^T * x) 
% where z = theta^T * x
% and g(z) = 1 / (1 + exp(-z)) (sigmoid function)

h = sigmoid(X * theta);

J = (1 / m) * (-y' * log(h) - (1-y)' * log(1 - h));

for j=1:n
  grad(j) = (1 / m) * sum((X(:, j)')*(h - y));
end 
```

## CostFunctionReg.m
Compute cost and gradient for logistic regression with regularization. COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters.

```matlab
m = length(y); % number of training examples
n = length(theta);

grad = zeros(size(theta));
h = zeros(size(X), 1);

h = sigmoid(X * theta);

J = (1 / m) * (-y' * log(h) - (1-y)' * log(1 - h)) + ((lambda / (2 * m)) * sum(theta(2:end).^2));

% Gradient of theta zero should not be regularized.
grad(1) = (1 / m) * sum((X(:, 1)')*(h - y));
% Gradient of theta 1 ... n 
for j=2:n
    grad(j) = (1 / m) * sum((X(:, j)')*(h - y)) + ((lambda / m) * theta(j));
end 
```

## predict.m
Predict whether the label is 0 or 1 using learned logistic regression parameters theta. PREDICT(theta, X) computes the predictions for X using a threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

```matlab
m = size(X, 1); % Number of training examples
p = zeros(m, 1);

for i=1:m
  p(i) = round(sigmoid(theta' * X(i, :)'));
end
```

## sigmoid.m
Compute sigmoid function. SIGMOID(z) computes the sigmoid of z.

```matlab
g = zeros(size(z));

% Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).

g = 1 ./ (1 + exp(-z));
```