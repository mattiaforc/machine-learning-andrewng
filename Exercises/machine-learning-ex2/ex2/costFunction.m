function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h = zeros(size(X), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
% h_theta(x) = g(theta^T * x) 
% where z = theta^T * x
% and g(z) = 1 / (1 + exp(-z)) (sigmoid function)

h = sigmoid(X * theta);

J = (1 / m) * (-y' * log(h) - (1-y)' * log(1 - h));
for j=1:n
  grad(j) = (1 / m) * sum((X(:, j)')*(h - y));
end 

% =============================================================

end
