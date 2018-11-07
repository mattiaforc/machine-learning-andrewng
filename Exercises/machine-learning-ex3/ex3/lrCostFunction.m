function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

h = sigmoid(X * theta);

J = (1 / m) * (-y' * log(h) - (1-y)' * log(1 - h)) + ((lambda / (2 * m)) * sum(theta(2:end).^2));

% Gradient of theta 1 ... n 
grad = (1 / m) * X'*(h-y) + ((lambda / m) * theta);
% Gradient of theta zero should not be regularized.
grad(1) = (1 / m) * sum((X(:, 1)')*(h - y));
% ------------------------------------------

grad = grad(:);

end
