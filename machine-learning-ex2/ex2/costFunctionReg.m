function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Set theta0 equal to 0 to avoid regularization
thetaT = theta;
thetaT(1) = 0;
%Use sigmoid function for hypothesis calculation
z = X*theta;
h = sigmoid(z);
%Regularization term for model - Subracting theta0
reg = lambda / (2*m) * sum(thetaT .^ 2);

J = (1/m) * ( (-y'*log(h)) - (1 - y') * log(1 - h) ) + reg;

grad = (1/m) * X' * (h-y) + thetaT * (lambda / m); 

% temp1 = -1 * (y .* log(sigmoid(X * theta))); 
% temp2 = (1 - y) .* log(1 - sigmoid(X * theta)); 
%  
%  
% thetaT = theta; 
% thetaT(1) = 0; 
% correction = sum(thetaT .^ 2) * (lambda / (2 * m)); 
%  
%  
% J = sum(temp1 - temp2) / m + correction; 
%  
% grad = (X' * (sigmoid(X * theta) - y)) * (1/m) + 

% =============================================================

end
