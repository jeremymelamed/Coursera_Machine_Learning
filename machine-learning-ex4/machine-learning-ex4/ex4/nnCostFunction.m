function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add column of 1s to training matrix
X = [ones(m, 1) X];

% Send training samples multiplied by Theta1 parameters through sigmoid
% function
z1 = X*Theta1';
a2 = sigmoid(z1);

% Add column of ones to hidden layer for theta0
a2 = [ones(size(a2, 1), 1) a2];

% Use hidden layer values to compute output layer
z2 = a2*Theta2';
h = sigmoid(z2);

% Convert y labels to binary vectors using values as vector indices
yk = zeros(num_labels, m);
for i=1:m,
    yk(y(i),i)=1;
end

% Regularization term
theta1_reg = sum( sum(Theta1(:, 2:end) .^ 2) );
theta2_reg = sum( sum(Theta2(:, 2:end) .^ 2) );
reg = (lambda / (2 * m)) * (theta1_reg + theta2_reg);

J = (1/m) * sum( sum(-yk' .* log(h) - (1 - yk)' .* log(1 - h) )) + reg;

% Backpropogation gradient calculation
delta_L = 0;
for t = 1:m
   % Set input layer equal sample vectors equal to a1
   a1 = X(t, :);
   
   % Calculate sigmoid of nodes for hidden layer
   z2 = Theta1 * a1';
   a2 = sigmoid(z2);
   a2 = [1 ; a2]; %Add bias term
  
   % Calculate sigmoid of nodes for output layer
   z3 = Theta2 * a2;
   a3 = sigmoid(z3);  
   z2 = [1 ; z2];
   
   % Calculate error for each layer
   delta_3 = (a3 - yk(:, t));   
   delta_2 = Theta2' * delta_3 .* sigmoidGradient(z2);
    
   % Accumulate gradients
   Theta2_grad = Theta2_grad + delta_3 * a2';
   Theta1_grad = Theta1_grad + delta_2(2:end) * a1;
end

% Gradient without regularization for first columns
Theta2_grad(:, 1) = (1 / m) .* Theta2_grad(:, 1);
Theta1_grad(:, 1) = (1 / m) .* Theta1_grad(:, 1);
% Gradients with regularization
Theta2_grad(:, 2:end) = (1 / m) * Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);
Theta1_grad(:, 2:end) = (1 / m) * Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
