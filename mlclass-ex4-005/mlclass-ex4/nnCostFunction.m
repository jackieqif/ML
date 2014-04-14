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
% Jackie Part 1 code:

% convert y into num_labels * m demensional matrix, with each column represent a num_labels dimension y vector:
y_matrix = zeros(num_labels, m);
for y_index = 1:m
	y_matrix(y(y_index), y_index) = 1;
endfor

% Add bias unit to X:
X = [ones(m, 1), X];

% Calculate hidden layer:
a = sigmoid(Theta1 * X');

% Add bias unit to hidden layer:
a = [ones(1, size(a)(2)); a];

% Calculate h(x) output:
h = sigmoid(Theta2 * a); 

% Calculate J:
J = (1 / m) .* sum(sum(-y_matrix .* log(h) - ((1 - y_matrix) .* log(1-h))));

% Adding regularization:
reg = lambda / (2 * m) * (sum((sum(Theta1(:, 2:end) .^ 2))) + sum(sum(Theta2(:, 2:end) .^ 2)));
J = J + reg;


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
% Jackie Part2 code:

DELTA_1 = zeros(size(Theta1));
DELTA_2 = zeros(size(Theta2));

for t = 1: m
	% feed forwarding:
	a_1 = X(t, :);
	y_vec = y_matrix(:, t);
	% calculate hidden layer 
	a_2 = sigmoid(Theta1 * a_1');
	a_2 = [1; a_2];
	% calculate output layer
	a_3 = sigmoid(Theta2 * a_2);
	% dealta3
	delta_3 = a_3 - y_vec;

	% back propagation:
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(a_2);
	delta_2 = delta_2(2:end);

	DELTA_1 = DELTA_1 + delta_2 * a_1;
	DELTA_2 = DELTA_2 + delta_3 * a_2';

Theta1 = (1 / m) .* DELTA_1;
Theta2 = (1 / m) .* DELTA_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
