function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Convert output data set to an m*num_labels matrix
Y = zeros(m, num_labels);
Y(sub2ind(size(Y), 1:length(y), y')) = 1;

% Calculate feedforward
A0 = [ones(m, 1), X];
Z1 = A0 * Theta1';
A1 = [ones(m, 1), sigmoid(Z1)];
Z2 = A1 * Theta2';
A2 = sigmoid(Z2);

% Calculate loss with regularization 
J = -1 / m * sum((Y.*log(A2) + (ones(m, num_labels)-Y).*log(ones(m, num_labels)-A2)), "all");
J = J + lambda / (2*m) * (sum(Theta1(:, 2:end) .^ 2, "all") + sum(Theta2(:, 2:end) .^ 2, "all"));

% Calculate gradients
delta2 = -1 / m * (Y./A2 - (1 - Y)./(1 - A2));
delta1 = (delta2 .* sigmoidGradient(Z2)) * Theta2;
Theta2_grad = (delta2 .* sigmoidGradient(Z2))' * A1;
Theta1_grad = (delta1(:, 2:end) .* sigmoidGradient(Z1))' * A0;

% Implemetn regularization to gradients
temp2 = Theta2 * lambda / m;
temp2(:, 1) = 0;
Theta2_grad = Theta2_grad + temp2;

temp1 = Theta1 * lambda / m;
temp1(:, 1) = 0;
Theta1_grad = Theta1_grad + temp1;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
