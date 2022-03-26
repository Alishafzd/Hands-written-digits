%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   

%% Loading and Visualizing Data 

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data set.mat');

% Shuffle the data set
ir = randperm(size(X, 1));
X = X(ir, :);
y = y(ir, :);

% Split data set into train, cv, and test sets
X_train = X(1:3000, :);
X_cv = X(3000:4000, :);
X_test = X(4000:5000, :);

y_train = y(1:3000, :);
y_cv = y(3000:4000, :);
y_test = y(4000:5000, :);

m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Initializing Pameters

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
nn_params = initial_nn_params;

%% Compute Cost (Feedforward)
% Weight regularization parameter 
lambda = 2;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X_train, y_train, lambda);

fprintf(('Cost at parameters: %f '), J);

%% Implement Backpropagation

%  Check gradients by running checkNNGradients
checkNNGradients;

%% Training NN 

%  Use "fmincg"
options = optimset('MaxIter', 200);

% Create "short hand" for the cost function 
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% Implement Predict

pred = predict(Theta1, Theta2, X_cv);

fprintf('\nCV Set Accuracy: %f\n', mean(double(pred == y_cv)) * 100);


%% Obtain the best lambda value

[J_cv, lambda_cv] = lambdaFind(nn_params, input_layer_size, hidden_layer_size, num_labels, X_cv, y_cv);

% Find minimum value and index of J_cv
[J_min, I] = min(J_cv);
lambda = lambda_cv(I);

%% Rerun the model with new lambda

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
% Implement predict on test set

pred = predict(Theta1, Theta2, X_test);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);



