function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z

g = zeros(size(z));

% Compute the gradient of the sigmoid function 
g = sigmoid(z).*(1 - sigmoid(z));

% =============================================================
end
