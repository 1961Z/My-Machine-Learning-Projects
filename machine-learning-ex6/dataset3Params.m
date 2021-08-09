function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec = [0 0.01 0.03 0.1 0.3 1 3 10 30]'; %create a 64 * 3 matrix 
sigma_vec = [0 0.01 0.03 0.1 0.3 1 3 10 30]';
counter = 1;
prediction_error = zeros(length(C_vec), length(sigma_vec));
findingValue = zeros(64, 3); 
lowestValue = zeros(64, 1); 
row = 0;


for i = 1:length(C_vec)
  for a = 1:length(sigma_vec)
    C_test = C_vec(i); 
    sigma_test = sigma_vec(a);
    model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test)); 
    predictions = svmPredict(model, Xval);
    prediction_error(i,a) = mean(double(predictions ~= yval));  
    findingValue(counter, :) = [C_test, sigma_test, prediction_error(i,a)]; 
    counter += 1; 
  endfor
endfor

findingValue

lowestValue = findingValue(:, 3); 
[minval, row] = min(min(lowestValue,[],2));
row
C = findingValue(row, 1); 
sigma = findingValue(row, 2);


% =========================================================================

end
