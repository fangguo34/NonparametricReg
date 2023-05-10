function [table, R2] = ols_reg_table(X, y, alpha)
K = size(X,2);
table = NaN(K,5);

% OLS coefficients
b_hat = X\y;
table(:,1) = b_hat; 

% Robust se 
e = y - X*b_hat; 
V_r = inv(X'*X)*X'*diag(e.^2)*X*inv(X'*X); 
se_r = sqrt(diag(V_r));
table(:, 2) = se_r;

% Confidence Interval 
table(:, 3) = table(:, 1) + norminv(alpha/2).*table(:, 2);
table(:, 4) = table(:, 1) + norminv(1-alpha/2).*table(:, 2);

% Significance Flag
table(:,5) = (table(:, 3).*table(:, 4) >0); 

% R squared
y_til = y - mean(y);
R2 = 1 - (e'*e)/ (y_til'*y_til);
end

