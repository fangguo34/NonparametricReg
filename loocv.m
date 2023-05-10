function [est_error] = loocv(X,Y,h_range,kernel,estimator)
M = length(h_range);
N = length(X);

est_error = NaN(M,1);

for j = 1:M
    h = h_range(j);
    Y_est = NaN(N,1);
    
    for i = 1:N
        xi = X(i);
    
        X_rest = X;
        X_rest(i) = [];
    
        Y_rest = Y;
        Y_rest(i) = [];
        
        if strcmp(estimator,'NW')
            Y_est(i,1) = nadaraya_est_1(xi, X_rest, Y_rest, h, kernel);
            
        elseif strcmp(estimator,'Local Linear')
            [theta0_hat, ~] = local_linear_est(xi, X_rest, Y_rest , h, kernel);
            Y_est(i,1) = theta0_hat;
            
        end
        
    end
    
    est_error(j,1) = mean((Y-Y_est).^2);
end
end

