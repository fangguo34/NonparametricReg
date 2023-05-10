function [theta0_hat, theta1_hat] = local_linear_est(x,X,Y,h,kernel)
% x is data for plot where I estimated the m(x_j) for each x_j
% X is data from the sample used for kernel density estimation

N_data = length(x);
N_sample = length(X);

theta0_hat = NaN(N_data,1);
theta1_hat = NaN(N_data,1);
    
for j = 1:N_data
    xj = x(j);
    
    % estimate m(x)
    if strcmp(kernel,'standard normal')
        A11 = sum(normpdf((xj-X)./h));
        A12 = sum(normpdf((xj-X)./h).*(X-xj));
        A21 = sum(normpdf((xj-X)./h).*(X-xj));
        A22 = sum(normpdf((xj-X)./h).*((X-xj).^2));
        
        A = [A11 A12; A21 A22];
        
        B1 = sum(normpdf((xj-X)./h).*Y);
        B2 = sum(normpdf((xj-X)./h).*(X-xj).*Y);
        
        B = [B1; B2];
        
        theta = A\B;
       
    elseif strcmp(kernel,'naive')
        break
        
    end
 
    theta0_hat(j,1) = theta(1);
    theta1_hat(j,1) = theta(2);
    
end


end
