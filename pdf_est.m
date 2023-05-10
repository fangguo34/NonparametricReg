function [f_hat] = pdf_est(x, X,h,kernel)
% x is data for plot where I estimated the kernel density for each x_j
% X is data from the sample used for kernel density estimation

N_data = length(x);
N_sample = length(X);

f_hat = NaN(N_data,1); 

for j = 1:N_data
    xj = x(j);
    
    if strcmp(kernel,'standard normal')
        k = normpdf((xj-X)./h);
        
    elseif strcmp(kernel,'epanechnikov')
        k_0 = 0.75*(1-(abs(xj-X)./h).^2);
        k_1 = (abs(xj-X)./h<=1);
        k = k_0.*k_1;
    
    elseif strcmp(kernel,'naive')
        k = 0.5*(abs(xj-X)./h<=1);      
    
    end
    
    f_hat(j) = (1/(N_sample*h))*sum(k);
end

end

