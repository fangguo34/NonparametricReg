function [m_hat] = nadaraya_est(x,X,Y,h,kernel)
% x is data for plot where I estimated the m(x_j) for each x_j
% X is data from the sample used for kernel density estimation

N_data = length(x);
N_sample = length(X);

m_hat = NaN(N_data,3); 
sigma2 = NaN(N_data,1);
margin = NaN(N_data,1);

% integral K(z) for standard normal K(z) is 1/sqrt(pi)
kz2 = 1/(2*sqrt(pi));

% use fx_hat to replace fx
fx_hat = pdf_est(x, X, h, kernel);    
    
for j = 1:N_data
    xj = x(j);
    
    % estimate m(x)
    if strcmp(kernel,'standard normal')
        num = sum(Y.*normpdf((xj-X)./h));
        den = sum(normpdf((xj-X)./h));
       
    elseif strcmp(kernel,'naive')
        break
        
    end
 
    m_hat(j,1) = num./den;
    
    % estimate sigma2(x)
    var = sum((Y-m_hat(j,1)).^2.*normpdf((xj-X)./h));
    sigma2(j) = var./den;
    
    % confidence interval 
    margin(j) = 1.96*sqrt(sigma2(j))./sqrt(N_sample*h).*sqrt(kz2/fx_hat(j));
    m_hat(j,2) = m_hat(j,1) -  margin(j);
    m_hat(j,3) = m_hat(j,1) +  margin(j);   
    
    
end


end

