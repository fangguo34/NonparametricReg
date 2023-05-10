function [m_hat] = nadaraya_est_1(x,X,Y,h,kernel)
% x is data for plot where I estimated the m(x_j) for each x_j
% X is data from the sample used for kernel density estimation

N_data = length(x);
N_sample = length(X);

m_hat = NaN(N_data,1); 

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
    
end


end