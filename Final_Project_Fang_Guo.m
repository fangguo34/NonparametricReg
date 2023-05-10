% MGTECON 605
% Final Project 
% Fang Guo

%% Q3
clear
clc

data = readtable("data_clean.csv");
data.Properties.VariableNames;

%% Data Variables
price_close = data{:,2};
sqft = data{:,4};
income_mean = data{:,17};
N = length(price_close);

% Scatter plot
figure
scatter(sqft, price_close)
xlabel('sqft')
ylabel('Property Sale Price')
title({'Property Sale Price and Square Footage','(Greater San Francisco Area)'})

%% Baseline Linear Regression
% OLS
X = [ones(N,1) sqft sqft.^2];
y = price_close;
alpha = 0.05;
[table1, R2_1] = ols_reg_table(X, y, alpha);
row_names = {'Slope', ...
                'Square Footage', ...
                'Square Footage ^ 2'};

col_names = {'Point Est', 'Robust SE', '95% CI Lower Bound', '95% CI Upper Bound', 'Significance Indicator'};
save_table(table1, row_names, col_names, 'table1.tex')


% Fitted curve plot
input = 0:10:7000;
y_plot = table1(1,1) + table1(2,1).*input + table1(3,1).*(input.^2);

figure
scatter(sqft, price_close)
hold on 
plot(input, y_plot);
xlabel('sqft')
ylabel('Property Sale Price')
title({'Property Sale Price and Square Footage','(Greater San Francisco Area)'})
legend('Data', 'OLS Prediction')
hold off

%% Baseline Linear Regression with sqft^2 and neighborhood income
% Scatter plot
figure
scatter(income_mean, price_close)
xlabel('Neighborhood Mean Income')
ylabel('Property Sale Price')
title({'Property Sale Price and Neighborhood Mean Income','(Greater San Francisco Area)'})

% OLS
X = [ones(N,1) sqft sqft.^2 income_mean];
y = price_close;
alpha = 0.05;
[table2, R2_2] = ols_reg_table(X, y, alpha);
row_names = {'Slope', ...
                'Square Footage', ...
                'Square Footage ^ 2', ...
                'Mean Income'};

col_names = {'Point Est', 'Robust SE', '95\% CI Lower', '95\% CI Upper', 'Significance Flag'};
save_table(table2, row_names, col_names, 'table2.tex')

%% Linear Regression excluding outliers
% Subset data
price_close_sub = price_close(sqft<=4000);
sqft_sub = sqft(sqft<=4000);
income_mean_sub = income_mean(sqft<=4000);
N_sub = length(price_close_sub);

% OLS
X = [ones(N_sub,1) sqft_sub sqft_sub.^2];
y = price_close_sub;
alpha = 0.05;
[table3, R2_3] = ols_reg_table(X, y, alpha);
row_names = {'Slope', ...
                 'Square Footage', ...
                 'Square Footage^2'};
col_names = {'Point Est', 'Robust SE', '95\% CI Lower', '95\% CI Upper', 'Significance Flag'};
save_table(table3, row_names, col_names, 'table3.tex')

% Fitted curve plot
input = 0:10:7000;
y_plot = table3(1,1) + table3(2,1).*input + table3(3,1).*(input.^2);

figure
scatter(sqft, price_close)
hold on 
plot(input, y_plot);
xlabel('sqft')
ylabel('Property Sale Price')
title({'Property Sale Price and Square Footage','(Greater San Francisco Area)'})
legend('Data', 'OLS Prediction (exclude outliers)')

%% Kernel Density Estimation - sqft
h = std(sqft)*1.06*(N)^(-1/5);
input = 0:100:7000;

% Density estimation 
sqft_pdf = pdf_est(input, sqft, h,'standard normal');

% Matlab built in density estimation
[sqft_pdf_m,input2,h_default] = ksdensity(sqft);

figure
plot(input, sqft_pdf)
hold on
plot(input2, sqft_pdf_m)
xlabel('sqft')
ylabel('Est PDF')
title({'Kernel Density Estimation of Property Square Footage','(Greater San Francisco Area)'})
legend('Rule-of-thumb Bandwidth', 'MATLAB default')


%% NW Estimator
% Determine bandwidth (use leave-one-out cross validation)
error_obj = @(h) loocv(sqft,price_close,h,'standard normal','NW');
h0 = 100;
h_cv_nw = fminsearch(error_obj,h0);

input = 0:10:7000;
m = nadaraya_est(input, sqft, price_close, h_cv_nw, 'standard normal');

figure
scatter(sqft, price_close);
hold on;
plot(input, m(:,1),'k','linewidth',3);
hold on;
plot(input, m(:,2),'-.r','linewidth',2);
hold on;
plot(input, m(:,3),'-.r','linewidth',2);
xlabel('sqft')
ylabel('Est Sale Price')
title('NW Nonparametric Regression')
legend('Data', 'NW Estimator', '95% CI Lower', '95% CI Upper')

%% Local Linear Estimator
error_obj = @(h) loocv(sqft,price_close,h,'standard normal','Local Linear');
h0 = 100;
h_cv_ll = fminsearch(error_obj,h0);
% h_cv_ll = 423;

input = 0:10:7000;
[theta0_hat, theta1_hat] = local_linear_est(input, sqft, price_close, h_cv_ll, 'standard normal');

figure
scatter(sqft, price_close);
hold on;
plot(input, m(:,1),'r','linewidth',2);
hold on;
plot(input, theta0_hat,'k','linewidth',2);
xlabel('sqft')
ylabel('Est Sale Price')
title('NW vs Local Linear Nonparametric Regression')
legend('Data', 'NW Estimator', 'LL Estimator')

% look at deriv
figure
plot(input, theta1_hat,'k','linewidth',2);
xlabel('sqft')
ylabel('Est Derivative of Sale Price')
title('Local Linear Nonparametric Regression')
legend('LL Estimator of Derivative')

%% Bootstrap CI for LL
m_s = NaN(500,length(input));
N = length(sqft) 

for s = 1:1000
    s
    ind_b = randi(N, N, 1);
    price_close_b = price_close(ind_b);
    sqft_b = sqft(ind_b);
    
    [theta0_hat_b, theta1_hat_b] = local_linear_est(input, sqft_b, price_close_b, h_cv_ll, 'standard normal');
    m_s(s,:) = theta0_hat_b;
end

m_upper = prctile(m_s,97.5)';
m_lower = prctile(m_s,2.5)';


figure
scatter(sqft, price_close);
hold on;
plot(input, theta0_hat,'k','linewidth',3);
hold on;
plot(input, m_lower,'-.r','linewidth',2);
hold on;
plot(input, m_upper,'-.r','linewidth',2);
xlabel('sqft')
ylabel('Est Sale Price')
title('Local Linear Nonparametric Regression')
legend('Data', 'LL Estimator', '95% CI Lower', '95% CI Upper')


%% Semiparametric Regression
% Y: price_close X: neighbor income Z: sqft

% E[Y|Z] 
error_obj = @(h) loocv(sqft,price_close,h,'standard normal','Local Linear');
h0 = 100;
h_cv_ll = fminsearch(error_obj,h0);

input = 0:10:7000;
[thetaYZ_0_hat, thetaYZ_1_hat] = local_linear_est(sqft, sqft, price_close, h_cv_ll, 'standard normal');

% E[X|Z]
error_obj = @(h) loocv(sqft,income_mean,h,'standard normal','Local Linear');
h0 = 100;
h_cv_ll = fminsearch(error_obj,h0);

input = 0:10:7000;
[thetaXZ_0_hat, thetaXZ_1_hat] = local_linear_est(sqft, sqft, income_mean, h_cv_ll, 'standard normal');

% 
X_til = income_mean - thetaXZ_0_hat;
Y_til = price_close - thetaYZ_0_hat;

% 
b_est = inv(X_til'*X_til)*X_til'*Y_til;

% 
m_Z = thetaYZ_0_hat - thetaXZ_0_hat.*b_est;

%
Y_est = income_mean.*b_est + m_Z;

%% Plot Intermediate Steps
tiledlayout(1,2)

nexttile
scatter(sqft, price_close)
hold on
scatter(sqft, thetaYZ_0_hat, 'k', 'MarkerFaceColor','k')
xlabel('sqft')
ylabel('Property Sale Price')
title('E[Price|Sqft]')
legend('Data', 'LL Estimator')
hold off

nexttile
scatter(sqft, income_mean)
hold on
scatter(sqft, thetaXZ_0_hat, 'k', 'MarkerFaceColor','k')
xlabel('sqft')
ylabel('Neighborhood Mean Income')
title('E[Income|Sqft]')
legend('Data', 'LL Estimator')
hold off



%% Plot Final Est
% Sort data
[sqft_s, id] = sort(sqft);
income_mean_s = income_mean(id);
m_Z_s = m_Z(id);

% Plot
figure
scatter(sqft, price_close)
hold on 
plot(sqft_s , m_Z_s, 'k','linewidth',3);
xlabel('sqft')
ylabel('Property Sale Price')
title({'Property Sale Price and Square Footage','(Greater San Francisco Area)'})
legend('Data', 'Semiparametric Estimates')
hold off

%%

% %% Linear Regression excluding outliers
% % Subset data
% price_close_sub = price_close(sqft<=4000);
% sqft_sub = sqft(sqft<=4000);
% income_mean_sub = income_mean(sqft<=4000);
% N_sub = length(price_close_sub);
% 
% % OLS
% X = [ones(N_sub,1) sqft_sub sqft_sub.^2];
% y = price_close_sub;
% alpha = 0.05;
% [table4, R2_4] = ols_reg_table(X, y, alpha);
% row_names = {'Slope', ...
%                 'Square Footage', ...
%                 'Square Footage^2'};
% 
% col_names = {'Point Est', 'Robust SE', '95\% CI Lower', '95\% CI Upper', 'Significance Flag'};
% save_table(table4, row_names, col_names, 'table4.tex')
% 
% 
% % Fitted curve plot
% input = 0:10:7000;
% y_plot = table4(1,1) + table4(2,1).*input + table4(3,1).*(input.^2);
% 
% figure
% scatter(sqft, price_close)
% hold on 
% plot(input, y_plot);
% xlabel('sqft')
% ylabel('Property Sale Price')
% title({'Property Sale Price and Square Footage','(Greater San Francisco Area)'})
% legend('Data', 'OLS Prediction (exclude outliers)')

% %% Local Linear Estimator on Subset of data
% error_obj = @(h) loocv(sqft_sub,price_close_sub,h,'standard normal','Local Linear');
% h0 = 100;
% h_cv_ll = fminsearch(error_obj,h0);
% 
% input = 0:10:4000;
% [theta0_hat, theta1_hat] = local_linear_est(input, sqft_sub, price_close_sub, h_cv_ll, 'standard normal');
% 
% figure
% scatter(sqft_sub, price_close_sub);
% hold on;
% plot(input, theta0_hat,'k','linewidth',2);
% xlabel('sqft')
% ylabel('Est Sale Price')
% title('Local Linear Nonparametric Regression')
% 
% % look at deriv


% %% Local Linear Estimator on income
% h_cv_ll = std(income_mean)*1.06*(N)^(-1/5);
% 
% input = 50000:1000:250000;
% [theta0_hat, theta1_hat] = local_linear_est(input, income_mean ,price_close, h_cv_ll, 'standard normal');
% 
% figure
% scatter(income_mean,price_close);
% hold on;
% plot(input, theta0_hat,'k','linewidth',2);
% xlabel('Neighborhood Mean Household Income')
% ylabel('Est Sale Price')
% title('Local Linear Nonparametric Regression')

% %% Comprehensive Linear Regression
% X = [ones(N,1) sqft sqft.^2 bed bath parking bld_age condo_dummy loft_dummy ...
%     sf_dummy pop income_mean];
% y = price_close;
% alpha = 0.05;
% [table5, R2_5] = ols_reg_table(X, y, alpha);
% 
% % R2_5_adj = 1 - (1-R2_1)*((N-1)/(N-12))
% 

% %% Baseline Linear Regression without intercept 
% % OLS
% X = [sqft sqft.^2];
% y = price_close;
% alpha = 0.05;
% [table2, R2_2] = ols_reg_table(X, y, alpha);
% row_names = {'Square Footage', ...
%                 'Square Footage ^ 2'};
% 
% col_names = {'Point Est', 'Robust SE', '95% CI Lower Bound', '95% CI Upper Bound', 'Significance Indicator'};
% save_table(table2, row_names, col_names, 'table2.tex')
% 
% 
% % Fitted curve plot
% input = 0:10:7000;
% y_plot = table2(1,1).*input + table2(2,1).*(input.^2);
% 
% figure
% scatter(sqft, price_close)
% hold on 
% plot(input, y_plot);
% xlabel('sqft')
% ylabel('Property Sale Price')
% title({'Property Sale Price and Square Footage','(Greater San Francisco Area)'})
% legend('Data', 'OLS Prediction without intercept')

