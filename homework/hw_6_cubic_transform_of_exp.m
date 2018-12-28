syms x y alpha;
assume(alpha>0)

alpha_case_1 = 2;
alpha_case_2 = 2;

pdf_exponential = piecewise(x<0, 0, x>=0, alpha * exp(-alpha * x));
pdf = subs(pdf_exponential, alpha, alpha_case_1);

% figure;
% fplot(pdf);
% axis([-2 8 0 1.5])

% -------------------
% Analytic transform
analytic_cubic_transform = subs(pdf, x, y^(1/3)) * (y^(-2/3) / 3);

figure;
fplot(analytic_cubic_transform, '--', 'MarkerSize', 10);
axis([-2 10 0 10]);
title('PDF of Exponential distribution transformed by y=x^3')

% -------------------
% Simulate the transform

num_datapoints = 1000000;
data = exprnd(1/alpha_case_1, 1, num_datapoints);

transformed_data = data .^ 3;

nbins = 10000;
[num_in_bin, edges] = histcounts(transformed_data, nbins);
x_vals = movmean(edges, 2, 'Endpoints','discard');

simulated_cubic_transform = num_in_bin ./ trapz(x_vals, num_in_bin);

% figure;
hold on
plot(x_vals, simulated_cubic_transform, 'MarkerSize',10);
% axis([-2 50 0 10]);
legend('Anlytic transformation PDF', 'Simulated transformation PDF')

% figure;
% [f,xi] = ksdensity(transformed_data);
% plot(xi,f);

