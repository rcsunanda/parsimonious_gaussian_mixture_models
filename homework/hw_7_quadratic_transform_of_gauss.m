syms x y sigma;
assume(sigma>0)

sigma_case_1 = 0.5;

% figure;
% fplot(pdf);
% axis([-2 8 0 1.5])

% -------------------
% Analytic transform
analytic_quadratic_transform = exp(-y/(2*sigma^2)) / (sqrt(y*2*pi*sigma^2));
analytic_quadratic_transform = subs(analytic_quadratic_transform, sigma, sigma_case_1);

figure;
fplot(analytic_quadratic_transform, '--', 'MarkerSize', 10);
axis([-1 5 0 5]);
title('PDF of Gaussian distribution transformed by y=x^2')

% -------------------
% Simulate the transform

num_datapoints = 100000;
data = normrnd(0, sigma_case_1, 1, num_datapoints);

transformed_data = data .^ 2;

nbins = 1000;
[num_in_bin, edges] = histcounts(transformed_data, nbins);
x_vals = movmean(edges, 2, 'Endpoints','discard');

bin_width = edges(2) - edges(1);
simulated_quadratic_transform = num_in_bin ./ trapz(x_vals, num_in_bin);

hold on
plot(x_vals, simulated_quadratic_transform, 'MarkerSize', 10);
% axis([-2 50 0 10]);
legend('Anlytic transformation PDF', 'Simulated transformation PDF');

% figure;
% hold on
% [f,xi] = ksdensity(transformed_data);
% plot(xi,f);
% legend('Anlytic transformation PDF', 'Simulated transformation PDF', 'kdensity');

