syms x y m sigma;
assume(sigma>0)

sigma_val = 1;
mean_val = 1;

pdf_gaussian = (1/sqrt(2*pi*sigma^2)) * exp(-(x - m)^2/(2*sigma^2));
pdf = subs(pdf_gaussian, m, mean_val);
pdf = subs(pdf, sigma, sigma_val);

left_amplitude = int(pdf, x, [-inf -sigma_val]);
right_amplitude = int(pdf, x, [sigma_val inf]);

left_delta = left_amplitude * dirac(y+1*sigma_val);
right_delta = right_amplitude * dirac(y-1*sigma_val);
middle = subs(pdf, x, y/1);

y_vals = -5:0.01:5;

% left_delta = subs(left_delta, y, y_vals);
% idx = left_delta == Inf;    % find Inf (where the peak of the delta is)
% left_delta(idx) = 1;    % replace with a finite value



% figure;
% fplot(pdf);
% axis([-2 8 0 1.5])

% -------------------
% Analytic transform
analytic_piecewise_transform = piecewise(y>=sigma_val, right_delta, ...
                                        -sigma_val < y < sigma_val, middle,...
                                        y <= -sigma_val, left_delta);

analytic_piecewise_transform = subs(analytic_piecewise_transform, y, y_vals);
idx = analytic_piecewise_transform == Inf;    % find Inf (where the peak of the delta is)
analytic_piecewise_transform(idx) = [left_amplitude, right_amplitude];    % replace with a finite values

figure;
plot(y_vals, analytic_piecewise_transform, '--', 'MarkerSize', 10);
axis([-5 5 0 2]);
title('PDF of Gaussian distribution transformed by y=x^2')

% -------------------
% Simulate the transform

num_datapoints = 1000000;
data = normrnd(mean_val, sigma_val, 1, num_datapoints);

% transformation_func = piecewise(x>=sigma_val, 1, ...
%                                 -sigma_val < x < sigma_val, x,...
%                                 x <= -sigma_val, -1);
% transformed_data = subs(transformation_func, x, data);

transformed_data = (1).*(data>=sigma_val) + ...
     (data).*((data>-sigma_val)&(data<sigma_val)) + ...
     (-1).*(data<=-sigma_val);
% plot(x, transformed_data);

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

