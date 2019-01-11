syms x y;

% -------------------
% Analytic transform
analytic_pdf = piecewise(0 <= y <= 1, 2*y, y > 1 | y < 0, 0);

figure;
fplot(analytic_pdf, '--', 'MarkerSize', 10);
axis([-2 4 0 4]);
title('PDF of two uniform RVs transformed by y=max(x1, x2)')

% -------------------
% Simulate the transform
% 
num_datapoints = 1000000;
data = rand(2, num_datapoints); % uniform in [0,1]

transformed_data = max(data);

nbins = 100;
[num_in_bin, edges] = histcounts(transformed_data, nbins);
x_vals = movmean(edges, 2, 'Endpoints','discard');

simulated_pdf = num_in_bin ./ trapz(x_vals, num_in_bin);

hold on
plot(x_vals, simulated_pdf, 'MarkerSize', 10);
% axis([-2 50 0 10]);
legend('Anlytic transformation PDF', 'Simulated transformation PDF');
