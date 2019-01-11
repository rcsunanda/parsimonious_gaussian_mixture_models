

% -------------------
% Simulate the transform

num_datapoints = 1000000;
figure;
% title('PDF of average of N uniform [0; 1] random variables')
title('PDF of average of N exponential (mean = 1) random variables')
style= {'-','--','-.', '-d', '-+'};
hold all

for N = 1: 5

% data = rand(N, num_datapoints); % uniform in [0,1]
data = exprnd(1, N, num_datapoints);    % exponential with mean = 1

transformed_data = mean(data, 1);

nbins = 50;
[num_in_bin, edges] = histcounts(transformed_data, nbins);
x_vals = movmean(edges, 2, 'Endpoints','discard');

simulated_transform = num_in_bin ./ trapz(x_vals, num_in_bin);


legend_text = ['N = ', num2str(N)];
plot(x_vals, simulated_transform, style{N}, 'MarkerSize', 4, 'DisplayName', legend_text);
% axis([-2 50 0 10]);

end

hold off
legend show

