
num_trails = 10000000;

prob_one = 0.5 + 0.045 * 8;
prob_zero = 1 - prob_one;

outcomes = [0, 1];
distribution = [prob_zero, prob_one];

% A = randi([0 1], 1, num_trails);

generated_idx = gendist(distribution, 1, num_trails);
generated_nums = outcomes(generated_idx);

num_ones = sum(generated_nums);
num_zeros = num_trails - num_ones;

freq_ones = num_ones / num_trails;
freq_zeros = num_zeros / num_trails;

disp(['num_trails = ', num2str(num_trails)])
disp(['actual probabilities, prob_one = ', num2str(prob_one), ', ', 'prob_zero = ', num2str(prob_zero)])
disp(['relative frequencies, freq_ones = ', num2str(freq_ones), ', ', 'freq_zeros = ', num2str(freq_zeros)])
