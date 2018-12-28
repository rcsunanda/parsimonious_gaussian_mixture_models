% In a binary digit string of length string_length, find 
% Prob(exactly n number of 1's in the string)
% for n in [1, string_length]

fig1 = figure;
title("Theoretic Prob(exactly n number of 1's in the string)")

fig2 = figure;
title("Prob(exactly n number of 1's in the string)")

iter = 1;
for string_length = [8, 16, 32]
    n_vec = 0:string_length;

    prob_zero = 0.3 + 0.01 * 8;
    prob_one = 1 - prob_zero;
    
    % ----------------------------------
    % Theoretic answers (for each n)

    combinations = arrayfun(@(k) nchoosek(string_length, k), n_vec);

    prob_zero_term = arrayfun(@(n) power(prob_zero, string_length - n), n_vec);
    prob_one_term = arrayfun(@(n) power(prob_one, n), n_vec);
    prob_single_seq = prob_zero_term .* prob_one_term;

    theoretic_answers = combinations .* prob_single_seq;

    set(0, 'CurrentFigure', fig1) % Make fig1 the current figure
    if (iter == 3); subplot(2, 2, [3,4]); else; subplot(2, 2, iter); end
    
    bar(n_vec, theoretic_answers);
    axis([min(n_vec) max(n_vec) 0 0.5])
    title(['string length = ', num2str(string_length)])
    xlabel("n = number of 1's")
    ylabel("Probability")
    legend('Theoretic probability', 'Location','northwest')
    
    disp(['sum(theoretic_answers) = ', num2str(sum(theoretic_answers))])
    % assert(isequal(sum(theoretic_answers), 1.));

    
    % ----------------------------------
    % Simulate for num_trials and take counts of each unique 1's count occurence

    outcomes = [0, 1];
    distribution = [prob_zero, prob_one];
    num_trials = 10000;
    seq_occurence_counts = zeros(1, string_length+1);   % first index corresponds to n = 0, last to n = string_length
    
    for t = 1:num_trials
        generated_idx = gendist(distribution, 1, string_length);
        generated_nums = outcomes(generated_idx);

        num_ones = sum(generated_nums);
        idx = num_ones + 1;
        seq_occurence_counts(idx) = seq_occurence_counts(idx) + 1;
    end
    
    simulated_answers = seq_occurence_counts ./ num_trials;
    
    set(0, 'CurrentFigure', fig2) % Make fig2 the current figure
    if (iter == 3); subplot(2, 2, [3,4]); else; subplot(2, 2, iter); end
    
    bar(n_vec, [theoretic_answers; simulated_answers]');
    axis([min(n_vec) max(n_vec) 0 0.5])
    title(['string length = ', num2str(string_length)])
    xlabel("n = number of 1's")
    ylabel("Probability")
    legend('Theoretic probability','Simulated probability', 'Location','northwest')
    
    iter = iter + 1;
end

disp(['Simulation complete.', ' Press Enter ']);
pause;
close all
