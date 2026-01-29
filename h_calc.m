function [h_coeff, h_func, fit_type] = h_calc(x_orig, y_orig, pp, fig_no, fitter_type, const)
% H_CALC fits the best monotonic function to data (x,y).
% It can use one of two methods, selected by the 'fitter_type' string.
%
% INPUTS:
%   x_orig      - Independent variable (vector)
%   y_orig      - Dependent variable (vector)
%   pp          - Polynomial degree (used only for 'poly' fitter and extrapolation in 'pava')
%   fig_no      - Figure number for plotting
%   fitter_type - String, either 'poly' or 'pava', to select the algorithm.
%
% OUTPUTS:
%   h_coeff     - The "coefficients" of the fit.
%                 - For 'poly': A 1x(pp+1) row vector of polynomial coefficients.
%                 - For 'pava': An Nx2 matrix [x_unique, y_unique_avg] defining the steps.
%   h_func      - A function handle @(x) for the best fit.
%   fit_type    - A string, either 'increasing' or 'decreasing'.

    % --- 1. Common Data Preparation ---
    x = x_orig(:);
    y = y_orig(:);
    [x_sorted, sort_idx] = sort(x);
    y_sorted = y(sort_idx);

    % --- 2. Select and Run the Chosen Fitter ---
    if strcmpi(fitter_type, 'poly')
        % --- POLYNOMIAL FITTER LOGIC ---
        C = x_sorted.^(0:pp);
        n_constraints = 200;
        x_cons = linspace(min(x_sorted), max(x_sorted), n_constraints)';
        A_inc_constraint = zeros(n_constraints, pp+1);
        for ai = 1:pp
            A_inc_constraint(:, ai+1) = -ai * x_cons.^(ai-1);
        end
        bb = zeros(n_constraints, 1);
        opt = optimoptions('lsqlin', 'Display', 'off');

        [p_inc, err_inc] = lsqlin(C, y_sorted, A_inc_constraint, bb, [], [], [], [], [], opt);
        [p_dec, err_dec] = lsqlin(C, y_sorted, -A_inc_constraint, bb, [], [], [], [], [], opt);

        if err_inc <= err_dec
            p_final_col = p_inc;
            fit_type = 'increasing';
        else
            p_final_col = p_dec;
            fit_type = 'decreasing';
        end

        h_coeff = p_final_col'; % Return row vector of coefficients
        h_func = @(x_in) x_in(:).^(0:pp) * p_final_col;

        % Plotting for polynomial
        figure(fig_no);
        plot(x_orig, y_orig, '.', 'DisplayName', 'Original Data');
        hold on;
        plot(x_sorted, h_func(x_sorted), '-', 'LineWidth', 2, 'DisplayName', ['Best Fit (Poly): ' fit_type]);
        hold off;

    elseif strcmpi(fitter_type, 'pava')
        % --- PAVA FITTER LOGIC ---
        y_inc_fit = pava_core(y_sorted);
        err_inc = sum((y_sorted - y_inc_fit).^2);

        y_dec_fit = -pava_core(-y_sorted);
        err_dec = sum((y_sorted - y_dec_fit).^2);

        if err_inc <= err_dec
            y_final_fit = y_inc_fit;
            fit_type = 'increasing';
        else
            y_final_fit = y_dec_fit;
            fit_type = 'decreasing';
        end

        % --- Create an unambiguous lookup table ---
        [x_unique, ~, ic] = unique(x_sorted);
        y_unique_avg = accumarray(ic, y_final_fit, [], @mean);
        
        % Validate x_unique and y_unique_avg
        if any(isnan(x_unique) | isinf(x_unique)) || any(isnan(y_unique_avg) | isinf(y_unique_avg))
            error('NaN or Inf in x_unique=%s or y_unique_avg=%s', mat2str(x_unique), mat2str(y_unique_avg));
        end

        % --- Fit a monotonic polynomial for extrapolation ---
        C = x_unique.^(0:pp);
        n_constraints = 200;
        x_cons = linspace(min(x_unique), max(x_unique), n_constraints)';
        A_inc_constraint = zeros(n_constraints, pp+1);
        for ai = 1:pp
            A_inc_constraint(:, ai+1) = -ai * x_cons.^(ai-1);
        end
        bb = zeros(n_constraints, 1);
        opt = optimoptions('lsqlin', 'Display', 'off');

        [p_inc, err_inc] = lsqlin(C, y_unique_avg, A_inc_constraint, bb, [], [], [], [], [], opt);
        [p_dec, err_dec] = lsqlin(C, y_unique_avg, -A_inc_constraint, bb, [], [], [], [], [], opt);

        if strcmp(fit_type, 'increasing')
            p_poly = p_inc;
        else
            p_poly = p_dec;
        end

        poly_func = @(x_in) x_in(:).^(0:pp) * p_poly;

        % --- Combine PAVA and polynomial extrapolation ---
%         h_func = @(x_in) (x_in(:) >= min(x_unique) & x_in(:) <= max(x_unique)) .* ...
%                          interp1(x_unique, y_unique_avg, x_in(:), 'previous') + ...
%                          (x_in(:) < min(x_unique) | x_in(:) > max(x_unique)) .* poly_func(x_in(:));

        h_func = @(x_in) hybrid_evaluator(x_in, x_unique, y_unique_avg, poly_func);
        h_coeff = [x_unique, y_unique_avg]; % PAVA steps as coefficients

        % Plotting for PAVA with polynomial extrapolation
        figure(fig_no);
        plot(x_orig, y_orig*const, '.', 'DisplayName', 'Original Data');
        hold on;
        stairs(x_unique, y_unique_avg*const, '-', 'LineWidth', 2, 'DisplayName', ['PAVA Fit: ' fit_type]);
        % Plot extrapolation
        x_extrap = linspace(min(x_unique)-0.5, max(x_unique)+0.5, 100);
        plot(x_extrap, poly_func(x_extrap)*, '--', 'LineWidth', 1, 'DisplayName', 'Polynomial Extrapolation');
        hold off;

    else
        error("Unknown fitter_type: Please choose 'poly' or 'pava'.");
    end

    % --- 3. Common Plotting Details ---
    figure(fig_no); % Bring figure to front
    grid on;
    legend;
    title(['h_calc Fit (Outer Iteration ' num2str(fig_no) ')']);
end

function g = pava_core(y)
% Core implementation of the Pool-Adjacent-Violators Algorithm (PAVA)
    n = length(y);
    g = y(:);
    while true
        violations_found = false;
        i = 1;
        while i < n
            if g(i) > g(i+1)
                violations_found = true;
                start_idx = i;
                while start_idx > 1 && g(start_idx-1) == g(i)
                    start_idx = start_idx - 1;
                end
                block_val_1 = g(start_idx);
                block_val_2 = g(i+1);
                w1 = i - start_idx + 1;
                w2 = 1;
                j = i + 2;
                while j <= n && g(j) == block_val_2
                    w2 = w2 + 1;
                    j = j + 1;
                end
                new_avg = (block_val_1 * w1 + block_val_2 * w2) / (w1 + w2);
                g(start_idx:(i+w2)) = new_avg;
                i = start_idx;
            else
                i = i + 1;
            end
        end
        if ~violations_found
            break;
        end
    end
end

function y_out = hybrid_evaluator(x_in, x_table, y_table, poly_extrap_func)
    % This function safely evaluates the hybrid PAVA/Poly model.
    % It avoids the "0 * NaN = NaN" problem.
    
    x_in = x_in(:); % Ensure input is a column
    y_out = zeros(size(x_in)); % Pre-allocate output
    
    min_x = min(x_table);
    max_x = max(x_table);
    
    % Identify which points are inside and outside the interpolation range
    is_inside = (x_in >= min_x & x_in <= max_x);
    is_outside = ~is_inside;
    
    % Evaluate each set of points using ONLY the appropriate function
    if any(is_inside)
        y_out(is_inside) = interp1(x_table, y_table, x_in(is_inside), 'previous');
    end
    if any(is_outside)
        y_out(is_outside) = poly_extrap_func(x_in(is_outside));
    end
end
