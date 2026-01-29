% % clear all;
% % close all;
% % gap=0.02;
% % t=0:gap:1;
% % n=50;
% % ScoSh=1;
% % stdER=0.1;
% % 
% % for i=1:n
% %     ci1=normrnd(0,1);
% %     ci2=normrnd(0,1);
% %     fi0{i}=@(t) ci1*sqrt(2)*sin(2*pi*t)+ci2*sqrt(2)*cos(2*pi*t);
% %     alpha=unifrnd(-1,1);
% %     gammai{i}=@(t) t+alpha.*(t).*(t-1);
% %     fi{i}=@(x) fi0{i}(gammai{i}(x));
% %     f(i,:)=fi{i}(t);
% %     qi(i,:)=curve_to_q(fi{i}(t));
% %     
% %     epsi(i)=normrnd(0,stdER);
% %     %Semi Real y_i
% %     y(i)=epsi(i)+max(f(i,:))-min(f(i,:));%sum(abs(diff([0 fi{i}(t)])/gap)*gap);%
% %     costtrue=costtrue+epsi(i)^2;
% % end
% % 
% % % Ensure y is a column vector
% % y = y(:);
% % 
% % %% 2. Split Data (80% Train, 20% Test)
% % % ---------------------------------------------------------
% % fraction = 0.8;
% % n_train = round(fraction * n);
% % n_test = n - n_train;
% % 
% % % Random shuffle indices to ensure random split
% % rng(1); % Fixed seed for reproducibility
% % rand_idx = randperm(n);
% % 
% % idx_train = rand_idx(1:n_train);
% % idx_test  = rand_idx(n_train+1:end);
% % 
% % % Create Sets
% % Q_train = qi(idx_train, :);
% % y_train = y(idx_train);
% % 
% % Q_test  = qi(idx_test, :);
% % y_test  = y(idx_test);
% % 
% % %% 3. Define Distance Function (Placeholder)
% % % ---------------------------------------------------------
% % % You said "Leave distance calc to me", so here is the wrapper.
% % % Input: two vectors (1xM). Output: Scalar Squared Distance.
% % gamSync(i,:)=DynamicProgrammingQ_Adam(qi(i,:),beta(t),0,0);
% %         dumf=interp1(t,f(i,:),gamSync(i,:),'spline');
% %         qi(i,:)=curve_to_q(dumf);
% % calc_dist_sq = @(q_a, q_b) sum((q_a - q_b).^2);
% % 
% % %% 4. Optimize Tau using Training Data (LOOCV)
% % % ---------------------------------------------------------
% % disp('Optimizing Tau...');
% % 
% % % A. Pre-calculate Distance Matrix for Training Set
% % % We do this once to speed up the optimization loop.
% % % DistMat(i,j) = distance between training point i and training point j
% % DistMat_Train = zeros(n_train, n_train);
% % for i = 1:n_train
% %     for j = 1:n_train
% %         if i ~= j
% %             DistMat_Train(i,j) = calc_dist_sq(Q_train(i,:), Q_train(j,:));
% %         end
% %     end
% % end
% % 
% % % B. Define the Cost Function (Leave-One-Out Cross Validation)
% % % J = Sum ( y_i - y_hat_i )^2
% % % y_hat_i is calculated using all points except i
% % loocv_cost_fun = @(tau) calculate_loocv_error(tau, DistMat_Train, y_train);
% % 
% % % C. Minimize Cost Function
% % % We use fminbnd because Tau must be > 0. 
% % % We search for Tau between 0.001 and 100 (adjust range based on data scale)
% % options = optimset('Display','iter');
% % tau_optimal = fminbnd(loocv_cost_fun, 0.001, 100, options);
% % 
% % fprintf('Optimal Tau found: %.4f\n', tau_optimal);
% % 
% % %% 5. Prediction on Test Set
% % % ---------------------------------------------------------
% % y_test_pred = zeros(n_test, 1);
% % 
% % for k = 1:n_test
% %     % Current test query
% %     q_new = Q_test(k,:);
% %     
% %     % 1. Calculate Distances from q_new to ALL Training points
% %     dists_sq = zeros(n_train, 1);
% %     for i = 1:n_train
% %         dists_sq(i) = calc_dist_sq(q_new, Q_train(i,:));
% %     end
% %     
% %     % 2. Calculate Kernels (Un-normalized weights)
% %     u_vec = exp(-dists_sq / tau_optimal);
% %     
% %     % 3. Normalize Weights
% %     w_vec = u_vec / sum(u_vec);
% %     
% %     % 4. Weighted Sum Prediction
% %     y_test_pred(k) = sum(w_vec .* y_train);
% % end
% % 
% % %% 6. Results Visualization
% % % ---------------------------------------------------------
% % figure;
% % plot(y_test, 'ko-', 'LineWidth', 1.5, 'DisplayName', 'Actual y');
% % hold on;
% % plot(y_test_pred, 'rx--', 'LineWidth', 1.5, 'DisplayName', 'Predicted y');
% % legend;
% % title(['Test Set Predictions (Tau_{opt} = ' num2str(tau_optimal) ')']);
% % xlabel('Test Sample Index');
% % ylabel('Response y');
% % grid on;
% % 
% % mse_test = mean((y_test - y_test_pred).^2);
% % fprintf('Mean Squared Error on Test Set: %.4f\n', mse_test);
% % 
% % 
% % %% --- HELPER FUNCTION: LOOCV COST ---
% % function J = calculate_loocv_error(tau, D_sq, y)
% %     % D_sq: Squared Distance Matrix (Pre-calculated)
% %     % y: Training responses
% %     
% %     n = length(y);
% %     J = 0;
% %     
% %     % Calculate Kernel Matrix
% %     K = exp(-D_sq / tau);
% %     
% %     % IMPORTANT: Set diagonal to 0. 
% %     % This ensures the point itself is NOT used to predict itself (i != new)
% %     K(logical(eye(n))) = 0;
% %     
% %     % Calculate Predictions for all i simultaneously
% %     % Sum of weights for each row
% %     sum_weights = sum(K, 2); 
% %     
% %     % Avoid division by zero
% %     sum_weights(sum_weights == 0) = eps; 
% %     
% %     % Normalized Weights Matrix
% %     W = K ./ sum_weights;
% %     
% %     % Predictions: y_hat = W * y
% %     y_hat = W * y;
% %     
% %     % Sum of Squared Errors
% %     J = sum((y - y_hat).^2);
% % end
% % % % clear all;
% % % % close all;
% % % % 
% % % % %% 1. Data Generation
% % % % gap=0.02;
% % % % t=0:gap:1;
% % % % n=100; 
% % % % stdER=0.01;
% % % % ScoSh=0;
% % % % 
% % % % % Pre-allocate F to store original curves
% % % % F = zeros(n, length(t)); 
% % % % qi = zeros(n, length(t)); 
% % % % y = zeros(n, 1);
% % % % costtrue = 0;
% % % % 
% % % % for i=1:n
% % % %     ci1=normrnd(0,1);
% % % %     ci2=normrnd(0,1);
% % % %     fi0{i}=@(t) ci1*sqrt(2)*sin(2*pi*t)+ci2*sqrt(2)*cos(2*pi*t);
% % % %     alpha=unifrnd(-1,1);
% % % %     gammai{i}=@(t) t+alpha.*(t).*(t-1);
% % % %     fi{i}=@(x) fi0{i}(gammai{i}(x));
% % % %     
% % % %     % Store f (Required for warping later)
% % % %     f(i,:)=fi{i}(t);
% % % %     F(i,:) = f(i,:); 
% % % %     
% % % %     % Placeholder for curve_to_q (Ensure this function is in your path)
% % % %     if exist('curve_to_q', 'file')
% % % %         qi(i,:)=curve_to_q(fi{i}(t));
% % % %     else
% % % %         % Fallback if function missing
% % % %         qi(i,:) = f(i,:); 
% % % %     end
% % % %     
% % % %     epsi(i)=normrnd(0,stdER);
% % % %     y(i)=epsi(i)+max(f(i,:))-min(f(i,:));%sum(abs(diff([0 fi{i}(t)])/gap)*gap);% 
% % % %     costtrue=costtrue+epsi(i)^2;
% % % % end
% % % % y = y(:);
% % % % 
% % % % %% 2. Split Data (80% Train, 20% Test)
% % % % fraction = 0.8;
% % % % n_train = round(fraction * n);
% % % % n_test = n - n_train;
% % % % rng(1); 
% % % % rand_idx = randperm(n);
% % % % idx_train = rand_idx(1:n_train);
% % % % idx_test  = rand_idx(n_train+1:end);
% % % % 
% % % % Q_train = qi(idx_train, :);
% % % % F_train = F(idx_train, :); % <--- NEW: Keep f for training set
% % % % y_train = y(idx_train);
% % % % 
% % % % Q_test  = qi(idx_test, :);
% % % % F_test  = F(idx_test, :);  % <--- NEW: Keep f for testing set
% % % % y_test  = y(idx_test);
% % % % 
% % % % %% 3. Run Analysis for Both Methods
% % % % % ---------------------------------------------------------
% % % % % check for parallel pool
% % % % if isempty(gcp('nocreate')), parpool; end 
% % % % 
% % % % disp('--- Running Method 1: Standard L2 (ScoSh = 0) ---');
% % % % [tau_0, y_pred_0, R2_0] = run_regression_pipeline(Q_train, F_train, y_train, Q_test, F_test, y_test, t, 0);
% % % % 
% % % % disp('--- Running Method 2: Shape Alignment (ScoSh = 1) ---');
% % % % [tau_1, y_pred_1, R2_1] = run_regression_pipeline(Q_train, F_train, y_train, Q_test, F_test, y_test, t, 1);
% % % % 
% % % % %% 4. Comparative Visualization
% % % % % ---------------------------------------------------------
% % % % figure;
% % % % hold on;
% % % % 
% % % % % 1. Plot Identity Line (Reference)
% % % % limits = [min(y_test), max(y_test)];
% % % % plot(limits, limits, 'k--', 'LineWidth', 2, 'DisplayName', 'Perfect Prediction');
% % % % 
% % % % % 2. Plot Standard L2 Results (Blue Circles)
% % % % plot(y_test, y_pred_0, 'bo', 'LineWidth', 1.5, 'DisplayName', ['Standard L2 (R^2=' num2str(R2_0, '%.2f') ')']);
% % % % 
% % % % % 3. Plot Shape Aligned Results (Red Crosses)
% % % % plot(y_test, y_pred_1, 'rx', 'LineWidth', 1.5, 'DisplayName', ['Shape Aligned (R^2=' num2str(R2_1, '%.2f') ')']);
% % % % 
% % % % % 4. Formatting
% % % % legend('Location', 'best');
% % % % title('Model Comparison: Standard vs. Shape Distance');
% % % % xlabel('True Response (y)');
% % % % ylabel('Predicted Response (y_{hat})');
% % % % grid on;
% % % % axis square;
% % % % hold off;
% % % % 
% % % % fprintf('\nFinal Results:\n');
% % % % fprintf('Standard L2 (ScoSh=0): Tau=%.4f, R2=%.4f\n', tau_0, R2_0);
% % % % fprintf('Shape Aligned (ScoSh=1): Tau=%.4f, R2=%.4f\n', tau_1, R2_1);


%% 

clear all;
close all;

% --- Setup Noise Levels to Test ---
std_vals = [0, 0.1, 0.5, 1, 3, 5, 7]; 
n_trials = length(std_vals);
results_table = zeros(n_trials, 3); % Columns: stdER, R2_L2, R2_Shape

% Check for parallel pool once
if isempty(gcp('nocreate')), parpool; end 

fprintf('Starting comparison over %d noise levels...\n', n_trials);

%% Loop over Standard Errors
for s_idx = 1:n_trials
    
    % 1. Set current Noise Level
    stdER = std_vals(s_idx);
    fprintf('\nProcessing stdER = %.2f ...\n', stdER);
    
    % --- DATA GENERATION (Inside loop to apply new noise) ---
    gap=0.02;
    t=0:gap:1;
    n=100; 
    
    F = zeros(n, length(t)); 
    qi = zeros(n, length(t)); 
    y = zeros(n, 1);
    
    % Reset RNG for consistent curves, but different noise if desired
    % rng(s_idx); % Optional: Uncomment to fix curves per noise level
    
    for i=1:n
        ci1=normrnd(0,1);
        ci2=normrnd(0,1);
        fi0{i}=@(t) ci1*sqrt(2)*sin(2*pi*t)+ci2*sqrt(2)*cos(2*pi*t);
        alpha=unifrnd(-1,1);
        gammai{i}=@(t) t+alpha.*(t).*(t-1);
        fi{i}=@(x) fi0{i}(gammai{i}(x));
        
        f(i,:)=fi{i}(t);
        F(i,:) = f(i,:); 
        
        qi(i,:)=curve_to_q(fi{i}(t));
        
        % Apply current stdER here
        epsi(i)=normrnd(0,stdER);
        y(i)=epsi(i)+sum(abs(diff([0 fi{i}(t)])/gap)*gap);
    end
    y = y(:);
    
    % --- SPLIT DATA ---
    fraction = 0.75;
    n_train = round(fraction * n);
    rng(1); % Keep split consistent
    rand_idx = randperm(n);
    idx_train = rand_idx(1:n_train);
    idx_test  = rand_idx(n_train+1:end);

    Q_train = qi(idx_train, :); F_train = F(idx_train, :); y_train = y(idx_train);
    Q_test  = qi(idx_test, :);  F_test  = F(idx_test, :);  y_test  = y(idx_test);
    
    % --- RUN ANALYSIS ---
    % Method 1: Standard L2 (ScoSh=0)
    [~, ~, R2_0] = run_regression_pipeline(Q_train, F_train, y_train, Q_test, F_test, y_test, t, 0);
    
    % Method 2: Shape Aligned (ScoSh=1)
    [~, ~, R2_1] = run_regression_pipeline(Q_train, F_train, y_train, Q_test, F_test, y_test, t, 1);
    
    % Store Results
    results_table(s_idx, :) = [stdER, R2_0, R2_1];
end

%% Display Final Table
fprintf('\n=========================================\n');
fprintf('       R^2 Comparison Table\n');
fprintf('=========================================\n');
fprintf(' %-10s | %-12s | %-12s \n', 'stdER', 'R2 (L2)', 'R2 (Shape)');
fprintf('-----------------------------------------\n');
for i = 1:n_trials
    fprintf(' %-10.2f | %-12.4f | %-12.4f \n', ...
        results_table(i,1), results_table(i,2), results_table(i,3));
end
fprintf('-----------------------------------------\n');

% (Keep your helper functions below this line)
%% --- HELPER FUNCTIONS ---

function [tau_opt, y_pred, R2] = run_regression_pipeline(Q_tr, F_tr, y_tr, Q_te, F_te, y_te, t, ScoSh)
    n_train = size(Q_tr, 1);
    n_test  = size(Q_te, 1);
    
    % A. Calculate Training Distance Matrix
    DistMat = zeros(n_train, n_train);
    
    parfor i = 1:n_train
        row_dists = zeros(1, n_train);
        for j = (i+1):n_train
            % Pass ScoSh to the distance function
            row_dists(j) = get_warped_dist_sq(Q_tr(i,:), Q_tr(j,:), F_tr(i,:), t, ScoSh);
        end
        DistMat(i,:) = row_dists;
    end
    DistMat = DistMat + DistMat'; % Mirror symmetry
    
    % B. Optimize Tau
    loocv_fun = @(tau) calculate_loocv_error(tau, DistMat, y_tr);
    tau_opt = fminbnd(loocv_fun, 0.001, 100);
    
    % C. Predict on Test Set
    y_pred = zeros(n_test, 1);
    
    parfor k = 1:n_test
        q_new = Q_te(k,:);
        f_new = F_te(k,:);
        dists_sq = zeros(n_train, 1);
        
        for i = 1:n_train
            dists_sq(i) = get_warped_dist_sq(q_new, Q_tr(i,:), f_new, t, ScoSh);
        end
        
        u_vec = exp(-dists_sq / tau_opt);
        w_vec = u_vec / sum(u_vec);
        y_pred(k) = sum(w_vec .* y_tr);
    end
    
    % D. Calculate R^2
    SS_res = sum((y_te - y_pred).^2);
    SS_tot = sum((y_te - mean(y_te)).^2);
    R2 = 1 - (SS_res / SS_tot);
end

% PERFORMANCE OPTIMIZED DISTANCE FUNCTION
function dist_sq = get_warped_dist_sq(q_a, q_b, f_a, t, ScoSh)
    if ScoSh == 1
        % Expensive Alignment Path
        gam = DynamicProgrammingQ_Adam(q_a, q_b, 0, 0); 
        f_a_warped = interp1(t, f_a, gam, 'linear', 'extrap');
        q_a_prime = curve_to_q(f_a_warped);
        dist_sq = sum((q_a_prime - q_b).^2); 
    else
        % Fast L2 Path (Skips DP and Interp)
        dist_sq = sum((q_a - q_b).^2); 
    end
end

function J = calculate_loocv_error(tau, D_sq, y)
    n = length(y);
    K = exp(-D_sq / tau);
    K(logical(eye(n))) = 0; 
    sum_weights = sum(K, 2); 
    sum_weights(sum_weights == 0) = eps; 
    W = K ./ sum_weights;
    y_hat = W * y;
    J = sum((y - y_hat).^2);
end
