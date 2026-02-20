>> LOOCVal
Starting comparison over 1 sample sizes...

Processing n = 100 ...

rep =

     1

Warning: Column headers from the file were modified to make them valid MATLAB identifiers before creating variable names for the table. The original
column headers are saved in the VariableDescriptions property.
Set 'PreserveVariableNames' to true to use the original column headers as table variable names. 

ans =

   1.6971e+08

Error using interp1>reshapeAndSortXandV (line 424)
X and V must be of the same length.

Error in interp1 (line 93)
    [X,V,orig_size_v] = reshapeAndSortXandV(varargin{1},varargin{2});

Error in LOOCVal>get_warped_dist_sq (line 623)
        f_a_warped = interp1(t, f_a, gam, 'linear', 'extrap');

Error in LOOCVal>run_regression_pipeline (line 581)
    parfor i = 1:n_train

Error in LOOCVal (line 531)
    [~, ~, R2_1] = run_regression_pipeline(Q_train, F_train, y_train, Q_test, F_test, y_test, t, 1);
