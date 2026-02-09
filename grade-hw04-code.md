# HW04 Code Grading Report

## Question 1A — TISP on Gisette (2 points): **Correct**

- **Standardization**: Correct. `standardize_train_test` computes mean and std from training data only and applies both to train and test.
- **Hard thresholding (η=0)**: Correct. `out[np.abs(out) <= lam] = 0.0` correctly implements the hard-thresholding operator.
- **TISP loop**: Correct. Performs gradient step then hard thresholding on β (not b0), for 100 iterations.
- **Logistic loss and gradient**: Correct. `loss = np.mean(np.log1p(np.exp(-yz)))` and gradient `(1/n) X^T(-y * σ(-yz))` are mathematically correct.
- **Lambda sweep and selection**: Correct approach — sweeps a log-spaced grid, picks λ yielding k closest to each target.
- **All required outputs present**: Train error vs iteration plot (k≈1000), semilogx plot of train/test error vs k, and table with errors, AUC, λ values, and training time.

## Question 1B — TISP on Dexter (2 points): **Correct**

- **Sparse data loading**: Correct. Custom loader properly converts 1-based indices to 0-based (`i = int(i_str) - 1`), and uses training dimensionality for test data.
- **Same correct TISP method** applied as in 1A with proper standardization.
- **All required outputs present**: Same plots and table as 1A. Results stored in `df_tisp_dexter`.

## Question 2A — FSA on Gisette (2 points): **Correct**

- **FSA implementation**: Correct. Uses gradient step followed by `keep_top_m` (retains top m coefficients by absolute value).
- **Parameters match specification**: s=0.001, μ=100, N^iter=100, β^(0)=0 — all match homework requirements.
- **Annealing schedule**: Correct. `m(t) = ⌈k + (p−k) · (1 − t/N_iter)^μ⌉` correctly anneals from ~p features down to k.
- **Comparison plot**: Correctly overlays FSA results with `df_tisp` (Gisette TISP results from Q1A).
- **All required outputs present**: Loss vs iteration (k=1000), table with errors/AUC/training time, comparison semilogx plot.

## Question 2B — FSA on Dexter (2 points): **Incorrect**

- **FSA training on Dexter**: Correct. Uses Dexter data, correct standardization, correct parameters.
- **BUG — Wrong TISP comparison data in semilogx plot**: The comparison plot extracts TISP results from `df_tisp` (the **Gisette** TISP dataframe created in Q1A) instead of `df_tisp_dexter` (the **Dexter** TISP dataframe created in Q1B). The erroneous lines are:
  - `k_tisp = df_tisp["k_selected"].values`
  - `tr_tisp = df_tisp["train_err"].values`
  - `te_tisp = df_tisp["test_err"].values`

  These should reference `df_tisp_dexter` instead of `df_tisp`. As a result, the Dexter FSA errors are plotted against Gisette TISP errors, producing a meaningless cross-dataset comparison.

## Summary

| Question | Status | Deduction | Notes |
|----------|--------|-----------|-------|
| 1A — TISP Gisette | Correct | 0 | All requirements met |
| 1B — TISP Dexter | Correct | 0 | All requirements met |
| 2A — FSA Gisette | Correct | 0 | All requirements met |
| 2B — FSA Dexter | Incorrect | Points off | Comparison plot uses wrong dataset (`df_tisp` instead of `df_tisp_dexter`) |
