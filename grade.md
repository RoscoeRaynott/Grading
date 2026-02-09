# HW04 Grading Report

## Question 1a: TISP on Gisette (2 points)

**Verdict: Correct**

- **Normalization:** Correct. Columns normalized to zero mean and unit variance using train statistics; same mean/std applied to test set. Zero-variance columns handled properly (`sigma_g[sigma_g == 0] = 1.0`).
- **Label conversion:** Correct. Converts -1/+1 labels to 0/1 for logistic loss.
- **TISP algorithm:** Correct. The iterative loop performs: (1) gradient step on logistic loss, (2) hard thresholding (`beta[np.abs(beta) < best_lam] = 0`), matching the hard-thresholding penalty with eta=0. Uses 100 iterations as required.
- **Gradient computation:** Correct. `grad = (X.T @ (prob - y)) / n` is the gradient of the average logistic loss.
- **Lambda search:** Reasonable approach — grid search over `np.logspace(-4, -0.3, 40)` with a fast 30-iteration pre-scan to find the lambda producing the closest feature count to each target.
- **Plots:** Train error vs iteration for k=1000 plotted correctly (error computed after each update). Semilogx plot of train/test error vs actual selected features present.
- **Table:** Reports k, lambda, train error, test error, AUC, and training time. All required columns present.

No errors found.

---

## Question 1b: TISP on Dexter (2 points)

**Verdict: Correct**

- **Sparse data loading:** Correct. Custom loader parses the `index:value` sparse format, properly adjusting from 1-indexed to 0-indexed (`int(index)-1`).
- **Normalization:** Correct. Same approach as 1a — train statistics applied to both train and test.
- **TISP algorithm:** Identical correct implementation as 1a, applied to Dexter data.
- **Plots and table:** All required outputs (convergence plot for k=1000, semilogx error plot, results table) are generated.

No errors found.

---

## Question 2a: FSA on Gisette (2 points)

**Verdict: Correct**

- **Parameters:** Correct. Uses s=0.001, mu=100, N_iter=100 as specified.
- **Initialization:** Correct. `beta = np.zeros(n_features)` matches beta(0)=0.
- **Annealing schedule:** Correct. `k_i = int(k + (n_features - k) * max(0, (N_iter - i) / (i * mu + N_iter)))` matches the formula k_i = k + (p-k) * max(0, (N^iter - i) / (mu*i + N^iter)).
- **Feature selection step:** Correct. Keeps top k_i features by absolute value of beta (`threshold = np.sort(np.abs(beta))[-k_i]`; zeroes out entries below threshold).
- **Gradient computation:** Correct. Same logistic loss gradient as TISP.
- **Loss plot (k=1000):** The training loss is computed using `prob` from before the gradient update. This means the plotted loss at iteration i reflects the state at the start of iteration i (loss before the i-th update). This is a standard convention and acceptable, though the loss after the final update is not captured.
- **Comparison plot:** Correct. Overlays TISP and FSA train/test errors on the same semilogx plot.
- **Table:** Reports k, train error, test error, AUC, and training time for all k values. All required columns present.
- **Training time comparison:** Tables for both TISP and FSA include timing data, enabling comparison. No explicit textual discussion is provided, but the code correctly generates the data needed for comparison.

No errors found.

---

## Question 2b: FSA on Dexter (2 points)

**Verdict: Correct**

- **Implementation:** Identical correct FSA algorithm as 2a, applied to Dexter data.
- **Plots and table:** All required outputs (loss convergence for k=1000, comparison semilogx plot overlaying TISP and FSA, results table) are generated.

No errors found.

---

## Summary

| Question | Score | Verdict |
|----------|-------|---------|
| 1a - TISP Gisette | 2/2 | Correct |
| 1b - TISP Dexter | 2/2 | Correct |
| 2a - FSA Gisette | 2/2 | Correct |
| 2b - FSA Dexter | 2/2 | Correct |
| **Total** | **8/8** | |
