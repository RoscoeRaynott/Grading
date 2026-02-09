# HW04 Grading Report

## Question 1a — TISP on Gisette (2 points): **Correct**

**Algorithm implementation:**

- **Lipschitz constant**: `L = 0.25 * spectral_norm_sq(Xtr) / n` correctly computes L = (1/4n)||X||\_op^2, the upper bound on the Lipschitz constant of the logistic loss gradient (since the second derivative of the logistic function is bounded by 1/4). The `spectral_norm_sq` function uses power iteration on X^T X to estimate ||X||\_op^2. Correct.
- **Step size**: `step = 1.0 / (L + 1e-12)` is correct.
- **Hard-thresholding threshold**: `thr = np.sqrt(2.0 * lam * step)` correctly computes sqrt(2*lambda/L), the threshold for the proximal operator of the L0 penalty (hard-thresholding with eta=0).
- **TISP iteration**: The loop computes the gradient, performs a gradient step on both `b0` and `beta`, then applies hard thresholding `beta[np.abs(beta) <= thr] = 0.0`. This correctly implements the ISTA update with L0 proximal operator. 100 iterations as required.
- **Gradient of logistic loss**: `s = sigmoid(-y * z)`, `g0 = np.mean(-y * s)`, `gb = (X.T @ (-y * s)) / n` correctly implements d/d(beta) of (1/n) sum log(1 + exp(-y_i * z_i)).
- **Standardization**: `standardize(Xtr, Xte)` uses training mean and std for both sets. Correct.
- **Lambda search**: `tisp_lambda_for_k` uses binary search over lambda to find thresholds giving approximately k selected features. Correct logic (larger lambda -> fewer features).

**Required outputs:**

- Plot of train misclassification error vs iteration for ~1000 features: present (`record_path=True` when `1000 in ks`).
- Semilogx plot of train/test error vs number of selected features: present (uses actual selected counts `sel`).
- Table with train/test errors, AUC, selected features, lambda, training time: CSV written with all required columns.

**Minor presentation issue:** The semilogx plot sets `label="train"` and `label="test"` but never calls `plt.legend()`, so the legend is not displayed.

---

## Question 1b — TISP on Dexter (2 points): **Correct**

The `run_one` function is called for both Gisette and Dexter, so the same (correct) TISP implementation applies. The Dexter sparse data loader (`load_dexter`) correctly reads `index:value` format, infers dimensionality, and constructs dense arrays. Labels are correctly converted to +1/-1. The `ks` list is filtered to only include values <= number of features, which correctly handles datasets with fewer than 3000 features.

---

## Question 2a — FSA on Gisette (2 points): **Correct**

**Algorithm implementation:**

- **Parameters**: `s=0.001`, `mu=100`, `n_iter=100` match the homework specification. `beta` is initialized to zeros as required (`beta^(0)=0`).
- **Gradient step**: `b0 -= s * g0; beta -= s * gb` performs fixed step-size gradient descent. Correct.
- **Cooling schedule**: `frac = max(0.0, (n_iter - e) / (n_iter + mu * e))` followed by `m = int(round(k + (p - k) * frac))` correctly implements m^(e) = k + (p-k) * max(0, (N_iter - e) / (N_iter + mu*e)). At e=N_iter, frac=0 so m=k. Correct.
- **Feature selection per iteration**: `np.argpartition(np.abs(beta), -m)[-m:]` keeps the top-m features by absolute value and zeros out the rest. Correct.
- **Final selection**: After the loop, top-k features are kept and the rest zeroed. Correct.

**Required outputs:**

- Plot of training logistic loss vs iteration for k=1000: present (`record_loss=True` when `k==1000`).
- Table with train/test errors, AUC, training time: FSA CSV written with all required columns.
- Combined semilogx plot of TISP and FSA train/test errors vs k: present (the comparison plot at the end of `run_one`).
- Training time comparison with TISP: times are printed and saved in both CSV tables for comparison.

**Minor presentation issue:** The comparison semilogx plot sets labels but never calls `plt.legend()`, so the four lines (TISP train/test, FSA train/test) cannot be distinguished visually.

---

## Question 2b — FSA on Dexter (2 points): **Correct**

Same `run_one` function handles Dexter with the same correct FSA implementation. All required outputs are generated for both datasets.

---

## Summary

| Question | Verdict | Score |
|----------|---------|-------|
| 1a) TISP Gisette | Correct | 2/2 |
| 1b) TISP Dexter | Correct | 2/2 |
| 2a) FSA Gisette | Correct | 2/2 |
| 2b) FSA Dexter | Correct | 2/2 |
| **Total** | | **8/8** |

**Note:** Two plots (TISP semilogx and TISP-vs-FSA comparison) are missing `plt.legend()` calls, making the multi-line plots harder to interpret. This is a cosmetic presentation issue and does not affect algorithmic correctness.
