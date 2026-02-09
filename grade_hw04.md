# HW04 Grading Report

## Q1a — TISP on Gisette (2 points): **Correct**

The TISP implementation is correct.

- **Logistic gradient** (`logistic_gradient`): Correctly computes `grad = -(1/n) X^T (y * q)` where `q = 1/(1+exp(y*(X@beta)))`. This is the proper gradient of the logistic loss `(1/n) sum log(1 + exp(-y_i * x_i^T beta))`.
- **Hard thresholding (eta=0)**: The lines `mask = np.abs(beta_next) > self.lam` followed by `self.beta = beta_next * mask` correctly implement `H_lambda(x) = x * I(|x| > lambda)` as specified in page 11 of the notes.
- **Normalization**: `StandardScaler` is fit on training data and used to transform both train and test. This correctly gives zero mean and unit variance columns, using the same parameters for both sets.
- **Lambda selection** (`get_lambda_for_k`): Uses one gradient step from beta=0 to heuristically estimate lambda for a target k. This is a crude but acceptable heuristic; the student correctly reports both `Target_k` and `Actual_k` in the results table to show any discrepancy.
- **Outputs**: All required deliverables are present — train error vs iteration plot for k=1000, semilogx plot of train/test error vs features, and the results table with train/test errors, AUC, number of selected features, lambda values, and training times.

## Q1b — TISP on Dexter (2 points): **Correct**

The same correct TISP implementation is applied to Dexter. The sparse data loader (`load_sparse_dexter`) correctly parses the `index:value` format with 1-to-0 index conversion and `n_features=20000`, matching the Dexter dataset specification.

## Q2a — FSA on Gisette (2 points): **Incorrect**

The FSA implementation contains an **off-by-one error in the annealing schedule** for M(t).

**The error**: In the `FSA.fit` method, the schedule is:
```python
alpha = t / self.mu
M_t = int(d + (self.k_target - d) * alpha)
```
With `n_iter=100` and `mu=100`, the loop variable `t` ranges from 0 to 99. The maximum value of `alpha` is `99/100 = 0.99`, so M(t) **never reaches the target k**. At the final iteration (t=99):

- For d=5000, k=10: M_99 = int(5000 + (10−5000) × 0.99) = int(59.9) = **59** instead of 10
- For d=5000, k=30: M_99 = int(5000 + (30−5000) × 0.99) = int(79.3) = **79** instead of 30
- For d=5000, k=100: M_99 = int(5000 + (100−5000) × 0.99) = int(149) = **149** instead of 100

The model finishes training with significantly more features than the target k, especially for small k values. The correct formula should be `alpha = (t + 1) / self.mu` so that at the last iteration (t=99), alpha = 100/100 = 1.0 and M_t = k exactly.

**Additional minor issues**:
- The FSA results table does not report the actual number of non-zero features (unlike the TISP table which reports `Actual_k`), which masks the off-by-one bug.
- The assignment asks "How do these training times compare with the corresponding TISP training times from 1a)?" — the code prints times in the table but provides no explicit comparison or discussion.

## Q2b — FSA on Dexter (2 points): **Incorrect**

Same off-by-one error as Q2a, with a worse impact due to the higher dimensionality (d=20000):

- For d=20000, k=10: M_99 = int(20000 + (10−20000) × 0.99) = int(209.9) = **209** instead of 10
- For d=20000, k=30: M_99 = int(20000 + (30−20000) × 0.99) = int(229.3) = **229** instead of 30

The final models retain over 20x the intended number of features for small k targets.
