# ======================================================
# âš™ï¸� Five Ensemble Strategies + Diagnostics (Matplotlib)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna

# ---------- 0) CONFIG ----------
# List your submission files and (optional) public RMSEs.
files = [
    '/kaggle/input/playground-5-10-top-submissions/submission aliffa agnur autogluon16.csv',
    '/kaggle/input/playground-5-10-top-submissions/submission anthonythierren cb ensemble.csv',
    '/kaggle/input/playground-5-10-top-submissions/submission sung hur.csv',
    '/kaggle/input/playground-5-10-top-submissions/submission.csv'
]
public_rmse = [0.05539, 0.05539, 0.05540, 0.05538]  # if unknown, set all equal

target_col = 'accident_risk'
id_col = 'id'
clip_min, clip_max = 1e-6, 1 - 1e-6
seed = 42
np.random.seed(seed)

# ---------- 1) LOAD ----------
subs = [pd.read_csv(f) for f in files]
base = subs[0][[id_col]].copy()
for i, s in enumerate(subs):
    base = base.merge(s[[id_col, target_col]].rename(columns={target_col: f'm{i}'}), on=id_col, how='inner')

ids = base[id_col].values
preds = base[[c for c in base.columns if c.startswith('m')]].values
M = preds.shape[1]
print(f"âœ… Loaded {M} submissions, {len(ids):,} rows.")



def minmax01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-12)

def safe_logit(p):
    p = np.clip(p, clip_min, clip_max)
    return np.log(p / (1 - p))

def safe_sigmoid(z):
    return 1 / (1 + np.exp(-z))

def to_ranks01(X):
    R = np.empty_like(X, dtype=float)
    n = X.shape[0]
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        R[:, j] = ranks / (n - 1)
    return R

def pearson_corr_matrix(X):
    Xc = X - X.mean(0)
    return (Xc.T @ Xc) / (X.shape[0] - 1 + 1e-9)





# ---------- 3) ENSEMBLE STRATEGIES ----------

# (1) Simple Average
ens_simple = preds.mean(axis=1)

# (2) Rank Averaging (robust)
ranks = to_ranks01(preds)
ens_rankavg = ranks.mean(axis=1)

# (3) Logit (Geometric) Averaging
norm01 = np.column_stack([minmax01(preds[:, j]) for j in range(M)])
ens_logit = safe_sigmoid(safe_logit(norm01).mean(axis=1))

# (4) Correlation-Aware Weighted Average
conf = np.exp(-np.array(public_rmse))
conf = conf / conf.sum()
C = pearson_corr_matrix(preds)
avg_corr = (C.sum(axis=1) - 1.0) / max(M - 1, 1)
diverse = 1.0 / (1e-6 + (avg_corr - avg_corr.min()) / (avg_corr.max() - avg_corr.min() + 1e-9) + 1.0)
raw_w = conf * diverse
w_corrconf = raw_w / raw_w.sum()
ens_corrconf = preds @ w_corrconf

# (5) Optuna Fine-Tuned Weights
def objective(trial):
    w = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(M)])
    w /= w.sum() + 1e-9
    prior_term = np.sqrt(((w - conf) ** 2).sum())
    corr_term = float(w @ C @ w)
    entropy = -np.sum(w * np.log(w + 1e-9))
    return prior_term + 0.25 * corr_term - 0.05 * entropy

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200, show_progress_bar=False)
w_opt = np.array([study.best_params[f"w{i}"] for i in range(M)])
w_opt /= w_opt.sum()
ens_optuna = preds @ w_opt




# ---------- 4) SAVE FIVE SUBMISSIONS ----------
ensembles = {
    'submission_simple.csv': ens_simple,
    'submission_rankavg.csv': ens_rankavg,
    'submission_logit.csv': ens_logit,
    'submission_corrconf.csv': ens_corrconf,
    'submission_optuna.csv': ens_optuna,
}

for fname, vec in ensembles.items():
    pd.DataFrame({id_col: ids, target_col: vec}).to_csv(fname, index=False)
    print(f"ğŸ’¾ Saved {fname}")

print("\nFinal Weights:")
print(f"  Conf:      {np.round(conf, 4)}")
print(f"  CorrConf:  {np.round(w_corrconf, 4)}")
print(f"  Optuna:    {np.round(w_opt, 4)}")


plt.figure(figsize=(7,4))
x = np.arange(M)
bw = 0.25
plt.bar(x - bw, conf, width=bw, label='Confidence prior')
plt.bar(x, w_corrconf, width=bw, label='CorrConf')
plt.bar(x + bw, w_opt, width=bw, label='Optuna')
plt.title('Model Weights Comparison')
plt.xticks(x, [f'm{i}' for i in range(M)])
plt.legend()
plt.show()

print("\nâœ… 5 submission files generated successfully.")


# ---------- 5) PLOTS ----------

plt.figure(figsize=(8,5))
for j in range(M):
    plt.hist(preds[:, j], bins=40, alpha=0.3, label=f'm{j}')
for name, vec in ensembles.items():
    plt.hist(vec, bins=40, alpha=0.5, label=name.split('.')[0], histtype='step')
plt.title('Distributions: Models vs Ensembles')
plt.legend()
plt.show()

plt.figure(figsize=(5,4))
im = plt.imshow(pearson_corr_matrix(preds), vmin=-1, vmax=1)
plt.title('Correlation Heatmap (Models)')
plt.colorbar(im)
plt.xticks(range(M), [f'm{i}' for i in range(M)], rotation=90)
plt.yticks(range(M), [f'm{i}' for i in range(M)])
plt.show()








