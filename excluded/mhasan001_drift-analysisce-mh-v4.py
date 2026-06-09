import pandas as pd
import numpy as np
from pathlib import Path



import matplotlib.pyplot as plt
from IPython import get_ipython

ip = get_ipython()
if ip is not None:
    ip.run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.dpi'] = 110
plt.rcParams['figure.facecolor'] = 'white'



from pathlib import Path
import pandas as pd

DATA_DIR = Path("/kaggle/input/playground-series-s5e12")

train = pd.read_csv(DATA_DIR / "train.csv") 
test  = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("Top missingness (train, fraction):")
print(train.isna().mean().sort_values(ascending=False).head(5))



import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

FOLDS = 18
SEED = 24

train_s = train.copy()
test_s = test.copy()
train_s['is_test'] = 0
test_s['is_test'] = 1
df_all = pd.concat([train_s, test_s], axis=0, ignore_index=True)

object_cols = df_all.select_dtypes(include=['object']).columns
for col in object_cols:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))

X_drift = df_all.drop(['diagnosed_diabetes', 'is_test', 'id'], axis=1)
y_drift = df_all['is_test'].astype(int)

params = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    tree_method='hist',
    random_state=SEED,
)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
scores = []
for i, (tr_idx, va_idx) in enumerate(skf.split(X_drift, y_drift), 1):
    X_tr, X_va = X_drift.iloc[tr_idx], X_drift.iloc[va_idx]
    y_tr, y_va = y_drift.iloc[tr_idx], y_drift.iloc[va_idx]
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr)
    pred = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    scores.append(auc)
    print(f'  Fold {i}: {auc:.4f}')
print(f'Mean AUC: {np.mean(scores):.4f}')
print(f'Std AUC: {np.std(scores):.4f}')



import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5, 3))
plt.bar(range(1, len(scores) + 1), scores, color='#1f77b4')
plt.axhline(np.mean(scores), color='red', linestyle='--', label='mean')
plt.title('Adversarial AUC by fold') 
plt.xlabel('Fold')
plt.ylabel('AUC')
plt.legend()
plt.tight_layout()
plt.show()



from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import stats

GLOBAL_SEED = SEED
rng = np.random.RandomState(GLOBAL_SEED)

n_train = len(train)
n_test = len(test)
X_train_enc = X_drift.iloc[:n_train].copy()
X_test_enc = X_drift.iloc[n_train:].copy()

C2ST_TRAIN_SAMPLE = 15000
C2ST_TEST_SAMPLE = 15000
C2ST_FOLDS = 3
C2ST_PERMUTATIONS = 20

MMD_SAMPLE = 2000
MMD_PERMUTATIONS = 50

ENERGY_SAMPLE = 1000
ENERGY_PERMUTATIONS = 50

KNN_SAMPLE = 2000
KNN_K = 5
KNN_PERMUTATIONS = 100

MST_SAMPLE = 1000
MST_PERMUTATIONS = 200

HOTELLING_SAMPLE = 5000
HOTELLING_FEATURES = 12



# Sample for C2ST
n_train = len(X_train_enc)
n_test = len(X_test_enc)
train_idx = rng.choice(n_train, min(C2ST_TRAIN_SAMPLE, n_train), replace=False)
test_idx = rng.choice(n_test, min(C2ST_TEST_SAMPLE, n_test), replace=False)

X_c2st = np.vstack([X_train_enc.iloc[train_idx].to_numpy(), X_test_enc.iloc[test_idx].to_numpy()])
y_c2st = np.concatenate([np.zeros(len(train_idx), dtype=int), np.ones(len(test_idx), dtype=int)])

c2st_params = dict(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    tree_method='hist',
    random_state=SEED,
)

def c2st_auc(X, y, folds=3):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    aucs = []
    for tr, va in skf.split(X, y):
        model = xgb.XGBClassifier(**c2st_params)
        model.fit(X[tr], y[tr])
        pred = model.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], pred))
    return float(np.mean(aucs))

observed_auc = c2st_auc(X_c2st, y_c2st, folds=C2ST_FOLDS)

perm_aucs = []
for _ in range(C2ST_PERMUTATIONS):
    y_perm = rng.permutation(y_c2st)
    perm_aucs.append(c2st_auc(X_c2st, y_perm, folds=C2ST_FOLDS))

perm_aucs = np.asarray(perm_aucs)
p_value = (1 + (perm_aucs >= observed_auc).sum()) / (1 + len(perm_aucs))

print(f'C2ST observed AUC: {observed_auc:.4f}')
print(f'C2ST permutation p-value: {p_value:.4f} (perms={C2ST_PERMUTATIONS})')
plt.figure(figsize=(5, 3))
plt.hist(perm_aucs, bins=10, color='#8c564b', alpha=0.7)
plt.axvline(observed_auc, color='red', linestyle='--', label='observed')
plt.title('C2ST permutation null (AUC)')
plt.xlabel('AUC')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.show()



# Sample for MMD
train_idx = rng.choice(n_train, min(MMD_SAMPLE, n_train), replace=False)
test_idx = rng.choice(n_test, min(MMD_SAMPLE, n_test), replace=False)
X_mmd = X_train_enc.iloc[train_idx].to_numpy()
Y_mmd = X_test_enc.iloc[test_idx].to_numpy()

# Median heuristic for gamma
pair_dists = pdist(np.vstack([X_mmd, Y_mmd])[:1000], metric='euclidean')
median_dist = np.median(pair_dists)
gamma = 1.0 / (2.0 * median_dist ** 2) if median_dist > 0 else 1.0

def mmd_rbf(X, Y, gamma):
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m = len(X)
    n = len(Y)
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    mmd2 += (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    mmd2 -= 2.0 * Kxy.mean()
    return float(mmd2)

mmd_obs = mmd_rbf(X_mmd, Y_mmd, gamma)

perm_stats = []
XY = np.vstack([X_mmd, Y_mmd])
labels = np.array([0] * len(X_mmd) + [1] * len(Y_mmd))
for _ in range(MMD_PERMUTATIONS):
    perm = rng.permutation(labels)
    Xp = XY[perm == 0]
    Yp = XY[perm == 1]
    perm_stats.append(mmd_rbf(Xp, Yp, gamma))

perm_stats = np.asarray(perm_stats)
p_value = (1 + (perm_stats >= mmd_obs).sum()) / (1 + len(perm_stats))

print(f'MMD^2 observed: {mmd_obs:.6f}')
print(f'MMD permutation p-value: {p_value:.4f} (perms={MMD_PERMUTATIONS})')
plt.figure(figsize=(5, 3))
plt.hist(perm_stats, bins=10, color='#1f77b4', alpha=0.7)
plt.axvline(mmd_obs, color='red', linestyle='--', label='observed')
plt.title('MMD permutation null (MMD^2)')
plt.xlabel('MMD^2')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.show()



train_idx = rng.choice(n_train, min(ENERGY_SAMPLE, n_train), replace=False)
test_idx = rng.choice(n_test, min(ENERGY_SAMPLE, n_test), replace=False)
X_e = X_train_enc.iloc[train_idx].to_numpy()
Y_e = X_test_enc.iloc[test_idx].to_numpy()

# Energy statistic
XX = cdist(X_e, X_e, metric='euclidean')
YY = cdist(Y_e, Y_e, metric='euclidean')
XY = cdist(X_e, Y_e, metric='euclidean')

energy_obs = 2.0 * XY.mean() - XX.mean() - YY.mean()

perm_stats = []
XY_pool = np.vstack([X_e, Y_e])
labels = np.array([0] * len(X_e) + [1] * len(Y_e))
for _ in range(ENERGY_PERMUTATIONS):
    perm = rng.permutation(labels)
    Xp = XY_pool[perm == 0]
    Yp = XY_pool[perm == 1]
    XXp = cdist(Xp, Xp, metric='euclidean')
    YYp = cdist(Yp, Yp, metric='euclidean')
    XYp = cdist(Xp, Yp, metric='euclidean')
    stat = 2.0 * XYp.mean() - XXp.mean() - YYp.mean()
    perm_stats.append(stat)

perm_stats = np.asarray(perm_stats)
p_value = (1 + (perm_stats >= energy_obs).sum()) / (1 + len(perm_stats))

print(f'Energy distance observed: {energy_obs:.6f}')
print(f'Energy permutation p-value: {p_value:.4f} (perms={ENERGY_PERMUTATIONS})')
plt.figure(figsize=(5, 3))
plt.hist(perm_stats, bins=10, color='#2ca02c', alpha=0.7)
plt.axvline(energy_obs, color='red', linestyle='--', label='observed')
plt.title('Energy permutation null')
plt.xlabel('energy distance')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.show()



train_idx = rng.choice(n_train, min(KNN_SAMPLE // 2, n_train), replace=False)
test_idx = rng.choice(n_test, min(KNN_SAMPLE // 2, n_test), replace=False)
X_knn = np.vstack([X_train_enc.iloc[train_idx].to_numpy(), X_test_enc.iloc[test_idx].to_numpy()])
labels = np.array([0] * len(train_idx) + [1] * len(test_idx))

nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(X_knn)
indices = nn.kneighbors(X_knn, return_distance=False)
neighbor_labels = labels[indices[:, 1:]]  # drop self
pred = (neighbor_labels.mean(axis=1) >= 0.5).astype(int)
acc_obs = (pred == labels).mean()

perm_stats = []
for _ in range(KNN_PERMUTATIONS):
    perm = rng.permutation(labels)
    neighbor_labels = perm[indices[:, 1:]]
    pred = (neighbor_labels.mean(axis=1) >= 0.5).astype(int)
    perm_stats.append((pred == perm).mean())

perm_stats = np.asarray(perm_stats)
p_value = (1 + (perm_stats >= acc_obs).sum()) / (1 + len(perm_stats))

print(f'kNN accuracy observed: {acc_obs:.4f}')
print(f'kNN permutation p-value: {p_value:.4f} (perms={KNN_PERMUTATIONS})')
plt.figure(figsize=(5, 3))
plt.hist(perm_stats, bins=10, color='#ff7f0e', alpha=0.7)
plt.axvline(acc_obs, color='red', linestyle='--', label='observed')
plt.title('kNN permutation null (accuracy)')
plt.xlabel('accuracy')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.show()



train_idx = rng.choice(n_train, min(MST_SAMPLE // 2, n_train), replace=False)
test_idx = rng.choice(n_test, min(MST_SAMPLE // 2, n_test), replace=False)
X_mst = np.vstack([X_train_enc.iloc[train_idx].to_numpy(), X_test_enc.iloc[test_idx].to_numpy()])
labels = np.array([0] * len(train_idx) + [1] * len(test_idx))

D = squareform(pdist(X_mst, metric='euclidean'))
mst = minimum_spanning_tree(D)
rows, cols = mst.nonzero()

cross_edges = np.sum(labels[rows] != labels[cols])

perm_stats = []
for _ in range(MST_PERMUTATIONS):
    perm = rng.permutation(labels)
    perm_cross = np.sum(perm[rows] != perm[cols])
    perm_stats.append(perm_cross)

perm_stats = np.asarray(perm_stats)
# In MST test, fewer cross-edges indicates stronger separation
p_value = (1 + (perm_stats <= cross_edges).sum()) / (1 + len(perm_stats))

print(f'MST cross-edges observed: {cross_edges}')
print(f'MST permutation p-value: {p_value:.4f} (perms={MST_PERMUTATIONS})')
plt.figure(figsize=(5, 3))
plt.hist(perm_stats, bins=10, color='#9467bd', alpha=0.7)
plt.axvline(cross_edges, color='red', linestyle='--', label='observed')
plt.title('MST permutation null (cross-edges)')
plt.xlabel('cross-edges')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.show()



# Choose numeric features for Hotelling's T2
num_cols_hot = [c for c in train.select_dtypes(include=[np.number]).columns if c not in ['id', 'diagnosed_diabetes']]
num_cols_hot = num_cols_hot[:min(HOTELLING_FEATURES, len(num_cols_hot))]

train_hot = train[num_cols_hot].dropna()
test_hot = test[num_cols_hot].dropna()

n1 = min(HOTELLING_SAMPLE, len(train_hot))
n2 = min(HOTELLING_SAMPLE, len(test_hot))
train_hot = train_hot.sample(n1, random_state=SEED)
test_hot = test_hot.sample(n2, random_state=SEED)

# Standardize with pooled mean/std
mean_pool = pd.concat([train_hot, test_hot], axis=0).mean()
std_pool = pd.concat([train_hot, test_hot], axis=0).std().replace(0, 1)
X1 = ((train_hot - mean_pool) / std_pool).to_numpy()
X2 = ((test_hot - mean_pool) / std_pool).to_numpy()

mean1 = X1.mean(axis=0)
mean2 = X2.mean(axis=0)
S1 = np.cov(X1, rowvar=False)
S2 = np.cov(X2, rowvar=False)
Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

Sp_inv = np.linalg.pinv(Sp)
diff = (mean1 - mean2)
T2 = (n1 * n2 / (n1 + n2)) * diff.T @ Sp_inv @ diff

p = len(num_cols_hot)
df2 = n1 + n2 - p - 1
if df2 > 0:
    F = (df2 * T2) / (p * (n1 + n2 - 2))
    p_value = 1 - stats.f.cdf(F, p, df2)
else:
    p_value = np.nan

print(f"Hotelling's T2: {T2:.4f}")
print(f"Hotelling p-value: {p_value if np.isfinite(p_value) else 'nan'}")
print(f"Features used: {num_cols_hot}")



from scipy import stats

def psi(expected, actual, buckets=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if np.all(expected == expected[0]) or np.all(actual == actual[0]):
        return 0.0
    quantiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)
    exp_pct = exp_counts / np.sum(exp_counts)
    act_pct = act_counts / np.sum(act_counts)
    eps = 1e-6
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ['diagnosed_diabetes', 'id']]

rows = []
for col in num_cols:
    tr = train_s[col].dropna()
    te = test_s[col].dropna()
    ks = stats.ks_2samp(tr, te)
    rows.append({
        'feature': col,
        'ks_stat': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
        'psi': psi(tr.values, te.values, buckets=10),
        'train_mean': float(tr.mean()),
        'test_mean': float(te.mean()),
    })

num_df = pd.DataFrame(rows).sort_values('ks_stat', ascending=False)
print(num_df.head(8).to_string(index=False))



top_num = num_df.head(8).copy()
plt.figure(figsize=(7, 4))
plt.barh(top_num['feature'][::-1], top_num['ks_stat'][::-1], color='#2ca02c')
plt.title('Top numeric drift (KS statistic)')
plt.xlabel('KS statistic')
plt.tight_layout()
plt.show()

# Overlay histograms for the top feature
top_feature = top_num.iloc[0]['feature']
plt.figure(figsize=(6, 4))
plt.hist(train_s[top_feature], bins=40, alpha=0.5, label='train')
plt.hist(test_s[top_feature], bins=40, alpha=0.5, label='test')
plt.title(f'Train vs Test: {top_feature}')
plt.xlabel(top_feature)
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

top_feature = num_df.iloc[0]['feature']
tr = train_s[top_feature].dropna().values
te = test_s[top_feature].dropna().values
tr_sorted = np.sort(tr)
te_sorted = np.sort(te)
tr_ecdf = np.arange(1, len(tr_sorted) + 1) / len(tr_sorted)
te_ecdf = np.arange(1, len(te_sorted) + 1) / len(te_sorted)

# Approximate max gap location
grid = np.quantile(np.concatenate([tr_sorted, te_sorted]), np.linspace(0, 1, 200))
tr_cdf = np.searchsorted(tr_sorted, grid, side='right') / len(tr_sorted)
te_cdf = np.searchsorted(te_sorted, grid, side='right') / len(te_sorted)
diff = np.abs(tr_cdf - te_cdf)
idx = int(np.argmax(diff))
x = grid[idx]
y1 = tr_cdf[idx]
y2 = te_cdf[idx]

ks_stat = stats.ks_2samp(tr, te).statistic

plt.figure(figsize=(6, 4))
plt.plot(tr_sorted, tr_ecdf, label='train')
plt.plot(te_sorted, te_ecdf, label='test')
plt.vlines(x, y1, y2, color='red', linestyle='--', label='max gap')
plt.annotate(
    f'KS ~ {ks_stat:.3f}',
    xy=(x, (y1 + y2) / 2),
    xytext=(x, min(0.95, (y1 + y2) / 2 + 0.15)),
    arrowprops=dict(arrowstyle='->', color='red'),
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)
plt.title(f'ECDF: {top_feature}')
plt.xlabel(top_feature)
plt.ylabel('ECDF')
plt.legend()
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

psi_df = num_df[['feature', 'psi']].head(12).copy()
plt.figure(figsize=(7, 4))
plt.barh(psi_df['feature'][::-1], psi_df['psi'][::-1], color='#9467bd')
plt.axvline(0.1, color='gray', linestyle='--', linewidth=1)
plt.axvline(0.25, color='gray', linestyle='--', linewidth=1)
plt.text(0.1, -0.5, '0.10', color='gray')
plt.text(0.25, -0.5, '0.25', color='gray')
for i, (feat, val) in enumerate(zip(psi_df['feature'], psi_df['psi'])):
    if i < 3:
        plt.text(val + 0.002, len(psi_df) - 1 - i, f'{val:.3f}', va='center')
plt.title('PSI for top numeric features')
plt.xlabel('PSI')
plt.tight_layout()
plt.show()



cat_cols = train.select_dtypes(include=['object']).columns.tolist()
rows = []
for col in cat_cols:
    tr = train_s[col].astype(str)
    te = test_s[col].astype(str)
    cats = sorted(set(tr.unique()).union(te.unique()))
    tr_counts = tr.value_counts().reindex(cats, fill_value=0)
    te_counts = te.value_counts().reindex(cats, fill_value=0)
    table = np.vstack([tr_counts.values, te_counts.values])
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.sum()
    r, k = table.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r-1, k-1)))) if min(r-1, k-1) > 0 else 0.0
    tr_pct = tr_counts / tr_counts.sum()
    te_pct = te_counts / te_counts.sum()
    tvd = float(0.5 * np.abs(tr_pct - te_pct).sum())
    rows.append({
        'feature': col,
        'chi2': float(chi2),
        'pvalue': float(p),
        'cramers_v': float(cramers_v),
        'tvd': tvd,
    })

cat_df = pd.DataFrame(rows).sort_values('cramers_v', ascending=False)
print(cat_df.head(8).to_string(index=False))



import numpy as np

top_cats = cat_df.head(3)['feature'].tolist()
for col in top_cats:
    tr = train_s[col].astype(str)
    te = test_s[col].astype(str)
    cats = tr.value_counts().index[:6]
    tr_pct = tr.value_counts(normalize=True).reindex(cats, fill_value=0)
    te_pct = te.value_counts(normalize=True).reindex(cats, fill_value=0)
    x = np.arange(len(cats))
    width = 0.35
    plt.figure(figsize=(6, 3.5))
    plt.bar(x - width/2, tr_pct.values, width, label='train')
    plt.bar(x + width/2, te_pct.values, width, label='test')
    plt.xticks(x, cats, rotation=30, ha='right')
    plt.title(f'Train vs Test proportions: {col}')
    plt.ylabel('proportion')
    plt.legend()
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import numpy as np

top_cat = cat_df.iloc[0]['feature']
tr = train_s[top_cat].astype(str)
te = test_s[top_cat].astype(str)
cats = tr.value_counts().index[:6]
tr_pct = tr.value_counts(normalize=True).reindex(cats, fill_value=0)
te_pct = te.value_counts(normalize=True).reindex(cats, fill_value=0)
delta = te_pct - tr_pct

plt.figure(figsize=(6, 3.5))
plt.barh(cats[::-1], delta.values[::-1], color='#ff7f0e')
plt.axvline(0, color='black', linewidth=1)
for i, val in enumerate(delta.values[::-1]):
    plt.text(val + (0.002 if val >= 0 else -0.002), i, f'{val:+.3f}', va='center')
plt.title(f'Delta proportions (test - train): {top_cat}')
plt.xlabel('proportion delta')
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

top_cat_v = cat_df.head(8).copy()
plt.figure(figsize=(6, 3.5))
plt.barh(top_cat_v['feature'][::-1], top_cat_v['cramers_v'][::-1], color='#8c564b')
plt.title('Cramers V for top categorical features')
plt.xlabel('Cramers V')
plt.tight_layout()
plt.show()



from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ADV_FINAL_PARAMS = dict(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    tree_method='hist',
    random_state=SEED,
)

TARGET_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    tree_method='hist',
    random_state=SEED,
)

TARGET_FOLDS = 3
SHAP_SAMPLE = 50000
WEIGHTED_CV_SAMPLE = None  # set to an int for faster weighted CV

n_train = len(train)
X_train_enc = X_drift.iloc[:n_train].copy()
X_test_enc = X_drift.iloc[n_train:].copy()
y_train = train['diagnosed_diabetes'].values

adv_final = xgb.XGBClassifier(**ADV_FINAL_PARAMS)
adv_final.fit(X_drift, y_drift)
adv_prob = adv_final.predict_proba(X_drift)[:, 1]

train_adv_prob = adv_prob[:n_train]
test_adv_prob = adv_prob[n_train:]

print(f'Adversarial prob summary (train): min={train_adv_prob.min():.4f} mean={train_adv_prob.mean():.4f} max={train_adv_prob.max():.4f}')
print(f'Adversarial prob summary (test):  min={test_adv_prob.min():.4f} mean={test_adv_prob.mean():.4f} max={test_adv_prob.max():.4f}')



skf_target = StratifiedKFold(n_splits=TARGET_FOLDS, shuffle=True, random_state=SEED)
oof_pred = np.zeros(n_train)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf_target.split(X_train_enc, y_train), 1):
    model = xgb.XGBClassifier(**TARGET_PARAMS)
    model.fit(X_train_enc.iloc[tr_idx], y_train[tr_idx], verbose=False)
    pred = model.predict_proba(X_train_enc.iloc[va_idx])[:, 1]
    oof_pred[va_idx] = pred
    auc = roc_auc_score(y_train[va_idx], pred)
    fold_scores.append(auc)
    print(f'Target model fold {fold}: AUC={auc:.4f}')

print(f'Target model mean AUC: {np.mean(fold_scores):.4f}')

train_deciles = pd.qcut(train_adv_prob, 10, labels=False, duplicates='drop')
rows = []
for d in np.unique(train_deciles):
    mask = train_deciles == d
    if mask.sum() == 0:
        continue
    if len(np.unique(y_train[mask])) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_train[mask], oof_pred[mask])
    rows.append({
        'decile': int(d) + 1,
        'count': int(mask.sum()),
        'auc': float(auc),
        'pos_rate': float(y_train[mask].mean()),
    })

decile_df = pd.DataFrame(rows)
print(decile_df.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(7, 4))
bar = ax1.bar(decile_df['decile'], decile_df['auc'], color='#1f77b4', alpha=0.75)
ax1.set_xlabel('Test-likeness decile (1=train-like, 10=test-like)')
ax1.set_ylabel('AUC (bars)')
ax1.set_ylim(0.5, 1.0)

for x, y in zip(decile_df['decile'], decile_df['auc']):
    if np.isfinite(y):
        ax1.text(x, y + 0.01, f'{y:.3f}', ha='center', fontsize=8)

ax2 = ax1.twinx()
line, = ax2.plot(decile_df['decile'], decile_df['pos_rate'], color='#ff7f0e', marker='o', label='Positive rate')
ax2.set_ylabel('Positive rate (line)')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')
ax1.legend([bar, line], ['AUC (bars)', 'Positive rate (line)'], loc='lower left')

ax1.set_title('AUC by drift decile with positive rate')
fig.tight_layout()
plt.show()



rng = np.random.RandomState(SEED)
shap_n = min(SHAP_SAMPLE, X_drift.shape[0])
shap_idx = rng.choice(X_drift.index, shap_n, replace=False)

shap_matrix = xgb.DMatrix(X_drift.loc[shap_idx], feature_names=[str(c) for c in X_drift.columns])
shap_vals = adv_final.get_booster().predict(shap_matrix, pred_contribs=True)
mean_abs = np.abs(shap_vals[:, :-1]).mean(axis=0)

shap_df = pd.DataFrame({
    'feature': X_drift.columns,
    'mean_abs_shap': mean_abs,
}).sort_values('mean_abs_shap', ascending=False)

print(shap_df.head(12).to_string(index=False))

top_shap = shap_df.head(12)[::-1]
plt.figure(figsize=(7, 4))
plt.barh(top_shap['feature'], top_shap['mean_abs_shap'], color='#9467bd')
plt.title('Top drift drivers (mean |SHAP| on adversarial model)')
plt.xlabel('Mean |SHAP|')
for i, v in enumerate(top_shap['mean_abs_shap']):
    plt.text(v, i, f' {v:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.show()



gap_rows = []
for col in cat_cols:
    tr_counts = train[col].astype(str).value_counts()
    te_counts = test[col].astype(str).value_counts()
    only_train = set(tr_counts.index) - set(te_counts.index)
    only_test = set(te_counts.index) - set(tr_counts.index)
    for cat in only_train:
        gap_rows.append({'feature': col, 'category': cat, 'where': 'train_only', 'count': int(tr_counts[cat])})
    for cat in only_test:
        gap_rows.append({'feature': col, 'category': cat, 'where': 'test_only', 'count': int(te_counts[cat])})

gap_df = pd.DataFrame(gap_rows)
if gap_df.empty:
    print('No category coverage gaps found.')
else:
    gap_df = gap_df.sort_values('count', ascending=False)
    print(gap_df.head(15).to_string(index=False))

    top_gaps = gap_df.head(12)
    colors = top_gaps['where'].map({'train_only': '#1f77b4', 'test_only': '#d62728'})
    plt.figure(figsize=(8, 4))
    plt.barh(top_gaps['feature'] + ' = ' + top_gaps['category'], top_gaps['count'], color=colors)
    plt.title('Largest category coverage gaps')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.show()



quantiles = np.linspace(0.01, 0.99, 31).round(2).tolist()
q_features = num_df.head(6)['feature'].tolist()

q_rows = []
for col in q_features:
    tr_q = train_s[col].quantile(quantiles)
    te_q = test_s[col].quantile(quantiles)
    for q in quantiles:
        q_rows.append({
            'feature': col,
            'quantile': q,
            'train_q': float(tr_q[q]),
            'test_q': float(te_q[q]),
            'diff': float(te_q[q] - tr_q[q]),
        })

quant_df = pd.DataFrame(q_rows)
print(quant_df.head(15).to_string(index=False))

fig, axes = plt.subplots(1, min(3, len(q_features)), figsize=(11, 3))
if len(q_features) == 1:
    axes = [axes]
for ax, col in zip(axes, q_features[:3]):
    tr_q = train_s[col].quantile(quantiles)
    te_q = test_s[col].quantile(quantiles)
    ax.plot(quantiles, tr_q.values, marker='o', label='train')
    ax.plot(quantiles, te_q.values, marker='o', label='test')
    ax.set_title(col)
    ax.set_xlabel('quantile')
    ax.set_ylabel('value')
    ax.legend()
    ax.annotate('tail gap', xy=(0.99, te_q.values[-1]), xytext=(0.6, te_q.values[-1]),
                arrowprops={'arrowstyle': '->', 'color': 'gray'}, fontsize=8)
fig.suptitle('Quantile drift view (center vs tails)')
fig.tight_layout()
plt.show()



pairs = [
    ('bmi', 'age'),
    ('physical_activity_minutes_per_week', 'bmi'),
    ('cholesterol_total', 'triglycerides'),
]

pairs = [p for p in pairs if p[0] in train_s.columns and p[1] in train_s.columns]

for x_col, y_col in pairs:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hexbin(train_s[x_col], train_s[y_col], gridsize=40, cmap='Blues', bins='log')
    axes[0].set_title(f'Train: {x_col} vs {y_col}')
    axes[1].hexbin(test_s[x_col], test_s[y_col], gridsize=40, cmap='Reds', bins='log')
    axes[1].set_title(f'Test: {x_col} vs {y_col}')
    for ax in axes:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.text(0.02, 0.95, 'density', transform=ax.transAxes, fontsize=8, va='top')
    fig.suptitle(f'2D drift view: {x_col} vs {y_col}')
    fig.tight_layout()
    plt.show()



full_model = xgb.XGBClassifier(**TARGET_PARAMS)
full_model.fit(X_train_enc, y_train, verbose=False)

p_train_full = full_model.predict_proba(X_train_enc)[:, 1]
p_test_full = full_model.predict_proba(X_test_enc)[:, 1]

train_prior = float(np.mean(y_train))

def estimate_label_shift(p_test, p_train_prior, max_iter=200, tol=1e-6):
    pi = p_train_prior
    for _ in range(max_iter):
        num = (pi / p_train_prior) * p_test
        den = num + ((1 - pi) / (1 - p_train_prior)) * (1 - p_test)
        w = num / den
        new_pi = float(w.mean())
        if abs(new_pi - pi) < tol:
            break
        pi = new_pi
    return pi

est_test_prior = estimate_label_shift(p_test_full, train_prior)
print(f'Train prior: {train_prior:.4f}')
print(f'Estimated test prior: {est_test_prior:.4f}')

plt.figure(figsize=(6, 3))
plt.hist(p_train_full, bins=40, alpha=0.5, label='train')
plt.hist(p_test_full, bins=40, alpha=0.5, label='test')
plt.axvline(p_train_full.mean(), color='blue', linestyle='--')
plt.axvline(p_test_full.mean(), color='red', linestyle='--')
plt.title('Predicted probability shift (train vs test)')
plt.xlabel('Predicted probability')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.show()



low_cut = np.quantile(train_adv_prob, 0.2)
high_cut = np.quantile(train_adv_prob, 0.8)

low_mask = train_adv_prob <= low_cut
high_mask = train_adv_prob >= high_cut

prob_true_low, prob_pred_low = calibration_curve(y_train[low_mask], oof_pred[low_mask], n_bins=10)
prob_true_high, prob_pred_high = calibration_curve(y_train[high_mask], oof_pred[high_mask], n_bins=10)

brier_low = brier_score_loss(y_train[low_mask], oof_pred[low_mask])
brier_high = brier_score_loss(y_train[high_mask], oof_pred[high_mask])

plt.figure(figsize=(5, 4))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.plot(prob_pred_low, prob_true_low, marker='o', label=f'train-like (brier={brier_low:.4f})')
plt.plot(prob_pred_high, prob_true_high, marker='o', label=f'test-like (brier={brier_high:.4f})')
plt.title('Calibration stability under drift')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.legend()
plt.tight_layout()
plt.show()



pi_test = len(test) / (len(train) + len(test))
eps = 1e-6
train_adv_clip = np.clip(train_adv_prob, eps, 1 - eps)
weights = (train_adv_clip / (1 - train_adv_clip)) * ((1 - pi_test) / pi_test)

ess = (weights.sum() ** 2) / (weights ** 2).sum()
print(f'Effective sample size after weighting: {ess:.0f} of {len(weights)}')

plt.figure(figsize=(6, 3))
plt.hist(np.log10(weights), bins=40, color='#2ca02c', alpha=0.7)
plt.title('Distribution of covariate-shift weights (log10)')
plt.xlabel('log10(weight)')
plt.ylabel('count')
plt.tight_layout()
plt.show()

X_cv = X_train_enc
w_cv = weights
y_cv = y_train

if WEIGHTED_CV_SAMPLE:
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X_train_enc), WEIGHTED_CV_SAMPLE, replace=False)
    X_cv = X_train_enc.iloc[idx]
    y_cv = y_train[idx]
    w_cv = weights[idx]

def make_lr():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver='lbfgs', tol=1e-3)
    )

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
auc_unweighted = []
auc_weighted = []

for tr_idx, va_idx in skf.split(X_cv, y_cv):
    m1 = make_lr()
    m1.fit(X_cv.iloc[tr_idx], y_cv[tr_idx])
    p1 = m1.predict_proba(X_cv.iloc[va_idx])[:, 1]
    auc_unweighted.append(roc_auc_score(y_cv[va_idx], p1))

    m2 = make_lr()
    m2.fit(X_cv.iloc[tr_idx], y_cv[tr_idx], logisticregression__sample_weight=w_cv[tr_idx])
    p2 = m2.predict_proba(X_cv.iloc[va_idx])[:, 1]
    auc_weighted.append(roc_auc_score(y_cv[va_idx], p2))

print(f'Unweighted CV AUC: {np.mean(auc_unweighted):.4f}')
print(f'Weighted CV AUC:   {np.mean(auc_weighted):.4f}')


