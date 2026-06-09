import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from collections import defaultdict
import os


SEED = 42
N_SPLITS = 5
TARGET = "loan_paid_back"
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
DATA_PATHS = {
    "train": "/kaggle/input/playground-series-s5e11/train.csv",
    "test": "/kaggle/input/playground-series-s5e11/test.csv",
    "orig": "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv"
}
OUT_DIR = "./"
os.makedirs(OUT_DIR, exist_ok=True)


# Load data
train = pd.read_csv(DATA_PATHS["train"])
test = pd.read_csv(DATA_PATHS["test"])
orig = pd.read_csv(DATA_PATHS["orig"])


train.info()


orig.info()


train.describe().style.background_gradient(cmap='rainbow')


orig.describe().style.background_gradient(cmap='rainbow')


print("Shapes:", train.shape, test.shape, orig.shape)


# Basic feature list

BASE = [c for c in train.columns if c not in ['id', TARGET]]

# Create interaction features
INTER_NAMES = []
for a, b in combinations(BASE, 2):
    INTER_NAMES.append(f"{a}__{b}")

def create_interactions(dfs, cols_a):
    print("Creating interactions (this may take a while)...")
    for col1, col2 in combinations(cols_a, 2):
        name = f"{col1}__{col2}"
        for df in dfs:
            df[name] = df[col1].astype(str) + "|" + df[col2].astype(str)

create_interactions([train, test, orig], BASE)
print(f"{len(INTER_NAMES)} interaction features created.")


# Aggregations from original dataset

ORIG_FEATURES = []
for col in BASE:
    mean_name = f"orig_mean_{col}"
    mean_map = orig.groupby(col)[TARGET].mean()
    for df in [train, test]:
        df[mean_name] = df[col].map(mean_map).fillna(orig[TARGET].mean())
    ORIG_FEATURES.append(mean_name)

    count_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size()
    for df in [train, test]:
        df[count_name] = df[col].map(count_map).fillna(0).astype(int)
    ORIG_FEATURES.append(count_name)

print(f"{len(ORIG_FEATURES)} orig-based features created.")


# Final feature list

FEATURES = BASE + ORIG_FEATURES + INTER_NAMES
FEATURES = [f for f in FEATURES if f in train.columns and f in test.columns]
print("Total features used:", len(FEATURES))

X = train[FEATURES].copy()
y = train[TARGET].copy()
X_test = test[FEATURES].copy()


# Target Encoder

class KFoldTargetEncoder:
    """KFold-based mean target encoder with smoothing."""
    def __init__(self, cols, n_splits=5, seed=42, smoothing='auto'):
        self.cols = cols
        self.n_splits = n_splits
        self.seed = seed
        self.smoothing = smoothing
        self._global_mean = None
        self._mappings = {}

    def _compute_m(self, counts):
        if self.smoothing == 'auto':
            return counts / (counts + counts.mean())
        else:
            return float(self.smoothing)

    def fit_transform(self, X, y):
        self._global_mean = y.mean()
        out = pd.DataFrame(index=X.index)
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for col in self.cols:
            encoded = pd.Series(index=X.index, dtype=float)
            for tr_idx, val_idx in kf.split(X, y):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                gp = X_tr.assign(target=y_tr).groupby(col)['target'].agg(['mean', 'count'])
                counts, means = gp['count'], gp['mean']
                m_arr = self._compute_m(counts)
                smooth_map = (counts * means + m_arr * y_tr.mean()) / (counts + m_arr)
                encoded.iloc[val_idx] = X.iloc[val_idx][col].map(smooth_map)
            encoded.fillna(self._global_mean, inplace=True)
            out[f"TE_{col}"] = encoded

            full_gp = X.assign(target=y).groupby(col)['target'].agg(['mean', 'count'])
            counts, means = full_gp['count'], full_gp['mean']
            m_arr = self._compute_m(counts)
            full_smooth = (counts * means + m_arr * y.mean()) / (counts + m_arr)
            self._mappings[col] = full_smooth
        return out

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for col in self.cols:
            mapped = X[col].map(self._mappings[col])
            mapped.fillna(self._global_mean, inplace=True)
            out[f"TE_{col}"] = mapped
        return out



# Apply Target Encoding

INTER_TO_ENCODE = [c for c in INTER_NAMES if c in X.columns]
print("Target-encoding", len(INTER_TO_ENCODE), "interaction features...")

TE = KFoldTargetEncoder(INTER_TO_ENCODE, n_splits=5, seed=SEED, smoothing='auto')
te_train = TE.fit_transform(X, y)
te_test = TE.transform(X_test)

X_enc = pd.concat([X.drop(columns=INTER_TO_ENCODE), te_train], axis=1)
X_test_enc = pd.concat([X_test.drop(columns=INTER_TO_ENCODE), te_test], axis=1)

for c in CATS:
    if c in X_enc.columns:
        X_enc[c] = X_enc[c].astype('category')
        X_test_enc[c] = X_test_enc[c].astype('category')

print("Encoding done. Shapes:", X_enc.shape, X_test_enc.shape)



# Model Training (LGBM)

params = dict(
    n_estimators=10000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=6,
    colsample_bytree=0.5,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=SEED,
    n_jobs=-1,
    metric='auc',
    objective='binary'
)

oof_preds = np.zeros(len(X_enc))
test_preds = np.zeros(len(X_test_enc))
fold_importances = defaultdict(float)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

roc_curves, fold_scores = [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_enc, y), start=1):
    print(f"--- Fold {fold}/{N_SPLITS} ---")
    X_tr, X_val = X_enc.iloc[tr_idx], X_enc.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LGBMClassifier(**params)

    # Compatible fit using callbacks
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=True),
            log_evaluation(period=500)
        ]
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test_enc)[:, 1] / N_SPLITS

    auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.4f}")

    if hasattr(model, 'feature_importances_'):
        for f, imp in zip(X_tr.columns, model.feature_importances_):
            fold_importances[f] += imp

    fpr, tpr, _ = roc_curve(y_val, val_pred)
    roc_curves.append((fpr, tpr, auc))

overall_auc = roc_auc_score(y, oof_preds)
print("Fold AUCs:", [round(s,4) for s in fold_scores])
print(f"Overall OOF AUC: {overall_auc:.5f}")



# Feature Importance & Visualization

fi_df = pd.DataFrame(
    [(f, imp / N_SPLITS) for f, imp in fold_importances.items()],
    columns=['feature', 'importance']
).sort_values('importance', ascending=False)

plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
(ax_roc, ax_fi), (ax_hist, ax_calib) = axes.reshape(2,2)

# 1) ROC Curves
for (fpr, tpr, auc) in roc_curves:
    ax_roc.plot(fpr, tpr, lw=1, alpha=0.6, label=f'Fold AUC={auc:.3f}')
fpr_all, tpr_all, _ = roc_curve(y, oof_preds)
ax_roc.plot(fpr_all, tpr_all, color='black', lw=2, label=f'OOF AUC={overall_auc:.3f}')
ax_roc.plot([0,1],[0,1],'--',color='grey',alpha=0.6)
ax_roc.set_title("ROC Curves (per fold + OOF)",fontsize=14, fontweight='bold')
ax_roc.set_xlabel("False Positive Rate",fontsize=12, fontweight='bold', color='darkblue')
ax_roc.set_ylabel("True Positive Rate",fontsize=12, fontweight='bold', color='darkblue')
ax_roc.legend(loc='lower right', fontsize='small')

# 2) Feature Importance
topk = 25
fi_plot = fi_df.head(topk).sort_values('importance', ascending=True)
ax_fi.barh(fi_plot['feature'], fi_plot['importance'])
ax_fi.set_title(f"Top {topk} Feature Importances",fontsize=14, fontweight='bold')
ax_fi.set_xlabel("Average Importance",fontsize=12, fontweight='bold', color='darkblue')
ax_fi.tick_params(axis='y', labelsize=8)

# 3) Prediction Distribution
ax_hist.hist(oof_preds, bins=50, alpha=0.6, label='OOF (train)', density=True)
ax_hist.hist(test_preds, bins=50, alpha=0.6, label='Test', density=True)
ax_hist.set_title("Prediction Distribution: OOF vs Test",fontsize=14, fontweight='bold')
ax_hist.set_xlabel("Predicted Probability",fontsize=12, fontweight='bold', color='darkblue')
ax_hist.legend()

# 4) Calibration Plot
n_bins = 10
bins = np.linspace(0,1,n_bins+1)
binids = np.digitize(oof_preds, bins) - 1
bin_centers = 0.5 * (bins[:-1] + bins[1:])
observed, pred_mean = [], []
for i in range(n_bins):
    mask = binids == i
    if mask.sum() == 0:
        observed.append(np.nan)
        pred_mean.append(np.nan)
    else:
        observed.append(y.iloc[mask].mean())
        pred_mean.append(oof_preds[mask].mean())

ax_calib.plot(bin_centers, observed, marker='o', label='Observed')
ax_calib.plot(bin_centers, pred_mean, marker='x', label='Predicted Mean')
ax_calib.plot([0,1],[0,1],'--',color='grey',alpha=0.6)
ax_calib.set_title("Calibration (Binned)",fontsize=14, fontweight='bold')
ax_calib.set_xlabel("Predicted Probability (Bin Center)",fontsize=12, fontweight='bold', color='darkblue')
ax_calib.set_ylabel("Observed Frequency",fontsize=12, fontweight='bold', color='darkblue')
ax_calib.legend()

for ax in np.ravel(axes):
    ax.set_facecolor('#ffe3ee')  
    ax.grid(False) 

plt.tight_layout()
plt.show()


# Save Outputs

oof_file = os.path.join(OUT_DIR, f"oof_lgb_cv_{overall_auc:.5f}.csv")
test_file = os.path.join(OUT_DIR, f"test_lgb_cv_{overall_auc:.5f}.csv")
submission_file = os.path.join(OUT_DIR, "submission.csv")

# Save OOF predictions
pd.DataFrame({
    'id': train['id'],
    TARGET: oof_preds
}).to_csv(oof_file, index=False)

# Save test predictions (full test results)
pd.DataFrame({
    'id': test['id'],
    TARGET: test_preds
}).to_csv(test_file, index=False)


submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_preds
})

submission.to_csv(submission_file, index=False)
print(f"Submission file saved: {submission_file}")
print(submission.head())





