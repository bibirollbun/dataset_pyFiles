# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# === Binary classification with Bank dataset (Playground Series S5E8) ===
# Works in Kaggle: auto-locates train/test/sample_submission.
# Produces submission.csv with probability predictions for y and plots for EDA & model diagnostics.

import os, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import itertools

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, auc, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import sparse

# Prefer xgboost; it's preinstalled on Kaggle
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise RuntimeError(
        "xgboost not available in this environment. Please add it via Kaggle Add-ons "
        "or choose another model."
    ) from e

# -------- Locate data (robust) --------
def find_path(name="train.csv"):
    if os.path.exists(name):
        return name
    candidates = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not candidates:
        raise FileNotFoundError(f"Could not find {name}. Make sure the dataset is added to the notebook.")
    candidates.sort(key=len)
    return candidates[0]

train_path = find_path("train.csv")
test_path = find_path("test.csv")
sub_path = find_path("sample_submission.csv")

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

# -------- Basic info --------
TARGET = "y"
ID_COL = "id"

assert TARGET in train.columns, "Target column 'y' not found in train."
assert ID_COL in train.columns and ID_COL in test.columns, "Column 'id' missing."

features = [c for c in train.columns if c not in [ID_COL, TARGET]]
cat_cols = [c for c in features if train[c].dtype == "object"]
num_cols = [c for c in features if c not in cat_cols]

X = train[features].copy()
y = train[TARGET].astype(int).copy()
X_test = test[features].copy()

# -------- Quick EDA --------
print("Train shape:", train.shape, "| Test shape:", test.shape)
print("Target distribution:\n", y.value_counts(normalize=True).rename("ratio"))

# Class balance bar
plt.figure(figsize=(4,3))
y.value_counts().sort_index().plot(kind="bar")
plt.title("Class counts")
plt.xlabel("y")
plt.ylabel("count")
plt.tight_layout()
plt.show()

# Numeric histograms by target
if len(num_cols) > 0:
    n = len(num_cols)
    cols = 3
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(4*cols, 3*rows))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(rows, cols, i)
        # overlay two histograms
        plt.hist(train.loc[y==0, col].dropna(), bins=30, alpha=0.6, label="y=0")
        plt.hist(train.loc[y==1, col].dropna(), bins=30, alpha=0.6, label="y=1")
        plt.title(col)
        plt.legend()
    plt.tight_layout()
    plt.show()

# Correlation heatmap (numerics)
if len(num_cols) > 1:
    corr = train[num_cols].corr()
    plt.figure(figsize=(1.2*len(num_cols), 1.0*len(num_cols)))
    plt.imshow(corr, interpolation='nearest')
    plt.title("Correlation (numeric features)")
    plt.colorbar()
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.tight_layout()
    plt.show()

# -------- Preprocess --------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
)

# Class imbalance weight
neg = (y == 0).sum()
pos = (y == 1).sum()
scale_pos_weight = neg / max(pos, 1)

# -------- Model --------
model = XGBClassifier(
    n_estimators=900,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=2.0,
    min_child_weight=2,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",  # set to "gpu_hist" if GPU is enabled
    random_state=42,
    scale_pos_weight=scale_pos_weight,
)

pipe = Pipeline(steps=[("prep", preprocess), ("clf", model)])

# -------- 5-fold CV with OOF, plots --------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))
roc_points = []   # store fpr/tpr per fold for plotting
pr_points  = []   # store precision/recall per fold
fold_aucs = []
fold_aps = []

for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]

    pipe_fold = Pipeline(steps=[("prep", preprocess), ("clf", model)])
    pipe_fold.fit(X_tr, y_tr)
    val_proba = pipe_fold.predict_proba(X_va)[:, 1]
    oof_pred[val_idx] = val_proba

    fpr, tpr, _ = roc_curve(y_va, val_proba)
    roc_points.append((fpr, tpr))
    fold_auc = auc(fpr, tpr)
    fold_aucs.append(fold_auc)

    prec, rec, _ = precision_recall_curve(y_va, val_proba)
    pr_points.append((prec, rec))
    fold_ap = average_precision_score(y_va, val_proba)
    fold_aps.append(fold_ap)

print(f"OOF ROC AUC: {roc_auc_score(y, oof_pred):.5f} | mean-fold AUC: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")
print(f"OOF PR AUC:  {average_precision_score(y, oof_pred):.5f} | mean-fold AP:  {np.mean(fold_aps):.5f} ± {np.std(fold_aps):.5f}")

# ROC curve per fold + mean curve (via simple averaging of TPR at common FPR grid)
plt.figure(figsize=(6,5))
fpr_grid = np.linspace(0,1,1000)
tpr_interp = []
for i, (fpr, tpr) in enumerate(roc_points, 1):
    plt.plot(fpr, tpr, alpha=0.35, label=f"Fold {i} (AUC={fold_aucs[i-1]:.3f})")
    tpr_interp.append(np.interp(fpr_grid, fpr, tpr))
mean_tpr = np.mean(tpr_interp, axis=0)
plt.plot(fpr_grid, mean_tpr, linewidth=2.0, label=f"Mean ROC (AUC={auc(fpr_grid, mean_tpr):.3f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves (5-fold)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

# Precision-Recall curve (OOF)
prec, rec, _ = precision_recall_curve(y, oof_pred)
ap = average_precision_score(y, oof_pred)
plt.figure(figsize=(6,5))
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall (OOF), AP={ap:.3f}")
plt.tight_layout()
plt.show()

# Calibration curve
prob_true, prob_pred = calibration_curve(y, oof_pred, n_bins=10, strategy="uniform")
plt.figure(figsize=(5,5))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], '--')
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration (Reliability) Curve")
plt.tight_layout()
plt.show()

print("OOF Brier score (lower is better):", round(brier_score_loss(y, oof_pred), 6))

# Confusion matrix at 0.5 (OOF)
y_pred_05 = (oof_pred >= 0.5).astype(int)
cm = confusion_matrix(y, y_pred_05, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
plt.figure(figsize=(4,4))
disp.plot(values_format="d", cmap="Blues", colorbar=False)
plt.title("Confusion Matrix (threshold=0.5, OOF)")
plt.tight_layout()
plt.show()

# -------- Fit on full data & predict test --------
pipe.fit(X, y)
test_proba = pipe.predict_proba(X_test)[:, 1]

submission = sample_sub.copy()
submission[TARGET] = test_proba
submission = submission[[ID_COL, TARGET]].sort_values(ID_COL).reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print("Done. Wrote submission.csv")
print(submission.head())

# -------- Feature importance (gain) mapped to real feature names --------
# After fitting, grab the fitted preprocessor to get feature names in the same order used by the model.
fitted_prep = pipe.named_steps["prep"]
feature_names = fitted_prep.get_feature_names_out()  # order matches transformed columns

# Get importance from booster (keys like 'f0', 'f1', ...)
booster = pipe.named_steps["clf"].get_booster()
raw_importance = booster.get_score(importance_type="gain")

# Map 'fN' -> actual name
imp_rows = []
for k, v in raw_importance.items():
    if k.startswith("f"):
        idx = int(k[1:])
        if 0 <= idx < len(feature_names):
            imp_rows.append((feature_names[idx], v))
        else:
            imp_rows.append((k, v))
    else:
        imp_rows.append((k, v))

imp_df = pd.DataFrame(imp_rows, columns=["feature", "gain"]).sort_values("gain", ascending=False)
print("\nTop 25 features by gain:")
print(imp_df.head(25))

# Bar plot of top-N importances
TOP_N = 25
top_imp = imp_df.head(TOP_N).iloc[::-1]  # reverse for horizontal bar
plt.figure(figsize=(8, max(3, TOP_N*0.3)))
plt.barh(top_imp["feature"], top_imp["gain"])
plt.title(f"XGBoost Feature Importance (gain) - Top {TOP_N}")
plt.xlabel("Gain")
plt.tight_layout()
plt.show()




