import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")



# Prefer Kaggle path; fall back to local files if running outside Kaggle
kaggle_dir = Path("/kaggle/input/santander-customer-satisfaction")
data_dir = kaggle_dir if kaggle_dir.exists() else Path(".")

train = pd.read_csv(data_dir / "train.csv")
test  = pd.read_csv(data_dir / "test.csv")
print("Loaded:", train.shape, test.shape)



# Small sample to eyeball scales and formatting
display(train.head(3))
display(test.head(3))



# Count classes 0 vs 1
train["TARGET"].value_counts()



train.isnull().sum()


# Compute class counts (0 vs 1)
vc = train["TARGET"].value_counts()


vc.plot(kind="bar")
plt.title("TARGET distribution (0=satisfied, 1=unsatisfied)")
plt.xlabel("TARGET")
plt.ylabel("Count")
plt.show()


# Count number of zero-features per row and plot histogram

# 1) New feature: how many columns are zero in each row
train["zero_count"] = (train == 0).sum(axis=1)
test["zero_count"]  = (test == 0).sum(axis=1)

# 2) Histogram of this new feature
train["zero_count"].hist(bins=30)
plt.title("Distribution of zero_count (per row)")
plt.xlabel("number of zero-features")
plt.ylabel("frequency")
plt.show()



# 1) % of zeros per column
zero_pct = (train.eq(0).mean()*100).sort_values(ascending=False)

# Plot: Top-20 
zero_pct.head(20).plot(kind="barh", figsize=(6,4))
plt.title("Share of zero-values per column (top 20)")
plt.xlabel("% zeros")
plt.show()



# Correlation heatmap for top-20 high-variance features
import matplotlib.pyplot as plt

# 1) pick features (exclude ID/TARGET if present)
drop_cols = [c for c in ["ID", "TARGET"] if c in train.columns]
num_cols  = [c for c in train.columns if c not in drop_cols]

# 2) select top-20 by variance
top20 = train[num_cols].var().sort_values(ascending=False).head(20).index

# 3) correlation matrix
corr = train[top20].corr()

# 4) heatmap
plt.figure(figsize=(8,6))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(top20)), top20, rotation=90)
plt.yticks(range(len(top20)), top20)
plt.title("Correlation (top 20 high-variance features)")
plt.tight_layout()
plt.show()



feat = "delta_num_aport_var13_1y3"
train.boxplot(column=feat, by="TARGET", figsize=(5,4))
plt.ylim(train[feat].quantile(0.0), train[feat].quantile(0.99))  # cut top 1%
plt.title(f"Boxplot of {feat} (capped at 99th percentile)")
plt.suptitle("")
plt.show()



# Define target and IDs
y = train["TARGET"]
train_id = train["ID"]
test_id = test["ID"]

# Define feature sets (drop TARGET + ID)
X = train.drop(columns=["TARGET", "ID"])
X_test = test.drop(columns=["ID"])

print(X.shape, X_test.shape, y.shape)



# 1) Drop constant columns
const_cols = [c for c in X.columns if X[c].nunique() == 1]
X = X.drop(columns=const_cols)
X_test = X_test.drop(columns=const_cols, errors="ignore")
print("Dropped constant cols:", len(const_cols))

# 2) (Optional) Drop duplicate columns
dup_mask = X.T.duplicated()
dup_cols = X.columns[dup_mask]
X = X.loc[:, ~dup_mask]
X_test = X_test.drop(columns=dup_cols, errors="ignore")
print("Dropped duplicate cols:", len(dup_cols))

# 3) Add zero_count feature
X["zero_count"] = (X == 0).sum(axis=1)
X_test["zero_count"] = (X_test == 0).sum(axis=1)
print("Added zero_count feature")
print("Final shapes:", X.shape, X_test.shape)



# LightGBM + StratifiedKFold 

# Ensure X_test has the same columns as X, in the same order
X_test = X_test.reindex(columns=X.columns, fill_value=0)
print("Aligned feature sets:", X.shape, X_test.shape)


# imbalance weight (negatives / positives)
pos_weight = (y == 0).sum() / (y == 1).sum()

params = dict(
    n_estimators=2000,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=-1,
    scale_pos_weight=pos_weight,
    random_state=42,
    n_jobs=-1
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
pred = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)

    # callbacks for early stopping + logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(0)  # 0 = silent, change to 100 for logs
    ]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=callbacks
    )



    oof[val_idx] = model.predict_proba(X_va)[:, 1]
    pred += model.predict_proba(X_test)[:, 1] / kf.n_splits
    print(f"Fold {fold} AUC: {roc_auc_score(y_va, oof[val_idx]):.5f}")

print("OOF AUC:", roc_auc_score(y, oof))


# build submission
submission = pd.DataFrame({"ID": test_id, "TARGET": pred})
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved")


# Refit LightGBM on full train data and save model
final_model = lgb.LGBMClassifier(**params)
final_model.fit(X, y)

# save as pickle
import joblib
joblib.dump(final_model, "lgbm_santander.pkl")
print("âœ… Model saved as lgbm_santander.pkl")





