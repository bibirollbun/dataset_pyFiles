# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  IMPORTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

# Tree & Boosting
import lightgbm as lgb
import catboost as cb
import xgboost as xgb



import warnings
warnings.filterwarnings("ignore")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ðŸ“Œ LOAD DATA
# (Ensure you place train.csv, test.csv, sample_submission.csv in Kaggle input)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



# Target distribution
sns.countplot(x="diagnosed_diabetes", data=train)
plt.title("Target Distribution")
plt.show()

# Missing value overview
print("Missing values per column:\n", train.isna().sum().sort_values(ascending=False))

# Summary
train.describe().T



TARGET = "diagnosed_diabetes"
features = [c for c in train.columns if c != TARGET]

# Combine train+test for consistent encoding if needed
combined = pd.concat([train[features], test[features]], axis=0)

# Numeric scaling
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined.select_dtypes(include=np.number))
combined_scaled = pd.DataFrame(combined_scaled, columns=combined.select_dtypes(include=np.number).columns)

train_scaled = combined_scaled.iloc[:train.shape[0], :]
test_scaled  = combined_scaled.iloc[train.shape[0]:, :]

y = train[TARGET].values

# Save final features
X = train_scaled.copy()
X_test = test_scaled.copy()



n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)



lgb_preds = np.zeros(X_test.shape[0])
lgb_oof  = np.zeros(X.shape[0])

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1} Training...")

    tr_X, tr_y = X.iloc[tr_idx], y[tr_idx]
    va_X, va_y = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_estimators=4000,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # ---- Version-agnostic training ----
    # Use callbacks instead of early_stopping_rounds or verbose in .fit()
    callbacks = [
        lgb.early_stopping(stopping_rounds=200, first_metric_only=True),
        lgb.log_evaluation(period=300)  # logs every 300 rounds
    ]

    model.fit(
        tr_X, tr_y,
        eval_set=[(va_X, va_y)],
        eval_metric='auc',
        callbacks=callbacks
    )

    lgb_oof[val_idx] += model.predict_proba(va_X)[:,1]
    lgb_preds += model.predict_proba(X_test)[:,1] / n_splits

print("LGB OOF ROCâ€‘AUC:", roc_auc_score(y, lgb_oof))



cb_preds = np.zeros(X_test.shape[0])
cb_oof  = np.zeros(X.shape[0])

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_X, tr_y = X.iloc[tr_idx], y[tr_idx]
    va_X, va_y = X.iloc[val_idx], y[val_idx]

    model = cb.CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        eval_metric='AUC',
        random_seed=42,
        verbose=300,
        task_type="CPU"
    )

    model.fit(tr_X, tr_y, eval_set=(va_X, va_y), use_best_model=True)
    cb_oof[val_idx] += model.predict_proba(va_X)[:,1]
    cb_preds += model.predict_proba(X_test)[:,1] / n_splits

print("CatBoost OOF ROCâ€‘AUC:", roc_auc_score(y, cb_oof))



final_pred = (0.6 * lgb_preds + 0.4 * cb_preds)

sub["diagnosed_diabetes"] = final_pred
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv â€” ready for Kaggle upload!")


