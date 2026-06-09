import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")



print(train_df.isnull().sum())
print(test_df.isnull().sum())



object_cols = train_df.select_dtypes(include="object").columns
print(object_cols)


for col_name in object_cols:
    print(f"{col_name} \n >>> {sorted(train_df[col_name].unique())} \n >>> {sorted(test_df[col_name].unique())} \n")


X = train_df.drop(["y", "id"], axis=1)
y = train_df["y"]
X_test = test_df.drop(["id"], axis=1)


from sklearn.preprocessing import LabelEncoder

for col_name in object_cols:
    le = LabelEncoder()
    X[col_name] = le.fit_transform(X[col_name])
    X_test[col_name] = le.transform(X_test[col_name])


# â”€â”€ Shell 1: LightGBM hold-out AUC â”€â”€
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1) create hold-out
X_train_base, X_hold, y_train_base, y_hold = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2) fit & score
lgbm = lgb.LGBMClassifier(
    n_estimators=20000,
    learning_rate=0.06,
    num_leaves=31,
    max_depth=7,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_alpha=0.8,
    reg_lambda=3.0,
    max_bin=255,
    random_state=42,
    verbosity=-1
)
lgbm.fit(X_train_base, y_train_base)
y_pred = lgbm.predict_proba(X_hold)[:, 1]
print("LightGBM hold-out AUC:", roc_auc_score(y_hold, y_pred))



# â”€â”€ Shell 2: XGBoost hold-out AUC â”€â”€
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# assume X_train_base, X_hold, y_train_base, y_hold from Shell 1
xgb = XGBClassifier(
    n_estimators=10000,
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.01,
    early_stopping_rounds=200,
    random_state=42,
    enable_categorical=True,
    tree_method='gpu_hist',  # or 'hist' if no GPU
    use_label_encoder=False,
    n_jobs=-1
)
xgb.fit(
    X_train_base, y_train_base,
    eval_set=[(X_hold, y_hold)],
    verbose=False
)
y_pred = xgb.predict_proba(X_hold)[:, 1]
print("XGBoost hold-out AUC:", roc_auc_score(y_hold, y_pred))



# â”€â”€ Shell 3: CatBoost hold-out AUC â”€â”€
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# assume X_train_base, X_hold, y_train_base, y_hold
cat = CatBoostClassifier(
    iterations=30000,
    learning_rate=0.4,
    depth=8,
    bootstrap_type='MVS',
    boosting_type='Plain',
    loss_function='Logloss',
    subsample=0.9,
    random_strength=2.0,
    task_type='CPU',
    eval_metric='AUC',
    random_seed=42,
    verbose=False
)
cat.fit(X_train_base, y_train_base, eval_set=(X_hold, y_hold))
y_pred = cat.predict_proba(X_hold)[:, 1]
print("CatBoost hold-out AUC:", roc_auc_score(y_hold, y_pred))



# â”€â”€ Shell 4: Stratified K-Fold Stacking â”€â”€
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# base models from above shells
base_models = {
    'lgbm': lgbm,
    'xgb': xgb,
    'cat': cat
}

# placeholders
oof = {name: np.zeros(len(X)) for name in base_models}
test_pred = {name: np.zeros(len(X_test)) for name in base_models}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in base_models.items():
    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )
        oof[name][val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred[name] += model.predict_proba(X_test)[:, 1] / kf.n_splits

    print(f"{name} CV AUC: {roc_auc_score(y, oof[name]):.4f}")

# build meta-features
meta_X = pd.DataFrame(oof)
meta_X_test = pd.DataFrame(test_pred)

# train meta-model
meta = LogisticRegression(random_state=42)
meta_oof = np.zeros(len(X))
meta_test = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(meta_X, y):
    mtr, mvl = meta_X.iloc[train_idx], meta_X.iloc[val_idx]
    ytr, yvl = y.iloc[train_idx], y.iloc[val_idx]

    meta.fit(mtr, ytr)
    meta_oof[val_idx] = meta.predict_proba(mvl)[:, 1]
    meta_test += meta.predict_proba(meta_X_test)[:, 1] / kf.n_splits
    print("Meta-model fold AUC:", roc_auc_score(yvl, meta_oof[val_idx]))

print("Final stacked OOF AUC:", roc_auc_score(y, meta_oof))


# final submission
submission = pd.DataFrame({'id': test_df['id'], 'prediction': meta_test})
submission.to_csv('stacked_submission.csv', index=False)
print("Saved stacked_submission.csv") 

