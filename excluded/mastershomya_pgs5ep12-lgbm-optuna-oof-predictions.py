import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_train.head()


df_tr = df_train.drop(columns=["id"])


df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
df_te = df_test.drop(columns=["id"])
df_te.head()


edu_map = {
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}

income_map = {
    'Low': 0,
    'Lower-Middle': 1,
    'Middle': 2,
    'Upper-Middle': 3,
    'High': 4
}

smoke_map = {
    'Never': 0,
    'Former': 1,
    'Current': 2
}

df_tr['education_level'] = df_tr['education_level'].map(edu_map)
df_tr['income_level'] = df_tr['income_level'].map(income_map)
df_tr['smoking_status'] = df_tr['smoking_status'].map(smoke_map)

df_te['education_level'] = df_te['education_level'].map(edu_map)
df_te['income_level'] = df_te['income_level'].map(income_map)
df_te['smoking_status'] = df_te['smoking_status'].map(smoke_map)


from sklearn.preprocessing import OneHotEncoder

nominal_cols = ['gender', 'ethnicity', 'employment_status']

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ohe.fit(df_tr[nominal_cols])

X_tr_ohe = ohe.transform(df_tr[nominal_cols])
X_te_ohe = ohe.transform(df_te[nominal_cols])

ohe_cols = ohe.get_feature_names_out(nominal_cols)
df_tr_ohe = pd.DataFrame(X_tr_ohe, columns=ohe_cols, index=df_tr.index)
df_te_ohe = pd.DataFrame(X_te_ohe, columns=ohe_cols, index=df_te.index)

df_tr = df_tr.drop(columns=nominal_cols)
df_te = df_te.drop(columns=nominal_cols)

df_tr = pd.concat([df_tr, df_tr_ohe], axis=1)
df_te = pd.concat([df_te, df_te_ohe], axis=1)

print("Train and Test OHE applied and aligned.")


print(df_tr.shape)
print(df_te.shape)


df_tr['diagnosed_diabetes'] = df_tr['diagnosed_diabetes'].astype(int)


X = df_tr.drop(columns=["diagnosed_diabetes"])
y = df_tr["diagnosed_diabetes"]


import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
sampler = optuna.samplers.TPESampler(seed=SEED)

X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'n_estimators': 20000,
        'random_state': SEED,
        'n_jobs': -1,
        'bagging_freq': 1,
        # --- LIGHTGBM GRID ---
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 130), # Key param for LGBM
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 7.0)
    }

    oof_preds = np.zeros(len(X_dev))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for train_idx, val_idx in kf.split(X_dev, y_dev):
        X_tr, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        
        # Early stopping via callbacks
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    return roc_auc_score(y_dev, oof_preds)

print("Starting LightGBM Tuning...")
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50)

print("\nBest Params Found:")
print(study.best_params)

# Sanity Check (Holdout)
print("\nChecking on Holdout...")
final_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'device': 'cpu',
    'n_estimators': 20000,
    'random_state': SEED,
    'n_jobs': -1,
    'bagging_freq': 1,
    **study.best_params
}

model_check = lgb.LGBMClassifier(**final_params)
model_check.fit(
    X_dev, y_dev,
    eval_set=[(X_holdout, y_holdout)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
)
holdout_auc = roc_auc_score(y_holdout, model_check.predict_proba(X_holdout)[:, 1])
print(f"Holdout AUC: {holdout_auc:.5f}")


print("\nGenerating Final OOFs (for Stacking) and Test Predictions...")

oof_preds_full = np.zeros(len(X))
test_preds_full = np.zeros(len(df_te))

kf_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf_full.split(X, y)):
    print(f"Fold {fold+1}...")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
    )

    oof_preds_full[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds_full += model.predict_proba(df_te)[:, 1] / 5

print(f"\nFinal Full OOF AUC: {roc_auc_score(y, oof_preds_full):.5f}")

# Saving Files
df_oof_save = pd.DataFrame({
    'id': df_train['id'],
    'diagnosed_diabetes': y,
    'lgbm_pred': oof_preds_full # Unique column name for stacking
})
df_oof_save.to_csv('oof_lgbm.csv', index=False)

df_test_save = pd.DataFrame({
    'id': df_test['id'],
    'lgbm_pred': test_preds_full
})
df_test_save.to_csv('test_lgbm.csv', index=False)

print("LightGBM Files saved!")


subm = pd.read_csv("/kaggle/working/test_lgbm.csv")
subm.head()


subm = subm.rename(columns={'lgbm_pred': 'diagnosed_diabetes'})
subm.to_csv('submission.csv', index=False)

