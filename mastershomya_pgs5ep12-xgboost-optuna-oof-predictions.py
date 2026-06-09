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


df_train.shape


df_train.describe().T


df_train.isna().sum()


df_tr = df_train.drop(columns=["id"])


numeric_df = df_tr.select_dtypes(include=['number'])
categorical_df = df_tr.select_dtypes(include=['object'])


numeric_df.head()


numeric_df.nunique()


print(numeric_df["alcohol_consumption_per_week"].value_counts())
print(numeric_df["family_history_diabetes"].value_counts())
print(numeric_df["hypertension_history"].value_counts())
print(numeric_df["cardiovascular_history"].value_counts())
print(numeric_df["diagnosed_diabetes"].value_counts())


categorical_df.head()


categorical_df.nunique()


print(categorical_df["gender"].value_counts())
print(categorical_df["ethnicity"].value_counts())
print(categorical_df["education_level"].value_counts())
print(categorical_df["income_level"].value_counts())
print(categorical_df["smoking_status"].value_counts())
print(categorical_df["employment_status"].value_counts())


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

# 1. Fit encoder on TRAIN
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ohe.fit(df_tr[nominal_cols])

# 2. Transform train and test
X_tr_ohe = ohe.transform(df_tr[nominal_cols])
X_te_ohe = ohe.transform(df_te[nominal_cols])

# 3. Convert to DataFrame
ohe_cols = ohe.get_feature_names_out(nominal_cols)
df_tr_ohe = pd.DataFrame(X_tr_ohe, columns=ohe_cols, index=df_tr.index)
df_te_ohe = pd.DataFrame(X_te_ohe, columns=ohe_cols, index=df_te.index)

# 4. Drop original nominal columns
df_tr = df_tr.drop(columns=nominal_cols)
df_te = df_te.drop(columns=nominal_cols)

# 5. Add new OHE columns
df_tr = pd.concat([df_tr, df_tr_ohe], axis=1)
df_te = pd.concat([df_te, df_te_ohe], axis=1)

print("Train and Test OHE applied and aligned.")


print(df_tr.shape)
print(df_te.shape)


df_tr['diagnosed_diabetes'] = df_tr['diagnosed_diabetes'].astype(int)


df_tr['diagnosed_diabetes']


X = df_tr.drop(columns=["diagnosed_diabetes"])
y = df_tr["diagnosed_diabetes"]


import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

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
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',
        'n_jobs': -1,
        'random_state': SEED,
        'n_estimators': 30000,
        'verbosity': 0,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0) 
    }
    
    oof_preds = np.zeros(len(X_dev))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for train_idx, val_idx in kf.split(X_dev, y_dev):
        X_tr, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=300, verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    return roc_auc_score(y_dev, oof_preds)

print("Starting Tuning...")
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100) 

print("\nBest Params Found:")
print(study.best_params)


print("\nChecking on Holdout...")
final_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'n_jobs': -1,
    'random_state': SEED,
    'n_estimators': 30000,
    **study.best_params
}

model_check = xgb.XGBClassifier(**final_params)
model_check.fit(X_dev, y_dev, eval_set=[(X_holdout, y_holdout)], early_stopping_rounds=300, verbose=False)
holdout_auc = roc_auc_score(y_holdout, model_check.predict_proba(X_holdout)[:, 1])
print(f"Holdout AUC: {holdout_auc:.5f}")

# ==========================================
# 4. FINAL OOF GENERATION (The "Pro" Workflow)
# ==========================================
print("\nGenerating Final OOFs and Test Predictions...")

# Containers for the results
oof_preds_full = np.zeros(len(X)) 
test_preds_full = np.zeros(len(df_test)) 

# We use the FULL X and y now
kf_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf_full.split(X, y)):
    print(f"Fold {fold+1}...")
    
    # 1. Train on 4 folds (80%)
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=300, verbose=False)
    
    # 2. Predict on the LEFT OUT fold (20%) -> This is the OOF for these rows
    oof_preds_full[val_idx] = model.predict_proba(X_val)[:, 1]
    
    # 3. Predict on the TEST set
    # Why? Because this model is trained on specific 80% of data.
    # Averaging 5 of these is better than training 1 model on 100%.
    test_preds_full += model.predict_proba(df_te)[:, 1] / 5

# Report Score
print(f"\nFinal Full OOF AUC: {roc_auc_score(y, oof_preds_full):.5f}")

# ==========================================
# 5. Saving (Using df_train['id'])
# ==========================================
# OOF File (For Training Stacker)
df_oof_save = pd.DataFrame({
    'id': df_train['id'],          # Use the original IDs
    'diagnosed_diabetes': y,       # Keep target for safety checks
    'xgboost_pred': oof_preds_full # The new feature
})
df_oof_save.to_csv('oof_xgboost.csv', index=False)

# Test File (For Stacker Submission)
df_test_save = pd.DataFrame({
    'id': df_test['id'],           # Assuming df_test still has the ID column
    'xgboost_pred': test_preds_full
})
df_test_save.to_csv('test_xgboost.csv', index=False)

print("Files saved!")


subm = pd.read_csv("/kaggle/working/test_xgboost.csv")
subm.head()


subm = subm.rename(columns={'xgboost_pred': 'diagnosed_diabetes'})
subm.to_csv('submission.csv', index=False)

