import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. ROBUST DATA LOADING
# ==========================================
print("Loading data...")
df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# Search for the original dataset
df_orig = None
possible_paths = [
    "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv",
    "/kaggle/input/loan-approval-classification-dataset/loan_approval_dataset.csv"
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"✓ Original dataset found: {path}")
        df_orig = pd.read_csv(path)
        break

if df_orig is not None:
    df_orig.columns = df_orig.columns.str.strip()
    if 'loan_status' in df_orig.columns:
        df_orig.rename(columns={'loan_status': 'loan_paid_back'}, inplace=True)
    
    # Align and Combine
    common_cols = [c for c in df.columns if c in df_orig.columns]
    df_orig = df_orig[common_cols]
    df = pd.concat([df, df_orig], axis=0, ignore_index=True)
    df.drop_duplicates(inplace=True)
    print(f"✓ Combined Data Shape: {df.shape}")

target = 'loan_paid_back'
df = df.reset_index(drop=True)

# ==========================================
# 2. ELITE FEATURE ENGINEERING (FIXED)
# ==========================================
def create_elite_features(train_df, test_df, target):
    print("Generating Elite Features...")
    train = train_df.copy()
    test = test_df.copy()
    
    # A. Financial Ratios
    for df_ in [train, test]:
        # Income leverage
        df_['loan_to_income'] = df_['loan_amount'] / (df_['annual_income'] + 1)
        
        # Monthly load estimation
        df_['monthly_load'] = df_['loan_amount'] * 0.04 # Proxy for monthly payment
        df_['payment_to_income'] = df_['monthly_load'] / (df_['annual_income']/12 + 1)
        
        # Log transforms
        for col in ['annual_income', 'loan_amount']:
            df_[f'log_{col}'] = np.log1p(df_[col])
            
        # Grade Parsing
        if 'grade_subgrade' in df_.columns:
            df_['grade'] = df_['grade_subgrade'].astype(str).str[0]
            df_['subgrade_num'] = pd.to_numeric(df_['grade_subgrade'].astype(str).str[1:], errors='coerce').fillna(0)

    # B. Binning (Quantiles)
    for col in ['annual_income', 'loan_amount', 'loan_to_income']:
        for q in [10]:
            try:
                train[f'{col}_q{q}'], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates='drop')
                test[f'{col}_q{q}'] = pd.cut(test[col], bins=bins, labels=False).fillna(0)
            except:
                pass

    # C. Target Encoding (OOF)
    te_cols = ['grade', 'loan_purpose', 'employment_status']
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in te_cols:
        if col in train.columns:
            train[f'{col}_te'] = 0.0
            test[f'{col}_te'] = 0.0
            
            for tr_idx, val_idx in kf.split(train, train[target]):
                means = train.iloc[tr_idx].groupby(col)[target].mean()
                train.loc[val_idx, f'{col}_te'] = train.loc[val_idx, col].map(means)
            
            # Fill NaNs
            global_mean = train[target].mean()
            train[f'{col}_te'].fillna(global_mean, inplace=True)
            
            # Test mapping
            means_full = train.groupby(col)[target].mean()
            test[f'{col}_te'] = test[col].map(means_full).fillna(global_mean)

    # D. Label Encoding (THE FIX: Assign first, then cast)
    cat_cols = [c for c in train.columns if train[c].dtype == 'object' or c == 'grade']
    
    for col in cat_cols:
        le = LabelEncoder()
        # 1. Fit
        full_seq = pd.concat([train[col].astype(str), test[col].astype(str)])
        le.fit(full_seq)
        
        # 2. Transform (Returns Int Array)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        
        # 3. Cast to Category (Pandas Operation)
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    return train, test

# Apply Engineering
df, df_test = create_elite_features(df, df_test, target)

# Split back
train_len = len(df)
X = df.drop(columns=[target, 'id'], errors='ignore')
y = df[target]
X_test = df_test.drop(columns=['id'], errors='ignore')

print(f"Final Feature Count: {X.shape[1]}")

# ==========================================
# 3. STACKING LOOP
# ==========================================
N_FOLDS = 10
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = {}
test_preds = {}

def train_cv(name, model, X, y, X_test):
    oof = np.zeros(len(X))
    test_p = np.zeros(len(X_test))
    scores = []
    
    print(f"\nTraining {name}...", end=" ")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        if name == 'cb':
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=100)
        elif name == 'lgb':
            callbacks = [lgb.early_stopping(100, verbose=False)]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=callbacks)
        else: # XGB
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=100)
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_p += model.predict_proba(X_test)[:, 1] / N_FOLDS
        
    score = roc_auc_score(y, oof)
    print(f" OOF AUC: {score:.5f}")
    return oof, test_p

# 1. XGBoost
xgb_params = {
    'n_estimators': 3000, 'learning_rate': 0.02, 'max_depth': 8, 
    'subsample': 0.8, 'colsample_bytree': 0.6, 
    'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cuda',
    'eval_metric': 'auc', 'enable_categorical': True
}
oof_preds['xgb'], test_preds['xgb'] = train_cv('xgb', XGBClassifier(**xgb_params), X, y, X_test)

# 2. LightGBM
lgb_params = {
    'n_estimators': 3000, 'learning_rate': 0.02, 'num_leaves': 45,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'binary',
    'metric': 'auc', 'verbosity': -1
}
oof_preds['lgb'], test_preds['lgb'] = train_cv('lgb', lgb.LGBMClassifier(**lgb_params), X, y, X_test)

# 3. CatBoost
cb_params = {
    'iterations': 2000, 'learning_rate': 0.03, 'depth': 6,
    'l2_leaf_reg': 5, 'loss_function': 'Logloss', 'eval_metric': 'AUC',
    'cat_features': list(X.select_dtypes(include=['category']).columns),
    'task_type': 'CPU', 'thread_count': 4
}
oof_preds['cb'], test_preds['cb'] = train_cv('cb', cb.CatBoostClassifier(**cb_params), X, y, X_test)

# ==========================================
# 4. META LEARNER
# ==========================================
print("\nTraining Meta-Learner...")
X_meta = pd.DataFrame(oof_preds)
X_test_meta = pd.DataFrame(test_preds)

meta_model = LogisticRegression(C=1.0) 
meta_model.fit(X_meta, y)

print("Stacking Weights:", dict(zip(X_meta.columns, meta_model.coef_[0])))

final_preds = meta_model.predict_proba(X_test_meta)[:, 1]

submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
submission[target] = final_preds
submission.to_csv("submission_hybrid_fixed.csv", index=False)
print("✓ Saved submission_hybrid_fixed.csv")




