from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from category_encoders import TargetEncoder

warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
print(f"Train: {train.shape} | Test: {test.shape}")


def engineer_v3(df):
    df = df.copy()
    df['log_income'] = np.log1p(df['annual_income'])
    df['log_loan']   = np.log1p(df['loan_amount'])
    df['dti'] = df['debt_to_income_ratio']
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['income_per_loan'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['rate_dti'] = df['interest_rate'] * df['dti']
    df['dti_bin'] = pd.cut(df['dti'], bins=[0, 15, 25, 35, 100], labels=[0,1,2,3]).astype(int)
    df['credit_bin'] = pd.cut(df['credit_score'], bins=[0, 650, 720, 780, 900], labels=[0,1,2,3]).astype(int)
    df['rate_bin'] = pd.cut(df['interest_rate'], bins=[0, 8, 12, 16, 25], labels=[0,1,2,3]).astype(int)
    df['high_risk'] = ((df['dti'] > 30) & (df['interest_rate'] > 15) & (df['credit_score'] < 680)).astype(int)
    df['safe_profile'] = ((df['dti'] < 20) & (df['interest_rate'] < 10) & (df['credit_score'] > 750)).astype(int)
    grades = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5',
              'D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5']
    df['grade_num'] = df['grade_subgrade'].map({g: i+1 for i, g in enumerate(grades)})
    return df

train = engineer_v3(train)
test  = engineer_v3(test)


te = TargetEncoder(cols=['grade_subgrade'], smoothing=50)
train['grade_te'] = te.fit_transform(train['grade_subgrade'], train['loan_paid_back'])
test['grade_te']  = te.transform(test['grade_subgrade'])


num_cols = ['dti', 'credit_score', 'interest_rate', 'loan_amount', 'log_income', 'log_loan',
            'loan_to_income', 'income_per_loan', 'rate_dti', 'dti_bin', 'credit_bin', 'rate_bin',
            'grade_num', 'grade_te', 'high_risk', 'safe_profile']
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']

X = pd.get_dummies(train[num_cols + cat_cols], columns=cat_cols, drop_first=True)
y = train['loan_paid_back']
X_test = pd.get_dummies(test[num_cols + cat_cols], columns=cat_cols, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)


n_folds = 10
seeds = [42] 
lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
val_aucs = []

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

pbar = tqdm(enumerate(skf.split(X, y)), total=n_folds, desc="Folds")

for fold, (train_idx, val_idx) in pbar:
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=8,
        num_leaves=128,
        colsample_bytree=0.7,
        subsample=0.8,
        n_jobs=-1,          
        random_state=42
    )
    lgb.fit(X_train, y_train)
    lgb_preds += lgb.predict_proba(X_test)[:, 1] / n_folds
    
    cat = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        random_seed=42,
        verbose=False
    )
    cat.fit(X_train, y_train)
    cat_preds += cat.predict_proba(X_test)[:, 1] / n_folds
    
    xgb = XGBClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=8,
        colsample_bytree=0.7,
        subsample=0.8,
        n_jobs=-1,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_preds += xgb.predict_proba(X_test)[:, 1] / n_folds
    
    val_pred = (lgb.predict_proba(X_val)[:, 1] + cat.predict_proba(X_val)[:, 1] + xgb.predict_proba(X_val)[:, 1]) / 3
    auc = roc_auc_score(y_val, val_pred)
    val_aucs.append(auc)
    pbar.set_postfix({"AUC": f"{auc:.5f}"})

print(f"\nFINAL CV AUC: {np.mean(val_aucs):.6f}")


final_pred = (lgb_preds * 0.5 + cat_preds * 0.3 + xgb_preds * 0.2)


submission = pd.DataFrame({"id": test["id"], "loan_paid_back": final_pred})
submission.to_csv("submission_optimized.csv", index=False)
print("submission_optimized.csv saved")

