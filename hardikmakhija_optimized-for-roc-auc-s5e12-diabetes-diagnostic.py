## FINAL FAST SPRINT (UNDER 3 MINS)
## OPTIMIZED FOR ROC AUC | SILENT & STABLE

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import warnings
import os

warnings.filterwarnings('ignore')

# 1. DATA LOADING
PATH = "/kaggle/input/playground-series-s5e12"
train = pd.read_csv(f"{PATH}/train.csv")
test = pd.read_csv(f"{PATH}/test.csv")
y = train['diagnosed_diabetes']
test_ids = test['id']

# 2. FAST FEATURE ENGINEERING
def engineer(df):
    df['bmi_age'] = df['bmi'] * df['age']
    df['mean_bp'] = (df['systolic_bp'] + df['diastolic_bp']) / 2
    # Simple Label Encoding for speed
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]
    return df

train_proc = engineer(train).drop(['id', 'diagnosed_diabetes'], axis=1)
test_proc = engineer(test).drop(['id'], axis=1)

# 3. LIGHTWEIGHT ENSEMBLE (3-FOLD FOR SPEED)
# ------------------------------------------------------------------------------
# We use 3 folds instead of 5 to ensure you get your submission in NOW.
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
lgb_preds, cat_preds, xgb_preds = [], [], []

print("ðŸš€ Final Fast Training Started...")

for fold, (t_idx, v_idx) in enumerate(skf.split(train_proc, y)):
    X_tr, X_va = train_proc.iloc[t_idx], train_proc.iloc[v_idx]
    y_tr, y_va = y.iloc[t_idx], y.iloc[v_idx]
    
    # Fast LGBM
    m1 = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, verbose=-1)
    m1.fit(X_tr, y_tr)
    lgb_preds.append(m1.predict_proba(test_proc)[:, 1])
    
    # Fast CatBoost
    m2 = CatBoostClassifier(iterations=800, learning_rate=0.05, verbose=0)
    m2.fit(X_tr, y_tr)
    cat_preds.append(m2.predict_proba(test_proc)[:, 1])
    
    # Fast XGBoost
    m3 = XGBClassifier(n_estimators=800, learning_rate=0.05, tree_method='hist')
    m3.fit(X_tr, y_tr)
    xgb_preds.append(m3.predict_proba(test_proc)[:, 1])
    
    print(f"âœ… Progress: {int((fold+1)/3*100)}%")

# 4. FINAL RANK BLENDING
lgb_final = rankdata(np.mean(lgb_preds, axis=0))
cat_final = rankdata(np.mean(cat_preds, axis=0))
xgb_final = rankdata(np.mean(xgb_preds, axis=0))

# Weighted Blend
final_rank = (lgb_final * 0.5) + (cat_final * 0.3) + (xgb_final * 0.2)
final_predictions = final_rank / final_rank.max()

# 5. SUBMISSION
pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': final_predictions}).to_csv('submission.csv', index=False)
print("ðŸŽ¯ FINISHED! 'submission.csv' is ready in the Output folder.")

