# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


submit_ids = test_df['id']


def feature_engineering(df):
    df = df.copy()
    
    # --- A. Basic business characteristics ---
    df['estimated_total_repayment'] = df['loan_amount'] * (1 + df['interest_rate'] / 100)
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['income_minus_loan'] = df['annual_income'] - df['loan_amount']
    
    # --- B.Log Transform ---
    df['log_annual_income'] = np.log1p(df['annual_income'])
    df['log_loan_amount'] = np.log1p(df['loan_amount'])
    
    # --- C. Risk Mapping ---
    # Patterns discovered through data exploration
    # Retired(99%) > Employed(89%) > Student(26%) > Unemployed(7%)
    risk_map = {
        'Retired': 0,       
        'Employed': 1,
        'Self-employed': 1,
        'Student': 2,
        'Unemployed': 3
    }
    # Map and fill in unknown values
    if 'employment_status' in df.columns:
        df['job_risk_level'] = df['employment_status'].map(risk_map).fillna(2)
        df['risk_income_interaction'] = df['job_risk_level'] / (df['log_annual_income'] + 1)

    # --- D. GroupBy Aggregation ---
    if 'grade_subgrade' in df.columns:
        df['income_div_grade_mean'] = df['annual_income'] / df.groupby('grade_subgrade')['annual_income'].transform('mean')
        df['loan_div_grade_mean'] = df['loan_amount'] / df.groupby('grade_subgrade')['loan_amount'].transform('mean')
        df['interest_div_grade_mean'] = df['interest_rate'] / df.groupby('grade_subgrade')['interest_rate'].transform('mean')
    
    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)


target_col = 'loan_paid_back'
drop_cols = ['id'] 

cat_cols = [
    'gender', 'marital_status', 'education_level', 
    'employment_status', 'loan_purpose', 'grade_subgrade'
]

# Convert to Category type
for col in cat_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype('category')
    if col in test_df.columns:
        test_df[col] = test_df[col].astype('category')

feature_cols = [c for c in train_df.columns if c not in drop_cols + [target_col]]
X = train_df[feature_cols]
y = train_df[target_col]
X_test = test_df[feature_cols]


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def train_xgb_model(X_data, y_data, X_test_data, model_name="Model", drop_features=[]):
    X_train_use = X_data.drop(columns=drop_features, errors='ignore')
    X_test_use = X_test_data.drop(columns=drop_features, errors='ignore')
    
    oof_preds = np.zeros(X_train_use.shape[0])
    test_preds = np.zeros(X_test_use.shape[0])
    cv_scores = []
    
    print(f"\n========== train {model_name} ==========")
    if drop_features:
        print(f"drop features: {drop_features}")
        
    max_depth = 8 if drop_features else 6
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_use, y_data)):
        X_tr, y_tr = X_train_use.iloc[train_idx], y_data.iloc[train_idx]
        X_val, y_val = X_train_use.iloc[val_idx], y_data.iloc[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=max_depth, 
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            enable_categorical=True,
            tree_method='hist',
            eval_metric='auc',
            early_stopping_rounds=100,
            n_jobs=-1,
            random_state=42 + fold
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        test_preds += model.predict_proba(X_test_use)[:, 1] / 5
        
        score = roc_auc_score(y_val, val_pred)
        cv_scores.append(score)
        print(f"Fold {fold+1} AUC: {score:.5f}")
        
    print(f"{model_name} Overall CV AUC: {np.mean(cv_scores):.5f}")
    return oof_preds, test_preds


# --- Model A: Full feature model ---
oof_full, preds_full = train_xgb_model(
    X, y, X_test, 
    model_name="Model A (Full Features)"
)


# --- Model B: Blind Model ---
drop_list = ['employment_status', 'job_risk_level', 'risk_income_interaction']
oof_blind, preds_blind = train_xgb_model(
    X, y, X_test, 
    model_name="Model B (Blind / No Job Info)",
    drop_features=[c for c in drop_list if c in X.columns]
)


print("========== Start searching for the optimal fusion weight ==========")

best_score = 0
best_weight = 0
scores = []
weights = np.linspace(0, 1, 101)

for w in weights:
    # Mixed validation set prediction results
    blended_oof = w * oof_full + (1 - w) * oof_blind
    score = roc_auc_score(y, blended_oof)
    scores.append(score)
    
    if score > best_score:
        best_score = score
        best_weight = w


# Plot Weight Search Curve
plt.figure(figsize=(10, 5))
plt.plot(weights, scores)
plt.axvline(x=best_weight, color='r', linestyle='--', label=f'Best Weight: {best_weight}')
plt.title(f'Blending Search (Best AUC: {best_score:.5f})')
plt.xlabel('Weight for Model A (Full)')
plt.ylabel('AUC Score')
plt.legend()
plt.show()

print(f"Best weight (Model A - Full): {best_weight}")
print(f"Best weight (Model B - Blind): {1 - best_weight}")
print(f"CV AUC: {best_score:.5f}")
print(f"Compared to Model A, improved: {best_score - roc_auc_score(y, oof_full):.6f}")


final_test_preds = best_weight * preds_full + (1 - best_weight) * preds_blind

submission = pd.DataFrame({
    'id': submit_ids,
    'loan_paid_back': final_test_preds
})


submission.to_csv('submission.csv', index=False)

