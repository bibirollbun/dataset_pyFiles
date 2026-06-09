
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(f"Train: {train.shape} | Test: {test.shape}")
print(f"Target rate: {train['loan_paid_back'].mean():.5f}")

# ----------------------------- 2. Target & ID -----------------------------
y = train['loan_paid_back'].copy()

if 'id' in train.columns:
    train = train.drop('id', axis=1)
if 'id' in test.columns:
    test_id = test['id']
    test = test.drop('id', axis=1)
else:
    test_id = None

train = train.drop('loan_paid_back', axis=1)




# ----------------------------- 3. Feature Engineering -----------------------------
def create_features(df):
    df = df.copy()
    # Powerful interactions
    df['income_x_score']      = df['annual_income'] * df['credit_score']
    df['loan_x_rate']         = df['loan_amount'] * df['interest_rate']
    df['rate_per_income']     = df['interest_rate'] / (df['annual_income'] + 1e6)
    df['loan_per_income']     = df['loan_amount'] / (df['annual_income'] + 1e6)
    df['dti_x_rate']          = df['debt_to_income_ratio'] * df['interest_rate']
    df['total_risk']          = df['loan_amount'] * df['interest_rate'] * df['debt_to_income_ratio']
    df['income_per_loan']     = df['annual_income'] / (df['loan_amount'] + 1e6)
    df['score_per_dti']       = df['credit_score'] / (df['debt_to_income_ratio'] + 1e-6)
    
    # Log transforms
    df['log_income'] = np.log1p(df['annual_income'])
    df['log_loan']   = np.log1p(df['loan_amount'])
    
    return df

train = create_features(train)
test  = create_features(test)



# ----------------------------- 4. Safe Categorical Encoding -----------------------------
cat_features = ['gender', 'marital_status', 'education_level',
                'employment_status', 'loan_purpose', 'grade_subgrade']

for col in cat_features:
    if col in train.columns:
        le = LabelEncoder()
        # Fit on train + test to avoid unseen categories
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

# Fill missing (should be none, but safe)
train = train.fillna(-999)
test  = test.fillna(-999)

X = train.values
X_test = test.values



# ----------------------------- 5. 5-Fold Ensemble with AUC -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))

print("\nStarting training...\n")
print(f"{'Fold':<6} {'LGB':<10} {'XGB':<10} {'CAT':<10} {'Blend':<10}")
print("-" * 50)

for fold, (idx_train, idx_valid) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}", end=" ")

    X_train, y_train = X[idx_train], y[idx_train]
    X_valid, y_valid = X[idx_valid], y[idx_valid]
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.025,
        num_leaves=128,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42+fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
    p_lgb = lgb_model.predict_proba(X_valid)[:, 1]
    auc_lgb = roc_auc_score(y_valid, p_lgb)
    oof_lgb[idx_valid] = p_lgb
    pred_lgb += lgb_model.predict_proba(X_test)[:, 1] / 5
   # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42+fold,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds=300,
                  verbose=False)
    p_xgb = xgb_model.predict_proba(X_valid)[:, 1]
    auc_xgb = roc_auc_score(y_valid, p_xgb)
    oof_xgb[idx_valid] = p_xgb
    pred_xgb += xgb_model.predict_proba(X_test)[:, 1] / 5
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.03,
        depth=8,
        random_seed=42+fold,
        verbose=0,
        early_stopping_rounds=300
    )
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    p_cat = cat_model.predict_proba(X_valid)[:, 1]
    auc_cat = roc_auc_score(y_valid, p_cat)
    oof_cat[idx_valid] = p_cat
    pred_cat += cat_model.predict_proba(X_test)[:, 1] / 5

    # Blend
    blend = 0.60 * p_lgb + 0.25 * p_xgb + 0.15 * p_cat
    auc_blend = roc_auc_score(y_valid, blend)

    print(f"→ LGB:{auc_lgb:.5f}  XGB:{auc_xgb:.5f}  CAT:{auc_cat:.5f}  Blend:{auc_blend:.5f}")




# ----------------------------- 6. Final Result -----------------------------
final_oof = 0.60 * oof_lgb + 0.25 * oof_xgb + 0.15 * oof_cat
final_cv = roc_auc_score(y, final_oof)

print("\n" + "="*60)
print(f"     FINAL 5-FOLD CV AUC = {final_cv:.6f}")
print(f"     EXPECTED PUBLIC LB  ≈ {final_cv + 0.0018:.6f} – {final_cv + 0.003:.6f}")
print("="*60)


#----------------------------- 7. Save submission -----------------------------
final_pred = 0.60 * pred_lgb + 0.25 * pred_xgb + 0.15 * pred_cat
submission['loan_paid_back'] = final_pred
submission.to_csv('submission.csv', index=False)

print("\nsubmission.csv is saved ")

