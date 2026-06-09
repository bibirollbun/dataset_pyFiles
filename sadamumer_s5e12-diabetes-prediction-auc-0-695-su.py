import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Config
SEED = 42
N_FOLDS = 10
np.random.seed(SEED)


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target rate: {train['diagnosed_diabetes'].mean():.4f}")


def add_features(df):
    df = df.copy()
    
    # Basic interactions & ratios
    df['bmi_x_waist'] = df['bmi'] * df['waist_to_hip_ratio']
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-3)
    df['activity_per_age'] = df['physical_activity_minutes_per_week'] / (df['age'] + 1)
    df['screen_sleep'] = df['screen_time_hours_per_day'] / (df['sleep_hours_per_day'] + 1)
    df['cholesterol_ratio'] = df['hdl_cholesterol'] / (df['ldl_cholesterol'] + 1e-3)
    
    # Simple risk count
    risk_cols = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
    df['risk_count'] = df[risk_cols].sum(axis=1)
    
    # Age & BMI groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 35, 50, 65, 100], labels=[0,1,2,3])
    df['bmi_group'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0,1,2,3])
    
    return df

train = add_features(train)
test = add_features(test)


cat_features = ['gender', 'ethnicity', 'education_level', 'income_level', 
                'smoking_status', 'employment_status', 'age_group', 'bmi_group']

for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Final feature list
features = [c for c in train.columns if c not in ['id', 'diagnosed_diabetes']]
print(f"Using {len(features)} features")


lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 100,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 30,
    'random_state': SEED,
    'verbose': -1,
    'early_stopping_rounds':200,
    'verbose': 0
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.03,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'random_state': SEED,
    'tree_method': 'hist',
    'enable_categorical': False,
    'early_stopping_rounds':200,    
    'verbose': 0
}

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.05,
    'iterations': 2000,
    'depth': 8,
    'random_seed': SEED,
    'verbose': 0
}


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = {}
test_preds = {}

models = {
    'lgb': (lgb.LGBMClassifier(**lgb_params), 3000),
    'xgb': (xgb.XGBClassifier(**xgb_params), 3000),
    'cat': (CatBoostClassifier(**cat_params), None)
}

for name, (clf, n_est) in models.items():
    print(f"\nTraining {name.upper()}...")
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train[features], train['diagnosed_diabetes'])):
        X_tr = train.iloc[tr_idx][features]
        y_tr = train.iloc[tr_idx]['diagnosed_diabetes']
        X_val = train.iloc[val_idx][features]
        y_val = train.iloc[val_idx]['diagnosed_diabetes']
        
        if name == 'lgb':
            clf.fit(X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    )
            val_pred = clf.predict_proba(X_val)[:, 1]
            test_pred += clf.predict_proba(test[features])[:, 1] / N_FOLDS
            
        elif name == 'xgb':
            clf.fit(X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    )
            val_pred = clf.predict_proba(X_val)[:, 1]
            test_pred += clf.predict_proba(test[features])[:, 1] / N_FOLDS
            
        elif name == 'cat':
            clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
            val_pred = clf.predict_proba(X_val)[:, 1]
            test_pred += clf.predict_proba(test[features])[:, 1] / N_FOLDS
        
        oof[val_idx] = val_pred
        print(f"  Fold {fold+1}: AUC = {roc_auc_score(y_val, val_pred):.5f}")
    
    auc = roc_auc_score(train['diagnosed_diabetes'], oof)
    print(f"{name.upper()} OOF AUC: {auc:.5f}")
    
    oof_preds[name] = oof
    test_preds[name] = test_pred

# Simple average ensemble
final_oof = np.mean(list(oof_preds.values()), axis=0)
final_test = np.mean(list(test_preds.values()), axis=0)

print(f"\nENSEMBLE OOF AUC: {roc_auc_score(train['diagnosed_diabetes'], final_oof):.5f}")


submission['diagnosed_diabetes'] = final_test
submission.to_csv('submission.csv', index=False)

print("\nSubmission preview:")
print(submission.head(10))
print(f"\nSubmission saved! Mean prediction: {final_test.mean():.4f}")










