import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ---------------------------
# 1. Reproducibility & Configurations
# ---------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)

class Config:
    train_path = "/kaggle/input/playground-series-s5e3/train.csv"
    test_path  = "/kaggle/input/playground-series-s5e3/test.csv"
    sub_path   = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

# ---------------------------
# 2. Data Loading and Feature Engineering
# ---------------------------
train = pd.read_csv(Config.train_path, index_col='id')
test = pd.read_csv(Config.test_path, index_col='id')

# Fill missing values
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

# Combine for joint feature engineering
combined = pd.concat([train, test], axis=0)

# One-hot encode categorical features (drop_first to avoid collinearity)
combined = pd.get_dummies(combined, columns=['winddirection','day'], drop_first=True)

# Create lag features for the 'cloud' column
for lag in [1, 2, 3]:
    combined[f'cloud_shift_{lag}'] = combined['cloud'].shift(lag)
combined.fillna(-1, inplace=True)

# Split back into train and test sets
train_fe = combined.iloc[:len(train)].copy()
test_fe  = combined.iloc[len(train):].copy()
test_fe.drop(['rainfall'], axis=1, inplace=True)

# ---------------------------
# 3. Data Normalization
# ---------------------------
num_feats = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
             'humidity', 'cloud', 'sunshine', 'windspeed']
scaler = StandardScaler()
scaler.fit(train_fe[num_feats])
train_fe[num_feats] = scaler.transform(train_fe[num_feats])
test_fe[num_feats]  = scaler.transform(test_fe[num_feats])

# ---------------------------
# 4. Prepare Data for Modeling
# ---------------------------
X = train_fe.drop('rainfall', axis=1)
y = train_fe['rainfall']

# ---------------------------
# 5. Define Model Parameters for Tree-Based Models
# ---------------------------
lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': seed,
    'verbose': -1
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'use_label_encoder': False,
    'random_state': seed
}

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.01,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': seed,
    'early_stopping_rounds': 50,
    'verbose': 0
}

# ---------------------------
# 6. Stacking Ensemble via Out-of-Fold Predictions
# ---------------------------
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
meta_train = np.zeros((len(X), 3))   # One column per base model: [LGB, XGB, CatBoost]
meta_test = np.zeros((len(test_fe), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model_lgb = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    lgb_val_pred = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    lgb_test_pred = model_lgb.predict(test_fe, num_iteration=model_lgb.best_iteration)
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    xgb_val_pred = model_xgb.predict_proba(X_val)[:, 1]
    xgb_test_pred = model_xgb.predict_proba(test_fe)[:, 1]
    
    # CatBoost
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    cat_val_pred = model_cat.predict_proba(X_val)[:, 1]
    cat_test_pred = model_cat.predict_proba(test_fe)[:, 1]
    
    # Store out-of-fold predictions for meta training
    meta_train[val_idx, 0] = lgb_val_pred
    meta_train[val_idx, 1] = xgb_val_pred
    meta_train[val_idx, 2] = cat_val_pred
    
    # Accumulate test predictions
    meta_test[:, 0] += lgb_test_pred / n_folds
    meta_test[:, 1] += xgb_test_pred / n_folds
    meta_test[:, 2] += cat_test_pred / n_folds
    
    fold_auc = roc_auc_score(y_val, (lgb_val_pred + xgb_val_pred + cat_val_pred) / 3)
    print(f"Fold {fold+1} Average AUC: {fold_auc:.4f}")

# Train meta-model on the meta features
meta_model = LogisticRegression(random_state=seed, max_iter=1000)
meta_model.fit(meta_train, y)
meta_train_pred = meta_model.predict_proba(meta_train)[:, 1]
meta_auc = roc_auc_score(y, meta_train_pred)
print(f"\nMeta-model CV AUC: {meta_auc:.4f}")

# Final test set prediction from meta-model
meta_test_pred = meta_model.predict_proba(meta_test)[:, 1]

# ---------------------------
# 7. Final Submission
# ---------------------------
sub = pd.read_csv(Config.sub_path)
sub['rainfall'] = meta_test_pred
sub.fillna(sub.mean(), inplace=True)
sub.to_csv('submission.csv', index=False)
print("\nSubmission file prepared.")


