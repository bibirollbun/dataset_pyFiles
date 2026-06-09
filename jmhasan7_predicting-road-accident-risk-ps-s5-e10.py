
# 1. Importing libraries
import os
import gc
import math
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb

# Set seed
RND = 42
np.random.seed(RND)



# 3. Load data (adjust path to your environment)
DATA_DIR = Path("../input/playground-series-s5e10")  # Kaggle path example
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
sub   = pd.read_csv(DATA_DIR / "sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



# 4. Sanity checks
print(train.columns.tolist())
print(train.isna().sum())

# Target distribution
plt.figure(figsize=(8,4))
sns.histplot(train['accident_risk'], bins=100, kde=True)
plt.title("accident_risk distribution")
plt.show()

print("Target stats:", train['accident_risk'].describe())



# 5. EDA - correlation on numeric features
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != "accident_risk" and c not in ["id"]]

corr = train[num_cols + ["accident_risk"]].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", center=0)
plt.title("Feature correlations with accident_risk")
plt.show()

# Missing values
plt.figure(figsize=(8,3))
sns.heatmap(train[num_cols].isna(), cbar=False)
plt.title("Missingness (first 200 rows)")
plt.show()



# 6. Feature engineering
def feature_engineering(df):
    df = df.copy()
    # Example: if 'timestamp' exists, extract hour/day/month
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day']  = df['timestamp'].dt.day
        df['weekday'] = df['timestamp'].dt.weekday
        df['month'] = df['timestamp'].dt.month
        df.drop(columns=['timestamp'], inplace=True)
    # Example: create interaction features for numeric pairs (only small set)
    num_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['id','accident_risk']]
    # Pairwise interactions (careful — keep limited)
    if 'speed_limit' in df.columns and 'road_width' in df.columns:
        df['speed_by_width'] = df['speed_limit'] / (df['road_width'] + 1e-3)
    # Convert categories (example columns)
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    for c in cat_cols:
        df[c] = df[c].astype('category').cat.codes
    return df

train_fe = feature_engineering(train)
test_fe  = feature_engineering(test)

print("After FE train shape:", train_fe.shape)
train_fe.head()



# 7. Aggregations and target encoding - done carefully inside CV
# We'll create some group stats using training data, but in CV we must compute them from CV folds only.
# For a demonstration we create global group stats (safe only for baseline).
group_cols = []
# Example: if 'road_type' exists
if 'road_type' in train_fe.columns:
    grp = train_fe.groupby('road_type')['accident_risk'].agg(['mean','median','count']).reset_index()
    grp.columns = ['road_type','rt_mean_risk','rt_median_risk','rt_count']
    train_fe = train_fe.merge(grp, on='road_type', how='left')
    test_fe  = test_fe.merge(grp, on='road_type', how='left')

train_fe.fillna(-999, inplace=True)
test_fe.fillna(-999, inplace=True)



target = "accident_risk"
drop_cols = ['id', target] if 'id' in train_fe.columns else [target]
features = [c for c in train_fe.columns if c not in drop_cols]

X = train_fe[features].copy()
y = train_fe[target].copy()
X_test = test_fe[features].copy()

print("Features used:", len(features))



def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def run_lgb_cv(X, y, X_test, params=None, n_splits=5, seed=RND):
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    feature_importance_df = pd.DataFrame()

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=200)
            ]
        )

        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

        fi = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain'),
            'fold': fold
        })
        feature_importance_df = pd.concat([feature_importance_df, fi], axis=0)

        print(f"Fold {fold+1} RMSE: {rmse(y_val, oof[val_idx]):.6f}")

    print("Overall CV RMSE:", rmse(y, oof))
    return oof, preds, feature_importance_df



# 10. LGB parameters (baseline)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': RND,
    'verbosity': -1
}

oof_lgb, preds_lgb, fi_lgb = run_lgb_cv(X, y, X_test, params=lgb_params, n_splits=5)



def run_xgb_cv(X, y, X_test, params=None, n_splits=5, seed=42):
    """Cross-validation training for XGBoost (stable for all XGBoost versions)."""
    from sklearn.model_selection import KFold
    import xgboost as xgb
    import numpy as np
    import pandas as pd

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    feature_importance_df = pd.DataFrame()

    # Ensure correct metric setup
    if 'eval_metric' not in params:
        params['eval_metric'] = 'rmse'

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"\n===== XGB Fold {fold + 1} =====")
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        # Create DMatrix (core XGBoost data format)
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        # Train model
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=watchlist,
            early_stopping_rounds=100,
            verbose_eval=200
        )

        # Predict using the best iteration
        oof[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / n_splits

        # Extract feature importance safely
        importance_dict = model.get_score(importance_type="gain")
        fi = pd.DataFrame({
            "feature": list(importance_dict.keys()),
            "importance": list(importance_dict.values()),
            "fold": fold + 1
        })
        feature_importance_df = pd.concat([feature_importance_df, fi], axis=0)

        # Compute RMSE for current fold
        fold_rmse = np.sqrt(np.mean((y_val - oof[val_idx]) ** 2))
        print(f"XGB Fold {fold + 1} RMSE: {fold_rmse:.6f}")

    # Overall CV RMSE
    overall_rmse = np.sqrt(np.mean((y - oof) ** 2))
    print("\nOverall XGB CV RMSE:", overall_rmse)

    # Average feature importance
    feature_importance_df = (
        feature_importance_df.groupby("feature", as_index=False)
        .importance.mean()
        .sort_values("importance", ascending=False)
    )

    return oof, preds, feature_importance_df



## 11a

xgb_params = {
    'objective': 'reg:squarederror',
    # 'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eta': 0.05,
    'lambda': 1.0,
    'alpha': 0.0,
    'tree_method': 'hist',
    'verbosity': 1
}

oof_xgb, preds_xgb, fi_xgb = run_xgb_cv(X, y, X_test, params=xgb_params, n_splits=5)



preds_lgb.shape, preds_xgb.shape


type(preds_lgb), type(preds_xgb)


# 12. Simple blending: weighted average
blend_preds = 0.5 * preds_lgb + 0.5 * preds_xgb
# If you have oof predictions available, compute CV RMSE of blend on holdout by stacking 2nd-level model.
print("Blend (0.5/0.5) sample stats:", np.min(blend_preds), np.max(blend_preds))

# Clip to [0,1] if target bounded
blend_preds = np.clip(blend_preds, 0, 1)



fi_lgb.head()



# 13. Feature importance
fi = fi_lgb.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(8,10))
sns.barplot(x='importance', y='feature', data=fi.head(30))
plt.title("LGB Average Feature Importance (top 30)")
plt.tight_layout()
plt.show()



print(len(blend_preds), len(test))



# # 14. create submission
# submission = test[['id']].copy() if 'id' in test.columns else pd.DataFrame({"id": test.index})
# submission['accident_risk'] = blend_preds
# submission.to_csv("submission.csv", index=False)
# print("Saved submission.csv")



def save_submission(preds, test_df, sample_path=None, filename='submission.csv'):
    import pandas as pd
    submission = test_df[['id']].copy() if 'id' in test_df.columns else pd.DataFrame({"id": test_df.index})
    if sample_path:
        try:
            sample = pd.read_csv(sample_path)
            target_col = [col for col in sample.columns if col != 'id'][0]
        except Exception as e:
            print(f"Warning: Could not read sample submission ({e}). Using default 'accident_risk'")
            target_col = 'accident_risk'
    else:
        target_col = 'accident_risk'
    submission[target_col] = preds
    submission.to_csv(filename, index=False)
    print(f"✅ Saved submission file as '{filename}' with target column '{target_col}'")





