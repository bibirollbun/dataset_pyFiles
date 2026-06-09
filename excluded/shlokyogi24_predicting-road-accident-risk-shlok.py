# Step 1 — Setup and imports
SEED = 42
import warnings
warnings.filterwarnings('ignore')
import os, gc, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
random.seed(SEED)
np.random.seed(SEED)



# Step 2 — Load data
INPUT = "/kaggle/input/playground-series-s5e10"
train = pd.read_csv(f"{INPUT}/train.csv")
test = pd.read_csv(f"{INPUT}/test.csv")
sample = pd.read_csv(f"{INPUT}/sample_submission.csv")



# Step 3 — Detect target and ID
TARGET = 'accident_risk'
if TARGET not in train.columns:
    cand = [c for c in train.columns if c.lower().startswith('acc') or 'risk' in c.lower() or c.lower().startswith('target')]
    TARGET = cand[0] if len(cand)>0 else train.columns[-1]
id_col = None
for c in train.columns:
    if c.lower() == 'id':
        id_col = c
        break



# Step 4 — Basic type alignment (bool -> int)
def basic_align_types(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for df in [train_df, test_df]:
        bool_cols = df.select_dtypes(include='bool').columns.tolist()
        for c in bool_cols:
            df[c] = df[c].astype(int)
    return train_df, test_df
train, test = basic_align_types(train, test)


# Step 5 — Feature engineering
def feature_engineer(train_df, test_df, target):
    train_df = train_df.copy()
    test_df = test_df.copy()
    # interactions
    for df in [train_df, test_df]:
        if 'speed_limit' in df.columns and 'curvature' in df.columns:
            df['speed_x_curv'] = df['speed_limit'] * df['curvature'].fillna(0)
        if 'speed_limit' in df.columns and 'num_lanes' in df.columns:
            df['speed_x_lanes'] = df['speed_limit'] * df['num_lanes'].fillna(0)
        if 'num_reported_accidents' in df.columns and 'num_lanes' in df.columns:
            df['acc_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'].fillna(0) + 1)
    # group features
    if 'road_type' in train_df.columns:
        grp = train_df.groupby('road_type')[target].agg(['mean','count']).rename(columns={'mean':'roadtype_risk_mean','count':'roadtype_count'})
        train_df = train_df.merge(grp, on='road_type', how='left')
        test_df = test_df.merge(grp, on='road_type', how='left')
        train_df['roadtype_risk_mean'].fillna(train_df[target].mean(), inplace=True)
        test_df['roadtype_risk_mean'].fillna(train_df[target].mean(), inplace=True)
        train_df['roadtype_count'].fillna(0, inplace=True)
        test_df['roadtype_count'].fillna(0, inplace=True)
    # time flags
    if 'time_of_day' in train_df.columns:
        for df in [train_df, test_df]:
            df['is_night'] = df['time_of_day'].astype(str).str.contains('night', case=False).astype(int)
            df['is_morning'] = df['time_of_day'].astype(str).str.contains('morning', case=False).astype(int)
            df['is_evening'] = df['time_of_day'].astype(str).str.contains('evening', case=False).astype(int)
    # boolean-like columns to int
    for col in ['holiday','school_season','public_road','road_signs_present']:
        if col in train_df.columns:
            for df in [train_df, test_df]:
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)
    # fill na with train median for numeric stability
    num_median = train_df.median(numeric_only=True)
    train_df.fillna(num_median, inplace=True)
    test_df.fillna(num_median, inplace=True)
    return train_df, test_df

train_fe, test_fe = feature_engineer(train, test, TARGET)



# Step 6 — Identify features and categorical columns
features = [c for c in train_fe.columns if c not in [TARGET, id_col]]
cat_cols = [c for c in features if train_fe[c].dtype == 'object']
num_cols = [c for c in features if c not in cat_cols]



# Step 7 — Encode categorical columns by train categories
def encode_cats(train_df, test_df, cat_columns):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in cat_columns:
        train_df[c] = train_df[c].astype('category')
        cats = train_df[c].cat.categories
        test_df[c] = pd.Categorical(test_df[c], categories=cats)
        train_df[c] = train_df[c].cat.codes
        test_df[c] = test_df[c].cat.codes.fillna(-1).astype(int)
    return train_df, test_df

train_enc, test_enc = encode_cats(train_fe, test_fe, cat_cols)




# Step 8 — Prepare training and test matrices
X = train_enc[features].reset_index(drop=True)
y = train_enc[TARGET].reset_index(drop=True)
X_test = test_enc[features].reset_index(drop=True)



# Step 9 — Train LightGBM with 5-fold OOF
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def train_lgbm_oof(X, y, X_test, features, params=None, n_splits=5, seed=SEED):
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'seed': seed,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
        }
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    feature_importance = pd.DataFrame()
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(200)]
        )
        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        preds += model.predict(X_test[features], num_iteration=model.best_iteration) / n_splits
        fi = pd.DataFrame({'feature': features, 'importance': model.feature_importance(importance_type='gain'), 'fold': fold+1})
        feature_importance = pd.concat([feature_importance, fi], axis=0)
        del model, dtrain, dval
        gc.collect()
    score = mean_squared_error(y, oof, squared=False)
    return oof, preds, score, feature_importance

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'seed': SEED,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
}

oof_lgb, preds_lgb, score_lgb, fi_lgb = train_lgbm_oof(X, y, X_test, features, params=lgb_params, n_splits=5)
print(f"OOF RMSE: {score_lgb:.6f}")



# Step 10 — Feature importance visualization
fi_agg = fi_lgb.groupby('feature')['importance'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,8))
fi_agg.head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



# Step 11 — OOF diagnostics plots
plt.figure(figsize=(8,6))
plt.scatter(y, oof_lgb, alpha=0.3, s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.tight_layout()
plt.show()

residuals = y - oof_lgb
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=50, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', lw=2)
plt.tight_layout()
plt.show()

print(f"Mean residual: {residuals.mean():.6f}")
print(f"Std residual: {residuals.std():.6f}")



# Step 12 — Create submission file
sub = sample.copy()
target_col = [c for c in sub.columns if c.lower() != id_col.lower() and c.lower()!=id_col and c.lower()!=str(id_col).lower()][0] if len(sub.columns)>1 else sub.columns[1]
sub[target_col] = preds_lgb
sub.to_csv("submission_lgb_baseline.csv", index=False)
sub.head()


