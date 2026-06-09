# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb

# Display settings
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 120)


CATEGORICAL_FEATURES = ['road_type', 'lighting', 'weather', 'time_of_day']
BOOLEAN_FEATURES = ['road_signs_present', 'public_road', 'holiday', 'school_season']
NUMERICAL_FEATURES = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
TARGET = 'accident_risk'
ID_COL = 'id'
RANDOM_STATE = 42


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission_template = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Training : {train_df.head()}")
print(f"Training set: {train_df.shape}")
print(f"Test set: {test_df.shape}")
print(f"Memory usage: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


# Basic validation checks

def basic_checks(df, name='data'):
    print(f"--- Basic checks for {name} ---")
    print('Columns:', df.columns.tolist())
    print('Null counts:\n', df.isnull().sum())
    print('Dtypes:\n', df.dtypes)
    print('Duplicate IDs:', df[ID_COL].duplicated().sum() if ID_COL in df.columns else 'No id column')
    print()

basic_checks(train_df, 'train')
basic_checks(test_df, 'test')

# Target checks
if TARGET in train_df.columns:
    print('Target value range: min=', train_df[TARGET].min(), 'max=', train_df[TARGET].max())
    assert train_df[TARGET].min() >= -1e-6 and train_df[TARGET].max() <= 1+1e-6, "Target values should be within [0,1]"



# Range validation for numerical features (quick domain heuristics)
ranges = {
    'num_lanes': (1, 10),            # typical lanes 1-10
    'curvature': (0, 1000),          # curvature measure scale may vary; check distribution
    'speed_limit': (10, 200),        # km/h or mph context - adjust as needed
    'num_reported_accidents': (0, 1000),
}

for col, (minv, maxv) in ranges.items():
    if col in train_df.columns:
        print(col, 'min, max in train:', train_df[col].min(), train_df[col].max())
        bad_low = train_df[train_df[col] < minv]
        bad_high = train_df[train_df[col] > maxv]
        print(f'  values < {minv}:', len(bad_low), f' values > {maxv}:', len(bad_high))
        print()



# Check categorical consistency and typos
for col in CATEGORICAL_FEATURES:
    if col in train_df.columns:
        print(f'--- {col} ---')
        vals_train = train_df[col].astype(str).str.lower().str.strip().value_counts()[:20]
        vals_test = test_df[col].astype(str).str.lower().str.strip().value_counts()[:20]
        print('Train unique:', vals_train.shape[0], 'Top values:\n', vals_train.head())
        print('Test unique:', vals_test.shape[0], 'Top values:\n', vals_test.head())
        print()


# Summary of missingness
print('Train missing %:')
print((train_df.isnull().sum() / len(train_df) * 100).sort_values(ascending=False).head(20))
print('\nTest missing %:')
print((test_df.isnull().sum() / len(test_df) * 100).sort_values(ascending=False).head(20))

# Strategy:
# - Drop rows with missing target
train_df = train_df.dropna(subset=[TARGET])
print('After dropping missing targets, train shape:', train_df.shape)



# EDA: numerical distributions
for col in NUMERICAL_FEATURES:
    if col in train_df.columns:
        plt.figure(figsize=(6,3))
        plt.hist(train_df[col].dropna(), bins=50)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('count')
        plt.tight_layout()
        plt.show()



# Target distribution
plt.figure(figsize=(6,3))
plt.hist(train_df[TARGET], bins=50)
plt.title('Distribution of target (accident_risk)')
plt.xlabel(TARGET)
plt.ylabel('count')
plt.tight_layout()
plt.show()

# Scatter plots numeric vs target
for col in NUMERICAL_FEATURES:
    if col in train_df.columns:
        plt.figure(figsize=(5,3))
        plt.scatter(train_df[col], train_df[TARGET], alpha=0.3, s=8)
        plt.title(f'{col} vs {TARGET}')
        plt.xlabel(col)
        plt.ylabel(TARGET)
        plt.tight_layout()
        plt.show()


# Categorical counts and average target by category
for col in CATEGORICAL_FEATURES:
    if col in train_df.columns:
        df = train_df[[col, TARGET]].copy()
        df[col] = df[col].astype(str).str.lower().str.strip()
        counts = df[col].value_counts()
        means = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
        print('---', col, '---')
        print('Top categories by count:')
        print(counts.head())
        print('Top categories by avg target:')
        print(means.head())
        # bar plot counts
        plt.figure(figsize=(6,3))
        counts.head(10).plot(kind='bar')
        plt.title(f'Top 10 {col} categories (count)')
        plt.tight_layout()
        plt.show()


# Correlation matrix (numerical features + target)
corr_cols = [c for c in NUMERICAL_FEATURES + [TARGET] if c in train_df.columns]
if len(corr_cols) >= 2:
    corr = train_df[corr_cols].corr()
    print('Correlation matrix:\n', corr)
    plt.figure(figsize=(6,5))
    plt.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.show()


# Preprocessing and feature engineering pipeline

def preprocess(df, is_train=True):
    df = df.copy()
    # Booleans -> int
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].map({True:1, False:0, 'True':1, 'False':0, 'true':1, 'false':0}).fillna(df[col])
            try:
                df[col] = df[col].astype(int)
            except:
                df[col] = df[col].fillna(0).astype(int)

    # Fix categorical case and strip
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip().fillna('unknown')

    # Simple imputations for numerical
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Feature: speed_limit * curvature (interaction)
    if 'speed_limit' in df.columns and 'curvature' in df.columns:
        df['speed_curv_interaction'] = df['speed_limit'] * df['curvature']

    # Feature: accidents per lane
    if 'num_reported_accidents' in df.columns and 'num_lanes' in df.columns:
        df['accidents_per_lane'] = df['num_reported_accidents'] / df['num_lanes'].replace(0,1)

    # One-hot encode categorical (use get_dummies — will align later)
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_FEATURES if c in df.columns], dummy_na=False)
    return df

train_fe = preprocess(train_df)
test_fe = preprocess(test_df)

# Align columns
train_fe, test_fe = train_fe.align(test_fe, join='left', axis=1, fill_value=0)
print('train_fe shape:', train_fe.shape)
print('test_fe shape:', test_fe.shape)



# Prepare data matrices
drop_cols = [ID_COL, TARGET] if TARGET in train_fe.columns else [ID_COL]
X = train_fe.drop(columns=[c for c in drop_cols if c in train_fe.columns])
y = train_fe[TARGET] if TARGET in train_fe.columns else None
X_test = test_fe.drop(columns=[c for c in [ID_COL, TARGET] if c in test_fe.columns])

print('X shape:', X.shape)
print('y shape:', None if y is None else y.shape)
print('X_test shape:', X_test.shape)



# Modeling: K-Fold CV with LightGBM and XGBoost, then average predictions
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
oof = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print('Fold', fold+1)
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    lgb_params = {
        'objective':'regression',
        'metric':'rmse',
        'verbosity':-1,
        'boosting_type':'gbdt',
        'learning_rate':0.05,
        'num_leaves':31,
        'feature_fraction':0.8,
        'bagging_fraction':0.8,
        'bagging_freq':5,
        'seed':RANDOM_STATE
    }
    lgb_model = lgb.train(lgb_params, 
                          lgb_train,
                          num_boost_round=2000, 
                          valid_sets=[lgb_val], 
                           callbacks=[
                                    lgb.early_stopping(50),
                                    lgb.log_evaluation(100) ]
                           )
    val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    print('  LGB RMSE:', mean_squared_error(y_val, val_pred, squared=False))
    lgb_preds += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / NFOLDS

    # XGBoost
    xgb_params = {
        'objective':'reg:squarederror',
        'eval_metric':'rmse',
        'seed':RANDOM_STATE,
        'eta':0.05,
        'max_depth':6,
        'subsample':0.8,
        'colsample_bytree':0.8,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dx_test = xgb.DMatrix(X_test)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=2000, evals=[(dval,'val')], early_stopping_rounds=50, verbose_eval=False)
    xgb_val_pred = xgb_model.predict(dval,iteration_range=(0, xgb_model.best_iteration))
    print('  XGB RMSE:', mean_squared_error(y_val, xgb_val_pred, squared=False))
    xgb_preds += xgb_model.predict(dx_test, iteration_range=(0, xgb_model.best_iteration)) / NFOLDS


    # OOF
    oof[val_idx] = (val_pred + xgb_val_pred) / 2

# Overall OOF score
print('OOF RMSE:', mean_squared_error(y, oof, squared=False))

# Average ensemble
final_preds = (lgb_preds + xgb_preds) / 2
final_preds = np.clip(final_preds, 0, 1)



# Save submission
submission = pd.DataFrame({
    ID_COL: test_df[ID_COL],
    TARGET: final_preds
})
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)
print('Saved submission to', submission_path)
display(submission.head())


# Feature importance from the last trained LightGBM model (if available)
try:
    importances = lgb_model.feature_importance(importance_type='gain')
    feat_names = X.columns
    fi = pd.DataFrame({'feature':feat_names, 'importance':importances})
    fi = fi.sort_values('importance', ascending=False).head(30)
    print(fi)
    plt.figure(figsize=(6,8))
    plt.barh(fi['feature'][::-1], fi['importance'][::-1])
    plt.title('Top 30 LightGBM feature importances')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print('Could not compute feature importance:', e)


