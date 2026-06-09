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


# === IMPORTS ===
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gc
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

# === CONFIG ===
TARGET = 'accident_risk'
SEED = 42
N_FOLDS = 5


# === LOAD DATA ===
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# Keep ID for final submission
test_ids = test['id'].copy()

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")


# === FEATURE ENGINEERING (FINAL - XGBoost READY) ===

cat_cols = ['road_type', 'lighting', 'weather', 'road_signs_present',
            'public_road', 'time_of_day', 'holiday', 'school_season']
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

# -------------------------------------------------
# 1. Frequency + Binning
# -------------------------------------------------
def add_freq_and_bins(train_df, test_df, cat_cols, num_cols):
    train, test = train_df.copy(), test_df.copy()
    orig_train = {col: train[col].copy() for col in cat_cols + num_cols}
    orig_test  = {col: test[col].copy()  for col in cat_cols + num_cols}

    for col in cat_cols:
        freq = orig_train[col].value_counts(normalize=True)
        train[f'{col}_freq'] = pd.Series(orig_train[col].map(freq), index=train.index, dtype='float64').fillna(freq.mean())
        test[f'{col}_freq']  = pd.Series(orig_test[col].map(freq),  index=test.index,  dtype='float64').fillna(freq.mean())

    for col in num_cols:
        try:
            values = orig_train[col].dropna()
            if len(values) == 0 or values.nunique() < 2:
                raise ValueError("Not enough variation")
            for q in [5, 10, 20]:
                bins = pd.qcut(values, q=q, duplicates='drop', retbins=True)[1]
                train[f'{col}_bin{q}'] = pd.cut(orig_train[col], bins=bins, duplicates='drop', include_lowest=True).cat.codes
                test[f'{col}_bin{q}']  = pd.cut(orig_test[col],  bins=bins, duplicates='drop', include_lowest=True).cat.codes
        except Exception as e:
            print(f"Binning failed for {col}: {e}")
            for q in [5, 10, 20]:
                train[f'{col}_bin{q}'] = 0
                test[f'{col}_bin{q}']  = 0

    # Convert base categoricals
    for col in cat_cols:
        train[col] = train[col].astype('category')
        test[col]  = test[col].astype('category')

    return train, test

train, test = add_freq_and_bins(train, test, cat_cols, num_cols)

# -------------------------------------------------
# 2. Target Encoding
# -------------------------------------------------
def target_encode(train_df, test_df, col, target, smooth=30):
    prior = train_df[target].mean()
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    smoothed = (agg['mean'] * agg['count'] + prior * smooth) / (agg['count'] + smooth)
    train_df[f'{col}_te'] = pd.Series(train_df[col].map(smoothed), index=train_df.index, dtype='float64').fillna(prior)
    test_df[f'{col}_te']  = pd.Series(test_df[col].map(smoothed),  index=test_df.index,  dtype='float64').fillna(prior)
    return train_df, test_df

for col in cat_cols:
    train, test = target_encode(train, test, col, TARGET, smooth=30)

# -------------------------------------------------
# 3. Interactions (MUST BE CATEGORY)
# -------------------------------------------------
def add_interactions(df):
    df = df.copy()
    df['lanes_x_curv']     = df['num_lanes'] * df['curvature']
    df['speed_div_curv']   = df['speed_limit'] / (df['curvature'] + 1e-6)
    df['weather_light']    = (df['weather'].astype(str) + '_' + df['lighting'].astype(str)).astype('category')
    df['roadtype_time']    = (df['road_type'].astype(str) + '_' + df['time_of_day'].astype(str)).astype('category')
    df['is_night']         = (df['lighting'] == 'night').astype(int)
    df['is_foggy_rainy']   = (df['weather'].isin(['foggy','rainy']) & (df['lighting'] != 'daylight')).astype(int)
    return df

train = add_interactions(train)
test  = add_interactions(test)

# -------------------------------------------------
# 4. Map num_reported_accidents
# -------------------------------------------------
acc_risk_map = train.groupby('num_reported_accidents')[TARGET].mean()
train['num_reported_accidents'] = train['num_reported_accidents'].map(acc_risk_map)
test['num_reported_accidents']  = test['num_reported_accidents'].map(acc_risk_map).fillna(acc_risk_map.mean())

# -------------------------------------------------
# 5. Drop ID
# -------------------------------------------------
train.drop(columns=['id'], errors='ignore', inplace=True)
test.drop(columns=['id'], errors='ignore', inplace=True)

# -------------------------------------------------
# 6. FINAL: ENSURE ALL CATEGORICAL COLUMNS ARE 'category'
# -------------------------------------------------
all_cat_cols = cat_cols + ['weather_light', 'roadtype_time']

for col in all_cat_cols:
    if col in train.columns:
        train[col] = train[col].astype('category')
    if col in test.columns:
        test[col] = test[col].astype('category')

print(f"Final train shape: {train.shape}")
print(f"Categorical columns ({len(all_cat_cols)}): {all_cat_cols}")
print(train[all_cat_cols].dtypes)


# === CV SETUP ===
X = train.drop(columns=[TARGET])
y = train[TARGET]

xgb_params = {
    'objective'          : 'reg:squarederror',
    'eval_metric'        : 'rmse',
    'tree_method'        : 'hist',
    'device'             : 'cuda',
    'max_depth'          : 12,
    'learning_rate'      : 0.0098,
    'subsample'          : 0.83,
    'colsample_bytree'   : 0.79,
    'colsample_bylevel'  : 0.86,
    'colsample_bynode'   : 0.88,
    'reg_alpha'          : 0.12,
    'reg_lambda'         : 0.41,
    'min_child_weight'   : 3,
    'max_bin'            : 512,
    'random_state'       : SEED,
    'enable_categorical' : True
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
cv_scores = []
best_iters = []

print("Starting 5-fold CV...\n")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
    dval   = xgb.DMatrix(X_val, y_val, enable_categorical=True)

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=80,
        verbose_eval=500
    )

    best_iter = model.best_iteration
    best_iters.append(best_iter)
    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, best_iter))
    score = model.best_score
    cv_scores.append(score)

    print(f"Fold {fold+1} | RMSE: {score:.6f} | Best iter: {best_iter}")
    del dtrain, dval, model
    gc.collect()

print(f"\nCV Mean RMSE: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
print(f"OOF RMSE: {mean_squared_error(y, oof_preds, squared=False):.6f}")


# ==============================================================
# === FINAL MODEL – XGBoost + TAIL-ONLY PSEUDO-LGBM BLEND ===
# ==============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# --------------------------------------------------------------
# 1. Train the final XGBoost (identical to CV)
# --------------------------------------------------------------
avg_iter = int(np.mean(best_iters))
print(f"\nTraining final XGBoost with {avg_iter} rounds...")

dtrain_full = xgb.DMatrix(X, y, enable_categorical=True)
final_xgb = xgb.train(xgb_params, dtrain_full, num_boost_round=avg_iter)

dtest = xgb.DMatrix(test, enable_categorical=True)
xgb_pred = final_xgb.predict(dtest)

# --------------------------------------------------------------
# 2. TAIL-ONLY PSEUDO-LGBM (5 lines, < 1 sec)
# --------------------------------------------------------------
print("Training 5-line tail-only model...")

# Select only high-risk training rows (top 5 % of target)
high_risk_mask = y >= np.percentile(y, 95)
X_tail = X[high_risk_mask]
y_tail = y[high_risk_mask]

# Tiny HistGBR – learns only the tail
tail_model = HistGradientBoostingRegressor(
    max_iter=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=SEED
)
tail_model.fit(X_tail, y_tail)

# Predict on full test set
tail_pred = tail_model.predict(test)

# --------------------------------------------------------------
# 3. BLEND: 70 % XGBoost + 30 % tail-only model
# --------------------------------------------------------------
final_risk = 0.70 * xgb_pred + 0.30 * tail_pred

# Optional: ultra-light clipping (0.001 % / 99.999 %)
lower = np.percentile(y, 0.001)
upper = np.percentile(y, 99.999)
final_risk = np.clip(final_risk, lower, upper)

print(f"Final blend applied – mean = {final_risk.mean():.6f}")


# === SAVE SUBMISSION ===
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': final_risk
})

submission.to_csv('submission.csv', index=False)
print("\nsubmission.csv saved – ready to upload!")
submission.head()

