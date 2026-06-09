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


import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, brier_score_loss


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv").drop('id', axis=1)
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
'''print(X.columns)
y = X.pop('accident_risk')
X.head(10)'''


# kaggle_playground_accident_risk.py
# Run with: python kaggle_playground_accident_risk.py
# Requires: pandas, numpy, scikit-learn, catboost, lightgbm
# pip install pandas numpy scikit-learn catboost lightgbm

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from datetime import datetime

SEED = 42
NFOLDS = 5
np.random.seed(SEED)

# ---------- 1) Load data ----------


# Ensure id exists in test
if "id" not in test.columns:
    test["id"] = np.arange(len(test)) + 1

TARGET = "accident_risk"
IDCOL = "id"

# Basic safety: if target missing in test, drop it
if TARGET not in test.columns:
    test[TARGET] = np.nan

# ---------- 2) Basic cleaning & types ----------
# Known columns (from your prompt)
expected_cols = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
       'weather', 'road_signs_present', 'public_road', 'time_of_day',
       'holiday', 'school_season', 'num_reported_accidents', 'accident_risk']

# If some columns missing, try to continue gracefully
for col in expected_cols:
    if col not in train.columns and col in test.columns:
        train[col] = np.nan
    if col not in test.columns and col in train.columns:
        test[col] = np.nan

# Combine for joint transformations
full = pd.concat([train.drop(columns=[TARGET]), test], sort=False).reset_index(drop=True)

# Fill obvious missing numeric values
numeric_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents']
for col in numeric_cols:
    if col in full.columns:
        full[col] = pd.to_numeric(full[col], errors='coerce')
        # median imputation (safe)
        full[col] = full[col].fillna(full[col].median())

# Normalize boolean/flag columns to 0/1 if not already
bool_like = ['road_signs_present','public_road','holiday','school_season']
for col in bool_like:
    if col in full.columns:
        full[col] = full[col].map({True:1, False:0})
        full[col] = full[col].fillna(0).astype(int)

# Cast object-like categorical columns to string
cat_cols = ['road_type','lighting','weather','time_of_day']
for col in cat_cols:
    if col in full.columns:
        full[col] = full[col].astype(str).fillna("missing")

# ---------- 3) Feature engineering ----------
def add_features(df):
    # interaction features
    df['lanes_x_speed'] = df['num_lanes'] * df['speed_limit']
    df['curvature_x_speed'] = df['curvature'] * df['speed_limit']
    # accidents per lane (smoothed)
    df['acc_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 0.1)
    # risk proxy: speed adjusted curvature
    df['speed_adj_curv'] = df['speed_limit'] / (1 + np.exp(- (df['curvature'] - df['curvature'].median())))
    # flag high speed
    df['is_high_speed'] = (df['speed_limit'] >= df['speed_limit'].quantile(0.75)).astype(int)
    # lane density (lanes relative to median)
    df['lanes_rel'] = df['num_lanes'] / (df['num_lanes'].median()+1e-6)
    # signage effect (if signs present maybe lower risk)
    df['signs_effect'] = df['road_signs_present'] * (1 / (1 + df['speed_limit']/50.0))
    return df

full = add_features(full)

# Encode "time_of_day" roughly if values like 'night','day','dawn','dusk' or numeric hours
def map_time_of_day(x):
    x = str(x).lower()
    if x.isdigit():
        h = int(x) % 24
        return h
    if 'night' in x:
        return 0
    if 'midnight' in x:
        return 0
    if 'dawn' in x or 'sunrise' in x:
        return 5
    if 'morning' in x:
        return 9
    if 'noon' in x:
        return 12
    if 'afternoon' in x:
        return 15
    if 'dusk' in x or 'sunset' in x:
        return 18
    if 'evening' in x:
        return 20
    return 12  # fallback

if 'time_of_day' in full.columns:
    full['time_hour'] = full['time_of_day'].apply(map_time_of_day)
    # cyclic encoding
    full['time_sin'] = np.sin(2 * np.pi * full['time_hour'] / 24)
    full['time_cos'] = np.cos(2 * np.pi * full['time_hour'] / 24)
    # drop text time_of_day from some models but keep for CatBoost
else:
    full['time_hour'] = 12
    full['time_sin'] = 0
    full['time_cos'] = 1

# simple label encode for lightgbm (ordinal)
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ord_columns = [c for c in ['road_type','lighting','weather','time_of_day'] if c in full.columns]
if ord_columns:
    full[ord_columns] = ord_enc.fit_transform(full[ord_columns])

# split back
train2 = full.iloc[:len(train)].copy().reset_index(drop=True)
test2  = full.iloc[len(train):].copy().reset_index(drop=True)

# Reattach target
train2[TARGET] = train[TARGET].values

# ---------- 4) Prepare model features ----------
# Common features for LightGBM: numeric + encoded categoricals (we encoded above)
features = [c for c in train2.columns if c not in [IDCOL, TARGET]]
# We will give CatBoost the original categorical names (text) - rebuild small cat df
cat_features_for_catboost = [c for c in ['road_type','lighting','weather','time_of_day'] if c in train.columns]

# For CatBoost we need text categories - build catboost-friendly DataFrame:
train_cat = train.copy()
test_cat = test.copy()
# Fill missing for numeric and cat as above
for col in numeric_cols:
    if col in train_cat.columns:
        train_cat[col] = pd.to_numeric(train_cat[col], errors='coerce').fillna(train_cat[col].median())
    if col in test_cat.columns:
        test_cat[col] = pd.to_numeric(test_cat[col], errors='coerce').fillna(test_cat[col].median())
for col in cat_features_for_catboost:
    if col in train_cat.columns:
        train_cat[col] = train_cat[col].astype(str).fillna("missing")
    if col in test_cat.columns:
        test_cat[col] = test_cat[col].astype(str).fillna("missing")

# Add engineered features to cat DataFrames
train_cat = add_features(train_cat)
test_cat = add_features(test_cat)

# Ensure same order and features
cat_features_for_catboost = [c for c in cat_features_for_catboost if c in train_cat.columns]
features_cat = [c for c in train_cat.columns if c not in [IDCOL, TARGET]]

# ---------- 5) Cross-validated training ----------
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train2))
oof_cat = np.zeros(len(train2))
preds_test_lgb = np.zeros(len(test2))
preds_test_cat = np.zeros(len(test2))

fold = 0
for tr_idx, vl_idx in kf.split(train2):
    fold += 1
    print(f"\n--- FOLD {fold} --- {datetime.now().isoformat()}")
    X_tr, X_val = train2.iloc[tr_idx][features], train2.iloc[vl_idx][features]
    y_tr, y_val = train2.iloc[tr_idx][TARGET].values, train2.iloc[vl_idx][TARGET].values

    # ---------- LightGBM ----------
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': SEED,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
    lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=5000,
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ]
)

    # Predict
    oof_lgb[vl_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    preds_test_lgb += lgb_model.predict(test2[features], num_iteration=lgb_model.best_iteration) / NFOLDS

    # ---------- CatBoost ----------
    # Use the original textual cat features and engineered numeric columns
    # Build CatBoost pools with categorical features indices
    X_tr_cat = train_cat.iloc[tr_idx][features_cat]
    X_val_cat = train_cat.iloc[vl_idx][features_cat]
    y_tr_cat = train_cat.iloc[tr_idx][TARGET]
    y_val_cat = train_cat.iloc[vl_idx][TARGET]

    # CatBoost params (robust defaults)
    cb_params = {
        'iterations': 2000,
        'learning_rate': 0.03,
        'random_seed': SEED,
        'depth': 6,
        'loss_function': 'RMSE',
        'early_stopping_rounds': 100,
        'verbose': False
    }
    # cat_features_for_catboost list is textual column names inside features_cat
    cat_indices = [X_tr_cat.columns.get_loc(c) for c in cat_features_for_catboost if c in X_tr_cat.columns]

    cb_train = Pool(X_tr_cat, y_tr_cat, cat_features=cat_indices)
    cb_val = Pool(X_val_cat, y_val_cat, cat_features=cat_indices)
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(cb_train, eval_set=cb_val, use_best_model=True)

    oof_cat[vl_idx] = cb_model.predict(X_val_cat)
    preds_test_cat += cb_model.predict(test_cat[features_cat]) / NFOLDS

    # Log fold RMSEs
    rmse_lgb = mean_squared_error(y_val, oof_lgb[vl_idx], squared=False)
    rmse_cat = mean_squared_error(y_val, oof_cat[vl_idx], squared=False)
    print(f"Fold {fold} RMSE - LightGBM: {rmse_lgb:.5f}, CatBoost: {rmse_cat:.5f}")

# ---------- 6) OOF evaluation and blending ----------
rmse_lgb_oof = mean_squared_error(train2[TARGET].values, oof_lgb, squared=False)
rmse_cat_oof = mean_squared_error(train2[TARGET].values, oof_cat, squared=False)
print(f"\nOOF RMSE - LightGBM: {rmse_lgb_oof:.5f}, CatBoost: {rmse_cat_oof:.5f}")

# Simple weight by inverse RMSE
inv = np.array([1.0/rmse_lgb_oof, 1.0/rmse_cat_oof])
weights = inv / inv.sum()
print(f"Ensemble weights (LGB, CAT): {weights}")

oof_ensemble = oof_lgb * weights[0] + oof_cat * weights[1]
rmse_ensemble = mean_squared_error(train2[TARGET].values, oof_ensemble, squared=False)
print(f"OOF RMSE - Ensemble: {rmse_ensemble:.5f}")

# Test preds
preds_test = preds_test_lgb * weights[0] + preds_test_cat * weights[1]

# Clip predictions to [0,1]
preds_test = np.clip(preds_test, 0, 1)

# ---------- 7) Prepare submission ----------
submission = pd.DataFrame({
    IDCOL: test[IDCOL].values,
    TARGET: preds_test
})
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")


'''X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)
preprocessor = ColumnTransformer(transformers = [('cat',OneHotEncoder(),make_column_selector(dtype_include=object))])
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)'''


'''model = XGBRegressor(learning_rate=0.03,objective='reg:logistic',n_estimators=500)
model.fit(X_train,y_train,
         early_stopping_rounds=10,
         eval_set = [(X_test,y_test)],
          eval_metric = 'logloss',
         verbose = True)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE is {mae: .4f}")
print(f"R2 score is {r2: .4f}")'''


#xgb.plot_importance(model, importance_type='gain', max_num_features=15)


'''import numpy as np

def brier_score(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

score = brier_score(y_test, y_pred)
print("Brier score:", score)'''


'''pred = model.predict(pd.DataFrame(X_test))
comparison = pd.DataFrame({
    'Predicted': pred,
    'Actual': y_test
}).reset_index(drop=True)
print(comparison.iloc[42:80])'''


'''from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

scaling = ColumnTransformer(transformers=[('scaler',StandardScaler(), make_column_selector(dtype_include=np.number))], remainder='passthrough')
X_train = scaling.fit_transform(pd.DataFrame(X_train))
X_test = scaling.transform(X_test)
input_shape = X_train.shape[1]
nn = Sequential([Dense(64,activation='relu', input_shape=[input_shape]),
                Dense(32,activation='relu'),
                Dense(1,activation='sigmoid')
                ])
nn.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mae','mse'])
history = nn.fit(X_train,y_train,
                 validation_data =[X_test, y_test],
                 batch_size= 500,
                 epochs = 50,
                 verbose = 1
)
pd.DataFrame(X_train).head()'''


