# ====================================================
# 1. Libraries & Settings
# ====================================================
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ====================================================
# 2. Load Data
# ====================================================
df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

print("✅ Train shape:", df.shape)
print("✅ Test shape:", df_test.shape)


# ====================================================
# 3. Preprocessing
# ====================================================
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    df[col] = df[col].astype(int)
    df_test[col] = df_test[col].astype(int)

cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

# One-hot encode
df_all = pd.concat([df[cat_cols], df_test[cat_cols]], axis=0)
df_all_encoded = pd.get_dummies(df_all, columns=cat_cols, drop_first=False)

df_encoded = df_all_encoded.iloc[:len(df)]
df_test_encoded = df_all_encoded.iloc[len(df):].reset_index(drop=True)

num_bool_cols = [c for c in df.columns if c not in ['id','accident_risk'] and c not in cat_cols]
df_final = pd.concat([df[num_bool_cols], df_encoded], axis=1)
df_test_final = pd.concat([df_test[num_bool_cols], df_test_encoded], axis=1)


# ====================================================
# 4. Feature Engineering
# ====================================================
def add_engineered_features(df):
    df = df.copy()
    df['curv_speed'] = df['curvature'] * df['speed_limit']
    df['lane_speed_risk'] = (5 - df['num_lanes']) * df['speed_limit']
    df['speed_curv_lane'] = df['speed_limit'] * df['curvature'] / (df['num_lanes'] + 1)
    df['risk_density'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    # Cyclical encoding of time (if available as hour-like variable)
    if 'time_of_day' in df.columns:
        df['time_sin'] = np.sin(2 * np.pi * (df['time_of_day']+1) / 24)
        df['time_cos'] = np.cos(2 * np.pi * (df['time_of_day']+1) / 24)
    return df

df_final = add_engineered_features(df_final)
df_test_final = add_engineered_features(df_test_final)

print("Final train features:", df_final.shape[1])


# ====================================================
# 5. Train - Validation Split (Stratified)
# ====================================================
X = df_final
y = df['accident_risk']
X_test = df_test_final

bins = pd.qcut(y, q=10, labels=False)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

preds_xgb = np.zeros(len(X_test))
preds_lgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))


# ====================================================
# 6. Models
# ====================================================
params_xgb = {
    'n_estimators': 5000,
    'learning_rate': 0.02,
    'max_depth': 7,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'random_state': 42,
    'tree_method': 'hist',
    'objective': 'reg:squarederror'
}

params_lgb = {
    'n_estimators': 5000,
    'learning_rate': 0.02,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 42
}

params_cat = {
    'iterations': 5000,
    'depth': 7,
    'learning_rate': 0.02,
    'random_seed': 42,
    'verbose': 0
}


# ====================================================
# 7. Cross-Validation Training
# ====================================================
from lightgbm import early_stopping

# ====================================================
# 7. Cross-Validation Training
# ====================================================
for fold, (trn_idx, val_idx) in enumerate(kf.split(X, bins)):
    print(f"▶ Fold {fold+1}")
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(**params_xgb)
    model_xgb.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=200,
                  verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    preds_xgb += model_xgb.predict(X_test) / kf.get_n_splits()
    
    # LightGBM (fixed)
    model_lgb = LGBMRegressor(**params_lgb)
    model_lgb.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(200)])
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    preds_lgb += model_lgb.predict(X_test) / kf.get_n_splits()
    
    # CatBoost
    model_cat = CatBoostRegressor(**params_cat)
    model_cat.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=200,
                  verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_val)
    preds_cat += model_cat.predict(X_test) / kf.get_n_splits()


# ====================================================
# 8. Stacking (Ridge as Meta-Model)
# ====================================================
stack_train = np.vstack([oof_xgb, oof_lgb, oof_cat]).T
stack_test = np.vstack([preds_xgb, preds_lgb, preds_cat]).T

meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_train, y)
final_preds = meta_model.predict(stack_test)

cv_rmse = np.sqrt(mean_squared_error(y, meta_model.predict(stack_train)))
print(f"\n✅ Final Stacking CV RMSE: {cv_rmse:.5f}")


# ====================================================
# 9. Submission
# ====================================================
submission = pd.DataFrame({
    'id': df_test['id'],
    'accident_risk': np.clip(final_preds, 0, 1)
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission saved!")
print(submission.head())




