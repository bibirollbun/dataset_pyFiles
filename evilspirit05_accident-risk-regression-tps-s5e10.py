import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings



# -------------------------------
# 1. Load data
# -------------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission_id = test['id'].copy()

# Drop id
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# -------------------------------
# 2. Feature Engineering
# -------------------------------
# Interaction: road_type + speed_limit
train['road_type_speed'] = train['road_type'].astype(str) + '_' + train['speed_limit'].astype(str)
test['road_type_speed'] = test['road_type'].astype(str) + '_' + test['speed_limit'].astype(str)

# Night flag
train['is_night'] = train['time_of_day'].isin(['evening', 'night']).astype(int)
test['is_night'] = test['time_of_day'].isin(['evening', 'night']).astype(int)

# Accident rate by road_type
accident_rate = train.groupby('road_type')['num_reported_accidents'].mean()
train['accident_rate_road_type'] = train['road_type'].map(accident_rate)
test['accident_rate_road_type'] = test['road_type'].map(accident_rate).fillna(accident_rate.mean())

# Define column types
numerical_cols = ['curvature', 'speed_limit', 'num_reported_accidents', 'accident_rate_road_type', 'is_night']
categorical_cols = ['road_type', 'num_lanes', 'lighting', 'weather', 'road_signs_present',
                    'public_road', 'time_of_day', 'holiday', 'school_season', 'road_type_speed']

# Scale numerical
scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

# Prepare X, y, X_test
X = train.drop('accident_risk', axis=1)
y = train['accident_risk']
X_test = test.copy()

# -------------------------------
# 3. Cross-Validation Setup
# -------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# OOF & test prediction holders
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

pred_cat = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))

# Metrics storage
metrics = {'cat': [], 'xgb': [], 'lgb': [], 'ens': []}

# -------------------------------
# 4. Helper: Label Encode for XGB/LGB
# -------------------------------
def encode_cats(X_train, X_val, X_test, cols):
    X_tr = X_train.copy()
    X_va = X_val.copy()
    X_te = X_test.copy()
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_train[col].astype(str))
        X_va[col] = le.transform(X_val[col].astype(str))
        X_te[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    return X_tr, X_va, X_te, encoders

# -------------------------------
# 5. 5-Fold Training Loop
# -------------------------------
for fold, (idx_tr, idx_va) in enumerate(kf.split(X)):
    print(f"\n=== Fold {fold + 1} ===")
    
    X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]
    y_tr, y_va = y.iloc[idx_tr], y.iloc[idx_va]

    # --- CatBoost (native categoricals) ---
    cat_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        cat_features=categorical_cols,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    oof_cat[idx_va] = cat_model.predict(X_va)
    pred_cat += cat_model.predict(X_test) / kf.n_splits

    # --- XGBoost & LightGBM (need label encoding) ---
    X_tr_enc, X_va_enc, X_test_enc, _ = encode_cats(X_tr, X_va, X_test, categorical_cols)

    xgb_model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=100,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method='gpu_hist' if 'gpu' in str(xgb.XGBRegressor()) else 'hist'
    )
    xgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)], verbose=False)
    oof_xgb[idx_va] = xgb_model.predict(X_va_enc)
    pred_xgb += xgb_model.predict(X_test_enc) / kf.n_splits

    lgb_model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        device='gpu' if 'gpu' in str(lgb.LGBMRegressor()) else 'cpu',
        verbosity=-1
    )
    lgb_model.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)])
    oof_lgb[idx_va] = lgb_model.predict(X_va_enc)
    pred_lgb += lgb_model.predict(X_test_enc) / kf.n_splits

    # --- Fold Metrics ---
    def rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

    metrics['cat'].append(rmse(y_va, oof_cat[idx_va]))
    metrics['xgb'].append(rmse(y_va, oof_xgb[idx_va]))
    metrics['lgb'].append(rmse(y_va, oof_lgb[idx_va]))

    ens_fold = 0.4 * oof_cat[idx_va] + 0.3 * oof_xgb[idx_va] + 0.3 * oof_lgb[idx_va]
    metrics['ens'].append(rmse(y_va, ens_fold))

    print(f"Fold {fold+1} RMSE → Cat: {metrics['cat'][-1]:.6f} | XGB: {metrics['xgb'][-1]:.6f} | LGB: {metrics['lgb'][-1]:.6f} | ENS: {metrics['ens'][-1]:.6f}")

# -------------------------------
# 6. Final CV Scores
# -------------------------------
print("\n" + "="*50)
print("FINAL CV RMSE SCORES:")
print(f"CatBoost : {np.mean(metrics['cat']):.6f} ± {np.std(metrics['cat']):.6f}")
print(f"XGBoost  : {np.mean(metrics['xgb']):.6f} ± {np.std(metrics['xgb']):.6f}")
print(f"LightGBM : {np.mean(metrics['lgb']):.6f} ± {np.std(metrics['lgb']):.6f}")
print(f"ENSEMBLE : {np.mean(metrics['ens']):.6f} ± {np.std(metrics['ens']):.6f}")
print("="*50)

# -------------------------------
# 7. Final Ensemble Prediction
# -------------------------------
final_pred = 0.4 * pred_cat + 0.3 * pred_xgb + 0.3 * pred_lgb
final_pred = np.clip(final_pred, 0, 1)

# -------------------------------
# 8. Submission
# -------------------------------
submission = pd.DataFrame({'id': submission_id,'accident_risk': final_pred})
submission.to_csv('submission_cv_ensemble.csv', index=False)
print("\nSubmission saved: submission_cv_ensemble.csv")
submission.head()


# Suppress warnings
warnings.filterwarnings('ignore')
lgb.basic._log_warning = lambda *args, **kwargs: None

# =======================
# Load Data
# =======================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

y = train['accident_risk']
train = train.drop(['id', 'accident_risk'], axis=1)
test_ids = test['id']
test = test.drop('id', axis=1)

# =======================
# Encoding & Feature Engineering
# =======================
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

for col in ['road_signs_present', 'public_road', 'holiday', 'school_season']:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

train['curvature'] = pd.to_numeric(train['curvature'], errors='coerce').fillna(0)
test['curvature'] = pd.to_numeric(test['curvature'], errors='coerce').fillna(0)
train['num_lanes'] = pd.to_numeric(train['num_lanes'], errors='coerce').fillna(2)
test['num_lanes'] = pd.to_numeric(test['num_lanes'], errors='coerce').fillna(2)

# Derived features
train['speed_curvature'] = train['speed_limit'] * train['curvature']
test['speed_curvature'] = test['speed_limit'] * test['curvature']

train['lane_curvature'] = train['num_lanes'] * train['curvature']
test['lane_curvature'] = test['num_lanes'] * test['curvature']

train['risky_road'] = ((train['road_type'] == 2) & (train['speed_limit'] > 50) & (train['curvature'] > 0.5)).astype(int)
test['risky_road'] = ((test['road_type'] == 2) & (test['speed_limit'] > 50) & (test['curvature'] > 0.5)).astype(int)

train['bad_conditions'] = ((train['lighting'] != 0) | (train['weather'] != 0)).astype(int)
test['bad_conditions'] = ((test['lighting'] != 0) | (test['weather'] != 0)).astype(int)

train['peak_hour'] = train['time_of_day'].isin([0, 3]).astype(int)
test['peak_hour'] = test['time_of_day'].isin([0, 3]).astype(int)

train['accident_per_lane'] = train['num_reported_accidents'] / (train['num_lanes'] + 1)
test['accident_per_lane'] = test['num_reported_accidents'] / (test['num_lanes'] + 1)

# Scaling
scaler = StandardScaler()
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
            'speed_curvature', 'lane_curvature', 'accident_per_lane']
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

X = train.copy()
X_test = test.copy()

# =======================
# Model Training with 5-Fold CV
# =======================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds_lgb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))
val_rmse = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n===== Fold {fold+1} / {kf.n_splits} =====")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ---- LightGBM ----
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 9,
        'num_leaves': 256,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'device': 'gpu',
        'verbose': -1
    }

    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    best_iter = lgb_model.best_iteration
    val_pred = lgb_model.predict(X_val, num_iteration=best_iter)
    val_rmse.append(np.sqrt(mean_squared_error(y_val, val_pred)))
    preds_lgb += lgb_model.predict(X_test, num_iteration=best_iter) / kf.n_splits

    # ---- XGBoost ----
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        tree_method='hist',
        device='cuda'  # ✅ fixed param
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds_xgb += xgb_model.predict(X_test) / kf.n_splits

    # ---- CatBoost ----
    cat_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=9,
        l2_leaf_reg=3,
        random_state=42,
        task_type='GPU',
        devices='0',
        verbose=False
    )

    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )

    preds_cat += cat_model.predict(X_test) / kf.n_splits

# =======================
# Final Prediction
# =======================
print(f"\nCV RMSE: {np.mean(val_rmse):.6f}")

final_pred = 0.4 * preds_lgb + 0.4 * preds_xgb + 0.2 * preds_cat
final_pred = np.clip(final_pred, 0, 1)

sub = pd.DataFrame({'id': test_ids, 'accident_risk': final_pred})
sub.to_csv('submission.csv', index=False)
print("\n✅ Submission saved as 'submission.csv'")



# import numpy as np
# import pandas as pd

# file_weights = {
#     "/kaggle/input/predicting-road-accident-risk-vault/autogluon15.csv": 1.3,
#     "/kaggle/input/predicting-road-accident-risk-vault/submission.csv": 0.6,
#     "/kaggle/input/predicting-road-accident-risk-vault/submission (1).csv": 0.1,
# }

# total = sum(file_weights.values())
# weights = {path: val / total for path, val in file_weights.items()}

# def find_target_col(df):
#     if "accident_risk" in df.columns:
#         return "accident_risk"
#     num_cols = df.select_dtypes(include=np.number).columns
#     if len(num_cols) == 0:
#         raise ValueError("No numeric column found")
#     return num_cols[0]

# submissions = {}
# target_cols = {}
# predictions = {}

# for csv_path in weights:
#     df = pd.read_csv(csv_path)
#     col = find_target_col(df)
#     submissions[csv_path] = df
#     target_cols[csv_path] = col
#     predictions[csv_path] = df[col].astype(float)

# ensemble = None
# for csv_path, w in weights.items():
#     if ensemble is None:
#         ensemble = predictions[csv_path] * w
#     else:
#         ensemble += predictions[csv_path] * w

# base = next(iter(submissions.values())).copy()
# base["accident_risk"] = ensemble

# out_path = "/kaggle/working/hello_submission.csv"
# base.to_csv(out_path, index=False)
# print(f"Submission saved: {out_path}")





