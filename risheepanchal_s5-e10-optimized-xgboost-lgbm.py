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


#system handling
import os
import time
import warnings
warnings.filterwarnings('ignore')

#data handling
import numpy as np # linear algebra
import pandas as pd # data processing, 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print('done')


train =pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


CATEGORICAL_FEATURES = ['road_type', 'lighting', 'weather', 'time_of_day']
BOOLEAN_FEATURES = ['road_signs_present', 'public_road', 'holiday', 'school_season']
NUMERICAL_FEATURES = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
TARGET = 'accident_risk'
ID_COL = 'id'


def engineer_features(df):
    """
    Create domain-informed feature interactions.
    """
    df_eng = df.copy()
    
    # Core interactions
    df_eng['curv_speed'] = df_eng['curvature'] * df_eng['speed_limit']
    df_eng['lane_speed'] = df_eng['num_lanes'] * df_eng['speed_limit']
    df_eng['accidents_speed'] = df_eng['num_reported_accidents'] * df_eng['speed_limit']
    df_eng['accidents_curv'] = df_eng['num_reported_accidents'] * df_eng['curvature']
    
    # Polynomial features
    df_eng['curvature_sq'] = df_eng['curvature'] ** 2
    df_eng['curvature_cube'] = df_eng['curvature'] ** 3
    df_eng['speed_sq'] = df_eng['speed_limit'] ** 2
    # Risk scores
    df_eng['risk_intensity'] = (df_eng['curvature'] * df_eng['speed_limit']) / 50
    df_eng['lane_capacity_risk'] = (5 - df_eng['num_lanes']) * df_eng['speed_limit']
    df_eng['accidents_per_lane'] = df_eng['num_reported_accidents'] / (df_eng['num_lanes'] + 1)
    
    # Binary indicators
    df_eng['high_risk_combo'] = ((df_eng['curvature'] > 0.5) & 
                                  (df_eng['speed_limit'] >= 60)).astype(int)
    
    return df_eng

train_processed = train.copy()
test_processed = test.copy()

# Convert booleans
for col in BOOLEAN_FEATURES:
    train_processed[col] = train_processed[col].astype(int)
    test_processed[col] = test_processed[col].astype(int)

# Label encode categoricals
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    train_processed[f'{col}_enc'] = le.fit_transform(train_processed[col])
    test_processed[f'{col}_enc'] = le.transform(test_processed[col])
    label_encoders[col] = le

# Apply feature engineering
train_engineered = engineer_features(train_processed)
test_engineered = engineer_features(test_processed)

print(f"Feature engineering complete")
print(f"Original features: {len(CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERICAL_FEATURES)}")
print(f"Engineered features: {train_engineered.shape[1]}")
print(f"New features created: {train_engineered.shape[1] - train_processed.shape[1]}")


# Prepare feature matrix
exclude_cols = [ID_COL, TARGET] + CATEGORICAL_FEATURES
feature_cols = [col for col in train_engineered.columns if col not in exclude_cols]

X_train = train_engineered[feature_cols].values
y_train = train_engineered[TARGET].values
X_test = test_engineered[feature_cols].values

print(f"Training matrix: {X_train.shape}")
print(f"Test matrix: {X_test.shape}")


N_SPLITS = 7
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# ==========================
# Storage for OOF predictions
# ==========================
oof_xgb1 = np.zeros(len(X_train))   # Optimized XGB
oof_xgb2 = np.zeros(len(X_train))   # Default XGB
oof_lgb  = np.zeros(len(X_train))   # LightGBM
test_xgb1 = np.zeros(len(X_test))
test_xgb2 = np.zeros(len(X_test))
test_lgb  = np.zeros(len(X_test))
scores_xgb1, scores_xgb2, scores_lgb = [], [], []
models_xgb1, models_xgb2, models_lgb = [], [], []


from lightgbm import early_stopping, log_evaluation


# ==========================
# Optimized XGBoost (Model 1)
# ==========================
xgb1_params = {
    'n_estimators': 8000,
    'learning_rate': 0.010,
    'max_depth': 9,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

print("Training Optimized XGBoost with GPU")
print("=" * 60)
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_SPLITS}", end=" ")

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model = xgb.XGBRegressor(**xgb1_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        early_stopping_rounds=200,
        verbose=False
    )
    oof_xgb1[val_idx] = model.predict(X_fold_val)
    test_xgb1 += model.predict(X_test) / N_SPLITS

    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_xgb1[val_idx]))
    scores_xgb1.append(fold_rmse)
    models_xgb1.append(model)

    print(f"RMSE: {fold_rmse:.6f} | Best iter: {model.best_iteration}")

xgb1_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_xgb1))
print(f"\nOptimized XGBoost OOF RMSE: {xgb1_oof_rmse:.6f}")
print(f"CV Std: {np.std(scores_xgb1):.6f}")
print("=" * 60)

xgb2_params = {
    'n_estimators': 6000,
    'learning_rate': 0.012,
    'max_depth': 8,
    'min_child_weight': 2,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

print("Training Default XGBoost with GPU")
print("=" * 60)
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_SPLITS}", end=" ")

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model = xgb.XGBRegressor(**xgb2_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        early_stopping_rounds=200,
        verbose=False
    )

    oof_xgb2[val_idx] = model.predict(X_fold_val)
    test_xgb2 += model.predict(X_test) / N_SPLITS

    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_xgb2[val_idx]))
    scores_xgb2.append(fold_rmse)
    models_xgb2.append(model)

    print(f"RMSE: {fold_rmse:.6f} | Best iter: {model.best_iteration}")

xgb2_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_xgb2))
print(f"\nDefault XGBoost OOF RMSE: {xgb2_oof_rmse:.6f}")
print(f"CV Std: {np.std(scores_xgb2):.6f}")
print("=" * 60)

lgb_params = {
    'n_estimators': 7000,
    'learning_rate': 0.012,
    'max_depth': -1,
    'num_leaves': 255,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'objective': 'regression',
    'metric': 'rmse',
    'device': 'gpu'
}
print("Training LightGBM with GPU")
print("=" * 60)
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_SPLITS}", end=" ")

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        callbacks=[early_stopping(stopping_rounds=200), log_evaluation(0)]
    )

    oof_lgb[val_idx] = model.predict(X_fold_val)
    test_lgb += model.predict(X_test) / N_SPLITS

    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_lgb[val_idx]))
    scores_lgb.append(fold_rmse)
    models_lgb.append(model)

    print(f"RMSE: {fold_rmse:.6f}")

lgb_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_lgb))
print(f"\nLightGBM OOF RMSE: {lgb_oof_rmse:.6f}")
print(f"CV Std: {np.std(scores_lgb):.6f}")
print("=" * 60)

best_rmse = 1e9
best_w = None
best_oof, best_test = None, None

for w1 in np.linspace(0, 1, 21):   # XGB1
    for w2 in np.linspace(0, 1-w1, 21):  # XGB2
        w3 = 1 - (w1 + w2)               # LGB
        if w3 < 0:
            continue

        oof_ens = w1 * oof_xgb1 + w2 * oof_xgb2 + w3 * oof_lgb
        test_ens = w1 * test_xgb1 + w2 * test_xgb2 + w3 * test_lgb

        rmse = np.sqrt(mean_squared_error(y_train, oof_ens))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = (w1, w2, w3)
            best_oof, best_test = oof_ens, test_ens

print("=" * 60)
print(f"Best Weights Found -> XGB1: {best_w[0]:.2f}, XGB2: {best_w[1]:.2f}, LGB: {best_w[2]:.2f}")
print(f"Best Weighted Ensemble OOF RMSE: {best_rmse:.6f}")
print("=" * 60)

submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': best_test
})

# Validation checks
assert submission.shape[0] == test.shape[0]
assert submission['accident_risk'].isna().sum() == 0
assert (submission['accident_risk'] >= 0).all()
assert (submission['accident_risk'] <= 1).all()

submission.to_csv('/kaggle/working/submission.csv', index=False)

print("âœ… Submission Created Successfully")
print("=" * 60)
print(f"Shape: {submission.shape}")
print(f"Prediction Mean: {submission['accident_risk'].mean():.4f}")
print(f"Prediction Std: {submission['accident_risk'].std():.4f}")
print(f"Prediction Min: {submission['accident_risk'].min():.4f}")
print(f"Prediction Max: {submission['accident_risk'].max():.4f}")
print("\nFirst 10 predictions:")
print(submission.head(10))




