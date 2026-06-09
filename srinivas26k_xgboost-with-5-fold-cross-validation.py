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


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print("Dataset shapes:", train_df.shape, test_df.shape)


def create_features(df):
    df = df.copy()
    
    # Numerical interactions
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['speed_limit_sq'] = df['speed_limit'] ** 2
    df['curvature_sq'] = df['curvature'] ** 2
    df['speed_lanes_ratio'] = df['speed_limit'] / (df['num_lanes'] + 1)
    
    # Weather risk
    weather_risk = {'clear': 0, 'rainy': 2, 'foggy': 1}
    df['weather_risk'] = df['weather'].map(weather_risk)
    
    # Lighting risk
    lighting_risk = {'daylight': 0, 'bright': 0, 'dim': 1, 'night': 2}
    df['lighting_risk'] = df['lighting'].map(lighting_risk)
    
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)


X_train = train_df.drop(['id', 'accident_risk'], axis=1)
y_train = train_df['accident_risk'].values
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id'].values

# Encode categorical variables
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Handle missing values
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

X_train = X_train.values
X_test = X_test.values

print(f"\nFeature matrix shapes: Train {X_train.shape}, Test {X_test.shape}")


print("\n" + "="*70)
print("LEVEL 1: K-FOLD XGBoost BASE MODEL")
print("="*70)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
level1_train_pred = np.zeros(len(X_train))
level1_test_pred = np.zeros((len(X_test), 5))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    print(f"\nFold {fold + 1}/5:")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Train XGBoost
    xgb1 = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0
    )
    
    xgb1.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
             callbacks=[xgb.callback.EarlyStopping(rounds=50)])
    
    # Get OOF predictions (out-of-fold)
    level1_train_pred[val_idx] = xgb1.predict(X_val)
    level1_test_pred[:, fold] = xgb1.predict(X_test)
    
    # Validation score
    val_pred = xgb1.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_pred - y_val)**2))
    print(f"  Validation RMSE: {val_rmse:.6f}")

# Average test predictions across folds
level1_test_pred = level1_test_pred.mean(axis=1)


print("\n" + "="*70)
print("CALCULATING RESIDUALS")
print("="*70)

residuals_train = y_train - level1_train_pred
residuals_test = level1_test_pred  # Will be refined in Level 2

print(f"\nResiduals Train - Min: {residuals_train.min():.6f}, Max: {residuals_train.max():.6f}")
print(f"Residuals Train - Mean: {residuals_train.mean():.6f}, Std: {residuals_train.std():.6f}")


print("LEVEL 2: K-FOLD XGBoost ON RESIDUALS (META-LEARNER)")
print("="*70)

level2_train_pred = np.zeros(len(X_train))
level2_test_pred = np.zeros((len(X_test), 5))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    print(f"\nFold {fold + 1}/5:")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    res_tr, res_val = residuals_train[train_idx], residuals_train[val_idx]
    
    # Train XGBoost on residuals
    xgb2 = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0
    )
    
    xgb2.fit(X_tr, res_tr, eval_set=[(X_val, res_val)],
             callbacks=[xgb.callback.EarlyStopping(rounds=50)])
    
    # Get OOF predictions
    level2_train_pred[val_idx] = xgb2.predict(X_val)
    level2_test_pred[:, fold] = xgb2.predict(X_test)
    
    # Validation score
    residual_pred = xgb2.predict(X_val)
    residual_rmse = np.sqrt(np.mean((residual_pred - res_val)**2))
    print(f"  Residual RMSE: {residual_rmse:.6f}")

# Average test predictions across folds
level2_test_pred = level2_test_pred.mean(axis=1)


print("\n" + "="*70)
print("FINAL PREDICTION: Level 1 + Level 2 (Stacking)")
print("="*70)

# Final prediction = Level 1 + Level 2 (residual correction)
final_test_pred = level1_test_pred + level2_test_pred

# Clip to valid range [0, 1]
final_test_pred = np.clip(final_test_pred, 0, 1)

print(f"\nFinal Ensemble Predictions:")
print(f"  Min: {final_test_pred.min():.6f}")
print(f"  Max: {final_test_pred.max():.6f}")
print(f"  Mean: {final_test_pred.mean():.6f}")
print(f"  Std: {final_test_pred.std():.6f}")


train_oof_pred = level1_train_pred + level2_train_pred
train_oof_pred = np.clip(train_oof_pred, 0, 1)
cv_rmse = np.sqrt(np.mean((train_oof_pred - y_train)**2))
print(f"\n" + "="*70)
print(f"CROSS-VALIDATION RMSE (OOF): {cv_rmse:.6f}")
print("="*70)


submission_df = pd.DataFrame({
    'id': test_ids,
    'accident_risk': final_test_pred
})

submission_df.to_csv('submission.csv', index=False)
print("\n✓ Submission saved to 'submission.csv'")
print(f"\nSubmission Preview:")
print(submission_df.head(10))
print(f"\nTotal rows: {len(submission_df)}")

