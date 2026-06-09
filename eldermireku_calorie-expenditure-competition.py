import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Feature Engineering
def create_features(df):
    # Basic encoding
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0}).astype(int)
    
    # Biometric features
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['BSA'] = 0.007184 * (df['Height'] ** 0.725) * (df['Weight'] ** 0.425)
    
    # Exercise intensity features
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    df['Cardio_Load'] = df['Heart_Rate'] * df['Duration'] / (220 - df['Age'])
    df['Power_Output'] = df['Weight'] / (df['Duration'] * df['Age'] ** 0.5)
    
    # Interaction features
    df['Male'] = df['Sex']
    df['Female'] = 1 - df['Sex']
    for feat in ['Duration', 'Heart_Rate', 'Body_Temp', 'Age']:
        df[f'{feat}_x_Male'] = df[feat] * df['Male']
        df[f'{feat}_x_Female'] = df[feat] * df['Female']
    
    # Polynomial features
    for col in ['Height', 'Weight', 'Age', 'Duration', 'Heart_Rate']:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(df[col])
    
    df.drop(['Male', 'Female'], axis=1, inplace=True)
    return df

train = create_features(train)
test = create_features(test)

# Target transformation
y = np.log1p(train['Calories'])
X = train.drop(['Calories'], axis=1)
test_data = test.copy()

# Create bins for stratified CV
bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile', subsample=None)
duration_bins = bins.fit_transform(train[['Duration']]).astype(int).flatten()

# Model configurations (CPU-only versions)
catboost_params = {
    'iterations': 4000,
    'learning_rate': 0.015,
    'depth': 10,
    'loss_function': 'RMSE',
    'l2_leaf_reg': 5,
    'random_seed': 42,
    'early_stopping_rounds': 200,
    'verbose': False
}

xgb_params = {
    'max_depth': 8,
    'colsample_bytree': 0.8,
    'subsample': 0.9,
    'n_estimators': 3500,
    'learning_rate': 0.01,
    'gamma': 0.1,
    'eval_metric': 'rmse',
    'random_state': 42,
    'early_stopping_rounds': 100
}

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 127,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_estimators': 3000,
    'verbose': -1  # LightGBM verbosity control here instead of in fit()
}

# Training with Cross-Validation
test_preds = []
val_scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_bins)):
    print(f"\nFold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train CatBoost
    cat_model = CatBoostRegressor(**catboost_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    cat_preds = np.expm1(cat_model.predict(X_val))
    
    # Train XGBoost
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = np.expm1(xgb_model.predict(X_val))
    
    # Train LightGBM (verbose parameter removed from fit())
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse'
    )
    lgb_preds = np.expm1(lgb_model.predict(X_val))
    
    # Ensemble predictions (weighted average)
    val_preds = (0.4 * cat_preds + 0.3 * xgb_preds + 0.3 * lgb_preds)
    fold_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_preds))
    val_scores.append(fold_rmse)
    print(f'Fold {fold+1} RMSLE: {fold_rmse:.5f}')
    
    # Get test predictions
    cat_test = np.expm1(cat_model.predict(test_data))
    xgb_test = np.expm1(xgb_model.predict(test_data))
    lgb_test = np.expm1(lgb_model.predict(test_data))
    test_preds.append(0.4 * cat_test + 0.3 * xgb_test + 0.3 * lgb_test)

print(f'\nAverage Validation RMSLE: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f}')

# Ensemble predictions
final_preds = np.mean(test_preds, axis=0)

# Post-processing
train_q1 = np.expm1(y).quantile(0.01)
train_q3 = np.expm1(y).quantile(0.99)
final_preds = np.clip(final_preds, train_q1, train_q3)

# Adjust scale to match training distribution
target_median = np.expm1(y).median()
final_preds = final_preds * (target_median / np.median(final_preds))

# Submission
submission['Calories'] = final_preds
submission.to_csv('submission.csv', index=False)

