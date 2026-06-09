# Imports
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
import warnings
warnings.filterwarnings("ignore")

# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Log-transform target
train['Log_Calories'] = np.log1p(train['Calories'])
X = train.drop(columns=['id', 'Calories', 'Log_Calories'])
y = train['Log_Calories']
X_test = test.drop(columns=['id'])

# Label Encoding for categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col], _ = pd.factorize(X[col])
        X_test[col], _ = pd.factorize(X_test[col])



# BMI Feature
X['BMI'] = X['Weight'] / (X['Height']/100)**2
X_test['BMI'] = X_test['Weight'] / (X_test['Height']/100)**2

# Cross Features
X['Age_Duration'] = X['Age'] * X['Duration']
X_test['Age_Duration'] = X_test['Age'] * X_test['Duration']

# Log Transformations
X['Log_Duration'] = np.log1p(X['Duration'])
X_test['Log_Duration'] = np.log1p(X_test['Duration'])

# Ratio Features
X['Weight_Height_Ratio'] = X['Weight'] / X['Height']
X_test['Weight_Height_Ratio'] = X_test['Weight'] / X_test['Height']


# LightGBM Parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# XGBoost Parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# CatBoost Parameters
cat_params = {
    'loss_function': 'RMSE',
    'learning_rate': 0.05,
    'depth': 6,
    'random_seed': 42,
    'verbose': 0
}


# Initialize models
lgb_model = lgb.LGBMRegressor(**lgb_params)
xgb_model = xgb.XGBRegressor(**xgb_params)
cat_model = cb.CatBoostRegressor(**cat_params)

# Ensemble
ensemble = VotingRegressor(
    estimators=[
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('cat', cat_model)
    ],
    weights=[0.4, 0.3, 0.3]
)

# Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_val)
    preds = np.maximum(0, np.expm1(preds))
    y_val_exp = np.expm1(y_val)
    rmse = np.sqrt(mean_squared_error(y_val_exp, preds))
    cv_scores.append(rmse)

print(f"Average CV RMSE: {np.mean(cv_scores):.5f}")


# Train on full data
ensemble.fit(X, y)
final_preds = ensemble.predict(X_test)
final_preds = np.maximum(0, np.expm1(final_preds))

# Prepare submission
submission['Calories'] = final_preds
submission.to_csv('submission.csv', index=False)
print("✅ Submission file created: submission.csv")

