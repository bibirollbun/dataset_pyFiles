!nvidia-smi


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import LabelEncoder


# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')



train.head()


X = train.drop(['Calories', 'id'], axis=1)
y = np.log1p(train['Calories'])  # log transform for RMSLE

X_test = test.drop(['id'], axis=1)


# Label Encoding (if any categorical)
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# Prepare arrays for storing predictions
lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))


# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)


for train_idx, valid_idx in kf.split(X):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # LightGBM
    lgb_model = LGBMRegressor(n_estimators=10, learning_rate=0.005, device='gpu')
    lgb_model.fit(X_train, y_train, 
                  eval_set=[(X_valid, y_valid)])
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    cat_model = CatBoostRegressor(iterations=10, learning_rate=0.005, task_type="GPU", devices='0')
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100)
    cat_preds += cat_model.predict(X_test) / kf.n_splits


# Final Prediction: simple average ensemble
final_preds = np.expm1((lgb_preds + cat_preds) / 2)  # reverse log1p
#final_preds = np.expm1(lgb_preds)  # reverse log1p


# Submission
submission['Calories'] = final_preds
submission.to_csv('submission-1.csv', index=False)

