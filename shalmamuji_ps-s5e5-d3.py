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


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')



# Add advanced features
def add_features(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['HR_per_BodyTemp'] = df['Heart_Rate'] / df['Body_Temp']
    df['Calories_per_kg'] = df['Duration'] * df['Heart_Rate'] / df['Weight']
    df['EnergyIndex'] = df['Duration'] * df['Heart_Rate'] * df['Body_Temp'] / df['BMI']
    df['HRxTemp'] = df['Heart_Rate'] * df['Body_Temp']
    df['Duration_per_Age'] = df['Duration'] / (df['Age'] + 1)
    return df

train = add_features(train)
test = add_features(test)


from sklearn.preprocessing import PowerTransformer, LabelEncoder

# Define target and features
target = 'Calories'
exclude = ['ID', target]

# Separate categorical and numerical features
cat_features = train.select_dtypes(include='object').columns.tolist()
num_features = [col for col in train.columns if col not in exclude + cat_features]

# Label encode categorical columns
for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Apply PowerTransformer to numerical features only
pt = PowerTransformer(method='yeo-johnson')
train[num_features] = pt.fit_transform(train[num_features])
test[num_features] = pt.transform(test[num_features])

# Recombine features
features = num_features + cat_features
X = train[features]
y = np.log1p(train['Calories'].clip(upper=train['Calories'].quantile(0.995)))
X_test = test[features]



# KFold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize OOF and prediction arrays
cat_oof = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))

xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))

lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))



# CatBoost parameters
cat_params = {
    'iterations': 4000,
    'learning_rate': 0.01,
    'depth': 8,
    'l2_leaf_reg': 10,
    'eval_metric': 'RMSE',
    'loss_function': 'RMSE',
    'verbose': False,
    'random_seed': 42
}

# Train CatBoost
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    model = CatBoostRegressor(**cat_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx], 
              eval_set=(X.iloc[val_idx], y.iloc[val_idx]), 
              use_best_model=True)
    cat_oof[val_idx] = model.predict(X.iloc[val_idx])
    cat_preds += model.predict(X_test) / kf.n_splits



from xgboost import XGBRegressor

# Initialize prediction arrays
xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))

# K-Fold training
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    model = XGBRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=100  # ✅ moved here
    )

    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        verbose=False
    )

    xgb_oof[val_idx] = model.predict(X.iloc[val_idx])
    xgb_preds += model.predict(X_test) / kf.n_splits



import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# Initialize prediction arrays
lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))

# Train LightGBM with callbacks
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    train_set = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
    val_set = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
    
    model = lgb.train(
        {
            'objective': 'rmse',
            'metric': 'rmse',
            'verbosity': -1,
            'learning_rate': 0.01,
            'num_leaves': 64,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'seed': 42
        },
        train_set,
        num_boost_round=5000,
        valid_sets=[val_set],
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(0)  # Disable evaluation logs; set to 100 if you want periodic updates
        ]
    )

    lgb_oof[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
    lgb_preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits



# Stack out-of-fold predictions as features for meta-model
stack_train = np.column_stack((xgb_oof, lgb_oof))
stack_test = np.column_stack((xgb_preds, lgb_preds))
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(stack_train, np.expm1(y))  # Use original scale of target
final_preds = ridge.predict(stack_test)



submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission['Calories'] = np.clip(final_preds, 1, 314)
submission.to_csv('submission.csv', index=False)
print("\n submission.csv saved ")

print("RMSLE (CatBoost):", np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(cat_oof))))
print("RMSLE (XGBoost):", np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(xgb_oof))))
print("RMSLE (LightGBM):", np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(lgb_oof))))


