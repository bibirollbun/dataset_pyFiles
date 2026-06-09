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
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


path1 = '/kaggle/input/playground-series-s5e5/train.csv'
path2 = '/kaggle/input/playground-series-s5e5/test.csv'

train = pd.read_csv(path1)

test = pd.read_csv(path2)


from ydata_profiling import ProfileReport

report = ProfileReport(train,title='Data Report')

report


train['BMI'] = train['Weight'] / (train['Height'] / 100) ** 2
test['BMI'] = test['Weight'] / (test['Height'] / 100) ** 2

def heart_rate_zone(hr):
    if hr < 100:
        return 0
    elif hr < 130:
        return 1
    elif hr < 160:
        return 2
    else:
        return 3

train['HR_Zone'] = train['Heart_Rate'].apply(heart_rate_zone)
test['HR_Zone'] = test['Heart_Rate'].apply(heart_rate_zone)

train['Age_Duration'] = train['Age'] * train['Duration']
test['Age_Duration'] = test['Age'] * test['Duration']

train['AgeGroup'] = pd.cut(train['Age'], bins=[0, 18, 30, 45, 60, 100], labels=False)
test['AgeGroup'] = pd.cut(test['Age'], bins=[0, 18, 30, 45, 60, 100], labels=False)

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1}).fillna(-1)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1}).fillna(-1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train['Body_Temp_scaled'] = scaler.fit_transform(train[['Body_Temp']])
test['Body_Temp_scaled'] = scaler.transform(test[['Body_Temp']])

train['Temp_Zone'] = pd.cut(train['Body_Temp'], bins=[35, 36.5, 37.5, 38.5, 40], labels=[0, 1, 2, 3]).astype(float)
test['Temp_Zone'] = pd.cut(test['Body_Temp'], bins=[35, 36.5, 37.5, 38.5, 40], labels=[0, 1, 2, 3]).astype(float)

train['Temp_HR'] = train['Body_Temp'] * train['Heart_Rate']
test['Temp_HR'] = test['Body_Temp'] * test['Heart_Rate']

train['Temp_Duration'] = train['Body_Temp'] * train['Duration']
test['Temp_Duration'] = test['Body_Temp'] * test['Duration']

train['Temp_Deviation'] = train['Body_Temp'] - 37.0
test['Temp_Deviation'] = test['Body_Temp'] - 37.0


train.info()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Assume train, test are already loaded DataFrames
target = 'Calories'
drop_cols = ['id', 'Calories']
features = [col for col in train.columns if col not in drop_cols]
X = train[features]
y = train[target]
X_test = test[features]

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'random_state': 42
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_models = 4
oof_preds = np.zeros((len(X), n_models))
test_preds = np.zeros((len(X_test), n_models))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.train(
        params,
        lgb.Dataset(X_tr, y_tr),
        valid_sets=[lgb.Dataset(X_val, y_val)],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )
    oof_preds[val_idx, 0] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_preds[:, 0] += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / kf.n_splits

    # # Random Forest
    # rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=fold)
    # rf_model.fit(X_tr, y_tr)
    # oof_preds[val_idx, 1] = rf_model.predict(X_val)
    # test_preds[:, 1] += rf_model.predict(X_test) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, verbosity=0,
                                 random_state=fold)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    oof_preds[val_idx, 2] = xgb_model.predict(X_val)
    test_preds[:, 2] += xgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    cat_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, verbose=0, random_seed=fold)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
    oof_preds[val_idx, 3] = cat_model.predict(X_val)
    test_preds[:, 3] += cat_model.predict(X_test) / kf.n_splits

# Train meta-model on out-of-fold predictions
meta_model = RidgeCV()
meta_model.fit(oof_preds, y)
final_test_preds = meta_model.predict(test_preds)

# Evaluate
rmse = mean_squared_error(y, meta_model.predict(oof_preds), squared=False)
print(f"\nStacked Model CV RMSE: {rmse:.4f}")


lgb_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_model.feature_importance()
})
xgb_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})
imp_df = lgb_imp.merge(xgb_imp, on='Feature', suffixes=('_lgb', '_xgb'))
imp_df['Avg_Importance'] = (imp_df['Importance_lgb'] + imp_df['Importance_xgb']) / 2
imp_df = imp_df.sort_values('Avg_Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df.head(15), x='Avg_Importance', y='Feature')
plt.title("Top 15 Feature Importances (Avg. LGB + XGB)")
plt.tight_layout()
plt.show()


final_preds = test_preds.mean(axis=1)
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': final_preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv is ready!")

