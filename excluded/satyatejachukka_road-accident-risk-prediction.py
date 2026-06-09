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

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor

import warnings
warnings.filterwarnings('ignore')



# Load train and test data
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


print(train.info())
print("\nMissing values in train:\n", train.isnull().sum())
print(test.info())
print("\nMissing values in test:\n", test.isnull().sum())



# Target distribution
sns.histplot(train['accident_risk'], kde=True, bins=30)
plt.title('accident_risk Distribution')
plt.show()

# Numeric feature stats and correlations
num_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
print(train[num_features].describe())

corr = train[num_features + ['accident_risk']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



cat_features = [
    'road_type', 'lighting', 'weather', 
    'road_signs_present', 'public_road', 'time_of_day', 
    'holiday', 'school_season'
]

for col in cat_features:
    print(f"\n{col} value counts:")
    print(train[col].value_counts())



# Ensure boolean features are converted to int
for col in ['road_signs_present', 'public_road', 'holiday', 'school_season']:
    for df in [train, test]:
        df[col] = df[col].astype(int)

# Label encode other categorical features
label_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    le_dict[col] = le

# Standard scale numerical features
scaler = StandardScaler()
train[num_features] = scaler.fit_transform(train[num_features])
test[num_features] = scaler.transform(test[num_features])

features = label_cols + ['road_signs_present', 'public_road', 'holiday', 'school_season'] + num_features
X_train = train[features].values
y_train = train['accident_risk'].values
X_test = test[features].values



# --- Feature Engineering Cell ---
# Interaction features
train['lane_curvature'] = train['num_lanes'] * train['curvature']
test['lane_curvature'] = test['num_lanes'] * test['curvature']

train['is_high_risk_time'] = train['time_of_day'].isin(['evening', 'night']).astype(int)
test['is_high_risk_time'] = test['time_of_day'].isin(['evening', 'night']).astype(int)

# Composite categorical features
train['road_weather'] = train['road_type'].astype(str) + '_' + train['weather'].astype(str)
test['road_weather'] = test['road_type'].astype(str) + '_' + test['weather'].astype(str)
le = LabelEncoder()
train['road_weather'] = le.fit_transform(train['road_weather'])
test['road_weather'] = le.transform(test['road_weather'])

# Add new features to your feature list
features += ['lane_curvature', 'is_high_risk_time', 'road_weather']
X_train = train[features].values
X_test = test[features].values



X_train = train[features].values
X_test = test[features].values


kf = KFold(n_splits=5, shuffle=True, random_state=42)



models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=150, random_state=42, tree_method='gpu_hist'),
    'LightGBM': lgb.LGBMRegressor(n_estimators=150, random_state=42, device='gpu', gpu_platform_id=0, gpu_device_id=0)
}

scores = {}

for name, model in models.items():
    fold_scores = []
    for train_ix, val_ix in kf.split(X_train):
        X_tr, X_val = X_train[train_ix], X_train[val_ix]
        y_tr, y_val = y_train[train_ix], y_train[val_ix]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = mean_squared_error(y_val, y_pred, squared=False)  # RMSE
        fold_scores.append(score)
    avg_score = np.mean(fold_scores)
    scores[name] = avg_score
    print(f"{name} mean RMSE: {avg_score:.4f}")
print("All scores:", scores)



voting_regressor = VotingRegressor(
    estimators=[
        ('ridge', models['Ridge']),
        ('rf', models['RandomForest']),
        ('xgb', models['XGBoost']),   # GPU
        ('lgb', models['LightGBM'])   # GPU
    ]
)

ensemble_scores = []
for train_ix, val_ix in kf.split(X_train):
    X_tr, X_val = X_train[train_ix], X_train[val_ix]
    y_tr, y_val = y_train[train_ix], y_train[val_ix]
    voting_regressor.fit(X_tr, y_tr)
    y_pred = voting_regressor.predict(X_val)
    score = mean_squared_error(y_val, y_pred, squared=False)
    ensemble_scores.append(score)
ensemble_mean = np.mean(ensemble_scores)
print(f"VotingRegressor mean RMSE: {ensemble_mean:.4f}")



from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'num_leaves': randint(20, 50),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(5, 10),
    'n_estimators': randint(150, 300)
}

lgbm = lgb.LGBMRegressor(random_state=42, device='gpu')
rs = RandomizedSearchCV(
    lgbm, param_distributions=param_dist, n_iter=12, 
    scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1, random_state=42
)
rs.fit(X_train, y_train)
print("Best params:", rs.best_params_)
print("Best mean RMSE:", -rs.best_score_)
best_lgbm = rs.best_estimator_



# --- Final Model Training & Submission Cell ---
best_lgbm.fit(X_train, y_train)
y_test_pred = best_lgbm.predict(X_test)
submission = pd.DataFrame({
    "id": test["id"] if "id" in test.columns else test.index,
    "accident_risk": np.clip(y_test_pred, 0, 1)
})
submission.to_csv("submission.csv", index=False)
print(submission.head())


