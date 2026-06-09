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


TRAIN_PATH  = "/kaggle/input/playground-series-s5e9/train.csv"
TEST_PATH   = "/kaggle/input/playground-series-s5e9/test.csv"
SAMPLE_PATH = "/kaggle/input/playground-series-s5e9/sample_submission.csv"



train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_PATH)


print("Train shape:", train_df.shape)
print("Test  shape:", test_df.shape)
print("\nTrain columns:", train_df.columns.tolist())
print("Test  columns:",  test_df.columns.tolist())


TARGET = "BeatsPerMinute"
ID_COL = "id"


assert TARGET in train_df.columns, f"{TARGET} doesn't exist in the train_df!"
assert TARGET not in test_df.columns, "shouldn't be in the test_df!"
assert ID_COL in train_df.columns and ID_COL in test_df.columns, "id column doesn't exist!"


sample_submission.head()


assert list(sample_submission.columns) == ["id", TARGET], "Sample submission column names are different!"


import matplotlib.pyplot as plt
import seaborn as sns
print('Null values in train_df', dict(train_df.isnull().sum()))
print('Null values in test_df: ',dict(test_df.isnull().sum()))

# Histogram for target
plt.figure(figsize=(8,4))
sns.histplot(train_df['BeatsPerMinute'], bins = 50, kde = False, color = 'blue')
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Frequency')
plt.show()

# Correlation map
plt.figure(figsize=(10,6))
corr = train_df.corr(numeric_only = True)
sns.heatmap(corr, cmap = 'coolwarm', center = 0, annot = True, fmt = '.2f')
plt.title('Correlation Heatmap')
plt.show()
target_corr = corr['BeatsPerMinute'].sort_values(ascending=False)
print(target_corr)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

features = [col for col in train_df.columns if col not in ['id', 'BeatsPerMinute']]
X = train_df[features]
y = train_df['BeatsPerMinute']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_df[features])
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)

y_val_pred = lr.predict(X_val)

mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)

print(f"Baseline Linear Regression - MSE: {mse:.2f}, RMSE: {rmse:.2f}")



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators = 200,
    max_depth = None,
    random_state = 42,
    n_jobs = -1
)
rf.fit(X_train, y_train)
y_val_pred = rf.predict(X_val)

mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)

print(f"RandomForest - RMSE: {rmse:.2f}")


from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)
xgb.fit(X_train, y_train)

y_val_pred = xgb.predict(X_val)

mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)

print(f"XGBoost - RMSE: {rmse:.2f}")


from lightgbm import LGBMRegressor

lgb = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgb.fit(X_train, y_train)

y_val_pred = lgb.predict(X_val)

mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)

print(f"LightGBM - RMSE: {rmse:.2f}")


from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [300, 500, 800],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
xgb = XGBRegressor(random_state = 42, n_jobs = -1, tree_method = 'hist')
search = RandomizedSearchCV(
    xgb,
    param_distributions = param_dist,
    n_iter = 20,
    scoring = 'neg_root_mean_squared_error',
    cv = 3,
    verbose = 2,
    random_state = 42,
    n_jobs = -1
)
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best RMSE (CV):", -search.best_score_)


xgb_best = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

xgb_best.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    verbose=False,
    early_stopping_rounds=100
)

y_val_pred = xgb_best.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"XGB (early_stopping) RMSE: {rmse_xgb:.4f}")


from sklearn.linear_model import ElasticNetCV
enet = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=None,
    cv=5,
    n_jobs=-1,
    random_state=42
)
enet.fit(X_train, y_train)
y_val_pred = enet.predict(X_val)
rmse_enet = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"ElasticNetCV RMSE: {rmse_enet:.4f}")
print("Chosen l1_ratio:", enet.l1_ratio_, " alpha:", enet.alpha_)


from numpy import vstack

best_model_name = "xgb" 
best_model = enet if best_model_name=="elasticnet" else xgb_best
X_all = vstack([X_train, X_val])
y_all = np.concatenate([y_train, y_val])
best_model.fit(X_all, y_all)
test_pred = best_model.predict(X_test_scaled)
submission = pd.DataFrame({
    "id": test_df["id"],
    "BeatsPerMinute": test_pred
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv is ready.")




