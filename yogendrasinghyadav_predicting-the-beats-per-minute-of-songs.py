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


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



Train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
Test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")



print('Train shape:',Train.shape)
print('Train info:',Train.info())
print('Test shape:',Test.shape)
print('Test info:',Test.info())


Train.head()


Test.head()


Test.describe()


Train.describe()


# droping the columns with ID from both train and test datasets
train = Train.drop(['id'], axis=1)
test = Test.drop(['id'], axis=1)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1) separate features and target
features = train.iloc[:, :-1]
predictions = train.iloc[:, -1] # (Last column contains the target variable)

# 2) split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, predictions, test_size=0.3, random_state=42)

# 3) scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test = scaler.transform(test)

# 4) convert scaled arrays back to DataFrames (optional, but useful for inspection)
X_train = pd.DataFrame(X_train, columns=features.columns)
X_val = pd.DataFrame(X_val, columns=features.columns)
test = pd.DataFrame(test, columns=features.columns)

print("Scaled shape of train data:", X_train.shape)
print("Scaled shape of validation data:", X_val.shape)
print("Scaled shape of test data:", test.shape)


print('scaled X_train description:',X_train.describe())
print('scaled X_val description:',X_val.describe())


st_train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
st_train.head()


plt.figure(figsize=(14, 8))
X_train.boxplot()
plt.title("Box Plot of x_train Features and Target")
plt.xticks(rotation=45)
plt.show()


import seaborn as sns
cor_matrix = st_train.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap='coolwarm')



import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
rf = RandomForestRegressor(   
    n_estimators=200,        # start with 100
    max_depth=15,            # limit depth (None = unlimited)
    max_features='sqrt',     # common choice
    min_samples_leaf=6,
    n_jobs=-1,               # use all cores
    random_state=42,
    verbose=0,
    criterion='squared_error')
rf.fit(X_train, y_train)
pre = rf.predict(X_val)

t0 = time.time()
rf.fit(X_train, y_train)
t1 = time.time()
print("Train time (s):", t1 - t0)

mse = mean_squared_error(y_val, pre)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, pre)

# Print the evaluation metrics
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2 ): {r2}")



# Feature importance
import pandas as pd
feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feat_imp.head(20))


# Hyperparameter tuning — RandomizedSearchCV
'''
from sklearn.model_selection import RandomizedSearchCV
param_dist = {
 'n_estimators': [50,100,200,400],
 'max_depth': [6,10,15,25],
 'min_samples_split': [2,5,10],
 'min_samples_leaf': [1,2,4,8],
 'max_features': ['sqrt','log2', 0.2, 0.5],
 'bootstrap': [True, False]
}
rfr = RandomForestRegressor(random_state=42, n_jobs=-1)
rs = RandomizedSearchCV(rfr, param_distributions=param_dist,
                        n_iter=30, scoring='neg_mean_squared_error',
                        cv=3, verbose=2, n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)
print("Best params:", rs.best_params_)
best_rf = rs.best_estimator_
'''


# Hyperparameter tuned randomforest model
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
rf = RandomForestRegressor(   
    n_estimators=100,        # start with 100
    max_depth=6,            # limit depth (None = unlimited)
    max_features='log2',     # common choice
    min_samples_leaf=1,
    min_samples_split=2,
    n_jobs=-1,               # use all cores
    random_state=42,
    verbose=0,
    criterion='squared_error')
rf.fit(X_train, y_train)
pre = rf.predict(X_val)

t0 = time.time()
rf.fit(X_train, y_train)
t1 = time.time()
print("Train time (s):", t1 - t0)

mse = mean_squared_error(y_val, pre)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, pre)

# Print the evaluation metrics
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2 ): {r2}")



pip install xgboost



from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=6,
                   subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
xgb.fit(X_train, y_train)

pre = xgb.predict(X_val)

t0 = time.time()
rf.fit(X_train, y_train)
t1 = time.time()
print("Train time (s):", t1 - t0)

mse = mean_squared_error(y_val, pre)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, pre)

# Print the evaluation metrics
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2 ): {r2}")



# Hyperparameter tuning — RandomizedSearchCV
'''
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the model
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# Parameter grid (based on Random Forest results + XGB extras)
param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 1, 5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 5, 10]
}

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit to training data
random_search.fit(X_train, y_train)

# Print results
print("Best parameters found: ", random_search.best_params_)

'''


# Hyperparameter tuned XGBoost model
from xgboost import XGBRegressor
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.01, 
    max_depth=4,
    subsample=0.6, 
    colsample_bytree=0.7,
    reg_lambda=1,
    reg_alpha=0,
    min_child_weight=5, 
    n_jobs=-1, 
    random_state=42)
xgb.fit(X_train, y_train)

pre = xgb.predict(X_val)

t0 = time.time()
rf.fit(X_train, y_train)
t1 = time.time()
print("Train time (s):", t1 - t0)

mse = mean_squared_error(y_val, pre)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, pre)

# Print the evaluation metrics
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2 ): {r2}")



predict = rf.predict(test)
submission = pd.DataFrame({
    "id": Test["id"],
    "target": predict
})


submission.to_csv("submission.csv", index=False)
submission.head()

