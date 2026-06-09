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
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


import warnings
# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col="id")

print("------- Train Data -------")
print(train.info())
print(train.shape)
train.head()


print("------- Test Data -------")
print(test.info())
print(test.shape)
test.head()


# plt.figure(figsize=(10,6))
# g = sns.pairplot(data=train[:1000], hue="Sex")
# g.map_lower(sns.kdeplot, levels=4, color=".2")


encoding_dict = {"male": 1,
                "female": 2}

train["Sex_num"] = train["Sex"].map(encoding_dict)
test["Sex_num"] = test["Sex"].map(encoding_dict)


df_corr = train.drop("Sex", axis=1)
corr = df_corr.corr()
corr.style.background_gradient(cmap='coolwarm')


from sklearn.model_selection import train_test_split


train, val = train_test_split(train, test_size=0.1, random_state=42)

X_train = train.drop(["Calories", "Sex"], axis=1)
y_train = train["Calories"]

X_val = val.drop(["Calories", "Sex"], axis=1)
y_val = val["Calories"]


import xgboost as xgb

xgboost_params = {
    'eta': 0.1288,
    "n_estimators": 173,
    "max_depth": 10,
    'minchild_weight': 5,
    "random_state": 42
}
scaler_minmax = MinMaxScaler()
xgboost = xgb.XGBRegressor(**xgboost_params)
model_xgboost = make_pipeline(scaler_minmax, xgboost)
print(model_xgboost)
model_xgboost.fit(X_train, y_train)
y_pred = model_xgboost.predict(X_val)


error = np.sqrt(mean_squared_log_error(y_pred = y_pred, y_true=y_val))
print(f"RMSLE: {error:.5f}")


try:
  from bayes_opt import BayesianOptimization
except:
  %pip install bayesian-optimization
  from bayes_opt import BayesianOptimization

import xgboost as xgb

# Define optimization function
def xgb_evaluate(eta, max_depth, min_child_weight, n_estimators):
    
    model = xgb.XGBRegressor(
        eta=eta,
        max_depth=int(max_depth),
        min_child_weight=int(min_child_weight),
        n_estimators=int(n_estimators),
        # device= "cuda" if xgb.get_config()["device"] == "gpu" else "cpu",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    y_pred[y_pred < 0] = 0

    return -np.sqrt(mean_squared_log_error(y_pred=y_pred, y_true=y_val))

optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={
        'eta': (0.01, 0.3),
        'max_depth': (5, 25),
        'min_child_weight': (1, 10),
        'n_estimators': (200, 250),
    },
    random_state=42
)
optimizer.maximize(init_points=5, n_iter=20)
print("Best parameters:", optimizer.max['params'])

xgb_params = optimizer.max['params']
xgb_params['max_depth'] = int(xgb_params['max_depth'])
xgb_params['min_child_weight'] = int(xgb_params['min_child_weight'])  
xgb_params['n_estimators'] = int(xgb_params['n_estimators'])

model_xgb = xgb.XGBRegressor(**xgb_params)

model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_val)

error = np.sqrt(mean_squared_log_error(y_pred = y_pred, y_true=y_val))
print(f"RMSLE: {error:.5f}")


plt.figure(figsize=(10,6))

sns.scatterplot(x=y_val, y=y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")  


val_pred = X_val.copy()
val_pred["y_pred"] = y_pred
val_pred["y_true"] = y_val
val_pred["error"] = val_pred["y_true"] - val_pred["y_pred"]


fig, ax  = plt.subplots(2,3, figsize=(16,10))
sns.scatterplot(data=val_pred[:750], x="Height", y="error", hue="Sex_num", ax=ax[0,0])
sns.scatterplot(data=val_pred[:750], x="Weight", y="error", hue="Sex_num", ax=ax[0,1])
sns.scatterplot(data=val_pred[:750], x="Duration", y="error", hue="Sex_num", ax=ax[0,2])  
sns.scatterplot(data=val_pred[:750], x="Heart_Rate", y="error", hue="Sex_num", ax=ax[1,0])
sns.scatterplot(data=val_pred[:750], x="Body_Temp", y="error", hue="Sex_num", ax=ax[1,1])
sns.scatterplot(data=val_pred[:750], x="Age", y="error", hue="Sex_num", ax=ax[1,2])


X_test = test.drop("Sex", axis=1)
test_ids = test.index
test_pred = model_xgb.predict(X_test)
submission = pd.DataFrame({'id': test_ids, 'Calories': test_pred})
submission.to_csv('submission.csv', index=False)

