import xgboost
import lightgbm
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

dftrain = pd.DataFrame(train_data)
dftest = pd.DataFrame(test_data)

dftrain.head()


print("="*17)
print(dftrain.isnull().sum())
print("="*17)
print(dftrain.info())
print("="*17)
print(dftrain.describe())


print("="*17)
print(dftest.isnull().sum())
print("="*17)
print(dftest.info())
print("="*17)
print(dftest.describe())


cols_to_encode = ['road_type', 'lighting', 'weather', 'road_signs_present', 
                  'public_road', 'time_of_day', 'holiday', 'school_season']

le = LabelEncoder()

for i in cols_to_encode:
    dftrain[i] = le.fit_transform(dftrain[i])
    dftest[i] = le.fit_transform(dftest[i])
dftest.head()


plt.figure(figsize=(8,6))
sns.barplot(x = 'road_type', y = 'speed_limit', data = dftrain)
plt.title("Speed limit in different road types")
plt.xlabel("No of Lanes")
plt.ylabel("Road type")
plt.show()


plt.figure(figsize=(8,6))
sns.histplot(x = 'speed_limit', data = dftrain, kde = True, bins = 20)
plt.title("Distribution of speed limit")
plt.xlabel("speed limit")
plt.show()


plt.figure(figsize=(8,6))
sns.barplot(x = 'time_of_day', y = 'speed_limit', hue = 'num_reported_accidents' ,data = dftrain)
plt.title("Speed limit VS Time of day")
plt.xlabel("time_of_day")
plt.ylabel("speed_limit")
plt.show()


plt.figure(figsize=(12,6))
sns.heatmap(
    dftrain.corr(),
    annot = True,
    fmt = '.2g',
    center = 0,
    cmap = 'coolwarm'
)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.show()


dftrain.head()


X = dftrain.drop(['id','accident_risk'], axis = 1)
y = dftrain['accident_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
print(X_train.shape)
print(X_test.shape)


hgb = HistGradientBoostingRegressor(
    random_state = 42
)

hgb.fit(X_train, y_train)

print("="*17)
print("Training Results of HistGradient")
hgb_train = hgb.predict(X_train)
print("Mean Absolute Error : ", mean_absolute_error(y_train, hgb_train))
print("Mean Squared Error : ", mean_squared_error(y_train, hgb_train))
print("RMSE : ", np.sqrt(mean_squared_error(y_train, hgb_train)))
print("R2 Score : ", r2_score(y_train, hgb_train))

print("="*17)
print("Testing Results of HistGradient")
hgb_test = hgb.predict(X_test)
print("Test Mean Absolute Error : ", mean_absolute_error(y_test, hgb_test))
print("Test Mean Squared Error : ", mean_squared_error(y_test, hgb_test))
print("Test RMSE : ", np.sqrt(mean_squared_error(y_test, hgb_test)))
print("Test R2 Score : ", r2_score(y_test, hgb_test))


xgb = XGBRegressor(
    random_state = 42,
    tree_method="hist",   
    device="cuda",
    subsample = 0.9,
    reg_lambda = 0.046,
    reg_alpha = 0.046,
    n_estimators = 600,
    max_depth = 10,
    learning_rate = 0.23,
    gamma = 0.021,
    colsample_bytree = 1.0
)

xgb.fit(X_train, y_train)

print("="*17)
print("Training Results of XGBosst")
xgb_train = xgb.predict(X_train)
print("Mean Absolute Error : ", mean_absolute_error(y_train, xgb_train))
print("Mean Squared Error : ", mean_squared_error(y_train, xgb_train))
print("RMSE : ", np.sqrt(mean_squared_error(y_train, xgb_train)))
print("R2 Score : ", r2_score(y_train, xgb_train))

print("="*17)
print("Testing Results of XGBosst")
xgb_test = xgb.predict(X_test)
print("Test Mean Absolute Error : ", mean_absolute_error(y_test, xgb_test))
print("Test Mean Squared Error : ", mean_squared_error(y_test, xgb_test))
print("Test RMSE : ", np.sqrt(mean_squared_error(y_test, xgb_test)))
print("Test R2 Score : ", r2_score(y_test, xgb_test))
print("="*17)

xg_cv = cross_val_score(xgb, X_train, y_train, cv = 5, scoring = 'r2')
print("cross val score : ", xg_cv)
print("cross val mean : ", xg_cv.mean())


lgb = LGBMRegressor(
    random_state = 42,
    device="gpu",            
    gpu_platform_id=0,      
    gpu_device_id=0,
    subsample = 1.0,
    reg_lambda = 0.00359,
    reg_alpha = 0.166,
    num_leaves = 280,
    n_estimators = 300,
    min_child_samples = 5,
    max_depth = -1,
    learning_rate = 0.018,
    colsample_bytree = 0.8
)

lgb.fit(X_train, y_train)

print("="*17)
print("Training Results of LightGBM")
lgb_train = lgb.predict(X_train)
print("Mean Absolute Error : ", mean_absolute_error(y_train, lgb_train))
print("Mean Squared Error : ", mean_squared_error(y_train, lgb_train))
print("RMSE : ", np.sqrt(mean_squared_error(y_train, lgb_train)))
print("R2 Score : ", r2_score(y_train, lgb_train))

print("="*17)
print("Testing Results of LightGBM")
lgb_test = lgb.predict(X_test)
print("Test Mean Absolute Error : ", mean_absolute_error(y_test, lgb_test))
print("Test Mean Squared Error : ", mean_squared_error(y_test, lgb_test))
print("Test RMSE : ", np.sqrt(mean_squared_error(y_test, lgb_test)))
print("Test R2 Score : ", r2_score(y_test, lgb_test))
print("="*17)
lgb_cv = cross_val_score(lgb, X_train, y_train, cv = 5, scoring = 'r2')
print("cross val score : ", lgb_cv)
print("cross val mean : ", lgb_cv.mean())


dftest.head()


test_ids = dftest['id']

# select only the columns used in training
X_test = dftest[X_train.columns]

predictions = lgb.predict(X_test)

submission = pd.DataFrame({
    "id": test_ids,
    "accedint_risk": predictions
})

submission.to_csv("submission.csv", index=False)


submission

