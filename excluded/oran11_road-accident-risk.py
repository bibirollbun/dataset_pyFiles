import warnings 

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
import seaborn as sns
import polars as pl
import os 
import shutil

#Suppress warnings
warnings.simplefilter("ignore")

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.decomposition import PCA

from xgboost import XGBRegressor
from catboost import CatBoostRegressor



from tensorflow.random import set_seed
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.regularizers import L2


DATA_PATH = "/kaggle/input/playground-series-s5e10"
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv")).drop("id", axis = 1)
original_data = pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv")).drop("id", axis = 1)
sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
target_col = "accident_risk"


train


original_data


test


train.info()


original_data.info()


test.info()


train.select_dtypes(include = [object]).columns


values_same = lambda series1, series2: set(series1.values) == set(series2.values) # lambda function to test if the values are same

cols = test.columns.to_list() # columns except "accident_risk"
for col in cols:
    print(f"{col} \n  Unique values inside train set: {train[col].nunique()} \n  Unique values inside test set: {test[col].nunique()}")
    print(f" * Does both dataset have same values? {values_same(train[col], test[col])} \n")


# Select categorical columns
cat_cols = train.select_dtypes(include = [object]).columns.to_list() + ["num_lanes", "speed_limit", "road_signs_present", "public_road", "holiday", "school_season"]
# Select continuous columns
con_cols = [col for col in test.columns if col not in cat_cols]


train.describe()


test.describe()


plt.hist(train[target_col], bins = 75)
plt.xlabel("Risk likelihood")
plt.ylabel("Counts")
plt.show()


#Create subplots
fig, ax = plt.subplots(len(cat_cols), 2, figsize = (15, 50))
#ax = ax.flatten() # 1D array of axes

# Generate plots
for fold, col_n in enumerate(cat_cols):
    sns.boxplot(data = train, x = col_n, y = target_col, ax = ax[fold, 0])
    sns.violinplot(data = train , x = col_n, y = target_col, ax = ax[fold, 1])
    #sns.swarmplot(data = train, x = col_n, y = target_col, ax = ax[fold, 2]) Change column size arguments in subplots => 3

plt.tight_layout()
plt.show()


indices_to_remove = train.loc[(train[target_col] == 0) | (train[target_col] == 1)].index
train.drop(indices_to_remove, inplace = True)


train["curvature_x_speed_limit"] = train["curvature"] * train["speed_limit"] 
train["curvature**2"] = train["curvature"] ** 2
train["curvature**3"] = train["curvature"] ** 3
train["num_reported_accidents**2"] = train["num_reported_accidents"] ** 2
train["num_reported_accidents**3"] = train["num_reported_accidents"] ** 3
train["lighting_x_weather"] = train["lighting"].map({"night": 3, "dim": 1.05, "daylight": 1}) * train["weather"].map({"rainy": 2, "foggy": 2.5, "clear": 1}) # I gave more importance on fogginess (You can refer the plots)
# I gave dim a slight more importance to test it out
train["lighting_x_weather**2"] = train["lighting_x_weather"] ** 2
train["lighting_x_weather**3"] = train["lighting_x_weather"] ** 3

test["curvature_x_speed_limit"] = test["curvature"] * test["speed_limit"] 
test["curvature**2"] = test["curvature"] ** 2
test["curvature**3"] = test["curvature"] ** 3
test["num_reported_accidents**2"] = test["num_reported_accidents"] ** 2
test["num_reported_accidents**3"] = test["num_reported_accidents"] ** 3
test["lighting_x_weather"] = test["lighting"].map({"night": 3, "dim": 1.05, "daylight": 1}) * test["weather"].map({"rainy": 2, "foggy": 2.5, "cleartest": 1})
test["lighting_x_weather**2"] = test["lighting_x_weather"] ** 2
test["lighting_x_weather**3"] = test["lighting_x_weather"] ** 3


X, y = train.drop(target_col, axis = 1), train[target_col]


X_train, X_, y_train, y_ = train_test_split(X, y, test_size = 0.4, shuffle = True, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size = 0.5, shuffle = True, random_state = 42)


encoder = OrdinalEncoder()
scaler = StandardScaler()


X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_train[con_cols] = scaler.fit_transform(X_train[con_cols])

X_val[cat_cols] = encoder.transform(X_val[cat_cols])
X_val[con_cols] = scaler.transform(X_val[con_cols])

X_test[cat_cols] = encoder.transform(X_test[cat_cols])
X_test[con_cols] = scaler.transform(X_test[con_cols])


models = [
    ("Linear Regression", LinearRegression()),
    ("SGDRegressor - 0.1 - Huber", SGDRegressor(eta0 = 0.1, loss = "huber", random_state = 42)),
    ("SGDRegressor - 0.01 - Huber", SGDRegressor(eta0 = 0.01, loss = "huber", random_state = 42)),
    ("SGDRegressor - 0.001 - Huber", SGDRegressor(eta0 = 0.001, loss = "huber", random_state = 42)),
    ("SGDRegressor - 0.0001 - Huber", SGDRegressor(eta0 = 0.0001, loss = "huber", random_state = 42)),
    #("SGDRegressor - 0.1 - Squared Error", SGDRegressor(eta0 = 0.1, random_state = 42)), Does not converge properly
    #("SGDRegressor - 0.01 - Squared Error", SGDRegressor(eta0 = 0.01, random_state = 42)), Does not converge properly
    #("SGDRegressor - 0.001 -  Squared Error", SGDRegressor(eta0 = 0.001, random_state = 42)), Does not converge properly
    #("SGDRegressor - 0.0001 -  Squared Error", SGDRegressor(eta0 = 0.0001, random_state = 42)), Does not converge properly

    ("XGB - 0.1 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.1)),
    ("XGB - 0.13 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.13)),
    ("XGB - 0.16 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.16)),
    ("XGB - 0.2 - Sqaared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.2)),
    ("XGB - 0.01 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.01)),
    ("XGB - 0.001 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.001)),
    ("XGB - 0.0001 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.0001)),
    
    ("XGB - 0.02 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.02)),
    ("XGB - 0.03 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.03)),
    ("XGB - 0.05 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.05)),
    ("XGB - 0.06 - Squared Error", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.06)),
    ("XGB - 0.05 - Squared Error - Depth: 7", XGBRegressor(random_state = 42, verbose = 0, learning_rate = 0.05, max_depth = 7)),
    
    ("CAT- 0.6 - Root Mean Squared Error", CatBoostRegressor(random_state = 42, verbose = 0, allow_writing_files = False, learning_rate = 0.6)),
    ("CAT- 0.3 - Root Mean Squared Error", CatBoostRegressor(random_state = 42, verbose = 0, allow_writing_files = False, learning_rate = 0.3)),
    ("CAT- 0.1 - Root Mean Squared Error", CatBoostRegressor(random_state = 42, verbose = 0, allow_writing_files = False, learning_rate = 0.1)),
    ("CAT- 0.09 - Root Mean Squared Error", CatBoostRegressor(random_state = 42, verbose = 0, allow_writing_files = False, learning_rate = 0.09)),
    ("CAT- 0.06 - Root Mean Squared Error", CatBoostRegressor(random_state = 42, verbose = 0, allow_writing_files = False, learning_rate = 0.06)),
    ("CAT- 0.03 - Root Mean Squared Error", CatBoostRegressor(random_state = 42, verbose = 0, allow_writing_files = False, learning_rate = 0.03)),
    ("CAT- 0.01 - Root Mean Squared Error", CatBoostRegressor(random_state = 42, verbose = 0, allow_writing_files = False, learning_rate = 0.01)),
    ("CAT- 0.001 - Root Mean Squared Error", CatBoostRegressor(random_state = 42, verbose = 0, allow_writing_files = False, learning_rate = 0.001)),
    
]


def rmse(y, y_hat):
        return mean_squared_error(y, y_hat, squared = False)

def evaluate_models(datasets: tuple, models):
    train_error_dict = {}
    val_error_dict = {}
    
    X_train, y_train = datasets[0]
    X_val, y_val = datasets[1]

    for name, model in models:
        print(name)
        model.fit(X_train, y_train)

        #evaluate train error
        train_pred = model.predict(X_train)
        train_error = rmse(y_train, train_pred)
        print(f"  Training error: {train_error: .5f}")
        train_error_dict[name] = train_error
        #evaluate test error
        val_pred = model.predict(X_val) 
        val_error = rmse(y_val, val_pred)
        print(f"  Validation error: {val_error: .5f} \n") # Empty line for next model prompts
        val_error_dict[name] = val_error
        
    return train_error_dict, val_error_dict


dataset_pack = ((X_train, y_train), (X_val, y_val))


train_errors, val_errors = evaluate_models(dataset_pack, models)


best_model_index = np.argmin(list(val_errors.values())) # np.argmin doesn't work properly with dict values
best_model = models[best_model_index][1]

if hasattr(best_model, "get_params"):
    print(best_model.get_params())
    print()
else:
    print(best_model) # Version 9 Catboost - 0.9

feature_importance = None
if hasattr(best_model, "get_feature_importance"):
    feature_importance = best_model.get_feature_importance() # Catboost implementation
    
else:
    feature_importance = best_model.feature_importances()  # XGBoost implementation

feat_imp = pd.DataFrame(data = feature_importance, index = test.columns.to_list(), columns = ["score"]).sort_values(by = "score", ascending = True)


feat_imp.sort_values(by = "score")


plt.figure(figsize = (10, 10))
plt.barh(width = feat_imp["score"], y = feat_imp.index)
plt.show()


y_hat_ = best_model.predict(X_test)
err = mean_squared_error(y_test, y_hat_, squared = False)
print(f"Test Error: {err: .5f}") # 0.05605  I need to do a lot more feature engineering 
# Version 10 also has 0.05605 didn't change anything? I don't really know let us check feature importance
# Version 11 Catboost - 0.09 with test scre 0.05602


plt.scatter(x = y_test, y = y_hat_)
plt.xlabel("Ground Truth")
plt.ylabel("Predicted Values")
plt.show()


encoder = OrdinalEncoder()
scaler = StandardScaler()


X[cat_cols] = encoder.fit_transform(X[cat_cols])
X[con_cols] = scaler.fit_transform(X[con_cols])

test[cat_cols] = encoder.transform(test[cat_cols])
test[con_cols] = scaler.transform(test[con_cols])


model = clone(best_model)
model.fit(X, y)
y_hat = model.predict(test)
sub[target_col] = y_hat


sub.describe() # Negative values 
# Need to clip it


sub[target_col] = sub[target_col].clip(0, 1)


sub.describe()


sub.to_csv("submission.csv", index = False)


sub




