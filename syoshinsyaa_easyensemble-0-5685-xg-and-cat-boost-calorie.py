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


import pandas as pd 
import numpy as np 
import os 
import time
import logging 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from category_encoders import TargetEncoder

from tqdm.auto import tqdm
from itertools import combinations
import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)


num_features = train.select_dtypes(include='number')


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train["Sex"] = train["Sex"].astype("category")
test["Sex"] = test["Sex"].astype("category")
categorical_features_indices = ['Sex']

X = train.drop(columns=["id", "Calories"])
y = np.log1p(train["Calories"])
X_test = test.drop(columns=["id"])


FOLDS = 5
FEATURES = X.columns.tolist()


# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof1 = np.zeros(len(train))
oof2 = np.zeros(len(train))
pred1 = np.zeros(len(test))
pred2 = np.zeros(len(test))

# Start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = X_test.copy()


    # Train two models which are xgboost and catboost in this case
    model1 = XGBRegressor(
        device="cuda" if XGBRegressor().get_params().get("device") == "cuda" else "cpu",
        max_depth=10,
        colsample_bytree=0.7,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=0.01, 
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True
    )

    model1.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    # Predict OOF and test
    oof1[valid_idx] = model1.predict(x_valid)
    pred1 += model1.predict(x_test)

    rmse1 = np.sqrt(mean_squared_error(y_valid, oof1[valid_idx]))
    print(f"Fold {i+1} RMSE1: {rmse1:.4f}")

    model2 = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=10,
        subsample=0.9,
        colsample_bylevel=0.7,
        random_state=42,
        loss_function='RMSE',
        eval_metric='RMSE',
        early_stopping_rounds=100,
        verbose=0 
    )

    model2.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        cat_features=categorical_features_indices
    )
    
    oof2[valid_idx] = model2.predict(x_valid)
    pred2 += model2.predict(x_test)

    rmse2 = np.sqrt(mean_squared_error(y_valid, oof2[valid_idx]))
    print(f"Fold {i+1} RMSE1: {rmse2:.4f}")
    

# Average test predictions
pred1 /= FOLDS
pred2 /= FOLDS

final_pred = (pred1+pred2)/2

# Final RMSE
full_rmse1 = np.sqrt(mean_squared_error(y, oof1))
print(f"\nFinal CV RMSE1: {full_rmse:.4f}")
full_rmse2 = np.sqrt(mean_squared_error(y, oof2))
print(f"\nFinal CV RMSE2: {full_rmse2:.4f}")

rmse_ensemble = np.sqrt(mean_squared_error(y, (oof1+oof2)/2))
print(f"\nEnsemble CV RMSE: {rmse_ensemble:.4f}")



y_preds = np.expm1(final_pred)
print('predict mean :',y_preds.mean())
print('predict median :',np.median(y_preds))

y_preds = np.clip(y_preds,1,314)
print('predict mean after clip:',y_preds.mean())
print('predict median after clip:',np.median(y_preds))

submission["Calories"] = y_preds
submission.to_csv("submission2.csv", index=False)
submission.head()

