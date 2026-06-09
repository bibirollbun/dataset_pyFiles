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
import matplotlib.pyplot as plt
import seaborn as sns


## import the datsets
train_data = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/train.csv')
test_data = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/test.csv')
lookup_data = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/feature_lookup.csv')


train_data.shape


test_data.shape


train_data.info()


train_data.head()


# checking the percentage of null values in the dataframe
train_data.isna().sum()*100


train_data["F5"].unique()


train_data["F7"].unique()


train_data["F8"].unique()


train_data["F9"].unique()


train_data["F10"].unique()


train_data["F13"].unique()


# ENCODING CONVERTING THE CATEGORICAL INTO NUMERICAL
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



categorical = ["F5","F7","F8","F9","F10","F13"]
X_train = train_data
X_test = test_data


le_dict = {}
for i in categorical:
    X_train[i] = le.fit_transform(X_train[i].astype(str))
    X_test[i] = le.transform(X_test[i].astype(str))
    le_dict[i] = le


# all are converted into the numerical values
X_train.info()


# standralide the numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = X_train.drop("relationship_probability",axis =1)
X_test_scaled = X_test


X_train_scaled.shape


X_test_scaled.shape


# learn from the data then fit the model
scaler.fit(X_train_scaled)     
X_train_scaled = scaler.transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)



print("categorical features converted into numerical")
print("------------------------")
print("all are strandalised")


X_train = pd.DataFrame(X_train_scaled, columns=X_train.drop("relationship_probability",axis =1).columns, index=X_train.drop("relationship_probability",axis =1).index)


X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


print(X_train.shape)
print(X_test.shape)


X_train.info()


X_train.head()


X_train   ## independent features
Y_train = train_data["relationship_probability"] # dependent feauttures


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_train,Y_train,test_size=0.4,random_state=42)


#import the model
import xgboost as xgb


model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


model.fit(x_train, y_train)


# Predict
y_pred = model.predict(x_test)


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


print("R2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


from lightgbm import LGBMRegressor


light_gbm = LGBMRegressor(
    n_estimators=600,
    learning_rate=0.04,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)


light_gbm.fit(x_train, y_train)


y_pred2 = light_gbm.predict(x_test)


print("R2:", r2_score(y_test, y_pred2))
print("RMSE:", mean_squared_error(y_test, y_pred2, squared=False))


from sklearn.ensemble import RandomForestRegressor


random_forest = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,      
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="auto", 
    bootstrap=True,
    random_state=42,
    n_jobs=-1           
)


random_forest.fit(x_train, y_train)



y_pred = random_forest.predict(x_test)



print("R2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


XG_Boost_preds = model.predict(X_train)


LIGHT_GBM_preds = light_gbm.predict(X_train)


random_forest_predicts = random_forest.predict(X_train)


ensemble_preds = (XG_Boost_preds + LIGHT_GBM_preds + random_forest_predicts) / 3


submission = pd.DataFrame({
    'ID': np.arange(1, 2001),
    'relationship_probability': ensemble_preds_trimmed
})

submission.to_csv('Shanmukh-Datta_submission.csv', index=False)

print("Submission file created: Shanmukh-Datta_submission.csv")
print(f"Submission shape: {submission.shape}")




