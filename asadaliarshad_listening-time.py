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
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.head()


podcast_counts = train['Podcast_Name'].value_counts()

print(podcast_counts)





train.info()


train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(), inplace=True)
train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median(), inplace=True)
train['Number_of_Ads'].fillna(train['Number_of_Ads'].mode()[0], inplace=True)

test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(), inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].median(), inplace=True)
test['Number_of_Ads'].fillna(test['Number_of_Ads'].mode()[0], inplace=True)


# Extract numeric part from 'Episode_Title'
train['Episode_Number'] = train['Episode_Title'].str.extract(r'(\d+)', expand=False).astype(float)
test['Episode_Number'] = test['Episode_Title'].str.extract(r'(\d+)', expand=False).astype(float)


train.drop('Episode_Title', axis=1, inplace=True)
test.drop('Episode_Title', axis=1, inplace=True)


train = pd.get_dummies(train, columns=['Genre'], drop_first=True)
test = pd.get_dummies(test, columns=['Genre'], drop_first=True)


from sklearn.preprocessing import LabelEncoder

label_cols = ['Podcast_Name','Episode_Sentiment', 'Publication_Day', 'Publication_Time']

le = LabelEncoder()
for col in label_cols:
    train[col] = le.fit_transform(train[col])


label_cols = ['Podcast_Name','Episode_Sentiment', 'Publication_Day', 'Publication_Time']

le = LabelEncoder()
for col in label_cols:
    test[col] = le.fit_transform(test[col])



train


test



# Separate features and target
X = train.drop('Listening_Time_minutes', axis=1)
y = train['Listening_Time_minutes']



X.info()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)



# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # or y_test if you're using test set
print("RMSE:", rmse)


from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)


y_pred1 = xgb.predict(X_test)


# Evaluate
print("R2 Score:", r2_score(y_test, y_pred1))
print("MSE:", mean_squared_error(y_test, y_pred1))
rmse = np.sqrt(mean_squared_error(y_test, y_pred1))  # or y_test if you're using test set
print("RMSE:", rmse)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import gc


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import gc

# Define X and y
TARGET = 'Listening_Time_minutes'
y = train[TARGET]

# Make sure the test set has the same columns as X
test_X = test[X.columns]  # ensure column order matches

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train))
pred = np.zeros(len(test))

for i, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"--- Fold {i+1} / {FOLDS} ---")
    
    X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
    X_valid, y_valid = X.iloc[valid_idx].reset_index(drop=True), y.iloc[valid_idx].reset_index(drop=True)
    X_test = test_X.reset_index(drop=True)

    model = XGBRegressor(
        tree_method='hist',
        max_depth=14,
        colsample_bytree=0.5,
        subsample=0.8,
        n_estimators=1000,
        learning_rate=0.04,
        early_stopping_rounds=100,
        min_child_weight=10,
    )

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    oof[valid_idx] = model.predict(X_valid)
    pred += model.predict(X_test)

    del X_train, X_valid, X_test, y_train, y_valid
    if i != FOLDS - 1:
        del model
    gc.collect()

pred /= FOLDS
rmse_score = mean_squared_error(y, oof, squared=False)
print(f"Final RMSE: {rmse_score:.5f}")



sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub[TARGET] = pred
sub.to_csv('submission.csv', index=False)


sub

