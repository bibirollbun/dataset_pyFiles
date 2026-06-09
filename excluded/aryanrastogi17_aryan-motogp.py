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

train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
sample = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


import seaborn as sns
import matplotlib.pyplot as plt

# Target Distribution
sns.histplot(train['Lap_Time_Seconds'], kde=True, bins=30)
plt.title("Target Distribution - Lap Time (Seconds)")
plt.show()

# Null Count
print("Missing values:\n", train.isnull().sum().sort_values(ascending=False).head(10))


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
corr = train[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr[['Lap_Time_Seconds']].sort_values(by='Lap_Time_Seconds', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlation with Lap_Time_Seconds")
plt.show()


TARGET = 'Lap_Time_Seconds'

DROP_COLS = [
    'Unique ID', 'rider_name', 'team_name', 'bike_name',
    'circuit_name', 'points', 'position'
]


def add_features(df):
    df['LapTime_Estimate'] = df['Circuit_Length_km'] / df['Avg_Speed_kmh'] * 3600
    df['Points_per_Year'] = df['Championship_Points'] / (df['years_active'] + 1)
    df['Finish_Rate'] = df['finishes'] / (df['starts'] + 1)
    df['Podium_Rate'] = df['podiums'] / (df['starts'] + 1)
    df['Win_Rate'] = df['wins'] / (df['starts'] + 1)
    df['Avg_Temp'] = (df['Ambient_Temperature_Celsius'] + df['Track_Temperature_Celsius']) / 2
    return df

train = add_features(train)
test = add_features(test)



X = train.drop(DROP_COLS + [TARGET], axis=1)
X_test = test.drop(DROP_COLS, axis=1)
y = train[TARGET]


all_data = pd.concat([X, X_test], axis=0)
all_data.fillna(-1, inplace=True)


cat_cols = all_data.select_dtypes(include='object').columns
for col in cat_cols:
    all_data[col] = all_data[col].astype('category').cat.codes


X = all_data.iloc[:len(train)]
X_test = all_data.iloc[len(train):]


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=70,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Validation RMSE:", rmse)


importances = model.feature_importances_
features = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=features[:20], y=features.index[:20])
plt.title("Top 20 Feature Importances")
plt.show()


X_train['Speed_Degradation'] = X_train['Avg_Speed_kmh'] * X_train['Tire_Degradation_Factor_per_Lap']
X_val['Speed_Degradation'] = X_val['Avg_Speed_kmh'] * X_val['Tire_Degradation_Factor_per_Lap']
X_test['Speed_Degradation'] = X_test['Avg_Speed_kmh'] * X_test['Tire_Degradation_Factor_per_Lap']


X_train['Temp_Condition'] = X_train['Track_Temperature_Celsius'] * X_train['Track_Condition']
X_val['Temp_Condition'] = X_val['Track_Temperature_Celsius'] * X_val['Track_Condition']
X_test['Temp_Condition'] = X_test['Track_Temperature_Celsius'] * X_test['Track_Condition']


model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=10,
    num_leaves=90,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)

model.fit(X_train, y_train)
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Validation RMSE:", rmse)


# Ratio of Circuit length to number of corners
X_train['Corners_per_Km'] = X_train['Corners_per_Lap'] / (X_train['Circuit_Length_km'] + 1e-3)
X_val['Corners_per_Km'] = X_val['Corners_per_Lap'] / (X_val['Circuit_Length_km'] + 1e-3)
X_test['Corners_per_Km'] = X_test['Corners_per_Lap'] / (X_test['Circuit_Length_km'] + 1e-3)

# LapTime Estimate - Actual difference
X_train['Est_Error'] = X_train['LapTime_Estimate'] - y_train
X_val['Est_Error'] = X_val['LapTime_Estimate'] - y_val



model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.025,
    max_depth=12,
    num_leaves=100,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=1.5,
    reg_lambda=2.0,
    random_state=42
)

model.fit(X_train, y_train)
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Validation RMSE:", rmse)




