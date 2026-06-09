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


import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor, callback
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
val_df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/val.csv')
test_df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')


for df in [train_df, val_df, test_df]:
    df['Penalty_Seconds'] = df['Penalty'].map({
        '+3s': 3,
        '+5s': 5,
        'Ride Through': 20,
        'DNS': 0,
        'DNF': 0
    }).fillna(0)

for df in [train_df, val_df, test_df]:
    df['Tire_Combo'] = df['Tire_Compound_Front'] + '_' + df['Tire_Compound_Rear']
    df['Temperature_Diff'] = df['Track_Temperature_Celsius'] - df['Ambient_Temperature_Celsius']

target = 'Lap_Time_Seconds'
features = [
    'Circuit_Length_km', 'Avg_Speed_kmh', 'Corners_per_Lap',
    'Tire_Degradation_Factor_per_Lap', 'Pit_Stop_Duration_Seconds',
    'Ambient_Temperature_Celsius', 'Track_Temperature_Celsius', 'Penalty_Seconds',
    'Temperature_Diff', 'Track_Condition', 'Tire_Compound_Front',
    'Tire_Compound_Rear', 'Tire_Combo', 'Session', 'circuit_name'
]

numerical_features = [
    'Circuit_Length_km', 'Avg_Speed_kmh', 'Corners_per_Lap',
    'Tire_Degradation_Factor_per_Lap', 'Pit_Stop_Duration_Seconds',
    'Ambient_Temperature_Celsius', 'Track_Temperature_Celsius', 'Penalty_Seconds',
    'Temperature_Diff'
]
low_cardinality_categorical = [
    'Track_Condition', 'Tire_Compound_Front', 'Tire_Compound_Rear',
    'Tire_Combo', 'Session'
]
high_cardinality_categorical = ['circuit_name']

for df in [train_df, val_df, test_df]:
    for col in numerical_features:
        df[col] = df[col].astype('float32')
    if target in df.columns:
        df[target] = df[target].astype('float32')

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]
X_test = test_df[features]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

low_cardinality_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

high_cardinality_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('target', ce.TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('low_cat', low_cardinality_transformer, low_cardinality_categorical),
        ('high_cat', high_cardinality_transformer, high_cardinality_categorical)
    ])

X_train_transformed = preprocessor.fit_transform(X_train, y_train)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)


xgb_model = XGBRegressor(
    n_estimators=10000,
    learning_rate=1,
    objective='reg:squarederror',
    tree_method='gpu_hist',              # use 'gpu_hist' if you have a GPU
    random_state=42,
    verbosity=0                      # use 0 to suppress log duplication
)

# Fit the model with early stopping and log evaluation every 20 rounds
xgb_model.fit(
    X_train_transformed, y_train,
    eval_set=[(X_val_transformed, y_val)],
    eval_metric='rmse',
    callbacks=[
        callback.EarlyStopping(rounds=100, save_best=True, metric_name='rmse'),
    ]
)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb_model)
])


y_val_pred = model.predict(X_val)
mean_squared_error(y_val, y_val_pred)
joblib.dump(xgb_model, 'lgbm_model1_0.6.pkl')
y_test_val = model.predict(X_test)
results_df = test_df[['Unique ID']].copy()
results_df['Lap_Time_Seconds'] = y_test_val 
results_df.to_csv('submission1.csv', index=False)



y_val_pred.shape


results_df.shape




