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


# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')


df.head()


df.shape


df.dtypes


df.isnull().sum().sort_values(ascending=False)


df.describe()


df['Penalty'].unique()


df['Tire_Compound_Front'].unique()


df['Penalty'].value_counts()


df[df['Penalty'] == 'DNF']['Lap_Time_Seconds'].unique()


df['category_x'].unique()


df['Session'].unique()


df['circuit_name'].unique()


df['rider_name'].unique()


df[(df['rider_name'] == 'Ogura, Ai')]['circuit_name'].unique()


silverstone2012gp = df[(df['circuit_name'] == 'Silverstone') & (df['category_x'] == 'MotoGP') & (df['year_x'] == 2012) & (df['Session'] == 'Race')]


silverstone2012gp


df[(df['circuit_name'] == 'Silverstone') & (df['rider_name'] == 'North, Alan') & (df['category_x'] == 'MotoGP')]


tire_factors = {
    'Soft': 0.98,
    'Medium': 1.00,
    'Hard': 1.02
}
track_factors = {
    'Dry': 1.00,
    'Wet': 1.05
}
time_loss_per_corner = 0.5
ride_through_time = 20

df['Penalty_Seconds'] = df['Penalty'].map({
    '+3s': 3,
    '+5s': 5,
    'Ride Through': ride_through_time,
    'DNS': np.nan,
    'DNF': np.nan
}).fillna(0)

df['Base_Lap_Time'] = (df['Circuit_Length_km'] * 3600) / df['Avg_Speed_kmh']
df['Tire_Compound_Factor'] = df['Tire_Compound_Front'].map(tire_factors).fillna(1.0)
df['Track_Condition_Factor'] = df['Track_Condition'].map(track_factors).fillna(1.0)
df['Corner_Penalty'] = df['Corners_per_Lap'] * time_loss_per_corner
df['Tire_Degradation_Penalty'] = df['Tire_Degradation_Factor_per_Lap'] * df['Laps']

df['Temperature_Adjustment'] = 1.0
df.loc[df['Track_Temperature_Celsius'] < 20, 'Temperature_Adjustment'] = 1.02
df.loc[df['Track_Temperature_Celsius'] > 40, 'Temperature_Adjustment'] = 1.01
df.loc[df['Humidity_%'] > 80, 'Base_Lap_Time'] += 0.1

df['Theoretical_Lap_Time'] = (
    df['Base_Lap_Time'] * 
    df['Tire_Compound_Factor'] * 
    df['Track_Condition_Factor'] * 
    df['Temperature_Adjustment'] + 
    df['Corner_Penalty'] + 
    df['Tire_Degradation_Penalty'] + 
    df['Penalty_Seconds']
)

df_valid = df[df['Penalty_Seconds'].notna()]
df_valid[['Lap_Time_Seconds', 'Theoretical_Lap_Time']]


df_valid[['Lap_Time_Seconds', 'Theoretical_Lap_Time']].describe()


!pip install -q tqdm_joblib


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


df['Penalty_Seconds'] = df['Penalty'].map({
    '+3s': 3,
    '+5s': 5,
    'Ride Through': 20,
    'DNS': np.nan,
    'DNF': np.nan
}).fillna(0)

df = df[df['Penalty_Seconds'].notna()]

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

for col in numerical_features:
    df[col] = df[col].astype('float32')

X = df[features]
y = df[target].astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

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




model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

model


# model.fit(X_train, y_train)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        device_type='gpu',
        verbosity=1,
        random_state=42
    ))
])
model


model.fit(
    X_train, y_train,
    regressor__eval_set=[(X_val, y_val)],
    regressor__eval_metric='mae',
    regressor__callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=10)
    ]
)


df['Penalty_Seconds'] = df['Penalty'].map({
    '+3s': 3,
    '+5s': 5,
    'Ride Through': 20,
    'DNS': np.nan,
    'DNF': np.nan
}).fillna(0)

df = df[df['Penalty_Seconds'].notna()]

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

for col in numerical_features:
    df[col] = df[col].astype('float32')
df[target] = df[target].astype('float32')

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

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

lgbm = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.1,
    device_type='gpu',
    verbosity=1,
    random_state=42
)

lgbm.fit(
    X_train_transformed,
    y_train,
    eval_set=[(X_val_transformed, y_val)],
    eval_metric='mae',
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=10)
    ]
)


y_pred = lgbm.predict(X_test)




