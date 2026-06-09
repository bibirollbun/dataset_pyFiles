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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR
import lightgbm as lgb



train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")



print(train.head())

print(test.head())

print(train_extra.head())


print(train.isnull().sum())
print(test.isnull().sum())
print(train_extra.isnull().sum())


# Combine train and train_extra
data = pd.concat([train, train_extra], axis=0).reset_index(drop=True)


# Handle missing values
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)']


(data.isnull().sum())


for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)
for col in numerical_cols:
    data[col].fillna(data[col].median(), inplace=True)
for col in categorical_cols:
    test[col].fillna(test[col].mode()[0], inplace=True)
for col in numerical_cols:
    test[col].fillna(test[col].median(), inplace=True)


(data.isnull().sum())
(test.isnull().sum())



data.head(10)


    # Encode binary categorical columns
    data['Laptop Compartment'] = data['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    data['Waterproof'] = data['Waterproof'].map({'Yes': 1, 'No': 0})

    test['Laptop Compartment'] = test['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    test['Waterproof'] = test['Waterproof'].map({'Yes': 1, 'No': 0})


print(data['Laptop Compartment'].unique())
print(data['Waterproof'].unique())



print(data.describe())
print(data.describe(include='object'))
print(test.describe())
print(test.describe(include='object'))


print(data['Brand'].unique())
print(data['Material'].unique())
print(data['Size'].unique())
print(data['Style'].unique())
print(data['Color'].unique())


size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
data["Size"] = data["Size"].map(size_mapping)


test["Size"] = test["Size"].map(size_mapping)


categorical_cols = ["Brand", "Material", "Style", "Color"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoders for inverse transformation later


    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le


data.head()
test.head()


print(test.isnull().sum())  # Should be all zeros
print(test.dtypes)  # Should match training data types



# Split train data and train_extra
train = data[:len(train)]
train_extra = data[len(train):]


# Define features and target
X = train.drop(columns=['id', 'Price'])
y = train['Price']
X_test = test.drop(columns=['id'])


# Normalize numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




# Train the model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01)
model.fit(X_train, y_train)

# Make predictions on the validation set (y_val)
y_pred = model.predict(X_val)

# Calculate RMSE on the validation set
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse}")





# Train the model
lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01)
lgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = lgb_model.predict(X_val)

# Calculate RMSE on the validation set
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse}")




# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = svr_model.predict(X_val)

# Calculate RMSE on the validation set
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse}")





# Train the CatBoost model
catboost_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=3, silent=True)
catboost_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = catboost_model.predict(X_val)

# Calculate RMSE on the validation set
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse}")



def build_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Train model
model = build_model()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)


