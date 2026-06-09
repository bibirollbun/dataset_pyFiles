# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


Train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
Test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


Train.head()


Test.head()


print(Train.shape)
print(Test.shape)


df = Train
X = df.drop('Calories', axis=1)  # Features (everything except 'Calories')
y = df['Calories']               # Target variable



X


# Assuming df is your DataFrame
X = pd.get_dummies(X, columns=['Sex'], dtype=int)

X = X.drop('Sex_female',axis=1)

# Rename 'Sex_male' to 'Sex'
X.rename(columns={'Sex_male': 'Sex'}, inplace=True)
# Sex male:1 female:0





from sklearn.preprocessing import MinMaxScaler


# Separate 'id' column
id_col = X['id']

# Select features to normalize (excluding 'id')
features = X.drop(columns=['id'])

# Apply MinMaxScaler
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(features)

# Create a new DataFrame with normalized values
normalized_df = pd.DataFrame(normalized_values, columns=features.columns)

# Add back the 'id' column
X = pd.concat([id_col, normalized_df], axis=1)


X


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)







RF_model = RandomForestRegressor(n_estimators=200)


# Train the model
RF_model.fit(X_train.iloc[:,1::], y_train)


import joblib

# Save the trained model to a file
# joblib.dump(RF_model, 'random_forest_model.pkl')
# Load the model from the file
# RF_model = joblib.load('random_forest_model.pkl')



y_pred = RF_model.predict(X_test.iloc[:,1:])

# 9. Evaluate the model
y_pred = np.maximum(0, y_pred)  # Prevent negative predictions for regression task

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSLE: {rmsle:.4f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.4f}")



xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression task
    booster='gbtree',  # Tree booster
    n_estimators=500,  # Number of trees
    learning_rate=0.1,  # Learning rate
    max_depth=8,  # Max depth of trees
    subsample=0.8,  # Subsample ratio of the training set
    colsample_bytree=0.8  # Subsample ratio of columns when building each tree
)

# 6. Train the model
xgb_model.fit(X_train.iloc[:,1::], y_train)



import joblib

# Save the trained model to a file
# joblib.dump(xgb_model, 'xgb_model.pkl')
# Load the model from the file
# xgb_model = joblib.load('xgb_model.pkl')



y_pred = xgb_model.predict(X_test.iloc[:,1::])

# 9. Evaluate the model
y_pred = np.maximum(0, y_pred)  # Prevent negative predictions for regression task

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSLE: {rmsle:.4f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.4f}")



xgb_pred = xgb_model.predict(X_test.iloc[:,1::])
rf_pred = RF_model.predict(X_test.iloc[:,1::])

# Simple average ensemble
ensemble_pred = (xgb_pred + rf_pred) / 2




rmsle = np.sqrt(mean_squared_log_error(y_test, ensemble_pred))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSLE: {rmsle:.4f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.4f}")



Test


# Assuming df is your DataFrame
X = pd.get_dummies(Test, columns=['Sex'], dtype=int)

X = X.drop('Sex_female',axis=1)

# Rename 'Sex_male' to 'Sex'
X.rename(columns={'Sex_male': 'Sex'}, inplace=True)
# Sex male:1 female:0


# Separate 'id' column
id_col = X['id']

# Select features to normalize (excluding 'id')
features = X.drop(columns=['id'])

# Apply MinMaxScaler
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(features)

# Create a new DataFrame with normalized values
normalized_df = pd.DataFrame(normalized_values, columns=features.columns)

# Add back the 'id' column
X = pd.concat([id_col, normalized_df], axis=1)


xgb_pred = xgb_model.predict(X.iloc[:,1::])
rf_pred = RF_model.predict(X.iloc[:,1::])

# Simple average ensemble
ensemble_pred = (xgb_pred + rf_pred) / 2


submission = pd.DataFrame({
    'id': X['id'].values,  # Ensure 'id' was retained in X_test
    'Calories': ensemble_pred
})
submission.to_csv("submission.csv", index=False)

submission







