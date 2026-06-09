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
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# Load training dataset (assume it's available as 'training_data.csv')
# Replace 'training_data.csv' with the correct path to your training data
training_file_path = '/kaggle/input/ccs-ml-wars/kaggle-wars-sat-24/Kaggle-Wars Train.csv'
training_data = pd.read_csv(training_file_path)


# Preprocessing
# Drop irrelevant columns and handle missing values
columns_to_drop = ['Unnamed: 0', 'LocationDesc', 'Data_Value_Footnote', 'Data_Value_Footnote_Symbol']
training_data = training_data.drop(columns=columns_to_drop, errors='ignore')
training_data = training_data.dropna()


# Encode categorical features
categorical_features = training_data.select_dtypes(include=['object']).columns
training_data = pd.get_dummies(training_data, columns=categorical_features, drop_first=True)


#print(training_data.columns)
#print(training_data.head())

# Split features and target
X = training_data.drop('Data_Value', axis=1)
y = training_data['Data_Value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Feature scaling (optional, helps Ridge regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# XGBoost model (base model)
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
xgb_model.fit(X_train_scaled, y_train)



# Ridge regression model (base model)
ridge_model = Ridge(alpha=1.0)

# Stacking Regressor
stacked_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('ridge', ridge_model)
    ],
    final_estimator=Ridge(alpha=0.1)  # Final meta-model
)

# Fit stacked model
stacked_model.fit(X_train_scaled, y_train) 


# Predictions
stacked_predictions = stacked_model.predict(X_test_scaled)

# Evaluate model
stacked_rmse = np.sqrt(mean_squared_error(y_test, stacked_predictions))
print(f"Stacked RMSE: {stacked_rmse}")


# Test data preprocessing (similar to training data)
test_file_path = '/kaggle/input/ccs-ml-wars/kaggle-wars-sat-24/Kaggle-Wars Test.csv'
test_data = pd.read_csv(test_file_path)
test_data = test_data.drop(columns=columns_to_drop, errors='ignore')

# Handle missing categorical features
missing_categorical_features = [col for col in categorical_features if col not in test_data.columns]
for col in missing_categorical_features:
    test_data[col] = None

test_data = pd.get_dummies(test_data, columns=categorical_features, drop_first=True)
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Scale test data
test_data_scaled = scaler.transform(test_data)


# Predictions on test data
test_predictions = stacked_model.predict(test_data_scaled)


# Save predictions
if 'Index' in test_data.columns:
    output = pd.DataFrame({'Index': test_data['Index'], 'Data_Value': test_predictions})
else:
    output = pd.DataFrame({'Index': range(len(test_predictions)), 'Data_Value': test_predictions})
output.to_csv('test_predictions.csv', index=False)
print("Predictions saved to test_predictions.csv")


