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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.preprocessing import LabelEncoder
from  sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


# Load the training dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
# Load the testing dataset
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


# Display the first 5 rows of  DataFrames
print("Training head:")
display(train.head())


# Display the first 5 rows of  DataFrames
print("\nTesting  head:")
display(test.head())


# Display basic info
print("Train shape:", train.shape)
print("Train shape:",train.shape)


print("\n Train \n")
print("Train info:",train.info())
print("\n Test  \n")
print("test info:",train.info())


print("\nMissing values in train data:\n")
display(train.isnull().sum())
print("\nMissing values in test data:\n")
display(test.isnull().sum())


plt.figure(figsize=(10,5))
sns.histplot(train['accident_risk'],bins=30, kde=True)
plt.title('Distribution of accident_risk')
plt.xlabel('accident_risk')
plt.ylabel('Frequency')
plt.show()


# Identify numeric and categorical columns based on the known schema
num_cols = ["num_lanes","curvature","speed_limit","num_reported_accidents","accident_risk"] # numerical features
cat_cols = ["road_type", "lighting", "weather", "road_signs_present", "public_road", "time_of_day", "holiday", "school_season"] # categorical features


for col in num_cols:
 plt.figure(figsize=(7,4))
 sns.histplot(train[col],bins=30,kde=True)
 plt.title(f"measurment of {col}")
 plt.xlabel(f"{col}")
 plt.ylabel("frequency")
 plt.tight_layout()


for col in cat_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, data=train)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


plt.figure(figsize=(10,7))
sns.heatmap(train[num_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Train Data ')
plt.show()


num_cols_test = ["num_lanes","curvature","speed_limit","num_reported_accidents"] # numerical 
plt.figure(figsize=(10,7))
sns.heatmap(test[num_cols_test].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Test Data')
plt.show()


for col in num_cols:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=train,x=col)
    plt.title(f'Out Layers in {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Frequency')
    plt.show()


num_cols_test = ["num_lanes","curvature","speed_limit","num_reported_accidents"] # numerical 
for col in num_cols_test:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=test,x=col)
    plt.title(f'Out Layers in {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Frequency')
    plt.show()
    


train


#train
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# 1. Get categorical columns automatically (object or category dtype)
cat_cols = train.select_dtypes(include=["object","bool", "category"]).columns.tolist()
print("Categorical columns:", cat_cols)

# 2. Create encoder
encoder = OneHotEncoder(sparse=False, drop="first")

# 3. Fit + transform categorical data
encoded = encoder.fit_transform(train[cat_cols])

# 4. Get new column names
encoded_cols = encoder.get_feature_names_out(cat_cols)

# 5. Convert to dataframe
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=train.index)

# 6. Drop original categorical columns and join new ones
train_encoded = train.drop(columns=cat_cols).join(encoded_df)

display(train_encoded.head())


#test
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# 1. Get categorical columns automatically (object or category dtype)
cat_cols = test.select_dtypes(include=["object","bool", "category"]).columns.tolist()
print("Categorical columns:", cat_cols)

# 2. Create encoder
encoder = OneHotEncoder(sparse=False, drop="first")

# 3. Fit + transform categorical data
encoded = encoder.fit_transform(test[cat_cols])

# 4. Get new column names
encoded_cols = encoder.get_feature_names_out(cat_cols)

# 5. Convert to dataframe
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=test.index)

# 6. Drop original categorical columns and join new ones
test_encoded = test.drop(columns=cat_cols).join(encoded_df)

display(test_encoded.head())


# Scale numerical features
numerical_cols = ["num_lanes","curvature","speed_limit","num_reported_accidents","accident_risk"] # numerical features
scaler = StandardScaler()
#train_encoded[numerical_cols] = scaler.fit_transform(train_encoded[numerical_cols])
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])


# Scale numerical features
numerical_cols = ["num_lanes","curvature","speed_limit","num_reported_accidents"] # numerical features
scaler = StandardScaler()
#test_encoded[numerical_cols] = scaler.fit_transform(test_encoded[numerical_cols])
test[numerical_cols] = scaler.fit_transform(test[numerical_cols])


X = train_encoded.drop(['id', 'accident_risk'], axis=1)
y = train_encoded['accident_risk']
X_test=test_encoded.drop(['id'], axis=1)


print("X shape:", X.shape)
print("y shape:", y.shape)
print(X_test.shape)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)


from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
# Initialize the CatBoost model
catboost = CatBoostRegressor(random_state=42, verbose=0) # verbose=0 to suppress output during training

# Train the CatBoost model
print("Training CatBoost model...")
catboost.fit(X_train, y_train)

# Evaluate the CatBoost model
y_pred_catboost = catboost.predict(X_val)
rmse_catboost = mean_squared_error(y_val, y_pred_catboost)**0.5
print(f"CatBoost RMSE on validation data: {rmse_catboost}")


# Make predictions on the test data using the trained models
catboost_test_pred = catboost.predict(X_test)


submission_df = pd.DataFrame({'id': test['id'], 'accident_risk': catboost_test_pred})
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")


import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 512),
        'max_depth': trial.suggest_int('max_depth', -1, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best RMSE:", study.best_value)
print("Best params:", study.best_params)


import pandas as pd

# Concatenate back train + val into full dataset
X_full = pd.concat([X_train, X_val], axis=0)
y_full = pd.concat([y_train, y_val], axis=0)


# 1. Get best params from Optuna
best_params = study.best_params

# 2. Retrain on full train data (combine train+val)
best_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)

best_model.fit(X_full, y_full, eval_metric="rmse")

# 3. Predict on X_test
y_test_pred = best_model.predict(X_test)

print("Test predictions:", y_test_pred[:10])  # show first 10 predictions




# Example using LGBM predictions:
submission_df = pd.DataFrame({'id': test['id'], 'accident_risk':y_test_pred})
# submission_df = pd.DataFrame({'id': test_df['id'], 'accident_risk': averaged_predictions})


# Save the submission file
submission_df.to_csv('submissionb.csv', index=False)

print("Submission file created successfully!")




