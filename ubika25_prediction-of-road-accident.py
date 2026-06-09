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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from  sklearn.model_selection import train_test_split



train=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


display(train.head())
test.head()


print("\n Train\n")
train.info()
print("\n Test\n")
test.info()


print("\n Train\n")
display(train.describe())
print("\n Test\n")
display(test.describe())


print(train.shape)
print(test.shape)


print("\ntrain\n")
print(train.isnull().sum())
print("\ntest\n")
print(test.isnull().sum())


print(train.duplicated().sum())


print(test.duplicated().sum())


plt.figure(figsize=(10,6))
sns.histplot(train['accident_risk'],bins=30,kde=True)
plt.title('Distribution of Accident Risk')
plt.show()


#Spliting the Data's into two types such as Numeric as num_col and Words as cat_col
num_col=("num_lanes","curvature","speed_limit","num_reported_accidents")
cat_col=("road_type","lighting","weather","road_signs_present","public_road","time_of_day","holiday","school_season")


for col in num_col:
    plt.figure(figsize=(10,5))
    plt.title('Accident Risk')
    sns.histplot(train[col],bins=30,kde=True)
    


for col in cat_col:
 plt.figure(figsize=(8,5))
 sns.countplot(x=col,data=train)
 plt.show()


sns.heatmap(train.corr(numeric_only=True).round(2),annot=True,cmap='coolwarm')
plt.show()


#Looking on dtypes to know which type to use
train.dtypes


#Creating one new dataframe
cat_col = train.select_dtypes(include=["bool","object"]).columns.tolist()
# Viewing the dataframe
cat_col



#  Create encoder
encoder = OneHotEncoder(sparse=False, drop="first")

#  Fit + transform categorical data
encoded = encoder.fit_transform(train[cat_col])

#  Get new column names
encoded_cols = encoder.get_feature_names_out(cat_col)

#  Convert to dataframe
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=train.index)

# Drop original categorical columns and join new ones
train_encoded = train.drop(columns=cat_col).join(encoded_df)

display(train_encoded.head())


#  Create encoder
encoder = OneHotEncoder(sparse=False, drop="first")

#  Fit + transform categorical data
encoded = encoder.fit_transform(test[cat_col])

#  Get new column names
encoded_cols = encoder.get_feature_names_out(cat_col)

#  Convert to dataframe
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=test.index)

# Drop original categorical columns and join new ones
test_encoded = test.drop(columns=cat_col).join(encoded_df)

display(test_encoded.head())


for col in num_col:
 plt.figure(figsize=(10,4))
 sns.boxplot(train[col])
 plt.title(" Outliers of Road Accident")
 plt.show()


num_cols=("num_lanes","curvature","speed_limit","num_reported_accidents")
for col in num_cols:
 plt.figure(figsize=(10,4))
 sns.boxplot(train[col])
 plt.title(" Outliers of Road Accident")
 plt.show()


numerical_cols = ["num_lanes","curvature","speed_limit","num_reported_accidents","accident_risk"] # numerical features
scaler = StandardScaler()
train_encoded[numerical_cols] = scaler.fit_transform(train_encoded[numerical_cols])
#train[numerical_cols] = scaler.fit_transform(train[numerical_cols])



X = train_encoded.drop(['id','accident_risk'], axis=1)
y = train_encoded['accident_risk']
X_test =test_encoded.drop(['id'], axis=1) 


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


# Predicting the Test data using the Trained Model
catboost_test_pred = catboost.predict(X_test)



submission_df = pd.DataFrame({'id': test['id'], 'accident_risk': catboost_test_pred})
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")



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





