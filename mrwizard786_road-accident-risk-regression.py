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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print("ðŸŸ¦ Training Data: ")
display(train_df.head())
print("\nðŸŸ¦ Test Data: ")
display(test_df.head())


train_df.info()


test_df.info()


train_df.describe()


test_df.describe()


train_df.columns


test_df.columns


numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64']]
print("Numerical Columns: ", numerical_cols)
categorical_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']
print("Categorical Columns: ", categorical_cols)
boolean_cols = [col for col in train_df.columns if train_df[col].dtype == 'bool']
print("Boolean Columns: ", boolean_cols)


print("Value Counts in Training Dataset: \n")
for col in train_df.columns:
    print(f"Value Counts of \"{col}\"")
    print(train_df[col].value_counts(),"\n")


print("Value Counts in Testing Dataset: \n")
for col in test_df.columns:
    print(f"Value Counts of \"{col}\"")
    print(test_df[col].value_counts(),"\n")


numerical_cols, categorical_cols, boolean_cols


train_df.isnull().sum()


test_df.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns

for col in categorical_cols:
    plt.title(f"Distribution of {col}")
    sns.countplot(x=train_df[col], data=train_df)
    plt.show()


for col in boolean_cols:
    plt.title(f"Distribution of {col}")
    sns.histplot(x=train_df[col], data=train_df, bins=2)
    plt.show()


for col in numerical_cols:
    plt.title(f"Distribution of {col}")
    sns.histplot(x=train_df[col], data=train_df, kde=True, bins=10)
    plt.show()


for col in categorical_cols:
    print(train_df[col].value_counts())


from sklearn.preprocessing import LabelEncoder

lb_encoder = LabelEncoder()

for col in categorical_cols:
    train_df[col] = lb_encoder.fit_transform(train_df[col])

for col in boolean_cols:
    train_df[col] = lb_encoder.fit_transform(train_df[col])


for col in categorical_cols:
    test_df[col] = lb_encoder.fit_transform(test_df[col])

for col in boolean_cols:
    test_df[col] = lb_encoder.fit_transform(test_df[col])


train_df.columns


from sklearn.model_selection import train_test_split

X = train_df.drop(columns='accident_risk')
y = train_df['accident_risk']

# Splitting data into Training and Validation part
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# display("X Train: ", X_train)
print("X Train Shape: ", X_train.shape, "\n")
# display("X Valid: ", X_valid)
print("X Valid Shape: ", X_valid.shape, "\n")
# display("y Train: ", y_train)
print("y Train Shape: ", y_train.shape, "\n")
# display("y Valid: ", y_valid)
print("y Valid Shape: ", y_valid.shape, "\n")


print("Training Set:", len(X_train)/len(train_df) * 100)
print("Validation Set:", len(X_valid)/len(train_df) * 100)


from xgboost import XGBRegressor

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(X_valid)

print("R2 Score: ", r2_score(y_valid, y_pred))
print("Mean Squared Error: ", mean_squared_error(y_valid, y_pred))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_valid, y_pred)))


y_pred_test = model.predict(test_df)
print("Testing Dataset Predictions:")
print(y_pred_test)


submission = pd.DataFrame({'id': test_df['id'], 'accident_risk': y_pred_test})
submission


submission.to_csv("submission.csv", index=False)


# sub = pd.read_csv("/kaggle/working/submission.csv")
# sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
# from sklearn.svm import SVR

X = train_df.drop(columns='accident_risk').copy()
y = train_df['accident_risk'].copy()

# Splitting dataset into training and validation:
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

models = [LinearRegression(), DecisionTreeRegressor(random_state=42), 
          CatBoostRegressor(iterations=100,           # Number of boosting rounds
                            learning_rate=0.1,        # Step size shrinkage
                            depth=6,                  # Depth of the tree
                            loss_function='RMSE',     # Loss function for regression
                            random_seed=42,           # Random seed for reproducibility
                            verbose=False),
          lgb.LGBMRegressor(objective='regression',  # Specify regression objective
                            metric='rmse',          # Evaluation metric
                            n_estimators=100,       # Number of boosting rounds
                            learning_rate=0.1,      # Step size shrinkage
                            num_leaves=31,          # Max number of leaves in one tree
                            random_state=42),
          RandomForestRegressor(n_estimators=100, random_state=42),
          GradientBoostingRegressor(n_estimators=100, random_state=42)]

def evaluating_different_models(models):
    for model in models:
        mod = model
        mod.fit(X_train, y_train)
        y_pred = mod.predict(X_valid)
        print(f"{model}:")
        print("R2_Score: ", r2_score(y_valid, y_pred))
        print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_valid, y_pred)))
        print("\n")

evaluating_different_models(models)

