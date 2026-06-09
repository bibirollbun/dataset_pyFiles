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


# Import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


# Load the data
file_path = "/kaggle/input/playground-series-s5e2/train.csv"
df = pd.read_csv(file_path)



# remove id columns
df = df.drop(["id"], axis=1)
df.head(5)


# check data shepe
df.shape


#check information 
df.info()


# describe the data 
df.describe()


# check null values
df.isnull().sum().sort_values(ascending=True)


# Identify categorical and numerical columns
cat_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
num_cols = ["Compartments", "Weight Capacity (kg)"]


# Handle missing values (Changed method)
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])


# Define features and target variable
X = df.drop(columns=["Price"])
y = df["Price"]



# Preprocessing pipelines
num_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ("num", num_preprocessor, num_cols),
    ("cat", cat_preprocessor, cat_cols)
])


rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])


lgbm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(n_estimators=100, random_state=42))
])


# Split the data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Train and Evaluate LightGBM Model
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_valid)
rmse_lgbm = np.sqrt(mean_squared_error(y_valid, y_pred_lgbm))
print(f"LightGBM RMSE: {rmse_lgbm:.4f}")



# Load and preprocess test data
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_df[num_cols] = num_imputer.transform(test_df[num_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])



test_ids = test_df["id"]
X_test = test_df.drop(columns=["id"])
test_predictions = lgbm_model.predict(X_test)


# Create submission file
submission = pd.DataFrame({"id": test_ids, "Price": test_predictions})
submission.to_csv("new_Submission.csv", index=False)
print("Submission file saved as submission.csv")

