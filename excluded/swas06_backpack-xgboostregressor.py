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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_trainextra= pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


print(df_train.head(3))
print(df_trainextra.head(3))


print(df_train.shape)
print(df_trainextra.shape)


merged_df = pd.concat([df_train, df_trainextra], ignore_index=True)

# Display the merged DataFrame
print(merged_df.head(5))

X = merged_df.iloc[:, :-1]
y = merged_df.iloc[:, -1]


merged_df.shape


print(df_test.head(3))
print(df_test.shape)


print(merged_df.info())
print(df_test.info())


print(merged_df.isnull().sum())
print(df_test.isnull().sum())


numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)
print(f"\nTotal number of numerical columns: {len(numerical_columns)}")

print("\nCategorical Columns:")
print(categorical_columns)
print(f"\nTotal number of categorical columns: {len(categorical_columns)}")


merged_df[numerical_columns] = merged_df[numerical_columns].fillna(merged_df[numerical_columns].mean())
df_test[numerical_columns] = df_test[numerical_columns].fillna(df_test[numerical_columns].mean())


merged_df[categorical_columns] = merged_df[categorical_columns].fillna("Missing")
df_test[categorical_columns] = df_test[categorical_columns].fillna("Missing")


print(merged_df.isnull().sum())
print(df_test.isnull().sum())


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    
    ("scaler", StandardScaler())  # Standardize numerical features
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
   
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # One-hot encoding for categorical variables
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create a pipeline with XGBoost Regressor
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)




from sklearn.metrics import mean_squared_error

# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print RMSE
print(f"Model RMSE: {rmse:.4f}")


y_test_pred = pipeline.predict(df_test)


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
submission_df['Price'] = y_test_pred
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

