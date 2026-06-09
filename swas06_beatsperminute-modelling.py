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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.head(3)


df_train.isnull().sum(),df_test.isnull().sum()


X = df_train.drop('beatsperminute',axis=1)
y = df_train['beatsperminute']



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
#categorical_features = X.select_dtypes(include=["object", "category"]).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
    ("scaler", StandardScaler())  # Standardize numerical features
])


# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features)
    
])

# Create a pipeline with XGBoost Regressor
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error
import numpy as np

# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print RMSE
print(f"Model RMSE: {rmse:.4f}")



y_test_pred = pipeline.predict(df_test)


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

