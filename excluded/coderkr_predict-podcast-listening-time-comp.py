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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# Drop high-cardinality columns (too many unique values, low signal)
drop_cols = ['Podcast_Name', 'Episode_Title']
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# Separate features and target
X = train.drop(['Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']

# Identify types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessors
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Processed data
X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.1, random_state=42)

# Models
ridge = Ridge()
gbr = GradientBoostingRegressor(n_estimators=50, max_depth=3)  # small, fast

ridge.fit(X_train, y_train)
gbr.fit(X_train, y_train)

# Quick eval
val_preds = (ridge.predict(X_val) + gbr.predict(X_val)) / 2
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")

# Final prediction
final_preds = (ridge.predict(test_processed) + gbr.predict(test_processed)) / 2

# Submission
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': final_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Fast submission saved.")





