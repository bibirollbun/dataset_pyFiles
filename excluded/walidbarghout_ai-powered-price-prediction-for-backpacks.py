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
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Load data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra_data = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train_data.head()


train_extra_data.head()


test_data.head()


train_data.info()


train_extra_data.info()


test_data.info()


train_data.dtypes


train_extra_data.dtypes


test_data.dtypes


train_data.isna().count()


train_extra_data.isna().count()


test_data.isna().count()


# Combine original and extra training data
combined_train_data = pd.concat([train_data, train_extra_data], axis=0)


# Separate features and target
X = combined_train_data.drop(['id', 'Price'], axis=1)
y = combined_train_data['Price']
test_ids = test_data['id']
X_test = test_data.drop(['id'], axis=1)


# Identify categorical and numerical columns
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]


# Preprocessing pipeline
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# Define model with early_stopping_rounds in the constructor
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=10  # Set early_stopping_rounds here
)


# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])


# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Preprocess the validation data separately
X_val_preprocessed = preprocessor.fit_transform(X_val)


# Train model with early stopping
pipeline.fit(X_train, y_train,
             model__eval_set=[(X_val_preprocessed, y_val)],
             model__verbose=False)


# Predict on validation set
val_pred = pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'Validation RMSE: {rmse}')


# Retrain on full training data
model.set_params(early_stopping_rounds=None)  # Disable early stopping
pipeline.fit(X, y)


# Predict on test data
test_pred = pipeline.predict(X_test)


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'Price': test_pred})
submission.to_csv('submission.csv', index=False)

