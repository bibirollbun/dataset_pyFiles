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


import warnings
warnings.filterwarnings("ignore")


train1=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train2=pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample=pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


train = pd.concat([train1, train2], axis=0, ignore_index=True)
train.isnull().sum()


train['Brand'].fillna('Unknown', inplace=True)
train['Material'].fillna(train['Material'].mode()[0], inplace=True)
train['Size'].fillna(train['Size'].mode()[0], inplace=True)
train['Laptop Compartment'].fillna(train['Laptop Compartment'].mode()[0], inplace=True)
train['Waterproof'].fillna(train['Waterproof'].mode()[0], inplace=True)
train['Style'].fillna(train['Style'].mode()[0], inplace=True)
train['Color'].fillna(train['Color'].mode()[0], inplace=True)
train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median(), inplace=True)



print(train.isnull().sum())



test.isnull().sum()


# Fill missing categorical values with the most frequent category (mode)
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for col in categorical_columns:
    test[col].fillna(test[col].mode()[0], inplace=True)

# Fill missing numerical values with median
test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].median(), inplace=True)

# Verify no missing values remain
print(test.isnull().sum())



from category_encoders import TargetEncoder

# Initialize the Target Encoder with correct parameters
TE = TargetEncoder(smoothing=20)  # Corrected parameter name

# Define categorical features to encode
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                       'Waterproof', 'Style', 'Color']

# Apply Target Encoding
for col in categorical_columns:
    TE.fit(train[col], train['Price'])  # Fit the encoder on the train data
    train[col] = TE.transform(train[col])  # Transform train set
    test[col] = TE.transform(test[col])  # Transform test set



# Define the target variable
y = train['Price']

# Drop the target column from training data to get feature matrix
X = train.drop(columns=['Price'])



from sklearn.model_selection import train_test_split

# Split the indices of the train data
train_id, val_id = train_test_split(train.index, test_size=0.2, random_state=42)

# Create train and validation sets
X_train, X_val = X.iloc[train_id], X.iloc[val_id]
y_train, y_val = y.iloc[train_id], y.iloc[val_id]



from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor,log_evaluation

# Define models with optimal parameters
xgb_model = XGBRegressor(
    device="cuda",
    max_depth=5,
    n_estimators=2000,
    learning_rate=0.015,
    random_state=42
)

cat_model = CatBoostRegressor(
    iterations=2000,
    depth=6,
    learning_rate=0.02,
    loss_function='RMSE',
    verbose=200,
    random_seed=42
)

lgb_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.015,
    max_depth=6,
    objective='rmse',
    random_state=42
)

# Train models
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', verbose=200)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=200)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[log_evaluation(200)])



# Validation predictions
y_pred_xgb = xgb_model.predict(X_val)
y_pred_cat = cat_model.predict(X_val)
y_pred_lgb = lgb_model.predict(X_val)

# Test set predictions
y_test_pred_xgb = xgb_model.predict(test)
y_test_pred_cat = cat_model.predict(test)
y_test_pred_lgb = lgb_model.predict(test)

# Weighted average of predictions (adjust weights as needed)
y_val_pred = (0.4 * y_pred_xgb) + (0.3 * y_pred_cat) + (0.3 * y_pred_lgb)
y_test_pred = (0.4 * y_test_pred_xgb) + (0.3 * y_test_pred_cat) + (0.3 * y_test_pred_lgb)



from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")



# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': test['id'],
    'Price': y_test_pred
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

# Display the first few rows
submission.head()


