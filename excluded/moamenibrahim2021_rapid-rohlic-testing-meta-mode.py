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


# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import joblib


# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Read the data
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
test = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
inventory = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
calendar = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
test_weights = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")
solution = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv")

# Merge data
train = pd.merge(train, inventory, how='left', on=['unique_id', 'warehouse'])
train = pd.merge(train, calendar, how='left', on=['date', 'warehouse'])
test = pd.merge(test, inventory, how='left', on=['unique_id', 'warehouse'])
test = pd.merge(test, calendar, how='left', on=['date', 'warehouse'])

# Clean up duplicate columns
y_columns = [col for col in train.columns if col.endswith('_y')]
train = train.drop(columns=y_columns)
test = test.drop(columns=y_columns)
train = train.rename(columns={col: col.replace('_x', '') for col in train.columns if col.endswith('_x')})
test = test.rename(columns={col: col.replace('_x', '') for col in test.columns if col.endswith('_x')})

# Check for any NaN values after merge
print("NaN values in train:", train.isnull().sum())

# Drop rows with missing total_orders or sales
train = train.dropna(subset=['total_orders', 'sales'])

# Fill missing holiday_name with 'No Holiday'
train['holiday_name'] = train['holiday_name'].fillna('No Holiday')
test['holiday_name'] = test['holiday_name'].fillna('No Holiday')

# Convert date columns to datetime
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
calendar["date"] = pd.to_datetime(calendar["date"])

# Check the basic info about train data
print("Train data shape: ", train.shape)
print("\nFirst few rows of training data:")
print(train.head())

# Check train data types
print(train.dtypes)

# Check for train date range
print(f"Train dates: {train['date'].min()} to {train['date'].max()}")
print(f"Test dates: {test['date'].min()} to {test['date'].max()}")

# Check basic train data
train.head()

# Check the test data types
print(test.dtypes)

# Check test date range
print(f"Test dates: {test['date'].min()} to {test['date'].max()}")

# Create the has_any_discount column
for df in [train, test]:
    df['has_any_discount'] = (df[['type_0_discount', 'type_1_discount', 'type_2_discount', 
                                  'type_3_discount', 'type_4_discount', 'type_5_discount', 
                                  'type_6_discount']] > 0).any(axis=1).astype(int)

# Sort data by ID and date
train = train.sort_values(['unique_id', 'date'])
test = test.sort_values(['unique_id', 'date'])

# Check shapes before lag features
print(f"Train shape before lag features: {train.shape}")
print(f"Test shape before lag features: {test.shape}")

# Combine train and test datasets
test['sales'] = np.nan  # Add empty sales column to test
combined_data = pd.concat([train, test]).sort_values(['unique_id', 'date'])

# Create lag features
combined_data['sales_7days_ago'] = combined_data.groupby('unique_id')['sales'].shift(7)
combined_data['sales_14days_ago'] = combined_data.groupby('unique_id')['sales'].shift(14)
combined_data['sales_28days_ago'] = combined_data.groupby('unique_id')['sales'].shift(28)
combined_data['sales_365days_ago'] = combined_data.groupby('unique_id')['sales'].shift(365)

# Fill NaN values in lag features
for lag_col in ['sales_7days_ago', 'sales_14days_ago', 'sales_28days_ago', 'sales_365days_ago']:
    combined_data[lag_col] = combined_data.groupby('unique_id')[lag_col].fillna(method='ffill')
    combined_data[lag_col] = combined_data[lag_col].fillna(0)  # Fill remaining NaNs with 0

# Split back into train and test
train = combined_data[combined_data['sales'].notna()].copy()
test = combined_data[combined_data['sales'].isna()].copy()

# Drop rows with missing lag features
train = train.dropna()

# Fill missing lag values in test with mean of recent values
for lag_col in ['sales_7days_ago', 'sales_14days_ago', 'sales_28days_ago', 'sales_365days_ago']:
    combined_data[lag_col] = combined_data.groupby('unique_id')[lag_col].fillna(method='ffill')
    # If still any NaNs, fill with 0
    combined_data[lag_col] = combined_data[lag_col].fillna(0)

# Shapes after lag features:
print(f"Train shape after lag features: {train.shape}")

# Use last 14 days as validation (same length as test period)
val_start_date = '2024-05-17'
val_mask = train['date'] >= val_start_date
train_data = train[~val_mask]  # Exclude validation period
val_data = train[val_mask]

# Verify the dates
print(f"Train data: {train_data['date'].min()} to {train_data['date'].max()}")
print(f"Validation: {val_data['date'].min()} to {val_data['date'].max()}")
print(f"Test: {test['date'].min()} to {test['date'].max()}")

# Prepare features for modeling
def prepare_features(df):
    # Prepare features for modeling
    
    # Extract day of week, month, and year from the date
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday = 0, Sunday = 6
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Discount feature if it doesn't exist
    if 'has_any_discount' not in df.columns:
        df['has_any_discount'] = (df[['type_0_discount', 'type_1_discount', 'type_2_discount', 
                                      'type_3_discount', 'type_4_discount', 'type_5_discount', 
                                      'type_6_discount']] > 0).any(axis=1).astype(int)
    
    feature_cols = [
        'day_of_week', 'month', 'year', 
        'has_any_discount', 'holiday', 'shops_closed',
        'winter_school_holidays', 'school_holidays',
        'warehouse', 'total_orders', 'sell_price_main',
        'L1_category_name_en', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en',  
        'sales_7days_ago', 'sales_14days_ago', 'sales_28days_ago', 'sales_365days_ago'  
    ]
    
    return df[feature_cols]

# Prepare features for each dataset
X_train = prepare_features(train_data)
y_train = train_data['sales']
X_val = prepare_features(val_data)
y_val = val_data['sales']
X_test = prepare_features(test)

# Merge weights before modeling
train_with_weights = pd.merge(train_data, test_weights, on='unique_id', how='left')
val_with_weights = pd.merge(val_data, test_weights, on='unique_id', how='left')

# Check that data looks good before modeling
print("Shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}")

# Check all have same columns
print("\nAll datasets have same columns?")
print(f"X_train columns: {X_train.columns.tolist()}")
print(f"X_val columns: {X_val.columns.tolist()}")
print(f"X_test columns: {X_test.columns.tolist()}")

# Check for any missing values
print("\nMissing values:")
print("X_train:", X_train.isnull().sum().sum())
print("X_val:", X_val.isnull().sum().sum())
print("X_test:", X_test.isnull().sum().sum())

# Quick look at feature ranges
print("\nFeature ranges in train vs test:")
for col in X_train.select_dtypes(include=['int64', 'float64']).columns:
    print(f"\n{col}:")
    print(f"Train range: {X_train[col].min():.2f} to {X_train[col].max():.2f}")
    print(f"Test range: {X_test[col].min():.2f} to {X_test[col].max():.2f}")

# Check data types before modeling
print(f"X_train types\n: {X_train.dtypes}\n")
print(f"y_train types\n: {y_train.dtypes}\n")
print(f"X_val types\n: {X_val.dtypes}\n")
print(f"y_val types\n: {y_val.dtypes}\n")
print(f"X_test types\n: {X_test.dtypes}\n")



from sklearn.preprocessing import LabelEncoder

# Define categorical columns
categorical_columns = ['warehouse', 'L1_category_name_en', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']

# Apply Label Encoding to categorical columns
encoder = LabelEncoder()

for col in categorical_columns:
    X_train[col] = encoder.fit_transform(X_train[col])
    X_val[col] = encoder.transform(X_val[col])
    X_test[col] = encoder.transform(X_test[col])



import joblib

# Load models
xgboost1 = joblib.load('/kaggle/input/models/xgboost1_model.pkl')
#xgboost2 = joblib.load('/kaggle/input/models/xgboost2_model.pkl')
lgbm1 = joblib.load('/kaggle/input/models/lgbm1_model.pkl')
#lgbm2 = joblib.load('/kaggle/input/models/lgbm2_model.pkl')
#lgbm3 = joblib.load('/kaggle/input/models/lgbm3_model.pkl')
lgbm4 = joblib.load('/kaggle/input/models/lgbm4_model.pkl')
catboost1 = joblib.load('/kaggle/input/models/catboost1_model.pkl')
#catboost2 = joblib.load('/kaggle/input/models/catboost2_model.pkl')
#hist_grad = joblib.load('/kaggle/input/models/hist_grad_model.pkl')


# from sklearn.linear_model import Lasso, ElasticNet

# # Step 2: Get predictions from base models (for meta-model training)
# preds_xgboost1 = xgboost1.predict(X_train)
# # preds_xgboost2 = xgboost2.predict(X_train)
# preds_lgbm1 = lgbm1.predict(X_train)
# # preds_lgbm2 = lgbm2.predict(X_train)
# # preds_lgbm3 = lgbm3.predict(X_train)
# preds_lgbm4 = lgbm4.predict(X_train)
# preds_catboost1 = catboost1.predict(X_train)
# # preds_catboost2 = catboost2.predict(X_train)
# # preds_hist_grad = hist_grad.predict(X_train)

# # Combine base model predictions into a new feature matrix (X_meta) for meta-model training
# X_meta_train = np.column_stack((
#     preds_xgboost1,
#     # preds_xgboost2,
#     preds_lgbm1,
#     # preds_lgbm2, 
#     # preds_lgbm3,
#     preds_lgbm4,
#     preds_catboost1,
#     # preds_catboost2,
#     # preds_hist_grad
# ))


# # Step 4: Make predictions using the base models on the test set
# preds_xgboost1_test = xgboost1.predict(X_test)
# # preds_xgboost2_test = xgboost2.predict(X_test)
# preds_lgbm1_test = lgbm1.predict(X_test)
# # preds_lgbm2_test = lgbm2.predict(X_test)
# # preds_lgbm3_test = lgbm3.predict(X_test)
# preds_lgbm4_test = lgbm4.predict(X_test)
# preds_catboost1_test = catboost1.predict(X_test)
# # preds_catboost2_test = catboost2.predict(X_test)
# # preds_hist_grad_test = hist_grad.predict(X_test)

# # Combine the test predictions into a new feature matrix (X_meta_test) for the meta-model
# X_meta_test = np.column_stack((
#     preds_xgboost1_test,
#     # preds_xgboost2_test,
#     preds_lgbm1_test,
#     # preds_lgbm2_test, 
#     # preds_lgbm3_test,
#     preds_lgbm4_test,
#     preds_catboost1_test,
#     # preds_catboost2_test,
#     # preds_hist_grad_test
# ))


# import pandas as pd

# # Convert to DataFrame and save as CSV
# pd.DataFrame(X_meta_train).to_csv("X_meta_train.csv", index=False)
# pd.DataFrame(X_meta_test).to_csv("X_meta_test.csv", index=False)

# print("✅ X_meta_train and X_meta_test saved as CSV files.")


X_meta_train = pd.read_csv("/kaggle/working/X_meta_train.csv").values
X_meta_test = pd.read_csv("/kaggle/working/X_meta_test.csv").values


# Convert labels to NumPy arrays
y_train = y_train.values

X_meta_train.shape


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os

# # ===========================
# # 2️⃣ Define the DNN Meta-Model (TensorFlow/Keras)
# # ===========================

# # Build the model
# meta_model = keras.Sequential([
#     keras.layers.Dense(32, activation='relu', input_shape=(X_meta_train.shape[1],)),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(16, activation='relu'),
#     keras.layers.Dense(1)  # Output layer for regression
# ])

# # Compile the model
# meta_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#                    loss='mae',
#                    metrics=['mae'])

# meta_model.summary()


# # ===========================
# # 3️⃣ Train the Meta-Model
# # ===========================

# # Define early stopping

# # Train the model
# history = meta_model.fit(
#     X_meta_train, y_train,
#     validation_split=0.2,
#     epochs=20,
#     batch_size=64,
#     verbose=1
# )
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()


# # Save model in HDF5 format
# meta_model.save("meta_model.h5")
# print("✅ Model saved as 'meta_model.h5'")


meta_model = keras.models.load_model("meta_model.h5", custom_objects={"mae": keras.metrics.MAE})


# ===========================
# 4️⃣ Make Predictions with the Meta-Model
# ===========================
# Step 5: Make the final predictions with the meta-model
# Make predictions
final_preds = meta_model.predict(X_meta_test)
final_preds = final_preds.flatten()  # Convert (n,1) to (n,)



# Step 6: Prepare a submission or evaluate the model
# Create a submission DataFrame for the test data
test_ids = test['unique_id'].astype(str) + '_' + test['date'].dt.strftime('%Y-%m-%d')
submission = pd.DataFrame({
    'id': test_ids,
    'sales_hat': final_preds
})


# Quick sanity check
print("Submission sample:")
print(submission.head())
print("\nShape:", submission.shape)
print("\nCheck for any negative predictions:")
print((submission['sales_hat'] < 0).sum())


# Save the submission
submission.to_csv('submission.csv', index=False)

