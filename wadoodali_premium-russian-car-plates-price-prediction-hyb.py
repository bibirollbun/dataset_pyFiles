import pandas as pd
import numpy as np
import ast
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib


# Load the datasets
train_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')



# Display the first few rows of each file
train_df.head(), test_df.head(), sample_submission_df.head()



# Convert 'date' column to datetime type
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])



def extract_plate_features(df):
    df['prefix'] = df['plate'].str[:1]  # First letter(s) of the plate
    df['numeric_part'] = df['plate'].str.extract('(\d+)').astype(int)  # Extract numeric part
    df['region_code'] = df['plate'].str[-3:]  # Last 3 characters as region code
    return df



def extract_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df



with open('/kaggle/input/russian-car-plates-prices-prediction/supplemental_english.py', 'r', encoding='utf-8') as f:
    supplemental_english = f.read()

with open('/kaggle/input/russian-car-plates-prices-prediction/supplemental_russian.py', 'r', encoding='utf-8') as f:
    supplemental_russian = f.read()


# Extract REGION_CODES from the supplemental files
english_region_codes = ast.literal_eval(supplemental_english.split("REGION_CODES = ")[1].split("\n\n")[0])
russian_region_codes = ast.literal_eval(supplemental_russian.split("REGION_CODES = ")[1].split("\n\n")[0])


# Create region-to-name mapping
region_to_name = {}
for region_name, codes in english_region_codes.items():
    for code in codes:
        region_to_name[code] = region_name



# Extract features from plate
train_df = extract_plate_features(train_df)
test_df = extract_plate_features(test_df)

# Extract date features
train_df = extract_date_features(train_df) # Calling the function to extract date features for train_df
test_df = extract_date_features(test_df)  # Calling the function to extract date features for test_df


# Function to map region codes to region names
def map_region_name(df):
    df['region_name'] = df['region_code'].map(region_to_name).fillna('Unknown')
    return df

# Apply region mapping
train_df = map_region_name(train_df)
test_df = map_region_name(test_df)




# Encode 'prefix' and 'region_name'
prefix_encoder = LabelEncoder()
region_name_encoder = LabelEncoder()
train_df['prefix_encoded'] = prefix_encoder.fit_transform(train_df['prefix'])
test_df['prefix_encoded'] = prefix_encoder.transform(test_df['prefix'])
train_df['region_name_encoded'] = region_name_encoder.fit_transform(train_df['region_name'])
test_df['region_name_encoded'] = region_name_encoder.transform(test_df['region_name'])


# Select features for training
features = ['prefix_encoded', 'numeric_part', 'region_name_encoded', 'year', 'month', 'day', 'day_of_week', 'is_weekend']
X_train = train_df[features]
y_train = train_df['price']
X_test = test_df[features]



# Train and evaluate XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_train_pred_xgb = xgb_model.predict(X_train)
xgb_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
print('XGBoost RMSE:', xgb_rmse)



# Train and evaluate LightGBM model
lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)
y_train_pred_lgb = lgb_model.predict(X_train)
lgb_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_lgb))
print('LightGBM RMSE:', lgb_rmse)


# Make predictions on the test set
xgb_test_pred = xgb_model.predict(X_test)
lgb_test_pred = lgb_model.predict(X_test)


# Prepare submission file
submission = sample_submission_df.copy()
submission['price'] = (xgb_test_pred + lgb_test_pred) / 2  # Averaging predictions
submission.to_csv('submission.csv', index=False)



# Save models
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(lgb_model, 'lgb_model.pkl')

















