import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# Separate target variable
y = train_df['Listening_Time_minutes']
train_df.drop(columns=['Listening_Time_minutes'], inplace=True)


# Identify categorical and numerical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns


# Fill missing values
train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].median())
test_df[numerical_cols] = test_df[numerical_cols].fillna(train_df[numerical_cols].median())

for col in categorical_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    test_df[col].fillna(train_df[col].mode()[0], inplace=True)


# One-hot encode categorical columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_train = pd.DataFrame(encoder.fit_transform(train_df[categorical_cols]))
encoded_test = pd.DataFrame(encoder.transform(test_df[categorical_cols]))


# Restore index after encoding
encoded_train.index = train_df.index
encoded_test.index = test_df.index


# Combine encoded and numerical columns
X_train = pd.concat([train_df[numerical_cols].reset_index(drop=True), encoded_train.reset_index(drop=True)], axis=1)
X_test = pd.concat([test_df[numerical_cols].reset_index(drop=True), encoded_test.reset_index(drop=True)], axis=1)


# Train-validation split
X_train_split, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)


# Standardize numerical features
scaler = StandardScaler()
X_train_split[numerical_cols] = scaler.fit_transform(X_train_split[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# Initialize models
xgb_model = XGBRegressor(learning_rate=0.05, max_depth=6, n_estimators=1500, random_state=42)
lgbm_model = LGBMRegressor(learning_rate=0.05, num_leaves=31, n_estimators=1500, random_state=42)
catboost_model = CatBoostRegressor(learning_rate=0.05, depth=6, iterations=1500, random_state=42, verbose=0)

# Train and evaluate models
models = {
    'XGBoost': xgb_model,
    'LightGBM': lgbm_model,
    'CatBoost': catboost_model
}

for name, model in models.items():
    model.fit(X_train_split, y_train)
    y_val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    print(f'{name} RMSE: {rmse}')

# Select the best model based on RMSE
best_model_name = min(models, key=lambda name: mean_squared_error(y_val, models[name].predict(X_val), squared=False))
best_model = models[best_model_name]
print(f'Best Model: {best_model_name}')


# Generate predictions for test data
test_predictions = best_model.predict(X_test)

df_sub['Listening_Time_minutes'] = test_predictions


df_sub.to_csv('submission.csv', index=False)


df_sub['Listening_Time_minutes'].hist()

