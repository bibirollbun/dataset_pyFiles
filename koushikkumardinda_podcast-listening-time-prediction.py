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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


print(df.head())


print(df.info())


print(df.describe())


# Convert categorical features to numerical

# Label encode 'Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'
categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Handle missing values (imputation)
df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].mean(), inplace=True)
df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].mean(), inplace=True)
df['Number_of_Ads'].fillna(df['Number_of_Ads'].mean(), inplace=True)


# Select features for prediction
features = [
    'Episode_Length_minutes',
    'Genre',
    'Host_Popularity_percentage',
    'Publication_Day',
    'Publication_Time',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
    'Episode_Sentiment'
    # Add more relevant features if available
]


# Target variable
target = 'Listening_Time_minutes'


# Remove rows with missing values in the target variable
df_cleaned = df[features + [target]].dropna()


X = df_cleaned[features]
y = df_cleaned[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)


# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)


# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)


def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} ---")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")


evaluate_model(y_test, linear_predictions, "Linear Regression")
evaluate_model(y_test, rf_predictions, "Random Forest Regressor")
evaluate_model(y_test, gb_predictions, "Gradient Boosting Regressor")


# --- Random Forest Regressor ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=rf_predictions)
plt.xlabel("Actual Listening Time (minutes)")
plt.ylabel("Predicted Listening Time (minutes) - Random Forest")
plt.title("Actual vs. Predicted Listening Time (Random Forest)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()


# --- Linear Regression ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=linear_predictions)
plt.xlabel("Actual Listening Time (minutes)")
plt.ylabel("Predicted Listening Time (minutes) - Linear Regression")
plt.title("Actual vs. Predicted Listening Time (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--') # Diagonal line for perfect predictions
plt.grid(True)
plt.show()


# --- Gradient Boosting Regressor ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=gb_predictions)
plt.xlabel("Actual Listening Time (minutes)")
plt.ylabel("Predicted Listening Time (minutes) - Gradient Boosting Regressor")
plt.title("Actual vs. Predicted Listening Time (Gradient Boosting Regressor)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--') # Diagonal line for perfect predictions
plt.grid(True)
plt.show()


# --- Linear Regression Residuals ---
residuals_linear = y_test - linear_predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals_linear)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Actual Listening Time (minutes)")
plt.ylabel("Residuals (Actual - Predicted) - Linear Regression")
plt.title("Residual Plot - Linear Regression")
plt.grid(True)
plt.show()


# --- Gradient Boosting Regressor Residuals ---
residuals_gb = y_test - gb_predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals_gb)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Actual Listening Time (minutes)")
plt.ylabel("Residuals (Actual - Predicted) - Gradient Boosting Regressor")
plt.title("Residual Plot - Gradient Boosting Regressor")
plt.grid(True)
plt.show()


#Feature Importance (for Gradient Boosting Regressor)
feature_importances = gb_model.feature_importances_
feature_names = X_train.columns

sorted_indices = feature_importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Gradient Boosting Regressor)")
plt.bar(range(X_train.shape[1]), feature_importances[sorted_indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()


df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


print(df.head())


print(df.info())


print(df.describe())


train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
train_df_original = pd.DataFrame(train_data)
test_df_original = pd.DataFrame(test_data)
test_ids = test_df_original['id']
test_df = test_df_original.drop('id', axis=1)


def feature_engineer(df, is_train=True):
    numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads']
    categorical_features = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    df['Length_x_Ads'] = df['Episode_Length_minutes'] * df['Number_of_Ads']
    if is_train:
        target = df['Listening_Time_minutes']
        df = df.drop('Listening_Time_minutes', axis=1)
        return df, target
    else:
        return df


# Check the columns of your DataFrames before calling feature_engineer
print(f"Columns in train_df_original before feature engineering: {train_df_original.columns}")
print(f"Columns in test_df_original before feature engineering: {test_df_original.columns}")

train_df, train_target = feature_engineer(train_df_original.drop('id', axis=1), is_train=True)
test_df_processed = feature_engineer(test_df_original.drop('id', axis=1), is_train=False)

print(f"Columns in train_df after feature engineering: {train_df.columns}")
print(f"Columns in test_df_processed after feature engineering: {test_df_processed.columns}")


# Feature Engineering
train_df, train_target = feature_engineer(train_df_original.drop('id', axis=1), is_train=True)
test_df_processed = feature_engineer(test_df, is_train=False)

# Identify numerical columns for imputation (important!)
numerical_cols = train_df.select_dtypes(include=['number']).columns

# Create an imputer object (you can choose 'mean', 'median', 'most_frequent', or 'constant')
imputer = SimpleImputer(strategy='mean') # Using mean imputation

# Fit the imputer on the numerical training data and transform it
train_df[numerical_cols] = imputer.fit_transform(train_df[numerical_cols])


# Train your model
model = RandomForestRegressor(random_state=42)
model.fit(train_df, train_target)


# Apply the SAME imputer to your test data BEFORE prediction
test_df_processed = feature_engineer(test_df_original.drop('id', axis=1), is_train=False)
test_df_processed[numerical_cols] = imputer.transform(test_df_processed[numerical_cols])


# Make predictions on the processed test data
predictions = model.predict(test_df_processed)


print(f"Length of test_ids: {len(test_ids)}")
print(f"Shape of processed test_df: {test_df_processed.shape}")
print(f"Length of predictions: {len(predictions)}")


# Create submission DataFrame
if len(predictions) == len(test_ids):
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Listening_Time_minutes': predictions.round(2)
    })
    print("Submission DataFrame created successfully.")
    print(submission_df.head())
else:
    print(f"Error: Length mismatch between predictions ({len(predictions)}) and test_ids ({len(test_ids)}).")

submission_df.to_csv('submission.csv', index=False)


submission_file_path = 'sample_submission.csv'
submission_df.to_csv(submission_file_path, index=False)


print(f"Submission file created successfully at: {submission_file_path}")
print("\nFirst few rows of the submission file:")
print(submission_df.head())

