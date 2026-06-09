import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Drop ID column if it exists
train = train.drop(columns=['id'], errors='ignore')
test = test.drop(columns=['id'], errors='ignore')

# Define categorical mappings
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}  # Convert to numbers

# Apply mapping using .loc[] to avoid warnings
for df in [train, test]:
    df.loc[:, 'Publication_Day'] = df['Publication_Day'].map(day_mapping)
    df.loc[:, 'Publication_Time'] = df['Publication_Time'].map(time_mapping)
    df.loc[:, 'Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_mapping)

# Identify categorical and numerical features
categorical_features = ['Podcast_Name', 'Genre']
numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                      'Guest_Popularity_percentage', 'Number_of_Ads',
                      'Episode_Sentiment', 'Publication_Day', 'Publication_Time']

# Fill missing values (fixing the FutureWarning)
train[numerical_features] = train[numerical_features].apply(lambda col: col.fillna(col.median()))
test[numerical_features] = test[numerical_features].apply(lambda col: col.fillna(col.median()))

train[categorical_features] = train[categorical_features].fillna("Unknown")
test[categorical_features] = test[categorical_features].fillna("Unknown")

# One-Hot Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_encoded = encoder.fit_transform(train[categorical_features])
test_encoded = encoder.transform(test[categorical_features])

# Convert encoded arrays to DataFrames
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_features))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Reset indices
train_encoded_df.index = train.index
test_encoded_df.index = test.index

# Drop original categorical columns and concatenate encoded features
train_final = pd.concat([train[numerical_features], train_encoded_df], axis=1)
test_final = pd.concat([test[numerical_features], test_encoded_df], axis=1)

# Separate features and target variable
X = train_final
y = train['Listening_Time_minutes']
X_test = test_final

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Validate model
y_val_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f'Validation RMSE: {rmse:.4f}')

# Predict and save submission
test['Listening_Time_minutes'] = model.predict(X_test)
submission = test[['Listening_Time_minutes']]
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")

