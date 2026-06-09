import pandas as pd


# Load the training and test datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


# Display the first 5 rows of the training data
print("Training Data Head:")
print("-" * 100)
print(train_df.head())
print("-" * 100)


# Display a summary of the training data
print("Training Data Info:")
print("-" * 100)
print(train_df.info())
print("-" * 100)


# Display the first 5 rows of the test data
print("Test Data Head:")
print("-" * 100)
print(test_df.head())
print("-" * 100)


# Display a summary of the test data
print("Test Data Info:")
print("-" * 100)
print(test_df.info())
print("-" * 100)


# Check for missing values in training datasets
print("Missing values in Training Data:")
print("-" * 100)
print(train_df.isnull().sum())
print("-" * 100)


# Check for missing values in test datasets
print("Missing values in Test Data:")
print("-" * 100)
print(test_df.isnull().sum())
print("-" * 100)


# Define the features (X) and the target (y)
features = [col for col in train_df.columns if col not in ['id', 'BeatsPerMinute']]
X_train = train_df[features]
y_train = train_df['BeatsPerMinute']


# Prepare the test data
X_test = test_df[features]


import lightgbm as lgb

# Initialize the LightGBM Regressor model
lgbm_model = lgb.LGBMRegressor(random_state=42, force_col_wise=True)

# Train the model on the entire training dataset
print("Training the LightGBM Model")
print("-" * 100)
lgbm_model.fit(X_train, y_train)
print("-" * 100)


# Use the trained model to make predictions on the test data
print("Making predictions on the test data")
print("-" * 100)
predictions = lgbm_model.predict(X_test)

# Display the first 10 predictions to verify the output
print(predictions[:10])
print("-" * 100)


# Create a new DataFrame for the submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'BeatsPerMinute': predictions
})


# Display the first 5 rows of the submission DataFrame to verify the format
print("Submission DataFrame Head")
print("-" * 100)
print(submission_df.head())
print("-" * 100)


# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")

