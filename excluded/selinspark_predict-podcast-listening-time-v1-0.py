!pip install lightgbm
# %% [code]
# Install LightGBM if not installed (uncomment if needed)
# !pip install lightgbm


# %% [code]
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set a random seed for reproducibility
RANDOM_STATE = 42



# %% [code]
# Data Loading
# Load the training, test, and sample_submission datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# Display the shape of the datasets to understand the structure
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample Submission shape:", sample_submission.shape)

# Display sample data for initial verification
print("\n--- Train Sample ---")
print(train.head())
print("\n--- Test Sample ---")
print(test.head())



# %% [code]
# Data Preprocessing & Imputation
# Handle missing values by filling with column mean for numeric and mode for categorical features
cat_features = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 
                'Publication_Time', 'Episode_Sentiment']
num_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                'Guest_Popularity_percentage', 'Number_of_Ads']

# Fill missing values in numeric features with the column mean
for col in num_features:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(train[col].mean())

# Fill missing values in categorical features with the mode
for col in cat_features:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])



# %% [code]
# Label Encoding for Categorical Features
# Label encode categorical columns to convert them into numeric format
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)  # Combine train and test to fit LabelEncoder
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le



# %% [code]
# Feature Engineering: Creating new features
# Create new features that could help improve model performance

# Popularity ratio: The ratio of Host Popularity to Guest Popularity
train['Popularity_Ratio'] = train['Host_Popularity_percentage'] / (train['Guest_Popularity_percentage'] + 1e-6)
test['Popularity_Ratio'] = test['Host_Popularity_percentage'] / (test['Guest_Popularity_percentage'] + 1e-6)

# Ads per minute: Number of ads per minute of episode length
train['Ads_per_Minute'] = train['Number_of_Ads'] / (train['Episode_Length_minutes'] + 1e-6)
test['Ads_per_Minute'] = test['Number_of_Ads'] / (test['Episode_Length_minutes'] + 1e-6)

# Adding the new features to the numeric features list
num_features.extend(['Popularity_Ratio', 'Ads_per_Minute'])



# %% [code]
# Preparing the data for training
# Split the data into features (X) and target variable (y)
X = train[num_features + cat_features]
y = train['Listening_Time_minutes']
X_test = test[num_features + cat_features]

# Split the training data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)



# %% [code]
# Training the Model: LightGBM
# Setting up the LightGBM parameters for regression
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'random_state': RANDOM_STATE
}

# Initialize and train the LightGBM model
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train)



# %% [code]
# Model Evaluation: Calculate RMSE on the validation set
lgb_preds = lgb_model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_preds))
print(f"\nLightGBM Validation RMSE: {lgb_rmse:.4f}")



# %% [code]
# Predictions on the Test Set
# Use the trained LightGBM model to make predictions on the test set
test_preds = lgb_model.predict(X_test)

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': test_preds
})

# Save the submission file to the local directory as "submission.csv"
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")

# List current directory files to verify the submission file creation
print("\nCurrent Directory Files:")
print(os.listdir('.'))


