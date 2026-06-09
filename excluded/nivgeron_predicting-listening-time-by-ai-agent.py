import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style('whitegrid')

# Load the training data from Kaggle input folder
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

# Display the first few rows
train_df.head()


# Get basic information about the dataframe
print("DataFrame Info:")
train_df.info()

print("\n" + "="*50 + "\n") # Separator

# Get summary statistics for numerical columns
print("Numerical Summary Statistics:")
print(train_df.describe())

print("\n" + "="*50 + "\n") # Separator

# Get summary statistics for categorical columns
print("Categorical Summary Statistics:")
print(train_df.describe(include='object'))


# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Listening_Time_minutes'], kde=True, bins=50)
plt.title('Distribution of Listening Time (minutes)')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.show()

# Also print descriptive statistics for the target variable
print("Descriptive Statistics for Listening Time:")
print(train_df['Listening_Time_minutes'].describe())


# Calculate the percentage of missing values for each column
missing_percentage = (train_df.isnull().sum() / len(train_df)) * 100

# Filter and display columns with missing values
missing_info = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

print("Percentage of Missing Values per Column:")
print(missing_info)

# Visualize the missing data percentages
if not missing_info.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_info.index, y=missing_info.values)
    plt.title('Percentage of Missing Values by Feature')
    plt.xlabel('Features')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("\nNo missing values found in the dataset.")


# Visualize the distribution of Genre
plt.figure(figsize=(12, 6))
sns.countplot(y='Genre', data=train_df, order=train_df['Genre'].value_counts().index, palette='viridis')
plt.title('Distribution of Podcast Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()

# Print value counts for Genre as well
print("Value Counts for Genre:")
print(train_df['Genre'].value_counts())


# Visualize the distribution of Episode_Sentiment
plt.figure(figsize=(8, 5))
sns.countplot(x='Episode_Sentiment', data=train_df, order=train_df['Episode_Sentiment'].value_counts().index, palette='magma')
plt.title('Distribution of Episode Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Print value counts for Episode_Sentiment as well
print("Value Counts for Episode Sentiment:")
print(train_df['Episode_Sentiment'].value_counts())


# Select numerical columns (excluding 'id')
numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('id') # Exclude the ID column

# Calculate the correlation matrix
correlation_matrix = train_df[numerical_cols].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Create a copy for preprocessing
train_processed_df = train_df.copy()

# Identify numerical columns with missing values (excluding ID and target)
numerical_cols_with_nan = ['Guest_Popularity_percentage', 'Episode_Length_minutes', 'Number_of_Ads']

# Impute missing values with the median
for col in numerical_cols_with_nan:
    median_val = train_processed_df[col].median()
    train_processed_df[col].fillna(median_val, inplace=True)
    print(f"Imputed missing values in '{col}' with median: {median_val}")

# Verify imputation
print("\nMissing values after imputation:")
print(train_processed_df[numerical_cols_with_nan].isnull().sum())


# Convert categorical columns to 'category' dtype
categorical_cols = ['Genre', 'Episode_Sentiment']

for col in categorical_cols:
    train_processed_df[col] = train_processed_df[col].astype('category')

# Verify conversion
print("Data types after conversion:")
print(train_processed_df.dtypes)


print(train_processed_df.columns)


print("Unique values in Publication_Day:", train_processed_df['Publication_Day'].unique())
print("Unique values in Publication_Time:", train_processed_df['Publication_Time'].unique())


# Feature Engineering: Numerical Encoding of Publication Time

# Create dictionaries for mapping
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}

# Apply the mappings
train_processed_df['Publication_Day_Encoded'] = train_processed_df['Publication_Day'].map(day_mapping)
train_processed_df['Publication_Time_Encoded'] = train_processed_df['Publication_Time'].map(time_mapping)

# Drop original columns
train_processed_df = train_processed_df.drop(['Publication_Day', 'Publication_Time'], axis=1)

# Display the encoded features
print(train_processed_df[['Publication_Day_Encoded', 'Publication_Time_Encoded']].head())


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare the data
X = train_processed_df.drop(['id', 'Listening_Time_minutes', 'Podcast_Name', 'Episode_Title'], axis=1)
y = train_processed_df['Listening_Time_minutes']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model parameters
lgbm_params = {
    'objective': 'rmse',
    'metric': 'rmse',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'n_jobs': -1,
    'seed': 42,
}

# Train the LightGBM model
model = lgb.LGBMRegressor(**lgbm_params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=10)])

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")


# Plot feature importances
lgb.plot_importance(model, figsize=(10, 8), max_num_features=20)
plt.title('LightGBM Feature Importances')
plt.tight_layout()
plt.show()


# Load the test data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_ids = test_df['id'] # Store IDs for submission file

# --- Apply Preprocessing ---

# 1. Impute missing numerical values using TRAINING data medians
numerical_cols_with_nan = ['Guest_Popularity_percentage', 'Episode_Length_minutes', 'Number_of_Ads']
for col in numerical_cols_with_nan:
    # IMPORTANT: Use median from the original training data (train_df)
    median_val = train_df[col].median()
    test_df[col].fillna(median_val, inplace=True)
    print(f"Imputed missing values in test '{col}' with training median: {median_val}")

# 2. Convert categorical columns to 'category' dtype
categorical_cols = ['Genre', 'Episode_Sentiment']
for col in categorical_cols:
    # Ensure consistency with training data categories
    test_df[col] = pd.Categorical(test_df[col], categories=train_processed_df[col].cat.categories)

# 3. Encode Publication_Day and Publication_Time using TRAINING data mappings
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
test_df['Publication_Day_Encoded'] = test_df['Publication_Day'].map(day_mapping)
test_df['Publication_Time_Encoded'] = test_df['Publication_Time'].map(time_mapping)

# 4. Drop original and unnecessary columns
test_processed_df = test_df.drop(['id', 'Podcast_Name', 'Episode_Title', 'Publication_Day', 'Publication_Time'], axis=1)

# Verify preprocessing
print("\nTest data info after preprocessing:")
test_processed_df.info()
print("\nTest data head after preprocessing:")
print(test_processed_df.head())

# --- Make Predictions ---

# Ensure columns are in the same order as training data
test_processed_df = test_processed_df[X_train.columns]

# Predict on the preprocessed test data
test_predictions = model.predict(test_processed_df)

# --- Create Submission File ---
submission_df = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")
print(submission_df.head())

