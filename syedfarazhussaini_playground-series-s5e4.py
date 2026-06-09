# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For enhanced plotting
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


#Ignore warnings from notebook
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


inputFiles = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        inputFiles[filename.replace(".csv","")] = os.path.join(dirname, filename)

inputFiles


sample_submission_df = pd.read_csv(inputFiles['sample_submission'])
print(sample_submission_df.head())


train_df = pd.read_csv(inputFiles["train"])
print(train_df.head())


test_df = pd.read_csv(inputFiles["test"])
print(test_df.head())


# Print shapes to see number of rows/columns
print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nSample Submission Format:")
print(sample_submission_df.head())


# Train data info
train_df.info()


train_df.describe()


train_df.isnull().sum()


# Distribution of the target variable
plt.figure(figsize=(12, 6))
sns.histplot(train_df['Listening_Time_minutes'], kde=True, bins=50)
plt.title('Distribution of Listening Time (minutes)')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.show()


# Box plot for outliers
plt.figure(figsize=(12, 2))
sns.boxplot(x=train_df['Listening_Time_minutes'])
plt.title('Box Plot of Listening Time (minutes)')
plt.xlabel('Listening Time (minutes)')
plt.show()


numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(train_df[col].dropna(), kde=False, bins=50) # Use dropna() for columns with missing values
    plt.title(f'Distribution of {col}')
    plt.show()
    # Optional: Add boxplots too if you suspect outliers
    # plt.figure(figsize=(10, 1))
    # sns.boxplot(x=train_df[col].dropna())
    # plt.title(f'Box Plot of {col}')
    # plt.show()

# Specifically investigate the Number_of_Ads outlier
print("\nHigh values for Number_of_Ads:")
print(train_df['Number_of_Ads'].sort_values(ascending=False).head())


categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(y=train_df[col], order = train_df[col].value_counts().index) # Order by frequency
    plt.title(f'Frequency of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.show()
    print(f"\nValue Counts for {col}:")
    print(train_df[col].value_counts())
    print("-" * 30)


# Scatter plot (can be slow for large data, maybe sample first if needed)
# sns.pairplot(train_df, vars=numerical_cols + ['Listening_Time_minutes'])
# plt.show()

# Correlation Heatmap (more efficient)
plt.figure(figsize=(10, 8))
# Select only numerical columns for correlation, including the target
corr_cols = numerical_cols + ['Listening_Time_minutes']
correlation_matrix = train_df[corr_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


for col in categorical_cols:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=train_df[col], y=train_df['Listening_Time_minutes'], order=train_df.groupby(col)['Listening_Time_minutes'].median().sort_values().index) # Order by median listening time
    plt.title(f'Listening Time vs {col}')
    plt.xlabel(col)
    plt.ylabel('Listening Time (minutes)')
    plt.xticks(rotation=45)
    plt.show()


# Compare listening time for rows with/without missing Guest_Popularity_percentage
train_df['Guest_Popularity_Missing'] = train_df['Guest_Popularity_percentage'].isnull()
plt.figure(figsize=(8, 5))
sns.boxplot(x='Guest_Popularity_Missing', y='Listening_Time_minutes', data=train_df)
plt.title('Listening Time vs Missing Guest Popularity')
plt.show()
train_df.drop('Guest_Popularity_Missing', axis=1, inplace=True)


# Store number of training rows
ntrain = train_df.shape[0]

# Separate target variable
target = train_df['Listening_Time_minutes']
train_ids = train_df['id']
test_ids = test_df['id']

# Drop target from train_df for now
train_df_features = train_df.drop(['id', 'Listening_Time_minutes'], axis=1)
test_df_features = test_df.drop('id', axis=1) # test_df doesn't have the target

# Combine for preprocessing
combined_df = pd.concat((train_df_features, test_df_features)).reset_index(drop=True)

print("Combined shape:", combined_df.shape)


# Median imputation (calculate median ONLY from train part)
episode_len_median = train_df['Episode_Length_minutes'].median()
combined_df.fillna({'Episode_Length_minutes':episode_len_median}, inplace=True)


# Indicator + Placeholder (-1)
combined_df['Guest_Popularity_IsMissing'] = combined_df['Guest_Popularity_percentage'].isnull().astype(int)
combined_df.fillna({'Guest_Popularity_percentage':-1}, inplace=True)


ads_median = train_df['Number_of_Ads'].median() # Calculate before combining/dropping target if needed
combined_df.fillna({'Number_of_Ads':ads_median}, inplace=True)


# Cap Number_of_Ads at a plausible max, e.g., 15
# max_ads_cap = 15
# combined_df['Number_of_Ads'] = combined_df['Number_of_Ads'].apply(lambda x: min(x, max_ads_cap))
# Or use percentile capping:
percentile_999 = combined_df.loc[:ntrain-1, 'Number_of_Ads'].quantile(0.999)
combined_df['Number_of_Ads'] = combined_df['Number_of_Ads'].apply(lambda x: min(x, percentile_999))


# Example: Cap popularities at 100
combined_df['Host_Popularity_percentage'] = combined_df['Host_Popularity_percentage'].apply(lambda x: min(x, 100.0))
# Don't cap Guest Popularity if you used -1 placeholder for missing! Handle that case.
combined_df['Guest_Popularity_percentage'] = combined_df['Guest_Popularity_percentage'].apply(lambda x: min(x, 100.0) if x != -1 else x)


# One-Hot Encoding
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=False) # drop_first=True can help reduce multicollinearity for linear models


# Careful with division by zero if Episode_Length_minutes could be 0 after imputation
# Add a small epsilon or handle 0s
epsilon = 1e-6
# This feature can ONLY be calculated for the training set initially, as test has no target
# You might skip this for the combined_df and calculate later just on train,
# or predict listening time first then calculate ratio (less useful as a feature then).
# Alternative: Use Episode_Length itself as a strong feature.

# Interaction Features?
combined_df['Host_Guest_Popularity'] = combined_df['Host_Popularity_percentage'] * combined_df['Guest_Popularity_percentage']

# Time/Day Features (if using OHE, this is partly covered, but numerical might help trees)
# day_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
# time_map = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
# combined_df['Publication_Day_Num'] = combined_df['Publication_Day'].map(day_map) # Need original column before OHE
# combined_df['Publication_Time_Num'] = combined_df['Publication_Time'].map(time_map) # Need original column before OHE


#Drop columns unlikely to be useful in raw form for initial model
cols_to_drop = ['Podcast_Name', 'Episode_Title']
# Add original categoricals if you did OHE/Ordinal and don't need them anymore
# cols_to_drop.extend(['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])
combined_df = combined_df.drop(cols_to_drop, axis=1)


train_processed = combined_df[:ntrain]
test_processed = combined_df[ntrain:]

print("Processed Train shape:", train_processed.shape)
print("Processed Test shape:", test_processed.shape)


# Assuming train_processed and target are ready
# 1. Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    train_processed, target, test_size=0.2, random_state=42 # Adjust test_size, random_state is for reproducibility
)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# 2. Model Selection & Initialization
final_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# 3. Training
print("\nTraining model...")
final_model.fit(X_train, y_train)
print("Training complete.")


# 4. Evaluation
y_pred = final_model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"\nValidation RMSE: {rmse}")


# 5. Prediction on Test Data
print("\nMaking predictions on the actual test data...")
test_preds = final_model.predict(test_processed)
print("Predictions complete.")


# 6. Create Submission File
submission_df = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': test_preds})

# Optional: Ensure no negative predictions if listening time can't be negative
submission_df['Listening_Time_minutes'] = submission_df['Listening_Time_minutes'].clip(lower=0)

submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())




