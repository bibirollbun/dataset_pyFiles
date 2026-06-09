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


import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets with the correct paths
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Display basic information
print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)

# Display first few rows of the training data
print("\nFirst 5 rows of training data:")
print(train_df.head())

# Check for missing values
print("\nMissing values in training data:")
print(train_df.isnull().sum())

# Basic statistics of the target variable
print("\nStatistics of the target variable (Listening_Time_minutes):")
print(train_df['Listening_Time_minutes'].describe())

# Display column names
print("\nColumn names in training data:")
print(train_df.columns.tolist())

# Check data types
print("\nData types in training data:")
print(train_df.dtypes)

# Check if test set has the same features (except target)
print("\nColumn names in test data:")
print(test_df.columns.tolist())



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Set up the figure size for plots
plt.figure(figsize=(15, 10))

# 1. Distribution of the target variable
plt.subplot(2, 2, 1)
sns.histplot(train_df['Listening_Time_minutes'], kde=True)
plt.title('Distribution of Listening Time')
plt.xlabel('Listening Time (minutes)')

# 2. Relationship between Episode Length and Listening Time
plt.subplot(2, 2, 2)
sns.scatterplot(data=train_df.sample(5000), x='Episode_Length_minutes', y='Listening_Time_minutes', alpha=0.5)
plt.title('Episode Length vs Listening Time')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')

# 3. Listening Time by Genre (top 10 genres)
plt.subplot(2, 2, 3)
top_genres = train_df['Genre'].value_counts().nlargest(10).index
genre_data = train_df[train_df['Genre'].isin(top_genres)]
sns.boxplot(data=genre_data, x='Genre', y='Listening_Time_minutes')
plt.title('Listening Time by Genre (Top 10)')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Genre')
plt.ylabel('Listening Time (minutes)')

# 4. Listening Time by Publication Day
plt.subplot(2, 2, 4)
sns.boxplot(data=train_df, x='Publication_Day', y='Listening_Time_minutes')
plt.title('Listening Time by Publication Day')
plt.xlabel('Publication Day')
plt.ylabel('Listening Time (minutes)')

plt.tight_layout()
plt.savefig('eda_plots_1.png')
plt.close()

# Additional plots
plt.figure(figsize=(15, 10))

# 5. Listening Time by Publication Time
plt.subplot(2, 2, 1)
sns.boxplot(data=train_df, x='Publication_Time', y='Listening_Time_minutes')
plt.title('Listening Time by Publication Time')
plt.xlabel('Publication Time')
plt.ylabel('Listening Time (minutes)')

# 6. Relationship between Host Popularity and Listening Time
plt.subplot(2, 2, 2)
sns.scatterplot(data=train_df.sample(5000), x='Host_Popularity_percentage', y='Listening_Time_minutes', alpha=0.5)
plt.title('Host Popularity vs Listening Time')
plt.xlabel('Host Popularity (%)')
plt.ylabel('Listening Time (minutes)')

# 7. Relationship between Guest Popularity and Listening Time
plt.subplot(2, 2, 3)
sns.scatterplot(data=train_df.dropna(subset=['Guest_Popularity_percentage']).sample(5000), 
                x='Guest_Popularity_percentage', y='Listening_Time_minutes', alpha=0.5)
plt.title('Guest Popularity vs Listening Time')
plt.xlabel('Guest Popularity (%)')
plt.ylabel('Listening Time (minutes)')

# 8. Listening Time by Episode Sentiment
plt.subplot(2, 2, 4)
sns.boxplot(data=train_df, x='Episode_Sentiment', y='Listening_Time_minutes')
plt.title('Listening Time by Episode Sentiment')
plt.xlabel('Episode Sentiment')
plt.ylabel('Listening Time (minutes)')

plt.tight_layout()
plt.savefig('eda_plots_2.png')
plt.close()

# Correlation analysis for numerical features
numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                      'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']
correlation_df = train_df[numerical_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_matrix.png')
plt.close()

# Print some additional statistics
print("Number of unique podcasts:", train_df['Podcast_Name'].nunique())
print("Number of unique genres:", train_df['Genre'].nunique())
print("\nTop 10 podcasts by frequency:")
print(train_df['Podcast_Name'].value_counts().nlargest(10))
print("\nGenre distribution:")
print(train_df['Genre'].value_counts())
print("\nPublication Day distribution:")
print(train_df['Publication_Day'].value_counts())
print("\nPublication Time distribution:")
print(train_df['Publication_Time'].value_counts())
print("\nEpisode Sentiment distribution:")
print(train_df['Episode_Sentiment'].value_counts())

# Check for any patterns in missing values - fixed approach
print("\nMissing Episode_Length_minutes by Genre:")
missing_by_genre = train_df.groupby('Genre').apply(lambda x: x['Episode_Length_minutes'].isna().mean()).sort_values(ascending=False)
print(missing_by_genre)

print("\nMissing Guest_Popularity_percentage by Genre:")
missing_guest_by_genre = train_df.groupby('Genre').apply(lambda x: x['Guest_Popularity_percentage'].isna().mean()).sort_values(ascending=False)
print(missing_guest_by_genre)

# Additional analysis: Average listening time by genre
print("\nAverage Listening Time by Genre:")
avg_listening_by_genre = train_df.groupby('Genre')['Listening_Time_minutes'].mean().sort_values(ascending=False)
print(avg_listening_by_genre)

# Average listening time by publication day
print("\nAverage Listening Time by Publication Day:")
avg_listening_by_day = train_df.groupby('Publication_Day')['Listening_Time_minutes'].mean().sort_values(ascending=False)
print(avg_listening_by_day)

# Average listening time by publication time
print("\nAverage Listening Time by Publication Time:")
avg_listening_by_time = train_df.groupby('Publication_Time')['Listening_Time_minutes'].mean().sort_values(ascending=False)
print(avg_listening_by_time)

# Average listening time by sentiment
print("\nAverage Listening Time by Episode Sentiment:")
avg_listening_by_sentiment = train_df.groupby('Episode_Sentiment')['Listening_Time_minutes'].mean().sort_values(ascending=False)
print(avg_listening_by_sentiment)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Function to preprocess data
def preprocess_data(df, is_training=True):
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Extract episode number from Episode_Title where available
    processed_df['Episode_Number'] = processed_df['Episode_Title'].str.extract(r'Episode\s+(\d+)').astype(float)
    
    # Handle missing values
    # For Episode_Length_minutes, impute with median grouped by Genre
    genre_median_length = processed_df.groupby('Genre')['Episode_Length_minutes'].transform('median')
    processed_df['Episode_Length_minutes'].fillna(genre_median_length, inplace=True)
    
    # For any remaining missing Episode_Length_minutes, use overall median
    processed_df['Episode_Length_minutes'].fillna(processed_df['Episode_Length_minutes'].median(), inplace=True)
    
    # For Guest_Popularity_percentage, create a flag for missing values and impute with median
    processed_df['Guest_Missing'] = processed_df['Guest_Popularity_percentage'].isna().astype(int)
    processed_df['Guest_Popularity_percentage'].fillna(processed_df['Guest_Popularity_percentage'].median(), inplace=True)
    
    # For Number_of_Ads, impute with 0 (assuming missing means no ads)
    processed_df['Number_of_Ads'].fillna(0, inplace=True)
    
    # Encode categorical variables
    categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    
    for col in categorical_cols:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])
    
    # Drop columns that won't be used in modeling
    processed_df.drop(['id', 'Episode_Title'], axis=1, inplace=True)
    
    # If training data, separate features and target
    if is_training:
        y = processed_df['Listening_Time_minutes']
        X = processed_df.drop('Listening_Time_minutes', axis=1)
        return X, y
    else:
        return processed_df

# Preprocess the training data
X, y = preprocess_data(train_df, is_training=True)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a baseline model (Random Forest)
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions on validation set
rf_val_preds = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val_preds))
print(f"Random Forest Validation RMSE: {rf_rmse:.4f}")

# Train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions on validation set
xgb_val_preds = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_preds))
print(f"XGBoost Validation RMSE: {xgb_rmse:.4f}")

# Train LightGBM model
print("\nTraining LightGBM model...")
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)

# Make predictions on validation set
lgb_val_preds = lgb_model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_preds))
print(f"LightGBM Validation RMSE: {lgb_rmse:.4f}")

# Feature importance for the best model (using LightGBM as an example)
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (LightGBM)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Select the best model (based on validation results)
best_model = lgb_model  # This will be updated based on the results

# Preprocess the test data
test_processed = preprocess_data(test_df, is_training=False)

# Make predictions on the test set
test_predictions = best_model.predict(test_processed)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': test_predictions
})

submission.to_csv('submission_1.csv', index=False)
print("\nSubmission file created.")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Function for feature engineering and preprocessing
def preprocess_data(df, is_training=True):
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Extract episode number from Episode_Title where available
    processed_df['Episode_Number'] = processed_df['Episode_Title'].str.extract(r'Episode\s+(\d+)').astype(float)
    
    # Create a feature for whether Episode_Length_minutes is missing
    processed_df['Length_Missing'] = processed_df['Episode_Length_minutes'].isna().astype(int)
    
    # Create a feature for whether Guest_Popularity_percentage is missing
    processed_df['Guest_Missing'] = processed_df['Guest_Popularity_percentage'].isna().astype(int)
    
    # Handle missing values
    # For Episode_Length_minutes, impute with median grouped by Genre
    genre_median_length = processed_df.groupby('Genre')['Episode_Length_minutes'].transform('median')
    processed_df['Episode_Length_minutes'] = processed_df['Episode_Length_minutes'].fillna(genre_median_length)
    
    # For any remaining missing Episode_Length_minutes, use overall median
    processed_df['Episode_Length_minutes'] = processed_df['Episode_Length_minutes'].fillna(processed_df['Episode_Length_minutes'].median())
    
    # For Guest_Popularity_percentage, impute with median grouped by Genre
    genre_median_guest = processed_df.groupby('Genre')['Guest_Popularity_percentage'].transform('median')
    processed_df['Guest_Popularity_percentage'] = processed_df['Guest_Popularity_percentage'].fillna(genre_median_guest)
    
    # For any remaining missing Guest_Popularity_percentage, use overall median
    processed_df['Guest_Popularity_percentage'] = processed_df['Guest_Popularity_percentage'].fillna(processed_df['Guest_Popularity_percentage'].median())
    
    # For Number_of_Ads, impute with 0 (assuming missing means no ads)
    processed_df['Number_of_Ads'] = processed_df['Number_of_Ads'].fillna(0)
    
    # Create interaction features
    # Ratio of Host to Guest popularity
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Popularity_percentage'] / processed_df['Guest_Popularity_percentage']
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Guest_Ratio'].replace([np.inf, -np.inf], 0)
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Guest_Ratio'].fillna(0)
    
    # Interaction between Episode Length and Number of Ads
    processed_df['Length_per_Ad'] = processed_df['Episode_Length_minutes'] / (processed_df['Number_of_Ads'] + 1)  # +1 to avoid division by zero
    
    # Day of week encoding (using cyclic features)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_to_num = {day: i for i, day in enumerate(days_order)}
    processed_df['Day_Num'] = processed_df['Publication_Day'].map(day_to_num)
    processed_df['Day_Sin'] = np.sin(2 * np.pi * processed_df['Day_Num'] / 7)
    processed_df['Day_Cos'] = np.cos(2 * np.pi * processed_df['Day_Num'] / 7)
    
    # Time of day encoding
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_to_num = {time: i for i, time in enumerate(time_order)}
    processed_df['Time_Num'] = processed_df['Publication_Time'].map(time_to_num)
    processed_df['Time_Sin'] = np.sin(2 * np.pi * processed_df['Time_Num'] / 4)
    processed_df['Time_Cos'] = np.cos(2 * np.pi * processed_df['Time_Num'] / 4)
    
    # Sentiment encoding (Ordinal: Negative < Neutral < Positive)
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    processed_df['Sentiment_Ordinal'] = processed_df['Episode_Sentiment'].map(sentiment_map)
    
    # Encode categorical variables
    categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    
    for col in categorical_cols:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])
    
    # Drop columns that won't be used in modeling
    processed_df.drop(['id', 'Episode_Title', 'Day_Num', 'Time_Num'], axis=1, inplace=True)
    
    # If training data, separate features and target
    if is_training:
        y = processed_df['Listening_Time_minutes']
        X = processed_df.drop('Listening_Time_minutes', axis=1)
        return X, y
    else:
        return processed_df

# Preprocess the training data
X, y = preprocess_data(train_df, is_training=True)

# Verify no NaN values remain
print("NaN values in X:", X.isna().sum().sum())
print("NaN values in y:", y.isna().sum())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with optimized hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

# Train models
print("Training Random Forest model...")
rf_model.fit(X_train, y_train)
rf_val_preds = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val_preds))
print(f"Random Forest Validation RMSE: {rf_rmse:.4f}")

print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)
xgb_val_preds = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_preds))
print(f"XGBoost Validation RMSE: {xgb_rmse:.4f}")

print("\nTraining LightGBM model...")
lgb_model.fit(X_train, y_train)
lgb_val_preds = lgb_model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_preds))
print(f"LightGBM Validation RMSE: {lgb_rmse:.4f}")

# Create an ensemble model (weighted average of predictions)
print("\nCreating ensemble model...")
ensemble_preds = (0.4 * rf_val_preds + 0.3 * xgb_val_preds + 0.3 * lgb_val_preds)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_preds))
print(f"Ensemble Validation RMSE: {ensemble_rmse:.4f}")

# Feature importance for Random Forest
plt.figure(figsize=(12, 10))
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

# Preprocess the test data
test_processed = preprocess_data(test_df, is_training=False)

# Make predictions with each model
rf_test_preds = rf_model.predict(test_processed)
xgb_test_preds = xgb_model.predict(test_processed)
lgb_test_preds = lgb_model.predict(test_processed)

# Create ensemble predictions
ensemble_test_preds = (0.4 * rf_test_preds + 0.3 * xgb_test_preds + 0.3 * lgb_test_preds)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': ensemble_test_preds
})

submission.to_csv('submission_2.csv', index=False)
print("\nSubmission file created.")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Function for feature engineering and preprocessing
def preprocess_data(df, podcast_stats=None, genre_stats=None, is_training=True):
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Extract episode number from Episode_Title where available
    processed_df['Episode_Number'] = processed_df['Episode_Title'].str.extract(r'Episode\s+(\d+)').astype(float)
    
    # Create a feature for whether Episode_Length_minutes is missing
    processed_df['Length_Missing'] = processed_df['Episode_Length_minutes'].isna().astype(int)
    
    # Create a feature for whether Guest_Popularity_percentage is missing
    processed_df['Guest_Missing'] = processed_df['Guest_Popularity_percentage'].isna().astype(int)
    
    # Handle missing values
    # For Episode_Length_minutes, impute with median grouped by Genre
    genre_median_length = processed_df.groupby('Genre')['Episode_Length_minutes'].transform('median')
    processed_df['Episode_Length_minutes'] = processed_df['Episode_Length_minutes'].fillna(genre_median_length)
    
    # For any remaining missing Episode_Length_minutes, use overall median
    processed_df['Episode_Length_minutes'] = processed_df['Episode_Length_minutes'].fillna(processed_df['Episode_Length_minutes'].median())
    
    # For Guest_Popularity_percentage, impute with median grouped by Genre
    genre_median_guest = processed_df.groupby('Genre')['Guest_Popularity_percentage'].transform('median')
    processed_df['Guest_Popularity_percentage'] = processed_df['Guest_Popularity_percentage'].fillna(genre_median_guest)
    
    # For any remaining missing Guest_Popularity_percentage, use overall median
    processed_df['Guest_Popularity_percentage'] = processed_df['Guest_Popularity_percentage'].fillna(processed_df['Guest_Popularity_percentage'].median())
    
    # For Number_of_Ads, impute with 0 (assuming missing means no ads)
    processed_df['Number_of_Ads'] = processed_df['Number_of_Ads'].fillna(0)
    
    # Create interaction features
    # Ratio of Host to Guest popularity
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Popularity_percentage'] / processed_df['Guest_Popularity_percentage']
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Guest_Ratio'].replace([np.inf, -np.inf], 0)
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Guest_Ratio'].fillna(0)
    
    # Interaction between Episode Length and Number of Ads
    processed_df['Length_per_Ad'] = processed_df['Episode_Length_minutes'] / (processed_df['Number_of_Ads'] + 1)  # +1 to avoid division by zero
    
    # Percentage of episode listened to (only for training data)
    if is_training and 'Listening_Time_minutes' in processed_df.columns:
        processed_df['Listen_Percentage'] = processed_df['Listening_Time_minutes'] / processed_df['Episode_Length_minutes']
        processed_df['Listen_Percentage'] = processed_df['Listen_Percentage'].replace([np.inf, -np.inf], 1)
        processed_df['Listen_Percentage'] = processed_df['Listen_Percentage'].fillna(0)
        processed_df['Listen_Percentage'] = processed_df['Listen_Percentage'].clip(0, 1)  # Cap at 100%
    
    # Day of week encoding (using cyclic features)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_to_num = {day: i for i, day in enumerate(days_order)}
    processed_df['Day_Num'] = processed_df['Publication_Day'].map(day_to_num)
    processed_df['Day_Sin'] = np.sin(2 * np.pi * processed_df['Day_Num'] / 7)
    processed_df['Day_Cos'] = np.cos(2 * np.pi * processed_df['Day_Num'] / 7)
    
    # Time of day encoding
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_to_num = {time: i for i, time in enumerate(time_order)}
    processed_df['Time_Num'] = processed_df['Publication_Time'].map(time_to_num)
    processed_df['Time_Sin'] = np.sin(2 * np.pi * processed_df['Time_Num'] / 4)
    processed_df['Time_Cos'] = np.cos(2 * np.pi * processed_df['Time_Num'] / 4)
    
    # Sentiment encoding (Ordinal: Negative < Neutral < Positive)
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    processed_df['Sentiment_Ordinal'] = processed_df['Episode_Sentiment'].map(sentiment_map)
    
    # Add podcast and genre statistics from training data
    if is_training:
        # Calculate podcast statistics
        podcast_stats = {}
        for podcast in processed_df['Podcast_Name'].unique():
            podcast_data = processed_df[processed_df['Podcast_Name'] == podcast]
            podcast_stats[podcast] = {
                'mean_listening': podcast_data['Listening_Time_minutes'].mean(),
                'std_listening': podcast_data['Listening_Time_minutes'].std(),
                'median_listening': podcast_data['Listening_Time_minutes'].median()
            }
        
        # Calculate genre statistics
        genre_stats = {}
        for genre in processed_df['Genre'].unique():
            genre_data = processed_df[processed_df['Genre'] == genre]
            genre_stats[genre] = {
                'mean_listening': genre_data['Listening_Time_minutes'].mean(),
                'std_listening': genre_data['Listening_Time_minutes'].std(),
                'median_listening': genre_data['Listening_Time_minutes'].median()
            }
    
    if podcast_stats is not None:
        # Add podcast statistics
        processed_df['Podcast_Mean_Listening'] = processed_df['Podcast_Name'].map(
            {k: v['mean_listening'] for k, v in podcast_stats.items()}
        )
        processed_df['Podcast_Std_Listening'] = processed_df['Podcast_Name'].map(
            {k: v['std_listening'] for k, v in podcast_stats.items()}
        )
        processed_df['Podcast_Median_Listening'] = processed_df['Podcast_Name'].map(
            {k: v['median_listening'] for k, v in podcast_stats.items()}
        )
    
    if genre_stats is not None:
        # Add genre statistics
        processed_df['Genre_Mean_Listening'] = processed_df['Genre'].map(
            {k: v['mean_listening'] for k, v in genre_stats.items()}
        )
        processed_df['Genre_Std_Listening'] = processed_df['Genre'].map(
            {k: v['std_listening'] for k, v in genre_stats.items()}
        )
        processed_df['Genre_Median_Listening'] = processed_df['Genre'].map(
            {k: v['median_listening'] for k, v in genre_stats.items()}
        )
    
    # Encode categorical variables
    categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    
    for col in categorical_cols:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])
    
    # Drop columns that won't be used in modeling
    processed_df.drop(['id', 'Episode_Title', 'Day_Num', 'Time_Num'], axis=1, inplace=True)
    
    # If training data, separate features and target
    if is_training:
        if 'Listen_Percentage' in processed_df.columns:
            processed_df.drop('Listen_Percentage', axis=1, inplace=True)
        y = processed_df['Listening_Time_minutes']
        X = processed_df.drop('Listening_Time_minutes', axis=1)
        return X, y, podcast_stats, genre_stats
    else:
        return processed_df

# Preprocess the training data
X, y, podcast_stats, genre_stats = preprocess_data(train_df, is_training=True)

# Verify no NaN values remain
print("NaN values in X:", X.isna().sum().sum())
print("NaN values in y:", y.isna().sum())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with optimized hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

# Train models
print("Training Random Forest model...")
rf_model.fit(X_train, y_train)
rf_val_preds = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val_preds))
print(f"Random Forest Validation RMSE: {rf_rmse:.4f}")

print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)
xgb_val_preds = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_preds))
print(f"XGBoost Validation RMSE: {xgb_rmse:.4f}")

print("\nTraining LightGBM model...")
lgb_model.fit(X_train, y_train)
lgb_val_preds = lgb_model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_preds))
print(f"LightGBM Validation RMSE: {lgb_rmse:.4f}")

# Create an ensemble model (stacking approach)
print("\nCreating stacked ensemble model...")
# Use a simple average for this example
ensemble_preds = (rf_val_preds + xgb_val_preds + lgb_val_preds) / 3
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_preds))
print(f"Ensemble Validation RMSE: {ensemble_rmse:.4f}")

# Feature importance for Random Forest
plt.figure(figsize=(12, 10))
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

# Preprocess the test data
test_processed = preprocess_data(test_df, podcast_stats, genre_stats, is_training=False)

# Make predictions with each model
rf_test_preds = rf_model.predict(test_processed)
xgb_test_preds = xgb_model.predict(test_processed)
lgb_test_preds = lgb_model.predict(test_processed)

# Create ensemble predictions
ensemble_test_preds = (rf_test_preds + xgb_test_preds + lgb_test_preds) / 3

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': ensemble_test_preds
})

submission.to_csv('submission_3.csv', index=False)
print("\nSubmission file created.")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Function for feature engineering and preprocessing
def preprocess_data(df, is_training=True):
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Extract episode number from Episode_Title where available
    processed_df['Episode_Number'] = processed_df['Episode_Title'].str.extract(r'Episode\s+(\d+)').astype(float)
    
    # Create a feature for whether Episode_Length_minutes is missing
    processed_df['Length_Missing'] = processed_df['Episode_Length_minutes'].isna().astype(int)
    
    # Create a feature for whether Guest_Popularity_percentage is missing
    processed_df['Guest_Missing'] = processed_df['Guest_Popularity_percentage'].isna().astype(int)
    
    # Handle missing values
    # For Episode_Length_minutes, impute with median grouped by Genre
    genre_median_length = processed_df.groupby('Genre')['Episode_Length_minutes'].transform('median')
    processed_df['Episode_Length_minutes'] = processed_df['Episode_Length_minutes'].fillna(genre_median_length)
    
    # For any remaining missing Episode_Length_minutes, use overall median
    processed_df['Episode_Length_minutes'] = processed_df['Episode_Length_minutes'].fillna(processed_df['Episode_Length_minutes'].median())
    
    # For Guest_Popularity_percentage, impute with median grouped by Genre
    genre_median_guest = processed_df.groupby('Genre')['Guest_Popularity_percentage'].transform('median')
    processed_df['Guest_Popularity_percentage'] = processed_df['Guest_Popularity_percentage'].fillna(genre_median_guest)
    
    # For any remaining missing Guest_Popularity_percentage, use overall median
    processed_df['Guest_Popularity_percentage'] = processed_df['Guest_Popularity_percentage'].fillna(processed_df['Guest_Popularity_percentage'].median())
    
    # For Number_of_Ads, impute with 0 (assuming missing means no ads)
    processed_df['Number_of_Ads'] = processed_df['Number_of_Ads'].fillna(0)
    
    # Create interaction features
    # Ratio of Host to Guest popularity
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Popularity_percentage'] / processed_df['Guest_Popularity_percentage']
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Guest_Ratio'].replace([np.inf, -np.inf], 0)
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Guest_Ratio'].fillna(0)
    
    # Interaction between Episode Length and Number of Ads
    processed_df['Length_per_Ad'] = processed_df['Episode_Length_minutes'] / (processed_df['Number_of_Ads'] + 1)  # +1 to avoid division by zero
    
    # Day of week encoding (using cyclic features)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_to_num = {day: i for i, day in enumerate(days_order)}
    processed_df['Day_Num'] = processed_df['Publication_Day'].map(day_to_num)
    processed_df['Day_Sin'] = np.sin(2 * np.pi * processed_df['Day_Num'] / 7)
    processed_df['Day_Cos'] = np.cos(2 * np.pi * processed_df['Day_Num'] / 7)
    
    # Time of day encoding
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_to_num = {time: i for i, time in enumerate(time_order)}
    processed_df['Time_Num'] = processed_df['Publication_Time'].map(time_to_num)
    processed_df['Time_Sin'] = np.sin(2 * np.pi * processed_df['Time_Num'] / 4)
    processed_df['Time_Cos'] = np.cos(2 * np.pi * processed_df['Time_Num'] / 4)
    
    # Sentiment encoding (Ordinal: Negative < Neutral < Positive)
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    processed_df['Sentiment_Ordinal'] = processed_df['Episode_Sentiment'].map(sentiment_map)
    
    # Encode categorical variables
    categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    
    for col in categorical_cols:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])
    
    # Drop columns that won't be used in modeling
    processed_df.drop(['id', 'Episode_Title', 'Day_Num', 'Time_Num'], axis=1, inplace=True)
    
    # If training data, separate features and target
    if is_training:
        y = processed_df['Listening_Time_minutes']
        X = processed_df.drop('Listening_Time_minutes', axis=1)
        return X, y
    else:
        return processed_df

# Preprocess the training data
X, y = preprocess_data(train_df, is_training=True)

# Verify no NaN values remain
print("NaN values in X:", X.isna().sum().sum())
print("NaN values in y:", y.isna().sum())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest model with optimized hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Train the model
print("Training Random Forest model...")
rf_model.fit(X_train, y_train)
rf_val_preds = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val_preds))
print(f"Random Forest Validation RMSE: {rf_rmse:.4f}")

# Feature importance for Random Forest
plt.figure(figsize=(12, 10))
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

# Preprocess the test data
test_processed = preprocess_data(test_df, is_training=False)

# Make predictions on the test set
test_predictions = rf_model.predict(test_processed)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created.")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import time

# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Function for feature engineering and preprocessing
def preprocess_data(df, is_training=True):
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Extract episode number from Episode_Title where available
    processed_df['Episode_Number'] = processed_df['Episode_Title'].str.extract(r'Episode\s+(\d+)').astype(float)
    
    # Create a feature for whether Episode_Length_minutes is missing
    processed_df['Length_Missing'] = processed_df['Episode_Length_minutes'].isna().astype(int)
    
    # Create a feature for whether Guest_Popularity_percentage is missing
    processed_df['Guest_Missing'] = processed_df['Guest_Popularity_percentage'].isna().astype(int)
    
    # Handle missing values
    # For Episode_Length_minutes, impute with median grouped by Genre
    genre_median_length = processed_df.groupby('Genre')['Episode_Length_minutes'].transform('median')
    processed_df['Episode_Length_minutes'] = processed_df['Episode_Length_minutes'].fillna(genre_median_length)
    
    # For any remaining missing Episode_Length_minutes, use overall median
    processed_df['Episode_Length_minutes'] = processed_df['Episode_Length_minutes'].fillna(processed_df['Episode_Length_minutes'].median())
    
    # For Guest_Popularity_percentage, impute with median grouped by Genre
    genre_median_guest = processed_df.groupby('Genre')['Guest_Popularity_percentage'].transform('median')
    processed_df['Guest_Popularity_percentage'] = processed_df['Guest_Popularity_percentage'].fillna(genre_median_guest)
    
    # For any remaining missing Guest_Popularity_percentage, use overall median
    processed_df['Guest_Popularity_percentage'] = processed_df['Guest_Popularity_percentage'].fillna(processed_df['Guest_Popularity_percentage'].median())
    
    # For Number_of_Ads, impute with 0 (assuming missing means no ads)
    processed_df['Number_of_Ads'] = processed_df['Number_of_Ads'].fillna(0)
    
    # Create interaction features
    # Ratio of Host to Guest popularity
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Popularity_percentage'] / processed_df['Guest_Popularity_percentage']
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Guest_Ratio'].replace([np.inf, -np.inf], 0)
    processed_df['Host_Guest_Ratio'] = processed_df['Host_Guest_Ratio'].fillna(0)
    
    # Interaction between Episode Length and Number of Ads
    processed_df['Length_per_Ad'] = processed_df['Episode_Length_minutes'] / (processed_df['Number_of_Ads'] + 1)  # +1 to avoid division by zero
    
    # Day of week encoding (using cyclic features)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_to_num = {day: i for i, day in enumerate(days_order)}
    processed_df['Day_Num'] = processed_df['Publication_Day'].map(day_to_num)
    processed_df['Day_Sin'] = np.sin(2 * np.pi * processed_df['Day_Num'] / 7)
    processed_df['Day_Cos'] = np.cos(2 * np.pi * processed_df['Day_Num'] / 7)
    
    # Time of day encoding
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_to_num = {time: i for i, time in enumerate(time_order)}
    processed_df['Time_Num'] = processed_df['Publication_Time'].map(time_to_num)
    processed_df['Time_Sin'] = np.sin(2 * np.pi * processed_df['Time_Num'] / 4)
    processed_df['Time_Cos'] = np.cos(2 * np.pi * processed_df['Time_Num'] / 4)
    
    # Sentiment encoding (Ordinal: Negative < Neutral < Positive)
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    processed_df['Sentiment_Ordinal'] = processed_df['Episode_Sentiment'].map(sentiment_map)
    
    # Encode categorical variables
    categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    
    for col in categorical_cols:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])
    
    # Drop columns that won't be used in modeling
    processed_df.drop(['id', 'Episode_Title', 'Day_Num', 'Time_Num'], axis=1, inplace=True)
    
    # If training data, separate features and target
    if is_training:
        y = processed_df['Listening_Time_minutes']
        X = processed_df.drop('Listening_Time_minutes', axis=1)
        return X, y
    else:
        return processed_df

# Preprocess the training data
print("Preprocessing data...")
X, y = preprocess_data(train_df, is_training=True)

# Verify no NaN values remain
print("NaN values in X:", X.isna().sum().sum())
print("NaN values in y:", y.isna().sum())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter search space for XGBoost
space = {
    'max_depth': scope.int(hp.quniform('max_depth', 3, 12, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 500, 50)),
    'gamma': hp.uniform('gamma', 0, 5),
    'min_child_weight': hp.uniform('min_child_weight', 1, 10),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-10), np.log(1)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-10), np.log(1))
}

# Define the objective function for Bayesian optimization
def objective(params):
    # Convert integer parameters
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
    # Create XGBoost model with the current hyperparameters
    model = xgb.XGBRegressor(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        gamma=params['gamma'],
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=42
    )
    
    # Perform cross-validation
    cv_score = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=3, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    ).mean()
    
    # Return the negative RMSE (since we want to maximize the objective)
    return {'loss': -cv_score, 'status': STATUS_OK}

# Run Bayesian optimization
print("Starting Bayesian optimization for XGBoost hyperparameters...")
start_time = time.time()
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=30,  # Number of iterations for optimization
    trials=trials,
    rstate=np.random.RandomState(42)
)
end_time = time.time()
print(f"Optimization completed in {end_time - start_time:.2f} seconds")

# Get the best hyperparameters
best_params = {
    'max_depth': int(best['max_depth']),
    'learning_rate': best['learning_rate'],
    'n_estimators': int(best['n_estimators']),
    'gamma': best['gamma'],
    'min_child_weight': best['min_child_weight'],
    'subsample': best['subsample'],
    'colsample_bytree': best['colsample_bytree'],
    'reg_alpha': best['reg_alpha'],
    'reg_lambda': best['reg_lambda']
}

print("\nBest hyperparameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Train the final XGBoost model with the best hyperparameters
print("\nTraining final XGBoost model with best hyperparameters...")
final_model = xgb.XGBRegressor(
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    n_estimators=best_params['n_estimators'],
    gamma=best_params['gamma'],
    min_child_weight=best_params['min_child_weight'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    random_state=42
)

final_model.fit(X_train, y_train)

# Evaluate on validation set
val_preds = final_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"XGBoost Validation RMSE with best hyperparameters: {val_rmse:.4f}")

# Feature importance for XGBoost
plt.figure(figsize=(12, 10))
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': final_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance (XGBoost)')
plt.tight_layout()
plt.savefig('xgb_feature_importance.png')
plt.close()

# Preprocess the test data
test_processed = preprocess_data(test_df, is_training=False)

# Make predictions on the test set
test_predictions = final_model.predict(test_processed)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created.")


