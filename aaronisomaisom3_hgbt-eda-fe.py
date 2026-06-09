!pip install optuna --quiet


# Author: Aaron Isom
# Kaggle Predict Podcast Listening Time
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Load Data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
original_df = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")

# Concatenate original data with train
train_df = pd.concat([train_df, original_df], axis=0, ignore_index=True)
train_df = train_df.drop_duplicates()
train_df = train_df[~train_df['Listening_Time_minutes'].isna()]

# Display first few rows
display(train_df.head(10))
display('Train Shape', train_df.shape)
display('Test Shape', test_df.shape)

display('Missing Train Values:', train_df.isnull().sum())
display('Missing Test Values:', test_df.isnull().sum())

# Describe the data
display(train_df.describe())

# Display information about dtypes
display('Train Data Info:', train_df.info())

# Target distribution
plt.figure(figsize=(12, 6))
sns.histplot(train_df['Listening_Time_minutes'], kde=True, bins=50)
plt.title('Distribution of Listening Time')
plt.xlabel('Minutes')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Episode_Length_minutes', y='Listening_Time_minutes', data=train_df, alpha=0.3)
sns.regplot(x='Episode_Length_minutes', y='Listening_Time_minutes', data=train_df, scatter=False, color='red')
plt.title('Listening Time vs Episode Length')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df, x='Genre', y='Listening_Time_minutes')
plt.xticks(rotation=45)
plt.title('Listening Time by Podcast Genre')
plt.show()

# Convert categorical columns to strings for plotting
categorical_cols = ['Genre', 'Podcast_Name', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in categorical_cols:
    train_df[col] = train_df[col].astype('category')

# Boxplot: Listening Time by Genre
plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df, x='Genre', y='Listening_Time_minutes')
plt.xticks(rotation=45)
plt.title('Listening Time Distribution by Genre')
plt.show()

# Violin Plot: Publication Time vs Listening Time
plt.figure(figsize=(12, 6))
sns.violinplot(data=train_df, x='Publication_Time', y='Listening_Time_minutes', inner='quart')
plt.title('Listening Time by Publication Time')
plt.show()



# Helper for mapping categories
def safe_map_category(df, col, mapping_dict, convert_to_category=True):
    df[col] = df[col].replace(mapping_dict)
    if convert_to_category:
        df[col] = df[col].astype('category')
    return df
   
# CITE: Based on original code from Masaya Kawamata's Notebook at https://www.kaggle.com/code/masayakawamata/single-xgboost-add-features
def feature_engineering(df):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5,
                 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11,
                 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17,
                 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23,
                 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29,
                 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34,
                 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39,
                 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44,
                 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4,
                 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
      
    # Episode_Num
    if 'Episode_Title' in df.columns:
        df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
        df = df.drop(columns=['Episode_Title'])

    # Replace mapped categories
    df = safe_map_category(df, 'Genre', genr_dict, convert_to_category=True)
    df = safe_map_category(df, 'Podcast_Name', podc_dict, convert_to_category=True)
    df = safe_map_category(df, 'Publication_Day', week_dict, convert_to_category=True)
    df = safe_map_category(df, 'Publication_Time', time_dict, convert_to_category=True)
    df = safe_map_category(df, 'Episode_Sentiment', sent_dict, convert_to_category=True)

    # Addtional feature interactions
    df['Len_x_Ads'] = df['Episode_Length_minutes'] * df['Number_of_Ads']

    return df


# Optuna objective for RMSE using cv=5
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.4, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 10.0, log=True),
        'max_bins': trial.suggest_int('max_bins', 64, 255)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

        model = HistGradientBoostingRegressor(**params, early_stopping=True, 
                                              validation_fraction=0.25,
                                              random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# Preprocess full dataset
train_df = feature_engineering(train_df)
train_df = train_df[train_df['Number_of_Ads']<10]
test_df = feature_engineering(test_df)

features = [col for col in train_df.columns 
            if col not in ['Listening_Time_minutes', 'Percent_Watched', 'Over_Watch', 'Deviation_From_Avg']]

X = train_df[features]
y = train_df['Listening_Time_minutes']

# Run Optuna trials for tuning
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=25, show_progress_bar=True)

# Best trial results
print("Best RMSE:", study.best_value)
print("Best Parameters:", study.best_params)

# Store best parameters for reuse
best_params = study.best_params

# best_params = {'learning_rate': 0.04763452713252261, 'max_iter': 1725, 'max_leaf_nodes': 99, 'min_samples_leaf': 41, 'l2_regularization': 6.920079272650521, 'max_bins': 255}


# Final model trained on full data for test prediction
final_model = HistGradientBoostingRegressor(**best_params, early_stopping=False, random_state=42)
final_model.fit(X, y)


# Predict on test set
test_preds = final_model.predict(test_df[features])

submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission['Listening_Time_minutes'] = test_preds
submission.to_csv('submission.csv', index=False)
print('Submission file saved.')

