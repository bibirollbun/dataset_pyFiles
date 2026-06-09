import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import warnings

from torch.backends.mkl import verbose

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
original_dataset = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
original_dataset = original_dataset.dropna(subset=['Listening_Time_minutes']) # Drop rows with missing values in the 'Listening_Time_minutes' column, as it is the target variable
train.head(3)


# Check for missing values
def missing_percentage(df):
    ''' Check missing percentage in a DataFrame for each column '''
    for col in df.columns:
        missing = df[col].isnull().mean()
        if missing > 0:
            print(f'{col} - {missing:.2%}')
            
print('Missing values in the datasets:\n')
print('Train dataset:')
missing_percentage(train)
print('-' * 50)
print('Original dataset:')
missing_percentage(test)
print('-' * 50)
print('Test dataset:')
missing_percentage(original_dataset)


# Check podcast names
original_podcast_names = original_dataset['Podcast_Name'].unique()
train_podcast_names = train['Podcast_Name'].unique()
print('-' * 50)
print(f'Podcast names are the same: {set(original_podcast_names) == set(train_podcast_names)}')
print(f'Number of unique podcasts in the original_dataset: {len(original_podcast_names)}')
print(f'Number of unique podcasts in the train: {len(train_podcast_names)}')

# Check podcast genres
original_genres = original_dataset['Genre'].unique()
train_genres = train['Genre'].unique()
print('-' * 50)
print(f'Genres are the same: {set(original_genres) == set(train_genres)}')
print(f'Number of unique genres in the original_dataset: {len(original_genres)}')
print(f'Number of unique genres in the train: {len(train_genres)}')


# Episode titles are effectively numerical values, but they are stored as strings.
# We convert them to integers for easier processing and further analysis.
train['Episode_Title'] = train['Episode_Title'].str.extract('(\d+)').astype(int)
original_dataset['Episode_Title'] = original_dataset['Episode_Title'].str.extract('(\d+)').astype(int)
test['Episode_Title'] = test['Episode_Title'].str.extract('(\d+)').astype(int)


# Day and time of publication could be represented as categorical variables, but in numerical form. 
# We convert them to numerical form for easier processing and further analysis.

# Map days and times to numerical values
def map_cols(df):
    # Create mappings for days and times
    day_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    time_mapping = {
        'Morning': 0,
        'Afternoon': 1,
        'Evening': 2,
        'Night': 3
    }

    df['Publication_Day'] = df['Publication_Day'].map(day_mapping)
    df['Publication_Time'] = df['Publication_Time'].map(time_mapping)

    # Create a combined day-time feature (0-27)
    # This creates a single number from 0 to 27 representing all possible day-time combinations
    df['Publication_DayTime'] = df['Publication_Day'] * 4 + df['Publication_Time']

    return df

train = map_cols(train)
original_dataset = map_cols(original_dataset)
test = map_cols(test)


# Episode sentiment could be represented as a categorical variable, but in boolean form.
# 1 - Positive, 0 - Neutral, -1 - Negative

# Map sentiment to numerical values
def map_sentiment(df):
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    return df

train = map_sentiment(train)
original_dataset = map_sentiment(original_dataset)
test = map_sentiment(test)


# Clip number of ads to a reasonable range
train['Number_of_Ads'] = train['Number_of_Ads'].clip(0, 5)
original_dataset['Number_of_Ads'] = original_dataset['Number_of_Ads'].clip(0, 5)
test['Number_of_Ads'] = test['Number_of_Ads'].clip(0, 5)


numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                'Guest_Popularity_percentage', 'Number_of_Ads', 
                'Listening_Time_minutes', 'Episode_Title', 
                'Publication_Day', 'Publication_Time', 'Publication_DayTime',
                'Episode_Sentiment']

# Calculate grid dimensions
n_cols = 3  # You can adjust this
n_rows = math.ceil(len(numeric_cols) / n_cols)

# Create a figure with subplots in a grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()  # Flatten to make indexing easier

# Loop through each column and create a normalized distribution plot
for i, col in enumerate(numeric_cols):
    # Plot KDE instead of histograms to avoid overlap
    sns.kdeplot(train[col], ax=axes[i], label='Synthetic', color='blue')
    sns.kdeplot(original_dataset[col], ax=axes[i], label='Original', color='red')

    # Add labels and title
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")
    axes[i].legend()

# Hide any unused subplots
for j in range(len(numeric_cols), n_rows * n_cols):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


# # Based on this knowledge, we can try to combine the datasets for bigger training dataset.
# Observation: distribtution varaies and there are a lot of missing values in the original dataset.
# My initial attempt to merge them resulted in worse model performance, so I decided to keep the datasets separate.
# train_max_id = train['id'].max()
# original_dataset['id'] = range(train_max_id + 1, train_max_id + 1 + len(original_dataset))
# 
# train = pd.concat([train, original_dataset], ignore_index=True)


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Split the data into features and target
X = train.drop(columns = ['Listening_Time_minutes', 'id'], axis=1)
y = train['Listening_Time_minutes']

# Define categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = ['Podcast_Name', 'Genre'] # 'Publication_Day', 'Publication_Time', Episode_Sentiment 

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    
    # set categorical type
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('scaler', StandardScaler(), numeric_cols),
        # ('cat', OneHotEncoder(), categorical_cols),
    ]
)


from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        # n_estimators=100, random_state=42, 
        iterations=5000,
        enable_categorical=True,
        verbose=1
    ))
])
# LightGBM
lgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(
        # n_estimators=100, random_state=42, 
        iterations=5000,
        enable_categorical=True,
        verbose=-1
    ))
])

# CatBoost
catboost_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(
        iterations=5000,
        # iterations=1000, learning_rate=0.1, depth=3,
        #                             loss_function='RMSE', 
                                    verbose=0
    ))
])

pipelines = [
             catboost_pipeline,
             xgb_pipeline, 
             lgb_pipeline
]


# Use KFold for regression tasks
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Hashmap to store RMSE values for each model
from collections import defaultdict
rmse_values = defaultdict(list)

# List to store models
models = []

# list to store predictions
all_predictions = []

# Cross validate the models
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for pipeline in pipelines:
        print(f'Training {pipeline.named_steps["regressor"].__class__.__name__} for fold {fold}')
        pipeline.fit(X_train, y_train)

        # Save model
        models.append(pipeline)

        # Evaluate
        predictions = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        rmse_values[pipeline.named_steps['regressor'].__class__.__name__].append(rmse)
        
        print(f'Fold {fold} RMSE: {rmse:.4f}')
        all_predictions.append(predictions)
        
    print('-' * 50)

# Calculate mean and standard deviation for each model
mean_rmse = {model: np.mean(rmse_values[model]) for model in rmse_values}

# Print summary statistics
print('\nCross-Validation Results:')
for model, rmse in mean_rmse.items():
    print(f'{model}: {rmse:.4f} | SD: {np.std(rmse_values[model]):.4f}')

# Cross-Validation Results:
# CatBoostRegressor: 13.0256 | SD: 0.0220
# XGBRegressor: 13.0286 | SD: 0.0193
# LGBMRegressor: 13.0894 | SD: 0.0229


# Train XGBRegressor on full dataset

# Train model
xgb_pipeline.fit(X, y)

# Predict on test data
xgb_preds = xgb_pipeline.predict(test)


# Train CatBoostRegressor on full dataset

# Train model
catboost_pipeline.fit(X, y)

# Predict on test data
catboost_preds = catboost_pipeline.predict(test)


# Create an ensemble of models
ensemble_preds = (xgb_preds + catboost_preds) / 2


# Save predictions to a CSV file
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sample_submission['Listening_Time_minutes'] = ensemble_preds
sample_submission.to_csv('submission.csv', index=False)
sample_submission




