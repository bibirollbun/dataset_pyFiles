import os
os.listdir('/kaggle/input')


import pandas as pd
import numpy as np

# Load train and test CSVs
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# Preview
train_df.head()


# Shape and data types
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nColumn types:")
train_df.dtypes


# Missing values
print("Missing values in train set:\n")
print(train_df.isnull().sum())

# Unique values per column
print("\nUnique values per column:")
train_df.nunique()


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of the target
plt.figure(figsize=(8, 5))
sns.histplot(train_df['Listening_Time_minutes'], bins=40, kde=True)
plt.title('Distribution of Listening Time (minutes)')
plt.xlabel('Listening_Time_minutes')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Stats
train_df['Listening_Time_minutes'].describe()


numeric_cols = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
]

train_df[numeric_cols].hist(bins=30, figsize=(12, 8), edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=14)
plt.tight_layout()
plt.show()


# Scatter plots
plt.figure(figsize=(14, 10))

for i, col in enumerate(numeric_cols):
    plt.subplot(2, 2, i + 1)
    sns.scatterplot(data=train_df, x=col, y='Listening_Time_minutes')
    plt.title(f'{col} vs Listening Time')
    plt.grid(True)

plt.tight_layout()
plt.show()


# Manual mapping of textual time labels to hour values
time_map = {
    'Morning': 9,
    'Afternoon': 14,
    'Evening': 18,
    'Night': 22
}

# Apply mapping
train_df['Publication_Hour'] = train_df['Publication_Time'].map(time_map)

import matplotlib.pyplot as plt
import seaborn as sns

# List of categorical features
categorical_cols = ['Genre', 'Episode_Sentiment', 'Publication_Day', 'Publication_Hour']

# Plot boxplots for each
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train_df, x=col, y='Listening_Time_minutes')
    plt.title(f'Listening Time by {col}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Numeric correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = train_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix")
plt.show()


# 1. Missing Value Handling


def clean_missing_values(df):
    df = df.copy()

    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(0)
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].median())

    return df


# 2. Basic Preprocessing


def preprocess(df, is_train=True):
    df = df.copy()

    # Map textual time to numeric hour
    time_map = {'Morning': 9, 'Afternoon': 14, 'Evening': 18, 'Night': 22}
    df['Publication_Hour'] = df['Publication_Time'].map(time_map)
    df.drop(columns=['Publication_Time'], inplace=True)

    # Is weekend (store before encoding)
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

    # Label encode Genre and Publication_Day
    df['Genre'] = df['Genre'].astype('category').cat.codes
    df['Publication_Day'] = df['Publication_Day'].astype('category').cat.codes

    # Drop high-cardinality or text columns
    df.drop(columns=['id', 'Podcast_Name', 'Episode_Title'], inplace=True)

    if is_train:
        X = df.drop(columns=['Listening_Time_minutes'])
        y = df['Listening_Time_minutes']
        return X, y
    else:
        return df


# 3. Feature Engineering


def add_features(df, is_train=True, genre_target_map=None):
    df = df.copy()

    # Ad density
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'] = df['Ad_Density'].replace([np.inf, -np.inf], 0).fillna(0)

    # Prime time
    df['Is_Prime_Time'] = df['Publication_Hour'].apply(lambda x: 1 if 17 <= x <= 21 else 0)

    # Host Ã— Guest popularity + ratio
    df['Host_Guest_Popularity'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Host_to_Guest'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1)

    # Log episode length
    df['log_Episode_Length'] = np.log1p(df['Episode_Length_minutes'])

    # Binary flags
    df['Has_Guest'] = (df['Guest_Popularity_percentage'] > 0).astype(int)
    df['Has_Ads'] = (df['Number_of_Ads'] > 0).astype(int)

    # Normalized popularity
    df['Host_Popularity_norm'] = df['Host_Popularity_percentage'] / 100
    df['Guest_Popularity_norm'] = df['Guest_Popularity_percentage'] / 100

    # Length category
    def length_category(mins):
        if mins < 15: return 0
        elif mins < 45: return 1
        else: return 2
    df['Episode_Length_Category'] = df['Episode_Length_minutes'].apply(length_category)

    # Hour â†’ cyclical encoding
    df['Hour_sin'] = np.sin(2 * np.pi * df['Publication_Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Publication_Hour'] / 24)

    # Time slot bin
    def time_bin(hour):
        if hour < 12: return 'Morning'
        elif hour < 17: return 'Afternoon'
        elif hour < 21: return 'Evening'
        else: return 'Night'

    df['Time_Slot'] = df['Publication_Hour'].apply(time_bin)
    df = pd.get_dummies(df, columns=['Time_Slot'], drop_first=True)

    # One-hot encode Episode Sentiment (no mapping needed)
    df = pd.get_dummies(df, columns=['Episode_Sentiment'], drop_first=True)

    # Target encoding for Genre
    if is_train:
        genre_target_map = df.groupby('Genre')['Listening_Time_minutes'].mean()
        df['Genre_Target'] = df['Genre'].map(genre_target_map)
        return df, genre_target_map
    else:
        df['Genre_Target'] = df['Genre'].map(genre_target_map).fillna(0)
        return df


# 4. Extract title-based features


def extract_title_features(df):
    df = df.copy()
    
    # Title length in characters
    df['Title_Length'] = df['Episode_Title'].astype(str).apply(len)

    # Word count
    df['Word_Count'] = df['Episode_Title'].astype(str).apply(lambda x: len(x.split()))

    # Check for "interview" (case-insensitive)
    df['Has_Interview'] = df['Episode_Title'].str.lower().str.contains('interview').astype(int)

    # Check for "Q&A" or "Q and A"
    df['Has_QA'] = df['Episode_Title'].str.lower().str.contains('q&a|q and a').astype(int)

    return df


# Step 1: Clean
train_df = clean_missing_values(train_df)
test_df = clean_missing_values(test_df)


# 2. Extract title-based features
train_df = extract_title_features(train_df)
test_df = extract_title_features(test_df)


# 3. Preprocess and encode
X_train, y_train = preprocess(train_df, is_train=True)
X_test = preprocess(test_df, is_train=False)


# 4. Combine back for add_features()
train_processed = X_train.copy()
train_processed['Listening_Time_minutes'] = y_train


# 5. Add additional features
X_train_fe, genre_target_map = add_features(train_processed, is_train=True)
X_test_fe = add_features(X_test, is_train=False, genre_target_map=genre_target_map)


# 6. Pop the target out
y_train = X_train_fe.pop('Listening_Time_minutes')


print("âœ… Shape after preprocessing:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)

print("\nâœ… Shape after feature engineering:")
print("X_train_fe:", X_train_fe.shape)
print("X_test_fe :", X_test_fe.shape)


X_train_fe.head


#!pip install xgboost


# Import & Setup
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import early_stopping, log_evaluation


# Split Data (for validation inside Optuna)
X_tr, X_val, y_tr, y_val = train_test_split(X_train_fe, y_train, test_size=0.2, random_state=42)


# Define the Optuna Objective
def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 10000,  # large number, we'll stop early
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
    }

    model = lgb.LGBMRegressor(**param)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50), log_evaluation(0)]  # early stop if no improvement in 50 rounds
    )

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse


# Run the Optuna Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # You can increase to 100+ for even better search


# Best Parameters & Final Training

print("âœ… Best RMSE:", study.best_value)
print("ğŸ“¦ Best Parameters:", study.best_params)

best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'n_estimators': 10000,  # large number for early stopping
    'random_state': 42
})

final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(
    X_train_fe, y_train,
    eval_set=[(X_train_fe, y_train)],
    callbacks=[early_stopping(100), log_evaluation(100)]
)


# Predict and Submit
X_test_fe = X_test_fe[X_train_fe.columns]

lgb_preds = final_model.predict(X_test_fe)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': lgb_preds
})

submission.to_csv('/kaggle/working/submission_lgbm_optuna.csv', index=False)
submission.head()


