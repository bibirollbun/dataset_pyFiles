!pip install optuna --quiet
!pip install optuna-integration[lightgbm] --quiet


# Author: Aaron Isom
# Kaggle Predict Podcast Listening Time
# LGBMRegressor and Optuna for hyperparameter tuning (RMSE) w/ Pruning

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')

train_df.drop_duplicates()

total_rows = len(train_df)
filtered_rows = (train_df['Number_of_Ads'] < 10).sum()
print(f"Dropping {total_rows - filtered_rows} rows ({(1 - filtered_rows/total_rows):.2%})\n")
train_df = train_df[train_df['Number_of_Ads']<10]

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

# EDA
# Target distribution
sns.histplot(train_df['Listening_Time_minutes'], kde=True, bins=50)
plt.title('Distribution of Listening Time')
plt.xlabel('Minutes')
plt.ylabel('Count')
plt.show()

sns.scatterplot(x='Episode_Length_minutes', y='Listening_Time_minutes', data=train_df, alpha=0.3)
sns.regplot(x='Episode_Length_minutes', y='Listening_Time_minutes', data=train_df, scatter=False, color='red')
plt.title('Listening Time vs Episode Length')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=train_df, x='Genre', y='Listening_Time_minutes')
plt.xticks(rotation=45)
plt.title('Listening Time by Podcast Genre')
plt.show()

# Podcast Feature Encoder
class PodcastFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for podcast datasets.
    Combines train and test to ensure consistent categorical encoding.
    """

    def __init__(self, unknown_value=-1):
        self.unknown_value = unknown_value
        self.mappings = {}
        self.columns_to_encode = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

    def fit(self, X_train, y=None):
        # Just store the mappings based on train only
        for col in self.columns_to_encode:
            if col in X_train.columns:
                unique_vals = X_train[col].dropna().unique()
                self.mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, X_train, X_test=None):
        if X_test is not None:
            # Combine train and test
            merged = pd.concat([X_train, X_test], axis=0).copy()
            split_index = len(X_train)
        else:
            merged = X_train.copy()
            split_index = None

        # Extract episode number
        if 'Episode_Title' in merged.columns:
            merged['Episode_Num'] = merged['Episode_Title'].str.extract(r'(\\d+)$')[0].fillna("0")
            merged['Episode_Num'] = merged['Episode_Num'].astype('category')
            merged.drop(columns=['Episode_Title'], inplace=True)

        # Apply categorical mappings
        for col in self.columns_to_encode:
            if col in merged.columns:
                merged[col] = merged[col].map(self.mappings.get(col, {}))\
                                         .fillna(self.unknown_value)\
                                         .astype('int32')\
                                         .astype('category')

        # Split back into train and test
        if split_index is not None:
            return merged.iloc[:split_index].copy(), merged.iloc[split_index:].copy()
        else:
            return merged

encoder = PodcastFeatureEncoder()
encoder.fit(train_df)

train_encoded, test_encoded = encoder.transform(train_df, test_df)

# Replacing null values by median
train_encoded['Episode_Length_minutes'].fillna(train_encoded['Episode_Length_minutes'].median(), inplace=True)
train_encoded['Guest_Popularity_percentage'].fillna(train_encoded['Guest_Popularity_percentage'].median(), inplace=True)
train_encoded.dropna(inplace=True)

test_encoded['Episode_Length_minutes'].fillna(test_encoded['Episode_Length_minutes'].median(), inplace=True)
test_encoded['Guest_Popularity_percentage'].fillna(test_encoded['Guest_Popularity_percentage'].median(), inplace=True)

# Get categoricals and features
display('Train Data Info:', train_encoded.info())
display('Test Data Info:', test_encoded.info())

categorical_cols = train_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
features = [col for col in train_encoded.columns if col != 'Listening_Time_minutes']

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Optuna objective for RMSE using cv=10
def objective(trial):
  
    params = {
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'num_leaves': trial.suggest_int('num_leaves', 64, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 255),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'max_bin': trial.suggest_int('max_bin', 96, 255),
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
    }

    k = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    # Cross-validation
    for train_idx, valid_idx in k.split(train_encoded):
        train_data = train_encoded.iloc[train_idx]
        valid_data = train_encoded.iloc[valid_idx]

        X_train = train_data[features]
        y_train = train_data['Listening_Time_minutes']
        X_valid = valid_data[features]
        y_valid = valid_data['Listening_Time_minutes']

        # Train model
        model = LGBMRegressor(**params, early_stopping_round=100, random_state=42, n_jobs=-1, device='cpu', force_col_wise=True)
        # Fit model
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[LightGBMPruningCallback(trial, 'rmse')], categorical_feature=categorical_cols)  

        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# Run Optuna trials for tuning. >25 results in most being pruned with no improvements
# study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
# study.optimize(objective, n_trials=25, show_progress_bar=True)
# print("Best Trial RMSE:", study.best_value)
# print("Best Hyperparameters:", study.best_params)

# Train using Optuna to tune with best_params
# best_model = LGBMRegressor(**study.best_params, callbacks=[lgb.early_stopping(stopping_rounds=100)], random_state=42, n_jobs=-1, device='cpu', force_col_wise=True)

# Use final tuned best_params for submission
best_params = {'n_estimators': 3187, 'learning_rate': 0.018130736206034486, 'max_depth': 15, 'num_leaves': 213, 'min_child_samples': 241, 'min_split_gain': 0.8948273504276488, 'subsample': 0.8793699936433255, 'subsample_freq': 10, 
 'colsample_bytree': 0.7265477506155759, 'reg_alpha': 1.959828624191452, 'reg_lambda': 0.45227288910538066, 'max_bin': 148}

# Define model
best_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1, metric='rmse', objective='regression', force_col_wise=True)

X = train_encoded[features]
y = train_encoded['Listening_Time_minutes']

# Fit model
best_model.fit(X, y, categorical_feature=categorical_cols)

# Cross-validated score (more realistic)
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring=rmse_scorer)
print(f"Cross-validated RMSE score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importances
importances = best_model.feature_importances_
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.show()

# Predict on test set
test_preds = best_model.predict(test_encoded[features])

# Create submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission['Listening_Time_minutes'] = test_preds
submission.to_csv('submission.csv', index=False)
display(submission)
print('Submission file saved.')


