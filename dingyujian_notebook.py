import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error

# 配置可视化样式
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')


# Auto-adapt Kaggle path
INPUT_DIR = '/kaggle/input/bike-sharing-demand/'
TRAIN_PATH = os.path.join(INPUT_DIR, 'train.csv')
TEST_PATH  = os.path.join(INPUT_DIR, 'test.csv')
SUB_PATH   = os.path.join(INPUT_DIR, 'sampleSubmission.csv')

# Load data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SUB_PATH)

# Data overview
print("===== Train Data Info =====")
print(train.info())
print("\n===== Train Data Descriptive Statistics =====")
print(train.describe().T)
print("\n===== Missing Value Check =====")
print(f"Train Missing Values:\n{train.isnull().sum()}")
print(f"\nTest Missing Values:\n{test.isnull().sum()}")


# 1. Visualize target variable distribution (count, casual, registered)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Raw count distribution
sns.histplot(train['count'], kde=True, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Distribution of Raw Count (Total Bike Rentals)')
axes[0, 0].set_xlabel('Count of Bike Rentals')
axes[0, 0].set_ylabel('Frequency')

# Raw casual distribution
sns.histplot(train['casual'], kde=True, ax=axes[0, 1], color='forestgreen')
axes[0, 1].set_title('Distribution of Raw Casual Rentals')
axes[0, 1].set_xlabel('Count of Casual Rentals')
axes[0, 1].set_ylabel('Frequency')

# Raw registered distribution
sns.histplot(train['registered'], kde=True, ax=axes[1, 0], color='coral')
axes[1, 0].set_title('Distribution of Raw Registered Rentals')
axes[1, 0].set_xlabel('Count of Registered Rentals')
axes[1, 0].set_ylabel('Frequency')

# Log-transformed count distribution (solve right skew)
train['count_log'] = np.log(train['count'] + 1)
sns.histplot(train['count_log'], kde=True, ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Distribution of Log-Transformed Count')
axes[1, 1].set_xlabel('Log(Count + 1)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('target_variable_distribution.png')
plt.show()

# 2. Correlation between casual, registered and count
print("\n===== Correlation Between Target Variables =====")
target_corr = train[['casual', 'registered', 'count']].corr()
print(target_corr)

sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=0, vmax=1, square=True)
plt.title('Correlation Heatmap of Target Variables')
plt.savefig('target_correlation_heatmap.png')
plt.show()


# 1. Convert datetime to datetime type (for subsequent processing)
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# 2. Calculate feature correlation with log-transformed count
# Select numerical features
num_features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                'humidity', 'windspeed', 'casual', 'registered', 'count_log']
feature_corr = train[num_features].corr()

# Visualize correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(feature_corr, annot=True, cmap='viridis', square=True, fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features and Log(Count)')
plt.savefig('feature_correlation_heatmap.png')
plt.show()

# 3. Visualize key feature vs target (temp vs count)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='count', data=train, alpha=0.5, color='steelblue')
sns.regplot(x='temp', y='count', data=train, scatter=False, color='red', line_kws={'lw': 2})
plt.title('Relationship Between Temperature and Total Bike Rentals')
plt.xlabel('Temperature (°C)')
plt.ylabel('Count of Bike Rentals')
plt.savefig('temp_vs_count.png')
plt.show()


# 1. Log-transform target variables (consistent with original code)
for col in ['casual', 'registered', 'count']:
    train[f'{col}_log'] = np.log(train[col] + 1)

# 2. Extract datetime features (prepare for feature engineering)
# Create a function to avoid code duplication
def extract_datetime_features(df):
    date = pd.DatetimeIndex(df['datetime'])
    df['year'] = date.year
    df['month'] = date.month
    df['hour'] = date.hour
    df['dayofweek'] = date.dayofweek
    return df

# Apply function to train and test
train = extract_datetime_features(train)
test = extract_datetime_features(test)

# 3. Visualize hourly rental trend (verify time feature importance)
hourly_count = train.groupby('hour')['count'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='hour', y='count', data=hourly_count, color='steelblue')
plt.title('Average Bike Rentals by Hour of the Day')
plt.xlabel('Hour of the Day (0-23)')
plt.ylabel('Average Count of Bike Rentals')
plt.xticks(range(24))
plt.savefig('hourly_rental_trend.png')
plt.show()


# Build year_season feature (consistent with original code)
for df in [train, test]:
    df['year_season'] = df['year'] + df['season'] / 10

# Visualize year_season vs average count
year_season_count = train.groupby('year_season')['count'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='year_season', y='count', data=year_season_count, color='forestgreen')
plt.title('Average Bike Rentals by Year-Season')
plt.xlabel('Year-Season (e.g., 2011.1 = 2011 Season 1)')
plt.ylabel('Average Count of Bike Rentals')
plt.xticks(rotation=45)
plt.savefig('year_season_vs_count.png')
plt.show()


def build_user_behavior_features(df):
    # Feature for casual users (10:00-19:00 is the peak period)
    df['hour_workingday_casual'] = df[['hour', 'workingday']].apply(
        lambda x: int(10 <= x['hour'] <= 19), axis=1)
    
    # Feature for registered users (peak on workday 8:00/17:00-18:00, weekend 10:00-19:00)
    df['hour_workingday_registered'] = df[['hour', 'workingday']].apply(
      lambda x: int(
        (x['workingday'] == 1 and (x['hour'] == 8 or 17 <= x['hour'] <= 18))
        or (x['workingday'] == 0 and 10 <= x['hour'] <= 19)), axis=1)
    return df

# Apply function to train and test
train = build_user_behavior_features(train)
test = build_user_behavior_features(test)

# Visualize user behavior features vs rental count
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Casual feature vs casual rentals
casual_feature_count = train.groupby('hour_workingday_casual')['casual'].mean().reset_index()
sns.barplot(x='hour_workingday_casual', y='casual', data=casual_feature_count, ax=axes[0], color='forestgreen')
axes[0].set_title('Average Casual Rentals by Casual User Behavior Feature')
axes[0].set_xlabel('Casual User Behavior Feature (0 = Non-Peak, 1 = Peak)')
axes[0].set_ylabel('Average Casual Rentals')

# Registered feature vs registered rentals
registered_feature_count = train.groupby('hour_workingday_registered')['registered'].mean().reset_index()
sns.barplot(x='hour_workingday_registered', y='registered', data=registered_feature_count, ax=axes[1], color='coral')
axes[1].set_title('Average Registered Rentals by Registered User Behavior Feature')
axes[1].set_xlabel('Registered User Behavior Feature (0 = Non-Peak, 1 = Peak)')
axes[1].set_ylabel('Average Registered Rentals')

plt.tight_layout()
plt.savefig('user_behavior_feature_vs_rentals.png')
plt.show()


# Target encoding: median count by year_season (consistent with original code)
by_season = train.groupby('year_season')[['count']].median()
by_season.columns = ['count_season']

# Merge encoded feature to train and test
train = train.join(by_season, on='year_season')
test = test.join(by_season, on='year_season')

# Fill missing values (test may have no matching year_season in train)
train_median_count_season = train['count_season'].median()
train['count_season'] = train['count_season'].fillna(train_median_count_season)
test['count_season'] = test['count_season'].fillna(train_median_count_season)

# Visualize target encoded feature vs count
plt.figure(figsize=(12, 6))
sns.scatterplot(x='count_season', y='count', data=train, alpha=0.5, color='purple')
sns.regplot(x='count_season', y='count', data=train, scatter=False, color='red', line_kws={'lw': 2})
plt.title('Relationship Between Target Encoded Feature (count_season) and Total Rentals')
plt.xlabel('Median Count by Year-Season (count_season)')
plt.ylabel('Total Bike Rentals')
plt.savefig('target_encoded_feature_vs_count.png')
plt.show()


# Define feature sets for casual and registered (consistent with original code)
casual_features = ['season', 'holiday', 'workingday', 'weather',
                   'temp', 'atemp', 'humidity', 'windspeed', 'year', 'hour',
                   'dayofweek', 'hour_workingday_casual', 'count_season']

registered_features = ['season', 'holiday', 'workingday', 'weather',
                       'temp', 'atemp', 'humidity', 'windspeed', 'year', 'hour',
                       'dayofweek', 'hour_workingday_registered', 'count_season']

print("===== Casual Feature Set =====")
print(casual_features)
print("\n===== Registered Feature Set =====")
print(registered_features)


# Initialize models (consistent with original code)
regs = {
    "gbdt": GradientBoostingRegressor(random_state=0),
    "rf": RandomForestRegressor(random_state=0, n_jobs=-1)
}

# Set model parameters
model_params = {
    "gbdt": {"n_estimators": 1000, "min_samples_leaf": 6},
    "rf": {"n_estimators": 1000, "min_samples_leaf": 2}
}

# Apply parameters to models
for name, reg in regs.items():
    reg.set_params(**model_params[name])

print("===== Model Initialization Completed =====")
print(f"GBDT Params: {regs['gbdt'].get_params()}")
print(f"RF Params: {regs['rf'].get_params()}")


preds = {}  # Store test set predictions

for name, reg in regs.items():
    print(f"\n===== Training {name.upper()} Model for Casual Rentals =====")
    # Train on casual features
    reg.fit(train[casual_features], train['casual_log'])
    # Predict on test set
    pred_casual = reg.predict(test[casual_features])
    pred_casual = np.exp(pred_casual) - 1  # Reverse log-transform
    pred_casual[pred_casual < 0] = 0  # Correct negative values
    
    print(f"===== Training {name.upper()} Model for Registered Rentals =====")
    # Re-initialize model to avoid parameter contamination
    if name == 'gbdt':
        reg = GradientBoostingRegressor(random_state=0, **model_params[name])
    elif name == 'rf':
        reg = RandomForestRegressor(random_state=0, n_jobs=-1, **model_params[name])
    # Train on registered features
    reg.fit(train[registered_features], train['registered_log'])
    # Predict on test set
    pred_registered = reg.predict(test[registered_features])
    pred_registered = np.exp(pred_registered) - 1  # Reverse log-transform
    pred_registered[pred_registered < 0] = 0  # Correct negative values
    
    # Sum casual and registered predictions
    preds[name] = pred_casual + pred_registered
    print(f"{name.upper()} Model Prediction Completed")


# Visualize feature importance for GBDT (take casual features as example)
gbdt_casual_model = GradientBoostingRegressor(random_state=0, **model_params['gbdt'])
gbdt_casual_model.fit(train[casual_features], train['casual_log'])

# Extract feature importance
feature_importance = pd.DataFrame({
    'feature': casual_features,
    'importance': gbdt_casual_model.feature_importances_
}).sort_values(by='importance', ascending=False)

# Visualize
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance, color='steelblue')
plt.title('GBDT Model Feature Importance (Casual Rentals)')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.savefig('gbdt_feature_importance.png')
plt.show()


# Weighted ensemble (70% GBDT, 30% RF) (consistent with original code)
pred = 0.7 * preds['gbdt'] + 0.3 * preds['rf']

# Visualize ensemble prediction distribution
plt.figure(figsize=(10, 6))
sns.histplot(pred, kde=True, color='purple')
plt.title('Distribution of Ensemble Prediction (Test Set)')
plt.xlabel('Predicted Count of Bike Rentals')
plt.ylabel('Frequency')
plt.savefig('ensemble_prediction_distribution.png')
plt.show()


# Generate submission file
sample_sub['count'] = pred
sample_sub.to_csv('submission.csv', index=False)

print("\n===== Submission File Generated =====")
print(f"Submission File Preview:\n{sample_sub.head(10)}")

