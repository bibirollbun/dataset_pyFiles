# Bike Sharing Demand - Final Project Notebook
# Goal: Top 5% Kaggle Score with Comprehensive EDA, Feature Engineering, and Hyperparameter Tuning

import warnings
warnings.filterwarnings('ignore')

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ML models and tools
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import optuna

# Load data (Kaggle environment assumes files are in ../input/...)
train = pd.read_csv("../input/bike-sharing-demand/train.csv")
test = pd.read_csv("../input/bike-sharing-demand/test.csv")
df = train.copy()
test_df = test.copy()

# ===============================
# 1. EDA
# ===============================
# Convert datetime
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.dayofweek
train['month'] = train['datetime'].dt.month
train['year'] = train['datetime'].dt.year.map({2011:0, 2012:1})

# Visualize rentals by hour
plt.figure(figsize=(12,6))
sns.barplot(x='hour', y='count', data=train, ci=None)
plt.title('Average Rentals by Hour')
plt.show()

# Visualize categorical features
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
sns.boxplot(x='season', y='count', data=train, ax=axs[0,0])
axs[0,0].set_title('Season vs Count')
sns.boxplot(x='workingday', y='count', data=train, ax=axs[0,1])
axs[0,1].set_title('Working Day vs Count')
sns.boxplot(x='holiday', y='count', data=train, ax=axs[1,0])
axs[1,0].set_title('Holiday vs Count')
sns.boxplot(x='weather', y='count', data=train, ax=axs[1,1])
axs[1,1].set_title('Weather vs Count')
plt.tight_layout()
plt.show()

# Boxplot for continuous variables
plt.figure(figsize=(10,10))
sns.boxplot(data=df[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])
plt.title("Distribution of Continuous Features")
plt.show()

# Correlation heatmap
cor_mat = df.select_dtypes(include=[np.number]).corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
plt.figure(figsize=(14, 10))
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix")
plt.show()

# ===============================
# 2. Feature Engineering
# ===============================
# Improved Feature Engineering to enhance predictive power
def feature_engineering(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year.map({2011: 0, 2012: 1})
    df['is_weekend'] = df['day'].apply(lambda x: 1 if x >= 5 else 0)
    df = pd.get_dummies(df, columns=['season', 'weather'], drop_first=True)
    df['rush_hour'] = df['hour'].apply(lambda x: 1 if x in [7,8,9,16,17,18] else 0)
    df['morning'] = df['hour'].apply(lambda x: 1 if 6 <= x < 12 else 0)
    df['evening'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 20 else 0)
    df['temp_diff'] = df['atemp'] - df['temp']
    df['humidity_bins'] = pd.cut(df['humidity'], bins=[0, 30, 60, 100], labels=[0, 1, 2])
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# Visualize new features
feature_cols = ['hour', 'day', 'month', 'year', 'is_weekend', 'rush_hour', 'morning', 'evening', 'temp_diff', 'humidity_bins']
fig, axs = plt.subplots(len(feature_cols), 1, figsize=(12, 20))
for i, col in enumerate(feature_cols):
    sns.boxplot(x=col, y='count', data=train, ax=axs[i])
    axs[i].set_title(f'{col} vs Count')
plt.tight_layout()
plt.show()

# Drop unnecessary columns
train.drop(['datetime', 'casual', 'registered'], axis=1, inplace=True)
test_datetime = test['datetime']
test.drop(['datetime'], axis=1, inplace=True)

X = train.drop("count", axis=1)
y = np.log1p(train['count'])

# ===============================
# 3. LightGBM Model with Optuna Tuning
# ===============================
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }

    model = lgb.LGBMRegressor(**param, n_estimators=500)
    score = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    return -1.0 * np.mean(score)

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best model
best_params = study.best_params
model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
model.fit(X, y)

# Predict
preds = model.predict(test)
preds = np.expm1(preds)

# Submission
submission = pd.DataFrame({"datetime": test_datetime, "count": preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

# Visualize prediction results
plt.figure(figsize=(12,6))
sns.histplot(preds, bins=50, kde=True)
plt.title("Distribution of Predicted Bike Counts")
plt.xlabel("Predicted Count")
plt.ylabel("Frequency")
plt.show()


