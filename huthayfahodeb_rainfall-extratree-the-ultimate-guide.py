import pandas as pd
import numpy as np
import time
import warnings
warnings.simplefilter('ignore')

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier

import optuna


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
train_extra = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


print(f"Train shape: {train.shape}")
print(f"Training extra shape: {train_extra.shape}")
print(f"Test shape: {test.shape}")


print("Train columns:", train.columns)
print("Training extra columns:", train_extra.columns)
print("Test columns:", test.columns)


train_extra.columns = train_extra.columns.str.strip()
train_extra['rainfall'] = train_extra['rainfall'].map({'yes': 1, 'no': 0})


train = pd.concat([train, train_extra], axis=0, ignore_index=False)


train.head()


target = 'rainfall'

discrete = [
    var for var in train.columns if train[var].dtype != 'O' and var != 'rainfall'
    and train[var].nunique() <= 10
]
continuous = [
    var for var in train.columns
    if train[var].dtype != 'O' and var != 'rainfall' and var not in discrete
]

print('There are {} discrete variables'.format(len(discrete)))
print('There are {} continuous variables'.format(len(continuous)))


train.isnull().mean()[train.isnull().mean() > 0]


test.isnull().mean()[test.isnull().mean() > 0]


def plot_box_and_distribution(data, continuous_vars):
    for var in continuous_vars:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot
        sns.boxplot(x=data[var], ax=axes[0], color='skyblue')
        axes[0].set_title(f'Boxplot of {var}', fontsize=14)
        
        # Distribution plot
        sns.histplot(data[var], kde=True, ax=axes[1], bins=30, color='steelblue')
        axes[1].set_title(f'Distribution of {var}', fontsize=14)
        
        plt.tight_layout()
        plt.show()


plot_box_and_distribution(train, continuous)


train.describe()


train[continuous] = train[continuous].fillna(train[continuous].median())

test[continuous] = test[continuous].fillna(test[continuous].median())


def feature_engineering(df):

    df['day'] = pd.to_datetime(df['day'], errors='coerce')
    df['month'] = df['day'].dt.month

    # Extract temporal features with cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['day'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['day'].dt.month / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day'].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day'].dt.dayofweek / 7)
    df['is_weekend'] = df['day'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Correct typos and calculate temperature features properly
    df['temperature'] = (df['maxtemp'] + df['mintemp']) / 2  # Assuming 'temperature' was the mean
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['heat_index'] = 13.12 + 0.6215 * df['temperature'] - 11.37 * (df['windspeed']**0.16) + 0.3965 * df['temperature'] * (df['windspeed']**0.16)
    df['heat_index'] = np.where(df['temperature'] >= 10, df['temperature'], df['heat_index'])
    
    # Add relative humidity based on dew point and temperature
    df['relative_humidity'] = (
        np.exp((17.625 * df['dewpoint']) / (243.04 + df['dewpoint'])) / 
        np.exp((17.625 * df['temperature']) / (243.04 + df['temperature']))
    ) * 100
    
    # Wind direction decomposition into vector components
    df['wind_dir_rad'] = np.deg2rad(df['winddirection'])
    df['wind_x'] = df['windspeed'] * np.sin(df['wind_dir_rad'])
    df['wind_y'] = df['windspeed'] * np.cos(df['wind_dir_rad'])
    df.drop(columns=['wind_dir_rad'], inplace=True)
    
    # Rolling statistical features with multiple window sizes
    windows = [7, 14, 30]
    for window in windows:
        df[f'rolling_temp_median_{window}d'] = df['temperature'].rolling(window).median()
        df[f'rolling_windspeed_std_{window}d'] = df['windspeed'].rolling(window).std()
        df[f'rolling_humidity_max_{window}d'] = df['humidity'].rolling(window).max()
    
    # Advanced interaction and polynomial features
    df['temp_pressure_ratio'] = df['temperature'] / df['pressure']
    df['humidity_temp_squared'] = df['humidity'] * (df['temperature'] ** 2)
    df['cloud_squared'] = df['cloud'] ** 2
    
    # Time since last event features (example with simulated 'high_wind' flag)
    df['high_wind'] = (df['windspeed'] >= df['windspeed'].quantile(0.9)).astype(int)
    df['days_since_high_wind'] = df['high_wind'].groupby(df['high_wind'].ne(df['high_wind'].shift()).cumsum()).cumcount() + 1
    df.drop(columns=['high_wind'], inplace=True)
    
    # Lag features with multiple gaps and difference levels
    def generate_lags(df, columns, lags):
        for column in columns:
            for lag in lags:
                df[f'{column}_lag_{lag}'] = df[column].shift(lag)
                df[f'{column}_diff_{lag}'] = df[column].diff(lag)
    
    generate_lags(df, ['temperature', 'humidity', 'windspeed', 'pressure'], [1,2,3])
    
    # Multi-scale exponential moving averages
    df['ema_temp_7d'] = df['temperature'].ewm(span=7, adjust=False).mean()
    df['ema_humidity_14d'] = df['humidity'].ewm(span=14, adjust=False).mean()
    
    # Hyperbolic temporal decay factors
    df['decay_temp'] = df['temperature'] / np.exp(df.groupby('month')['temperature'].cumcount() / 10)
    
    # Encoding cyclical patterns for time features
    df['day_of_year'] = df['day'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Drop original date and redundant columns
    df.drop(columns=['day', 'day_of_year'], inplace=True)
    
    return df


train = feature_engineering(train)
test = feature_engineering(test)


train.dropna(inplace = True)
test.dropna(inplace = True)


def objective(trial):
    # Define the hyperparameter space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
    }
    
    # Initialize the Extra Trees model with the suggested hyperparameters
    model = ExtraTreesClassifier(
        random_state=42,
        **params
    )
    
    # Define the cross-validation strategy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000)
    
    # Perform cross-validation and calculate the ROC AUC score
    scores = cross_val_score(
        model, 
        train.drop(columns=['rainfall']),  # Features
        train['rainfall'],  # Target variable
        cv=skf,  # Cross-validation strategy
        scoring="roc_auc"  # Scoring metric
    )
    
    # Return the mean ROC AUC score
    return scores.mean()


study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=25)


best_trial = study.best_trial
print(f"Best trial: {best_trial.value}")
print(f"Best parameters: {best_trial.params}")


model = ExtraTreesClassifier(
    random_state=42,
    **best_trial.params
)

model.fit(
    train.drop(columns=['rainfall']), 
    train['rainfall']
)


importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
    
plt.figure(figsize=(15, 10))
sns.barplot(x=importance[sorted_idx], y=[train.columns[i] for i in sorted_idx], palette='Blues_r')
plt.xlabel("Feature Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.title("XGBoost Feature Importance", fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()



train_columns = [col for col in train.columns if col!=target]

test = test[train_columns]


test_predictions_prob = model.predict_proba(test)[:, 1]


test_predictions_prob[:10]


submission = pd.DataFrame({ "id": test.index, target: test_predictions_prob })

submission.to_csv("submission-ex.csv", index=False)

