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


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns
# from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

import xgboost as xgb
import lightgbm as lgb
import optuna


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
subm_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train_df.isnull().sum()


class FeatureCreator(BaseEstimator, TransformerMixin):
    
    def construct_date(self, row):
        year = int(row['year'])  # Explicit conversion to int
        day_of_year = int(row['day'])  # Explicit conversion to int
        return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    
    def get_season(self, month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Fall"
        
    # def create_rolling_features(self, df, windows, target_col='rainfall'):
    #     """Creates rolling window features for a DataFrame."""
    #     for window in windows:
    #         df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
    #         df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    #     return df
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        encoder = LabelEncoder()
        # Create a new feature: salary-to-age ratio
        X_new = X.copy()
        start_year = 2000
        X_new['year'] = start_year + (X_new['id']) // 365
        X_new['date'] = X_new.apply(lambda row: self.construct_date(row), axis=1)
        X_new['month'] = X_new['date'].dt.month
        X_new['weekday'] = X_new['date'].dt.weekday
        # Sine and cosine transformations to capture cyclical nature of days in a year
        X_new['sin_day'] = np.sin(2 * np.pi * X_new['day'] / 365)
        X_new['cos_day'] = np.cos(2 * np.pi * X_new['day'] / 365)
        
        # Create new features based on meteorological understanding and data analysis
        # Get Season
        X_new['season_name'] = X_new['month'].apply(self.get_season)
        X_new['season'] = encoder.fit_transform(X_new['season_name'])
        # Temperature range (difference between max and min temperatures)
        # X_new['temp_range'] = X_new['maxtemp'] - X_new['mintemp']
        # Dew point depression (difference between temperature and dew point)
        # X_new['dewpoint_depression'] = X_new['temparature'] - X_new['dewpoint']
        # Pressure change from previous day
        # X_new['pressure_change'] = X_new['pressure'].diff().fillna(0)
        # Cloud-Humidity Relationships
        # X_new['cloud_humidity_sum'] = X_new['cloud'] + X_new['humidity']
        # X_new['cloud_humidity'] = X_new['humidity'] * X_new['cloud']
        # X_new['cloud_humidity_squared'] = X_new['cloud_humidity'] ** 2
        # X_new['cloud_humidity_sqrt'] = np.sqrt(X_new['cloud_humidity'])
        # X_new['cloud_humidity_cubert'] = X_new['cloud_humidity'] ** (1/3)
        # X_new['cloud_humidity_log'] = np.log1p(X_new['cloud_humidity'])
        # X_new['cloud_humidity_ratio'] = X_new['cloud'] / X_new['humidity'].clip(lower=1)
        # X_new['cloud_humidity_ratio_squared'] = X_new['cloud_humidity_ratio'] ** 2
        # Cloud Features
        # X_new['log_cloud'] = np.log1p(X_new['cloud'])
        # X_new['cloud_squared'] = X_new['cloud'] ** 2
        # Cloud-Humidity-Pressure Interactions
        # X_new['cloud_humidity_pressure_ratio'] = (X_new['cloud_humidity']) / X_new['pressure']
        # Humidity to dew point ratio
        # X_new['humidity_dewpoint_ratio'] = X_new['humidity'] / X_new['dewpoint'].clip(lower=0.1)
        # Cloud coverage to sunshine ratio (inverse relationship)
        # X_new['cloud_sunshine_ratio'] = X_new['cloud'] / X_new['sunshine'].clip(lower=0.1)
        # Wind intensity factor (combination of speed and humidity)
        # X_new['wind_humidity_factor'] = X_new['windspeed'] * (X_new['humidity'] / 100)
        # Temperature-humidity index (simple version of heat index)
        X_new['temp_humidity_index'] = (
            (0.8 * X_new['temparature']) 
            + ((X_new['humidity'] / 100) * (X_new['temparature'] - 14.3)) 
            + 46.4
            )
        # Pressure change rate (acceleration)
        # X_new['pressure_acceleration'] = X_new['pressure_change'].diff().fillna(0)
        
        # X_new.drop(columns=['day', 'date', 'year', 'maxtemp', 'mintemp', 'season_name'], inplace=True)
        X_new.drop(columns=['id', 'day', 'date', 'year', 'season_name'], inplace=True)
        
        # windows = [3, 7]  # Create rolling windows for 3 and 7 days
        # X_new = self.create_rolling_features(X_new, windows)
        return X_new
    


logit_pipeline = Pipeline(steps=[
    ('feature_creator', FeatureCreator()),  # Feature creation step
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))  # Classification model
])


X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


logit_pipeline.fit(X_train, y_train)


logit_probs_train = logit_pipeline.predict_proba(X_train)[:, 1]
logit_probs = logit_pipeline.predict_proba(X_val)[:, 1]


logit_auc_train = roc_auc_score(y_train, logit_probs_train)
logit_auc = roc_auc_score(y_val, logit_probs)
print(f'Logistic AUC (Train): {logit_auc_train:.4f}')
print(f'Logistic AUC: {logit_auc:.4f}')


rf_pipeline = Pipeline(steps=[
    ('feature_creator', FeatureCreator()),  # Feature creation step
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))  # Classification model
])


rf_pipeline.fit(X_train, y_train)


rf_probs_train = rf_pipeline.predict_proba(X_train)[:, 1]
rf_probs = rf_pipeline.predict_proba(X_val)[:, 1]

rf_auc_train = roc_auc_score(y_train, rf_probs_train)
rf_auc = roc_auc_score(y_val, rf_probs)

print(f'Random Forest AUC (Train): {rf_auc_train:.4f}')
print(f'Random Forest AUC: {rf_auc:.4f}')


xgb_pipeline = Pipeline(steps=[
    ('feature_creator', FeatureCreator()),  # Feature creation step
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42))  # Classification model
])


xgb_pipeline.fit(X_train, y_train)


xgb_probs_train = xgb_pipeline.predict_proba(X_train)[:, 1]
xgb_probs = xgb_pipeline.predict_proba(X_val)[:, 1]

xgb_auc_train = roc_auc_score(y_train, xgb_probs_train)
xgb_auc = roc_auc_score(y_val, xgb_probs)

print(f'XGB AUC (Train): {xgb_auc_train:.4f}')
print(f'XGB AUC: {xgb_auc:.4f}')


lgb_pipeline = Pipeline(steps=[
    ('feature_creator', FeatureCreator()),  # Feature creation step
    ('scaler', StandardScaler()),
    ('classifier', lgb.LGBMClassifier(random_state=42))  # Classification model
])


lgb_pipeline.fit(X_train, y_train)


lgb_probs_train = lgb_pipeline.predict_proba(X_train)[:, 1]
lgb_probs = lgb_pipeline.predict_proba(X_val)[:, 1]

lgb_auc_train = roc_auc_score(y_train, lgb_probs_train)
lgb_auc = roc_auc_score(y_val, lgb_probs)

print(f'LightGBM AUC (Train): {lgb_auc_train:.4f}')
print(f'LightGBM AUC: {lgb_auc:.4f}')


# Define the objective function
def logit_objective(trial):
    # Suggest hyperparameters
    C = trial.suggest_loguniform('C', 1e-5, 100)
    solver = trial.suggest_categorical('solver', ['saga'])
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1) if penalty == 'elasticnet' else None

    logit_pipeline = Pipeline(steps=[
        ('feature_creator', FeatureCreator()),  # Feature creation step
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=C, solver=solver, penalty=penalty, l1_ratio=l1_ratio, max_iter=1000, random_state=42))  # Classification model
    ])
    
    # Perform cross-validation to evaluate model performance using AUC
    auc_scores = cross_val_score(logit_pipeline, X_train, y_train, cv=3, scoring='roc_auc')
    
    return auc_scores.mean()
    
    

# Create a study and optimize the objective function
logit_study = optuna.create_study(direction='maximize')
logit_study.optimize(logit_objective, n_trials=50)


# logit_study.best_trial.params
# logit_study.best_params


logit_best_params = logit_study.best_trial.params


logit_final_pipeline = Pipeline(steps=[
        ('feature_creator', FeatureCreator()),  # Feature creation step
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(**logit_best_params, random_state=42))  # Classification model
    ])


logit_final_pipeline.fit(X_train, y_train)


logit_final_probs_train = logit_final_pipeline.predict_proba(X_train)[:, 1]
logit_final_probs = logit_final_pipeline.predict_proba(X_val)[:, 1]

logit_final_auc_train = roc_auc_score(y_train, logit_final_probs_train)
logit_final_auc = roc_auc_score(y_val, logit_final_probs)

print(f'Logistic Final AUC (Train): {logit_final_auc_train:.4f}')
print(f'Logistic Final AUC: {logit_final_auc:.4f}')


# Define the objective function
def rf_objective(trial):
    # Suggest hyperparameters
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

    rf_pipeline = Pipeline(steps=[
        ('feature_creator', FeatureCreator()),  # Feature creation step
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, **param))  # Classification model
    ])
    
    # Perform cross-validation to evaluate model performance using AUC
    auc_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=3, scoring='roc_auc')
    
    return auc_scores.mean()
    
    

# Create a study and optimize the objective function
rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(rf_objective, n_trials=50)


rf_best_params = rf_study.best_trial.params


rf_final_pipeline = Pipeline(steps=[
        ('feature_creator', FeatureCreator()),  # Feature creation step
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**rf_best_params, random_state=42))  # Classification model
    ])


rf_final_pipeline.fit(X_train, y_train)


rf_final_probs_train = rf_final_pipeline.predict_proba(X_train)[:, 1]
rf_final_probs = rf_final_pipeline.predict_proba(X_val)[:, 1]

rf_final_auc_train = roc_auc_score(y_train, rf_final_probs_train)
rf_final_auc = roc_auc_score(y_val, rf_final_probs)

print(f'Random Forest Final AUC (Train): {rf_final_auc_train:.4f}')
print(f'Random Forest Final AUC: {rf_final_auc:.4f}')


# Define the objective function
def xgb_objective(trial):
    # Suggest hyperparameters
    param = {
        "objective": "binary:logistic",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
        # "random_state": 42,
        "eval_metric": "logloss",
    }

    xgb_pipeline = Pipeline(steps=[
        ('feature_creator', FeatureCreator()),  # Feature creation step
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(**param, random_state=42))  # Classification model
    ])
    
    # Perform cross-validation to evaluate model performance using AUC
    auc_scores = cross_val_score(xgb_pipeline, X_train, y_train, cv=3, scoring='roc_auc')
    
    return auc_scores.mean()
    
    

# Create a study and optimize the objective function
xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=50)


xgb_best_params = xgb_study.best_trial.params


xgb_final_pipeline = Pipeline(steps=[
        ('feature_creator', FeatureCreator()),  # Feature creation step
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(**xgb_best_params, random_state=42))  # Classification model
    ])


xgb_final_pipeline.fit(X_train, y_train)


xgb_final_probs_train = xgb_final_pipeline.predict_proba(X_train)[:, 1]
xgb_final_probs = xgb_final_pipeline.predict_proba(X_val)[:, 1]

xgb_final_auc_train = roc_auc_score(y_train, xgb_final_probs_train)
xgb_final_auc = roc_auc_score(y_val, xgb_final_probs)

print(f'XGB Final AUC (Train): {xgb_final_auc_train:.4f}')
print(f'XGB Final AUC: {xgb_final_auc:.4f}')


# Define the objective function
def lgb_objective(trial):
    # Suggest hyperparameters
    param = {
        'objective': 'binary',
        'metric': 'binary_error',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    }

    lgb_pipeline = Pipeline(steps=[
        ('feature_creator', FeatureCreator()),  # Feature creation step
        ('scaler', StandardScaler()),
        ('classifier', lgb.LGBMClassifier(**param, random_state=42))  # Classification model
    ])
    
    # Perform cross-validation to evaluate model performance using AUC
    auc_scores = cross_val_score(lgb_pipeline, X_train, y_train, cv=3, scoring='roc_auc')
    
    return auc_scores.mean()
    
    

# Create a study and optimize the objective function
lgb_study = optuna.create_study(direction='maximize')
lgb_study.optimize(lgb_objective, n_trials=50)


lgb_best_params = lgb_study.best_trial.params


lgb_final_pipeline = Pipeline(steps=[
        ('feature_creator', FeatureCreator()),  # Feature creation step
        ('scaler', StandardScaler()),
        ('classifier', lgb.LGBMClassifier(**lgb_best_params, random_state=42))  # Classification model
    ])


lgb_final_pipeline.fit(X_train, y_train)


lgb_final_probs_train = lgb_final_pipeline.predict_proba(X_train)[:, 1]
lgb_final_probs = lgb_final_pipeline.predict_proba(X_val)[:, 1]

lgb_final_auc_train = roc_auc_score(y_train, lgb_final_probs_train)
lgb_final_auc = roc_auc_score(y_val, lgb_final_probs)

print(f'LightGBM Final AUC (Train): {lgb_final_auc_train:.4f}')
print(f'LightGBM Final AUC: {lgb_final_auc:.4f}')


voting_clf = VotingClassifier(
    estimators=[
        ('logit', logit_final_pipeline),  # Logistic Regression model
        ('rf', rf_final_pipeline),  # Random Forest model
        ('xgb', xgb_final_pipeline),  # XGBoost model
        ('lgb', lgb_final_pipeline)   # LightGBM model
    ],
    voting='soft'  # 'hard' for majority voting, 'soft' for probability-based voting
)


# Fit the voting classifier using your training data
voting_clf.fit(X_train, y_train)


voting_clf_probs_train = voting_clf.predict_proba(X_train)[:, 1]
voting_clf_final_probs = voting_clf.predict_proba(X_val)[:, 1]

voting_clf_auc_train = roc_auc_score(y_train, voting_clf_probs_train)
voting_clf_auc = roc_auc_score(y_val, voting_clf_final_probs)

print(f'Voting AUC (Train): {voting_clf_auc_train:.4f}')
print(f'Voting AUC: {voting_clf_auc:.4f}')


# stacking_clf = StackingClassifier(
#     estimators=[
#         ('logit', logit_final_pipeline),  # Logistic Regression model
#         ('rf', rf_final_pipeline),  # Random Forest model
#         ('xgb', xgb_final_pipeline),  # XGBoost model
#         ('lgb', lgb_final_pipeline)   # LightGBM model
#     ],
#     # final_estimator=LogisticRegression()  # Meta-model to combine the predictions
#     final_estimator = logit_final_pipeline
# )


# # Fit the voting classifier using your training data
# stacking_clf.fit(X_train, y_train)


# stacking_clf_probs_train = stacking_clf.predict_proba(X_train)[:, 1]
# stacking_clf_final_probs = stacking_clf.predict_proba(X_val)[:, 1]

# stacking_clf_auc_train = roc_auc_score(y_train, stacking_clf_probs_train)
# stacking_clf_auc = roc_auc_score(y_val, stacking_clf_final_probs)

# print(f'Stacking AUC (Train): {stacking_clf_auc_train:.4f}')
# print(f'Stacking AUC: {stacking_clf_auc:.4f}')


test_df.isnull().sum()


test_df.fillna(test_df.mean(), inplace=True)


final_preds = voting_clf.predict_proba(test_df)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': final_preds
})

submission.head()


# Save to CSV

submission.to_csv("submission.csv", index=False)

