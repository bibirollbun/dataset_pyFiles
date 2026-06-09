import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
from IPython.display import clear_output

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, cross_val_score, cross_validate, GridSearchCV
from random import random, randint, randrange, uniform
from lightgbm import LGBMRegressor
from lightgbm import log_evaluation, early_stopping

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import *
from sklearn.metrics import make_scorer, mean_absolute_percentage_error


rs = 9


dfTrain = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
dfTest = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
dfSub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


dfTrain.head()


dfTrain.tail()


dfTrain.shape


dfTest.shape


dfTrain.duplicated().sum()


dfTest.duplicated().sum()


dfTrain.nunique()


dfTrain['date'] = pd.to_datetime(dfTrain['date'])
dfTrain['year'] = dfTrain['date'].dt.year
dfTrain['month'] = dfTrain['date'].dt.month
dfTrain['day'] = dfTrain['date'].dt.day


cols = ['country', 'store', 'product', 'num_sold', 'year', 'month', 'day']


for col in cols:
    print(f"Column: {col}")
    print(dfTrain[col].unique())
    print()


dfTrain.isnull().sum()


yearSales = dfTrain.groupby('year')['num_sold'].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(data=yearSales, marker='o', linewidth=2, color='gold')
plt.title('Average Sales (Year)')
plt.xlabel('Year')
plt.ylabel('Average num_sold')
plt.grid(True)
plt.tight_layout()
plt.show()


monthSales = dfTrain.groupby('month')['num_sold'].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthSales, marker='o', linewidth=2, color='gold')
plt.title('Average Sales (Month)')
plt.xlabel('Month')
plt.ylabel('Average num_sold')
plt.grid(True)
plt.tight_layout()
plt.show()


daySales = dfTrain.groupby('day')['num_sold'].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(data=daySales, marker='o', linewidth=2, color='gold')
plt.title('Average Sales (Month)')
plt.xlabel('Day')
plt.ylabel('Average num_sold')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(data=dfTrain, x='date', y='num_sold', hue='country')
plt.title('Sales Trends Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Average Sales')
plt.grid(True)
plt.tight_layout()
plt.show()


# plt.figure(figsize=(12, 6))
# sns.lineplot(data=dfTrain, x='date', y='num_sold', hue='product')
# plt.title('Sales Trends Over Time by Product')
# plt.xlabel('Date')
# plt.ylabel('Average Sales')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(12, 6))
# sns.lineplot(data=dfTrain, x='date', y='num_sold', hue='store')
# plt.title('Sales Trends Over Time by Store')
# plt.xlabel('Date')
# plt.ylabel('Average Sales')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


dfTrain.isnull().mean()


dfTest.isnull().sum()


plt.figure(figsize=(14, 8))
sns.histplot(dfTrain['num_sold'], kde=True, bins=100, color='lightgreen')
plt.title('Distribution of num_sold')
plt.xlabel('num_sold')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
dfTrain['missing'] = dfTrain['num_sold'].isnull()
sns.countplot(data=dfTrain, x='missing', color='mediumseagreen')
plt.title('Missing Values in num_sold')
plt.show()


dfTrain['missing'] = dfTrain['missing'].astype(int)


cols2 = ['country', 'store', 'product']

fig, axes = plt.subplots(3, 1, figsize=(10, 10))  
axes = axes.flatten() 

for i, col in enumerate(cols2):
    sns.barplot(data=dfTrain, x='missing', y=col, palette='bright', ax=axes[i])
    axes[i].set_title(f'Relationship Between Missing Values and {col}')
    axes[i].set_xlabel('Missing Value Ratio in Each Unique Value')

plt.tight_layout() 
plt.show()


cols3 = ['year', 'month', 'day']

fig, axes = plt.subplots(3, 1, figsize=(10, 10))  
axes = axes.flatten() 

for i, col in enumerate(cols3):
    sns.boxplot(data=dfTrain, x='missing', y=col, palette='pastel', ax=axes[i])
    axes[i].set_title(f'Relationship Between Missing Values and {col}')
    axes[i].set_xlabel('Missing Value')

plt.tight_layout() 
plt.show()


cols4 = ['country', 'store', 'product', 'year', 'month', 'day']

fig, axes = plt.subplots(3, 2, figsize=(15, 12))  
axes = axes.flatten() 

for i, col in enumerate(cols4):
    sns.countplot(x=col, hue='missing', palette='muted', data=dfTrain, ax=axes[i])
    axes[i].set_title(f'Countplot of {col}')

plt.tight_layout() 
plt.show()


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.min_date_ = None  

    def fit(self, X, y=None):
        X['date'] = pd.to_datetime(X['date'])
        
        self.min_date_ = X['date'].min()
        
        for col in ['country', 'store', 'product']:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        
        return self

    def transform(self, X):
        X['date'] = pd.to_datetime(X['date'])
                
        X['month'] = X['date'].dt.month
        X['day'] = X['date'].dt.day
        X['quarter'] = X['date'].dt.quarter
        X['season'] = X['month'].apply(self.get_season)

        X.drop('date', axis=1, inplace=True)

        for col in ['country', 'store', 'product']:
            le = self.label_encoders[col]
            X[col] = le.transform(X[col])

        X['month'] = X['month'].astype(int)
        X['day'] = X['day'].astype(int)
        X['quarter'] = X['quarter'].astype(int)

        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

        X['day_sin'] = np.sin(2 * np.pi * X['day'] / 365)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / 365)

        X['quarter_sin'] = np.sin(2 * np.pi * X['quarter'] / 4)
        X['quarter_cos'] = np.cos(2 * np.pi * X['quarter'] / 4)

        return X

    @staticmethod
    def get_season(month: int) -> int:
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        elif month in [9, 10, 11]:
            return 3  # Autumn
        return 4


prep = Pipeline([
    ('preprocessor', Preprocessor())
])


%%capture
!pip install optuna
import optuna


dfTrain = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

dfTrain = dfTrain.drop('id', axis=1)

dfTrain = dfTrain.dropna(subset=['num_sold'])

X = dfTrain.drop('num_sold', axis=1)
y = np.log(dfTrain["num_sold"])


def objective(trial):
    params = {
        "boosting_type": trial.suggest_categorical("boosting_type", ["dart"]), 
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "num_leaves": trial.suggest_int("num_leaves", 50, 600),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-3, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1e-3, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
    }
    
    model = lgb.LGBMRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=rs)
    mape_scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

        X_train_prep = prep.fit_transform(X_train)
        X_val_prep   = prep.transform(X_val)

        model.fit(X_train_prep, y_train)

        y_pred = np.exp(model.predict(X_val_prep))
        mape_val = mean_absolute_percentage_error(np.exp(y_val), y_pred)
        mape_scores.append(mape_val)

    return np.mean(mape_scores)

study = optuna.create_study(direction='minimize')  
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params

print("Best hyperparameters found by Optuna:")
print(best_params)

modelLGB_optimized = lgb.LGBMRegressor(**best_params)

X_all_processed = prep.fit_transform(X, y)
modelLGB_optimized.fit(X_all_processed, y)


dfTest = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


dfTest = dfTest.drop('id', axis=1)


dfTest = prep.transform(dfTest)


yPred = np.exp(modelLGB_optimized.predict(dfTest))


submission = pd.DataFrame({
    'id': dfSub['id'], 
    'num_sold': yPred
})
submission.to_csv('submission.csv', index=False)


dfConfirm = pd.read_csv('submission.csv')
dfConfirm.head()

