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


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from matplotlib.pyplot import figure

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import mstats
from scipy.stats.mstats import winsorize

from sklearn import preprocessing
from sklearn.preprocessing import (
    LabelEncoder,
    QuantileTransformer,
    StandardScaler,
    PowerTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
    OrdinalEncoder,
    OneHotEncoder,
    FunctionTransformer,
    KBinsDiscretizer,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_regression,
    SequentialFeatureSelector,
)
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    StratifiedGroupKFold,
    RepeatedStratifiedKFold,
    RepeatedKFold,
    cross_validate,
    train_test_split,
    TimeSeriesSplit,
)
from sklearn.linear_model import (
    SGDOneClassSVM,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    IsolationForest,
    BaggingRegressor,
    RandomForestRegressor,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error, 
    r2_score
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from tensorflow.keras.models import clone_model
import keras
from keras_tuner import RandomSearch
from keras import layers
from keras.layers import (
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    Activation,
)
from tensorflow.keras.models import Sequential
from keras import backend as K
import keras_tuner
from keras_tuner import Hyperband
from functools import partial

import optuna
from optuna.samplers import CmaEsSampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis

from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor
from category_encoders import TargetEncoder, MEstimateEncoder
#from cuml.preprocessing import TargetEncoder

import requests
import holidays
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
import re
import time
import logging
from functools import partial
from itertools import combinations
from IPython.display import Image

from functools import partial

# Visualization settings
plt.style.use('ggplot')
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

# Pandas settings
pd.options.mode.chained_assignment = None

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Warnings configuration
warnings.filterwarnings('ignore')


def plot_numerical_features(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.histplot(df[feature], bins=30, kde=True, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel(feature, fontsize=14)
        axes[i].set_ylabel('Frequency', fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)  

        mean_value = df[feature].mean()
        axes[i].axvline(mean_value, color='red', linestyle='--', label='Mean')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_numerical_boxplots(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.boxplot(x=df[feature], ax=axes[i], color='lightgreen')
        axes[i].set_title(f'Boxplot of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel(feature, fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)  

        median_value = df[feature].median()
        axes[i].axvline(median_value, color='orange', linestyle='--', label='Median')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_qq_plot(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        stats.probplot(df[feature], dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ Plot of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel('Theoretical Quantiles', fontsize=14)
        axes[i].set_ylabel('Sample Quantiles', fontsize=14)
        axes[i].grid(True, linestyle='--', alpha= 0.7)  

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, method='spearman'):
    num_df = df.select_dtypes(include=[np.number])
    
    corr = num_df.corr(method=method)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, linewidths=.5)
    plt.title(f'Correlation Matrix ({method.capitalize()} Correlation)', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def plot_pairplot(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    sns.pairplot(df[num_features], diag_kind='kde', plot_kws={'alpha': 0.6, 'edgecolor': 'k'}, height=2.5)
    plt.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=18, fontweight='bold')
    plt.show()

def plot_categorical_features(df, ncols=2, top_n=None):
    cat_features = df.select_dtypes(include=[object]).columns
    nrows = (len(cat_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(cat_features):
        if top_n is not None:
            top_categories = df[feature].value_counts().nlargest(top_n).index
            sns.countplot(data=df[df[feature].isin(top_categories)], y=feature, ax=axes[i], palette='viridis', order=top_categories)
        else:
            sns.countplot(data=df, y=feature, ax=axes[i], palette='viridis')
        
        axes[i].set_title(f'Count of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel('Count', fontsize=14)
        axes[i].set_ylabel(feature, fontsize=14)
        axes[i].tick_params(axis='y', rotation=0)
        axes[i].grid(True, linestyle='--', alpha=0.7)  
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def PolynomialFeatures_labeled(input_df,power):
   
    poly = preprocessing.PolynomialFeatures(power)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(input_feature_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s+%d" % (variable,power)
                if final_label == "":         #If the final label isn't yet specified
                    final_label = intermediary_label
                else:
                    final_label = final_label + "x" + intermediary_label
        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
    return output_df

def variance_threshold(df,th):
    var_thres=VarianceThreshold(threshold=th)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:,new_cols]
   
def optimize_memory_usage(df, print_size=True):
    """
    Optimizes memory usage in a DataFrame by downcasting numeric columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to optimize.
        print_size (bool): If True, prints memory usage before and after optimization.

    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    # Types for optimization.
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Memory usage size before optimize (Mb).
    before_size = df.memory_usage().sum() / 1024**2
    
    for column in df.columns:
        column_type = df[column].dtype
        
        if column_type in numerics:
            try:
                if str(column_type).startswith('int'):
                    df[column] = pd.to_numeric(df[column], downcast='integer')
                else:
                    df[column] = pd.to_numeric(df[column], downcast='float')
                logger.info(f"Optimized column {column}: {column_type} -> {df[column].dtype}")
            except Exception as e:
                logger.error(f"Failed to optimize column {column}: {e}")
    
    # Memory usage size after optimize (Mb).
    after_size = df.memory_usage().sum() / 1024**2
    
    if print_size:
        print(
            'Memory usage size: before {:5.4f} Mb - after {:5.4f} Mb ({:.1f}%).'.format(
                before_size, after_size, 100 * (before_size - after_size) / before_size
            )
        )
    
    return df


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

train.shape, test.shape


test = test.drop(['id'], axis =1)
train = train.drop(['id'], axis =1)

train.shape, test.shape


train.describe().T


duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

train = train.drop_duplicates()


for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


for col in test.columns:
    pct_missing = np.mean(test[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


train = optimize_memory_usage(train)
test = optimize_memory_usage(test)


plot_numerical_features(train)


plot_numerical_boxplots(train)


plot_qq_plot(train)


plot_correlation_matrix(train)


plot_categorical_features(train)


from statsmodels.graphics.mosaicplot import mosaic

plt.figure(figsize=(16, 10))
mosaic(train, ['Publication_Time', 'Genre'], title='Allocation of time')
plt.show()


plt.figure(figsize=(16, 10))
mosaic(train, ['Publication_Time', 'Episode_Sentiment'], title='Allocation of Sentiment time')
plt.show()


plt.figure(figsize=(12, 5))
sns.boxplot(x='Genre', y='Listening_Time_minutes', data=train)
plt.xticks(rotation=90)
plt.title('Listening time by genre')
plt.show()


train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median())
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median())
train['Number_of_Ads'] = train['Number_of_Ads'].fillna(train['Number_of_Ads'].median())

test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median())
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].median())
test['Number_of_Ads'] = test['Number_of_Ads'].fillna(test['Number_of_Ads'].median())


# # for col in train.select_dtypes(include=['number']).columns:
# #     train[col] = winsorize(train[col], limits=[0.05, 0.05])
# Q1 = train['Number_of_Ads'].quantile(0.25)
# Q3 = train['Number_of_Ads'].quantile(0.75)
# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# train = train[
#     (train['Number_of_Ads'] >= lower) &
#     (train['Number_of_Ads'] <= upper)
# ]
# train.shape, test.shape


def categorize_variable(df, column, labels):
    
    if len(labels) != 3:
        raise ValueError("3 type")
    
    bins = [-float('inf'), 
            df[column].quantile(0.25), 
            df[column].quantile(0.75), 
            float('inf')]
    
    df[f'{column}_group'] = pd.cut(df[column], bins=bins, labels=labels)
    return df


categorize_variable(train, 'Episode_Length_minutes', ["short", "mean", 'long'])
categorize_variable(train, 'Host_Popularity_percentage', ["low", "normal", 'high'])
categorize_variable(train, 'Guest_Popularity_percentage', ["low_important", "mean", 'top'])

categorize_variable(test, 'Episode_Length_minutes', ["short", "mean", 'long'])
categorize_variable(test, 'Host_Popularity_percentage', ["low", "normal", 'high'])
categorize_variable(test, 'Guest_Popularity_percentage', ["low_important", "mean", 'top'])

train.shape, test.shape


for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


def n_fe(df):
    import numpy as np
    
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Is_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 70).astype(int)
    df['Is_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 70).astype(int)
    df['Host_Guest_Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 60).astype(int)
    
    return df


train = n_fe(train)
test = n_fe(test)


col = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
       'Publication_Time', 'Episode_Sentiment', 'Episode_Length_minutes_group',
      'Host_Popularity_percentage_group','Guest_Popularity_percentage_group']
col_num = ['Episode_Length_minutes', 'Host_Popularity_percentage',
       'Guest_Popularity_percentage', 'Number_of_Ads']

TE = MEstimateEncoder(cols=col, m=5.0)


train[col] = TE.fit_transform(train[col], train['Listening_Time_minutes'])
test[col] = TE.transform(test[col])

train.shape, test.shape


X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']
print('before threshold:',X.shape, y.shape)

X = variance_threshold(X,0.01)
list_name = (X.columns)
test = test[list_name]

print('after threshold:',X.shape, y.shape)


scaler = StandardScaler()

X[X.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(X[X.select_dtypes(include=[np.number]).columns])
test[X.select_dtypes(include=[np.number]).columns] = scaler.transform(test[X.select_dtypes(include=[np.number]).columns])

X.shape, y.shape, test.shape


catboost_params = [
    {'iterations': 2000, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3, 'loss_function': 'RMSE', 'border_count': 32, 'bagging_temperature': 1, 'random_strength': 1},
    {'iterations': 3000, 'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'loss_function': 'RMSE', 'border_count': 64, 'bagging_temperature': 0.5, 'random_strength': 2},
    {'iterations': 1600, 'depth': 10, 'learning_rate': 0.01, 'l2_leaf_reg': 1, 'loss_function': 'RMSE', 'border_count': 16, 'bagging_temperature': 1.5, 'random_strength': 0.5},
    {'iterations': 4000, 'depth': 4, 'learning_rate': 0.1, 'l2_leaf_reg': 10, 'loss_function': 'RMSE', 'border_count': 32, 'bagging_temperature': 0.8, 'random_strength': 1.5},
    {'iterations': 2400, 'depth': 7, 'learning_rate': 0.02, 'l2_leaf_reg': 7, 'loss_function': 'RMSE', 'border_count': 64, 'bagging_temperature': 1, 'random_strength': 1},
    {'iterations': 1000, 'depth': 12, 'learning_rate': 0.005, 'l2_leaf_reg': 2, 'loss_function': 'RMSE', 'border_count': 16, 'bagging_temperature': 0.3, 'random_strength': 2},
    {'iterations': 1800, 'depth': 5, 'learning_rate': 0.04, 'l2_leaf_reg': 4, 'loss_function': 'RMSE','border_count': 48, 'bagging_temperature': 0.7, 'random_strength': 0.8,
     'grow_policy': 'SymmetricTree'},
    {'iterations': 2200, 'depth': 9, 'learning_rate': 0.025, 'l2_leaf_reg': 6, 'loss_function': 'RMSE', 'border_count': 24, 'bagging_temperature': 1.2, 'random_strength': 1.2, 'bootstrap_type': 'Bernoulli'}
]

xgb_params = [
    {'n_estimators': 2000, 'max_depth': 6, 'learning_rate': 0.01, 'subsample': 0.8, 'eval_metric': 'rmse', 'colsample_bytree': 0.8, 'gamma': 0, 'min_child_weight': 1},
    {'n_estimators': 3000, 'max_depth': 8, 'learning_rate': 0.03, 'subsample': 0.9, 'eval_metric': 'rmse', 'colsample_bytree': 0.9, 'gamma': 0.1, 'min_child_weight': 2},
    {'n_estimators': 1600, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.7, 'eval_metric': 'rmse', 'colsample_bytree': 0.7, 'gamma': 0.2, 'min_child_weight': 3},
    {'n_estimators': 4000, 'max_depth': 3, 'learning_rate': 0.005, 'subsample': 0.6, 'eval_metric': 'rmse', 'colsample_bytree': 0.6, 'gamma': 0.3, 'min_child_weight': 1},
    {'n_estimators': 2400, 'max_depth': 10, 'learning_rate': 0.02, 'subsample': 0.75, 'eval_metric': 'rmse', 'colsample_bytree': 0.75, 'gamma': 0.4, 'min_child_weight': 2},
    {'n_estimators': 1000, 'max_depth': 12, 'learning_rate': 0.1, 'subsample': 0.85, 'eval_metric': 'rmse', 'colsample_bytree': 0.85, 'gamma': 0.5, 'min_child_weight': 1},
    {'n_estimators': 2800, 'max_depth': 7, 'learning_rate': 0.015, 'subsample': 0.82, 'eval_metric': 'rmse', 'colsample_bytree': 0.82, 'gamma': 0.15, 'min_child_weight': 2,
     'reg_alpha': 0.1, 'reg_lambda': 0.5},
    {'n_estimators': 3500, 'max_depth': 9, 'learning_rate': 0.025, 'subsample': 0.88, 'eval_metric': 'rmse', 'colsample_bytree': 0.88, 'gamma': 0.25, 'min_child_weight': 1, 
     'tree_method': 'hist', 'grow_policy': 'lossguide'}
]

lgbm_params = [
    {'n_estimators': 2000, 'max_depth': 5, 'learning_rate': 0.01, 'num_leaves': 31, 'metric': 'rmse', 'min_data_in_leaf': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8},
    {'n_estimators': 3000, 'max_depth': 8, 'learning_rate': 0.03, 'num_leaves': 63, 'metric': 'rmse', 'min_data_in_leaf': 15, 'feature_fraction':  0.9, 'bagging_fraction': 0.9},
    {'n_estimators': 1600, 'max_depth': 3, 'learning_rate': 0.05, 'num_leaves': 15, 'metric': 'rmse', 'min_data_in_leaf': 10, 'feature_fraction': 0.7, 'bagging_fraction': 0.7},
    {'n_estimators': 4000, 'max_depth': 12, 'learning_rate': 0.005, 'num_leaves': 127, 'metric': 'rmse', 'min_data_in_leaf': 5, 'feature_fraction': 0.6, 'bagging_fraction': 0.6},
    {'n_estimators': 2400, 'max_depth': 6, 'learning_rate': 0.02, 'num_leaves': 255, 'metric': 'rmse', 'min_data_in_leaf': 25, 'feature_fraction': 0.75, 'bagging_fraction': 0.75},
    {'n_estimators': 1000, 'max_depth': 15, 'learning_rate': 0.1, 'num_leaves': 511, 'metric': 'rmse', 'min_data_in_leaf': 30, 'feature_fraction': 0.85, 'bagging_fraction': 0.85},
     {'n_estimators': 2700, 'max_depth': 7, 'learning_rate': 0.018, 'num_leaves': 95, 'metric': 'rmse', 'min_data_in_leaf': 18, 'feature_fraction': 0.83, 'bagging_fraction': 0.83, 
      'lambda_l1': 0.1, 'lambda_l2': 0.2},
    {'n_estimators': 3200, 'max_depth': 9, 'learning_rate': 0.035, 'num_leaves': 191, 'metric': 'rmse', 'min_data_in_leaf': 22, 'feature_fraction': 0.92, 'bagging_fraction': 0.92,
     'min_gain_to_split': 0.1, 'boosting_type': 'goss'}
]



def create_ensemble(X, y, test, n_folds=5):
    
    FOLDS = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_oof = {}
    all_predictions = {}
    
    models = []
    
    for i, params in enumerate(catboost_params, 1):
        models.append((f'cat_{i}', CatBoostRegressor(**params, verbose=0, thread_count=1)))
    
    for i, params in enumerate(xgb_params, 1):
        models.append((f'xgb_{i}', xgb.XGBRegressor(**params, n_jobs=1)))
    
    for i, params in enumerate(lgbm_params, 1):
        models.append((f'lgb_{i}', LGBMRegressor(**params, verbose=-1, n_jobs=1)))
    
    for name, model in models:
        try:
            print(f"\nTraining {name}...")
            oof = np.zeros(len(X))
            pred = np.zeros(len(test))
            
            for fold, (trn_idx, val_idx) in enumerate(FOLDS.split(X, y)):
                X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                oof[val_idx] = model.predict(X_val)
                pred += model.predict(test) / FOLDS.n_splits
                
                fold_rmse = np.sqrt(mean_squared_error(y_val, oof[val_idx]))
                print(f'{name} - Fold {fold} RMSE: {fold_rmse:.4f}')
            
            all_oof[name] = oof
            all_predictions[name] = pred
            
            full_rmse = np.sqrt(mean_squared_error(y, oof))
            print(f'{name} - Full OOF RMSE: {full_rmse:.4f}')
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    oof_df = pd.DataFrame(all_oof)
    predictions_df = pd.DataFrame(all_predictions)
    
    oof_df['target'] = y.values
    
    model_info = {
        'model_names': [name for name, _ in models],
        'num_models': len(models),
        'features_used': list(X.columns)
    }
    
    return oof_df, predictions_df, model_info


oof_results, test_predictions, model_info = create_ensemble(X, y, test)
    
oof_results.to_csv('oof_predictions.csv', index=False)
test_predictions.to_csv('test_predictions.csv', index=False)

print("\nModeling completed successfully!")
print(f"Trained {model_info['num_models']} models")
print("OOF predictions shape:", oof_results.shape)
print("Test predictions shape:", test_predictions.shape)


X = oof_results.drop(columns=['target'])
y = oof_results['target']

FOLDs = KFold(n_splits=5, shuffle=True,random_state=42)

oof_blend = np.zeros(len(X))
predictions_blend = np.zeros(len(test_predictions))
for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(X,y)):
    X.iloc[trn_idx], y.iloc[trn_idx]
    X.iloc[val_idx], y.iloc[val_idx]

    blend =  LGBMRegressor(objective = 'mae',
                           n_estimators = 4000,
                           max_depth = 15,
                           learning_rate = 0.01,
                           verbose=-1)
    blend.fit(X.iloc[trn_idx],y.iloc[trn_idx])
    
    oof_blend[val_idx] = blend.predict(X.iloc[val_idx])
    predictions_blend += blend.predict(test_predictions)/FOLDs.n_splits
    blend_score = mean_absolute_error(oof_blend, y)
    
    print('Fold', fold_, '-- Blend oof MAE is ---',blend_score)


sample = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sample['Listening_Time_minutes'] = test_predictions.mean(axis=1)
sample.to_csv('submission.csv', index=False)
sample.head(10)

