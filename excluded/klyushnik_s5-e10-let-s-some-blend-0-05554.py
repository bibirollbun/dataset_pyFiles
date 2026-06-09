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
    SelectFromModel
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
    cross_val_score
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
    AdaBoostRegressor
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error, 
    r2_score,
    make_scorer
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

from category_encoders import CatBoostEncoder, LeaveOneOutEncoder

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

def categorize_variable(df, column, labels):
    
    if len(labels) != 3:
        raise ValueError("3 type")
    
    bins = [-float('inf'), 
            df[column].quantile(0.25), 
            df[column].quantile(0.75), 
            float('inf')]
    
    df[f'{column}_group'] = pd.cut(df[column], bins=bins, labels=labels)
    return df

def replace_outliers_with_mean(df, threshold=3):

    df_clean = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        
        z_scores = np.abs(stats.zscore(df[col], nan_policy='omit')) 
        
        mean_val = df[col][z_scores <= threshold].mean()
        
        df_clean[col] = np.where(z_scores > threshold, mean_val, df[col])
        
    return df_clean


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

display(train.shape, test.shape)
display(train.info(), test.info())

test = test.drop(['id'], axis =1)
train = train.drop(['id'], axis =1)

display(train.describe().T)
display(test.describe().T)

duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

duplicates = test.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

train = train.drop_duplicates()

for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

display(train.head(5))


train = optimize_memory_usage(train)
test = optimize_memory_usage(test)


plot_numerical_features(train)


plot_numerical_boxplots(train)


plot_qq_plot(train)


plot_pairplot(train)


plot_correlation_matrix(train)


X = sm.add_constant(train.select_dtypes(include=[np.number]).iloc [:, 1:])

VIFs = pd.DataFrame()
VIFs['Variable'] = X.columns
VIFs['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(VIFs)


plot_categorical_features(train)


def create_features(df):
    df = df.copy()
    
    df['traffic_intensity'] = df['num_lanes'] * df['speed_limit'] / 10
    df['road_complexity'] = df['curvature'] * df['num_lanes']
    df['speed_curve_risk'] = df['curvature'] * (df['speed_limit'] / 50)
    
    df['poor_visibility'] = ((df['weather'].isin(['rainy', 'foggy', 'snowy'])) & 
                            (df['lighting'].isin(['dim', 'dark']))).astype(int)
    
    df['high_speed_curve'] = ((df['speed_limit'] > 60) & (df['curvature'] > 0.7)).astype(int)
    df['adverse_conditions'] = (df['poor_visibility'] | df['high_speed_curve']).astype(int)
    
    return df

train = create_features(train)
test = create_features(test)

display(train.shape, test.shape)
display(train.info(), test.info())


def encode_time(time_str):
    time_mapping = {'morning': 6, 'afternoon': 12, 'evening': 18, 'night': 0}
    time_val = time_mapping[time_str]
    
    sin_time = np.sin(2 * np.pi * time_val / 24)
    cos_time = np.cos(2 * np.pi * time_val / 24)
    return sin_time, cos_time

train['time_sin'], train['time_cos'] = zip(*train['time_of_day'].apply(encode_time))
test['time_sin'], test['time_cos'] = zip(*test['time_of_day'].apply(encode_time))

display(train.shape, test.shape)


def add_target_encoding(train_df, test_df, categorical_cols):
    for col in categorical_cols:
        target_mean = train_df.groupby(col)['accident_risk'].mean()
        
        train_df[f'{col}_target_enc'] = train_df[col].map(target_mean)
        
        test_df[f'{col}_target_enc'] = test_df[col].map(target_mean)
        
        overall_mean = train_df['accident_risk'].mean()
        test_df[f'{col}_target_enc'].fillna(overall_mean, inplace=True)
    
    return train_df, test_df

categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
train, test = add_target_encoding(train, test, categorical_cols)

display(train.shape, test.shape)
display(train.info(), test.info())


def add_statistical_features(train_df, test_df, group_cols):
    for col in group_cols:
        stats = train_df.groupby(col).agg({
            'accident_risk': ['mean', 'std', 'count']
        }).round(4)
        stats.columns = [f'{col}_risk_mean', f'{col}_risk_std', f'{col}_count']
        stats = stats.reset_index()
        
        train_df = train_df.merge(stats, on=col, how='left')
        
        test_df = test_df.merge(stats, on=col, how='left')
        
        overall_stats = train_df['accident_risk'].agg(['mean', 'std']).values
        test_df[f'{col}_risk_mean'].fillna(overall_stats[0], inplace=True)
        test_df[f'{col}_risk_std'].fillna(overall_stats[1], inplace=True)
        test_df[f'{col}_count'].fillna(len(train_df), inplace=True)
    
    return train_df, test_df

group_cols = ['road_type', 'weather', 'lighting', 'num_lanes']
train, test = add_statistical_features(train, test, group_cols)

display(train.shape, test.shape)


def add_interaction_features(df):
    df = df.copy()
    
    df['road_weather'] = df['road_type'] + '_' + df['weather']
    df['lighting_weather'] = df['lighting'] + '_' + df['weather']
    df['time_weather'] = df['time_of_day'] + '_' + df['weather']
    
    df['lanes_speed_interaction'] = df['num_lanes'] * df['speed_limit']
    df['curve_speed_ratio'] = df['curvature'] / (df['speed_limit'] + 1)  # +1 чтобы избежать деления на 0
    
    return df

train = add_interaction_features(train)
test = add_interaction_features(test)

display(train.shape, test.shape)


def add_binned_features(df):
    df = df.copy()
    
    speed_bins = [0, 30, 50, 60, 70]
    speed_labels = ['low', 'medium', 'high', 'very_high']
    df['speed_binned'] = pd.cut(df['speed_limit'], bins=speed_bins, labels=speed_labels)
    
    curvature_bins = [0, 0.3, 0.6, 1.0]
    curvature_labels = ['straight', 'moderate', 'curvy']
    df['curvature_binned'] = pd.cut(df['curvature'], bins=curvature_bins, labels=curvature_labels)
    
    return df

train = add_binned_features(train)
test = add_binned_features(test)

display(train.shape, test.shape)


train_cols = set(train.columns)
test_cols = set(test.columns)

print("Columns in test but not in train:", test_cols - train_cols)
print("Columns in train but not in test:", train_cols - test_cols)


train.head()


train.info()


object_category_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
print(object_category_cols)


for col in object_category_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined)
        
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

display(train.shape, test.shape)
display(train.info(), test.info())


X = train.drop(columns=['accident_risk'])
y = train['accident_risk']

#not today
X = variance_threshold(X,0.01)
list_name = (X.columns)
test = test[list_name]

display(X.shape, y.shape, test.shape)


def optimize_catboost_regression(X, y, n_trials=40, cv=5):
  
    def rmse_scorer(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    rmse_score = make_scorer(rmse_scorer, greater_is_better=False)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 0, 2),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'task_type': 'GPU', 
            'verbose': False,
            'early_stopping_rounds': 100
        }

        model = CatBoostRegressor(**params)
        
        
        scores = cross_val_score(model, X, y, cv=cv, 
                               scoring=rmse_score, n_jobs=1)
        
       
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')  
    study.optimize(objective, n_trials=n_trials)
    
    return study

catboost_studies = []
for i in range(3):
    print(f"\nRunning CatBoost Regression optimization {i+1}/3")
    study = optimize_catboost_regression(X, y, n_trials=40)
    catboost_studies.append(study)
    print(f"Best trial {i+1}:")
    print(f"  Value (-RMSE): {study.best_value:.5f}")
    print(f"  Actual RMSE: {-study.best_value:.5f}")  
    print(f"  Params: {study.best_params}")


catboost_rmse_params = []

for i, study in enumerate(catboost_studies):
    params = study.best_params.copy()
    params['loss_function'] = 'RMSE'
    params['eval_metric'] = 'RMSE'
    params['verbose'] = False
    catboost_rmse_params.append(params)
    print(f"\nBest parameters for model {i+1}:")
    for key, value in params.items():
        print(f"  {key}: {value}")


print("\n" + "="*50)
print("OPTIMIZATION SUMMARY")
print("="*50)
for i, (study, params) in enumerate(zip(catboost_studies, catboost_rmse_params)):
    print(f"Model {i+1}: RMSE = {-study.best_value:.5f}")

config_0 = {
  'iterations': 537,
  'depth': 8,
  'learning_rate': 0.017842162028402962,
  'l2_leaf_reg': 7.370412244865911,
  'border_count': 191,
  'random_strength': 1.1087121186242526,
  'bagging_temperature': 0.6239080010677365,
  'grow_policy': 'Depthwise',
  'min_data_in_leaf': 27,
  'loss_function': 'RMSE',
  'eval_metric': 'RMSE',
  'verbose': False
}

config_1 = {
  'iterations': 950,
  'depth': 8,
  'learning_rate': 0.018571742536927362,
  'l2_leaf_reg': 4.524357457928256,
  'border_count': 153,
  'random_strength': 0.4862818740678978,
  'bagging_temperature': 0.4014679810273516,
  'grow_policy': 'Depthwise',
  'min_data_in_leaf': 25,
  'loss_function': 'RMSE',
  'eval_metric': 'RMSE',
  'verbose': False
}

config_2 = {
  'iterations': 301,
  'depth': 8,
  'learning_rate': 0.04484782966119428,
  'l2_leaf_reg': 2.8119862594836165,
  'border_count': 102,
  'random_strength': 0.8375799223678608,
  'bagging_temperature': 0.874858527606345,
  'grow_policy': 'Depthwise',
  'min_data_in_leaf': 3,
  'loss_function': 'RMSE',
  'eval_metric': 'RMSE',
  'verbose': False

}

config_3 = {
    
  'iterations': 180,
  'depth': 8,
  'learning_rate': 0.06730149449365055,
  'l2_leaf_reg': 7.448233402386369,
  'border_count': 131,
  'random_strength': 0.5274500030545808,
  'bagging_temperature': 0.5049369701743718,
  'grow_policy': 'Depthwise',
  'min_data_in_leaf': 47,
  'loss_function': 'RMSE',
  'eval_metric': 'RMSE',
  'verbose': False
    
}

catboost_rmse_params.append(config_0)
catboost_rmse_params.append(config_1)
catboost_rmse_params.append(config_2)
catboost_rmse_params.append(config_3)


def optimize_xgboost_regression(X, y, n_trials=40, cv=5):
   
    def rmse_scorer(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    rmse_score = make_scorer(rmse_scorer, greater_is_better=False)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
            'eval_metric': 'rmse',
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'max_leaves': trial.suggest_int('max_leaves', 32, 256),
            'max_bin': trial.suggest_int('max_bin', 128, 256),
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'sampling_method': trial.suggest_categorical('sampling_method', ['uniform', 'gradient_based'])
        }
        
        model = xgb.XGBRegressor(**params)
        
        scores = cross_val_score(model, X, y, cv=cv, 
                               scoring=rmse_score, n_jobs=1)
        
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')  
    study.optimize(objective, n_trials=n_trials)
    
    return study


xgb_studies = []
for i in range(3):
    print(f"\nRunning XGBoost Regression optimization {i+1}/3")
    study = optimize_xgboost_regression(X, y, n_trials=40)
    xgb_studies.append(study)
    print(f"Best trial {i+1}:")
    print(f"  Value (-RMSE): {study.best_value:.5f}")
    print(f"  Actual RMSE: {-study.best_value:.5f}")  
    print(f"  Params: {study.best_params}")

xgb_rmse_params = []
for i, study in enumerate(xgb_studies):
    params = study.best_params.copy()
    params.update({
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'rmse'
    })
    xgb_rmse_params.append(params)
    print(f"\nXGBoost config {i+1}:")
    for key, value in params.items():
        print(f"  {key}: {value}")

print("\n" + "="*50)
print("XGBOOST OPTIMIZATION SUMMARY")
print("="*50)
for i, (study, params) in enumerate(zip(xgb_studies, xgb_rmse_params)):
    print(f"Model {i+1}: RMSE = {-study.best_value:.5f}")

config_0 = {
    
  'n_estimators': 760,
  'max_depth': 10,
  'learning_rate': 0.013795917955898896,
  'subsample': 0.7267734213015471,
  'colsample_bytree': 0.711424372302688,
  'gamma': 0.004090103222241464,
  'min_child_weight': 8,
  'reg_lambda': 0.3471752674396913,
  'reg_alpha': 0.6283614659091622,
  'max_delta_step': 2,
  'grow_policy': 'lossguide',
  'max_leaves': 215,
  'max_bin': 243,
  'sampling_method': 'gradient_based',
  'tree_method': 'gpu_hist',
  'predictor': 'gpu_predictor',
  'eval_metric': 'rmse',
    
}

config_2 = {
    
  'n_estimators': 396,
  'max_depth': 12,
  'learning_rate': 0.02054786311244503,
  'subsample': 0.8006661157443362,
  'colsample_bytree': 0.926330943646608,
  'gamma': 0.020881547833851988,
  'min_child_weight': 2,
  'reg_lambda': 0.9947304284610006,
  'reg_alpha': 0.8364558915124849,
  'max_delta_step': 5,
  'grow_policy': 'lossguide',
  'max_leaves': 148,
  'max_bin': 219,
  'sampling_method': 'uniform',
  'tree_method': 'gpu_hist',
  'predictor': 'gpu_predictor',
  'eval_metric': 'rmse',
    
}

config_2 = {
    
  'n_estimators': 551,
  'max_depth': 6,
  'learning_rate': 0.0503127818403867,
  'subsample': 0.6017365244797307,
  'colsample_bytree': 0.7777128943498162,
  'gamma': 0.0018954277087720513,
  'min_child_weight': 2,
  'reg_lambda': 0.8471823203750417,
  'reg_alpha': 0.6262587504812429,
  'max_delta_step': 5,
  'grow_policy': 'depthwise',
  'max_leaves': 59,
  'max_bin': 130,
  'sampling_method': 'gradient_based',
  'tree_method': 'gpu_hist',
  'predictor': 'gpu_predictor',
  'eval_metric': 'rmse',
}

config_3 = {

  'n_estimators': 328,
  'max_depth': 11,
  'learning_rate': 0.031634147945890305,
  'subsample': 0.6900450298316714,
  'colsample_bytree': 0.7809160542619278,
  'gamma': 0.0005155545903676355,
  'min_child_weight': 7,
  'reg_lambda': 1.6398728691452662,
  'reg_alpha': 0.4243594149698391,
  'max_delta_step': 2,
  'grow_policy': 'depthwise',
  'max_leaves': 197,
  'max_bin': 169,
  'sampling_method': 'uniform',
  'tree_method': 'gpu_hist',
  'predictor': 'gpu_predictor',
  'eval_metric': 'rmse',
}

xgb_rmse_params.append(config_0)
xgb_rmse_params.append(config_1)
xgb_rmse_params.append(config_2)
xgb_rmse_params.append(config_3)


def optimize_lightgbm_regression(X, y, n_trials=40, cv=5):
    
    def rmse_scorer(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    rmse_score = make_scorer(rmse_scorer, greater_is_better=False)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
            'min_split_gain': trial.suggest_float('min_split_gain', 0, 0.2),
            'path_smooth': trial.suggest_float('path_smooth', 0, 1),
            'max_bin': trial.suggest_int('max_bin', 64, 255),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1
        }
        
        model = LGBMRegressor(**params)
        
        scores = cross_val_score(model, X, y, cv=cv, 
                               scoring=rmse_score, n_jobs=1)
        
        return scores.mean()
    
    study = optuna.create_study(direction='maximize') 
    study.optimize(objective, n_trials=n_trials)
    
    return study


lgbm_studies = []
for i in range(3):
    print(f"\nRunning LightGBM Regression optimization {i+1}/3")
    study = optimize_lightgbm_regression(X, y, n_trials=40)
    lgbm_studies.append(study)
    print(f"Best trial {i+1}:")
    print(f"  Value (-RMSE): {study.best_value:.5f}")
    print(f"  Actual RMSE: {-study.best_value:.5f}")  
    print(f"  Params: {study.best_params}")


lgbm_rmse_params = []
for i, study in enumerate(lgbm_studies):
    params = study.best_params.copy()
    params.update({
        'device': 'gpu',
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1
    })
    lgbm_rmse_params.append(params)
    print(f"\nLightGBM config {i+1}:")
    for key, value in params.items():
        print(f"  {key}: {value}")


print("\n" + "="*50)
print("LIGHTGBM OPTIMIZATION SUMMARY")
print("="*50)
for i, (study, params) in enumerate(zip(lgbm_studies, lgbm_rmse_params)):
    print(f"Model {i+1}: RMSE = {-study.best_value:.5f}")

config_0 = {
    
  'n_estimators': 475,
  'max_depth': 6,
  'learning_rate': 0.02609566703640587,
  'num_leaves': 126,
  'min_child_samples': 10,
  'min_child_weight': 0.08696250243662902,
  'feature_fraction': 0.6040602458875166,
  'bagging_fraction': 0.8208724518623647,
  'bagging_freq': 3,
  'lambda_l1': 0.361298537906501,
  'lambda_l2': 0.3227120302947219,
  'min_split_gain': 0.0001271057985094303,
  'path_smooth': 0.02245717701896824,
  'max_bin': 243,
  'extra_trees': False,
  'device': 'gpu',
  'objective': 'regression',
  'metric': 'rmse',
  'verbose': -1,

}

config_1 = {
    
  'n_estimators': 729,
  'max_depth': 11,
  'learning_rate': 0.019579347467589887,
  'num_leaves': 51,
  'min_child_samples': 36,
  'min_child_weight': 0.07029393408032038,
  'feature_fraction': 0.8046195627507132,
  'bagging_fraction': 0.9548143084700726,
  'bagging_freq': 3,
  'lambda_l1': 0.08652123482408915,
  'lambda_l2': 0.11178179927007909,
  'min_split_gain': 0.000188284436315936,
  'path_smooth': 0.5054225516315782,
  'max_bin': 182,
  'extra_trees': False,
  'device': 'gpu',
  'objective': 'regression',
  'metric': 'rmse',
  'verbose': -1

}

config_2 = {
    
  'n_estimators': 830,
  'max_depth': 10,
  'learning_rate': 0.032717852791327814,
  'num_leaves': 58,
  'min_child_samples': 29,
  'min_child_weight': 0.022821245946439715,
  'feature_fraction': 0.9419990343992585,
  'bagging_fraction': 0.9090572699720074,
  'bagging_freq': 6,
  'lambda_l1': 0.16396068014180826,
  'lambda_l2': 0.6153088612059665,
  'min_split_gain': 0.000978174955847403,
  'path_smooth': 0.7739653148026253,
  'max_bin': 114,
  'extra_trees': False,
  'device': 'gpu',
  'objective': 'regression',
  'metric': 'rmse',
  'verbose': -1

}

config_3 = {

  'n_estimators': 475,
  'max_depth': 6,
  'learning_rate': 0.02609566703640587,
  'num_leaves': 126,
  'min_child_samples': 10,
  'min_child_weight': 0.08696250243662902,
  'feature_fraction': 0.6040602458875166,
  'bagging_fraction': 0.8208724518623647,
  'bagging_freq': 3,
  'lambda_l1': 0.361298537906501,
  'lambda_l2': 0.3227120302947219,
  'min_split_gain': 0.0001271057985094303,
  'path_smooth': 0.02245717701896824,
  'max_bin': 243,
  'extra_trees': False,
  'device': 'gpu',
  'objective': 'regression',
  'metric': 'rmse',
  'verbose': -1,
}

lgbm_rmse_params.append(config_0)
lgbm_rmse_params.append(config_1)
lgbm_rmse_params.append(config_2)
lgbm_rmse_params.append(config_3)


gb_params_list = [
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 
         'random_state': 42, 'subsample': 0.8},
        {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 7, 
         'random_state': 42, 'subsample': 0.9},
        {'n_estimators': 150, 'learning_rate': 0.2, 'max_depth': 3, 
         'random_state': 42, 'subsample': 0.7},
       {'n_estimators': 100, 'learning_rate': 0.02, 'max_depth': 2, 
         'random_state': 42, 'subsample': 0.7}
    ]

et_params_list = [
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 
         'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2, 
         'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 
         'random_state': 42, 'n_jobs': -1}
    ]

hgb_params_list = [
        {'max_iter': 200, 'learning_rate': 0.1, 'max_depth': 10, 
         'random_state': 42, 'l2_regularization': 0.1},
        {'max_iter': 300, 'learning_rate': 0.05, 'max_depth': 15, 
         'random_state': 42, 'l2_regularization': 0.01},
        {'max_iter': 100, 'learning_rate': 0.1, 'max_depth': 5, 
         'random_state': 42, 'l2_regularization': 0.01}
    ]

ada_params_list = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
        {'n_estimators': 200, 'learning_rate': 0.05, 'random_state': 42},
        {'n_estimators': 300, 'learning_rate': 0.01, 'random_state': 42}
    ]

bagging_params = [
        {'n_estimators': 20, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 50, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    ]

linear_models = [
        ('ridge', Ridge(alpha=1.0, random_state=42)),
        ('lasso', Lasso(alpha=0.1, random_state=42)),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
    ]



def create_ensemble(X, y, test, n_folds=5):
    FOLDS = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_oof = {}
    all_predictions = {}
    models = []

    for i, params in enumerate(catboost_rmse_params, 1):
        models.append((f'cat_{i}', CatBoostRegressor(**params)))
    
    for i, params in enumerate(xgb_rmse_params, 1):
        models.append((f'xgb_{i}', xgb.XGBRegressor(**params, n_jobs=1)))
    
    for i, params in enumerate(lgbm_rmse_params, 1):
        models.append((f'lgb_{i}', LGBMRegressor(**params, n_jobs=1)))

    for i, params in enumerate(gb_params_list, 1):
        models.append((f'gb_{i}', GradientBoostingRegressor(**params)))

    for i, params in enumerate(et_params_list, 1):
        models.append((f'et_{i}', ExtraTreesRegressor(**params)))

    for i, params in enumerate(hgb_params_list, 1):
        models.append((f'hgb_{i}', HistGradientBoostingRegressor(**params)))

    for i, params in enumerate(ada_params_list, 1):
        models.append((f'ada_{i}', AdaBoostRegressor(**params)))

    for i, params in enumerate(bagging_params, 1):
        models.append((f'bag_{i}', BaggingRegressor(**params)))

    for name, model in linear_models:
        models.append((f'linear_{name}', model))
    
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
        'features_used': list(X.columns),
        'used_log_transform': False
    }
    
    return oof_df, predictions_df, model_info

oof_results, test_predictions, model_info = create_ensemble(X, y, test)
    
oof_results.to_csv('oof_predictions.csv', index=False)
test_predictions.to_csv('test_predictions.csv', index=False)

print("\nModeling completed successfully!")
print(f"Trained {model_info['num_models']} models")
print("OOF predictions shape:", oof_results.shape)
print("Test predictions shape:", test_predictions.shape)


def create_optimal_ensemble(oof_results, test_predictions, y):
   
    oof_predictions = oof_results.drop(['target'], axis=1, errors='ignore')
    y_true = oof_results['target'] if 'target' in oof_results else y
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_predictions, y_true)
    ridge_pred = ridge.predict(oof_predictions)
    ridge_rmse = np.sqrt(mean_squared_error(y_true, ridge_pred))
    
    def objective(weights):
        weighted_pred = np.dot(oof_predictions.values, weights)
        return np.sqrt(mean_squared_error(y_true, weighted_pred))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * len(oof_predictions.columns)
    initial_weights = np.ones(len(oof_predictions.columns)) / len(oof_predictions.columns)
    
    result = minimize(objective, initial_weights, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = result.x
    optimized_pred = np.dot(oof_predictions.values, optimized_weights)
    optimized_rmse = np.sqrt(mean_squared_error(y_true, optimized_pred))
    
    if ridge_rmse <= optimized_rmse:
        print(f"Using Ridge (RMSE: {ridge_rmse:.4f})")
        test_pred = ridge.predict(test_predictions)
        weights = ridge.coef_
    else:
        print(f"Using Optimized Weights (RMSE: {optimized_rmse:.4f})")
        test_pred = np.dot(test_predictions.values, optimized_weights)
        weights = optimized_weights
    
    weights = weights / weights.sum()
    
    return test_pred, weights, min(ridge_rmse, optimized_rmse)

test_pred, weights, best_rmse = create_optimal_ensemble(oof_results, test_predictions, y)

print(f"\nBest Ensemble RMSE: {best_rmse:.4f}")
print("Final weights:")
for model, weight in zip(test_predictions.columns, weights):
    print(f"  {model}: {weight:.4f}")


sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
sample['accident_risk'] = test_pred
sample.to_csv('submission.csv', index=False)
sample.head(10)

