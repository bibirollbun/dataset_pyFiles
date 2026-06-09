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


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sub_28v = pd.read_csv('/kaggle/input/subv28/submission.csv')

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

for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

display(train.head(5))


train = optimize_memory_usage(train)
test = optimize_memory_usage(test)


plot_numerical_features(train)


plot_numerical_boxplots(train)


plot_qq_plot(train)


plot_pairplot(test)


plot_correlation_matrix(train)


X = sm.add_constant(train.select_dtypes(include=[np.number]).iloc [:, 1:])

VIFs = pd.DataFrame()
VIFs['Variable'] = X.columns
VIFs['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(VIFs)


train.columns


train['LoudnessEnergy_Interaction'] = train['AudioLoudness'] * train['Energy']
train['LoudnessEnergy_Ratio'] = train['Energy'] / (-1*train['AudioLoudness'] + 1e-7)
train['EnergyPerMinute'] = train['Energy'] * (60000 / train['TrackDurationMs'])

train['VocalInstrumental_Balance'] = train['VocalContent'] / (train['InstrumentalScore'] + 1e-7)
train['VocalInstrumental_Interaction'] = train['VocalContent'] * train['InstrumentalScore']

train['AcousticnessEnergy_Interaction'] = train['AcousticQuality'] * train['Energy'] 
train['MoodEnergy_Interaction'] = train['MoodScore'] * train['Energy'] 
train['LiveEnergy_Interaction'] = train['LivePerformanceLikelihood'] * train['Energy'] 

train['RhythmStability'] = train['RhythmScore'] * train['TrackDurationMs'] 
train['RhythmEnergy_Interaction'] = train['RhythmScore'] * train['Energy']

train['Energy_Squared'] = train['Energy'] ** 2
train['Log_Duration'] = np.log1p(train['TrackDurationMs']) 
train['Log_Loudness'] = np.log1p(-1*train['AudioLoudness'])

train['Energy_sin'] = np.sin(2 * np.pi * train['Energy'])
train['Energy_cos'] = np.cos(2 * np.pi * train['Energy'])

train['LogLoudness_Energy_Interaction'] = train['Log_Loudness'] * train['Energy']
train['LogDuration_Energy_Interaction'] = train['Log_Duration'] * train['Energy']

test['LoudnessEnergy_Interaction'] = test['AudioLoudness'] * test['Energy']
test['LoudnessEnergy_Ratio'] = test['Energy'] / (-1*test['AudioLoudness'] + 1e-7)
test['EnergyPerMinute'] = test['Energy'] * (60000 / test['TrackDurationMs'])

test['VocalInstrumental_Balance'] = test['VocalContent'] / (test['InstrumentalScore'] + 1e-7)
test['VocalInstrumental_Interaction'] = test['VocalContent'] * test['InstrumentalScore']

test['AcousticnessEnergy_Interaction'] = test['AcousticQuality'] * test['Energy'] 
test['MoodEnergy_Interaction'] = test['MoodScore'] * test['Energy'] 
test['LiveEnergy_Interaction'] = test['LivePerformanceLikelihood'] * test['Energy'] 

test['RhythmStability'] = test['RhythmScore'] * test['TrackDurationMs'] 
test['RhythmEnergy_Interaction'] = test['RhythmScore'] * test['Energy']

test['Energy_Squared'] = test['Energy'] ** 2
test['Log_Duration'] = np.log1p(test['TrackDurationMs']) 
test['Log_Loudness'] = np.log1p(-1*test['AudioLoudness'])

test['LogLoudness_Energy_Interaction'] = test['Log_Loudness'] * test['Energy']
test['LogDuration_Energy_Interaction'] = test['Log_Duration'] * test['Energy']

test['Energy_sin'] = np.sin(2 * np.pi * test['Energy'])
test['Energy_cos'] = np.cos(2 * np.pi * test['Energy'])


display(train.shape, test.shape)


X = train.drop(columns=['BeatsPerMinute'])
y = train['BeatsPerMinute']

#not today
X = variance_threshold(X,0.01)
list_name = (X.columns)
test = test[list_name]

display(X.shape, y.shape, test.shape)


scaler = StandardScaler()

X[X.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(X[X.select_dtypes(include=[np.number]).columns])
test[X.select_dtypes(include=[np.number]).columns] = scaler.transform(test[X.select_dtypes(include=[np.number]).columns])

X.shape, y.shape, test.shape


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

parameters = { 
               'iterations': 357,
                'depth': 4,
              'learning_rate': 0.019293830422619496,
              'l2_leaf_reg': 1.7030277055187715,
              'border_count': 68,
              'random_strength': 0.6543682482313615,
              'bagging_temperature': 0.9769334546523897,
              'grow_policy': 'SymmetricTree',
              'min_data_in_leaf': 15,
              'loss_function': 'RMSE',
              'eval_metric': 'RMSE',
              'verbose': False,
}


parameters_2 = {
                  'iterations': 270,
                  'depth': 6,
                  'learning_rate': 0.02773583289422086,
                  'l2_leaf_reg': 5.453251706129398,
                  'border_count': 145,
                  'random_strength': 0.3971144579660211,
                  'bagging_temperature': 0.6160360201200582,
                  'grow_policy': 'SymmetricTree',
                  'min_data_in_leaf': 4,
                  'loss_function': 'RMSE',
                  'eval_metric': 'RMSE',
                  'verbose': False
}

parameters_3 = {
                  'iterations': 455,
                  'depth': 6,
                  'learning_rate': 0.011395973420244683,
                  'l2_leaf_reg': 4.358597404490118,
                  'border_count': 229,
                  'random_strength': 0.428844179505764,
                  'bagging_temperature': 0.12315921245762332,
                  'grow_policy': 'SymmetricTree',
                  'min_data_in_leaf': 6,
                  'loss_function': 'RMSE',
                  'eval_metric': 'RMSE',
                  'verbose': False
}


catboost_rmse_params.append(parameters)
catboost_rmse_params.append(parameters_2)
catboost_rmse_params.append(parameters_3)


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

config_1 = {
              'n_estimators': 374,
              'max_depth': 7,
              'learning_rate': 0.0050247034923121625,
              'subsample': 0.6881223558971589,
              'colsample_bytree': 0.7869461512231557,
              'gamma': 0.828135543683264,
              'min_child_weight': 6,
              'reg_lambda': 1.667240816732193,
              'reg_alpha': 0.44092922531878276,
              'max_delta_step': 4,
              'grow_policy': 'lossguide',
              'max_leaves': 117,
              'max_bin': 147,
              'eval_metric': 'rmse'
}

config_2 = {
              'n_estimators': 136,
              'max_depth': 4,
              'learning_rate': 0.027359333604422675,
              'subsample': 0.6825248494688766,
              'colsample_bytree': 0.7955947575775649,
              'gamma': 0.9730566106537739,
              'min_child_weight': 6,
              'reg_lambda': 1.3545740379218356,
              'reg_alpha': 0.6652666597117963,
              'max_delta_step': 4,
              'grow_policy': 'depthwise',
              'max_leaves': 174,
              'max_bin': 171,
              'eval_metric': 'rmse'
}

config_3 = {
              'n_estimators': 317,
              'max_depth': 3,
              'learning_rate': 0.0105144370995476,
              'subsample': 0.8418008197832596,
              'colsample_bytree': 0.6029589209120716,
              'gamma': 0.3235479454568891,
              'min_child_weight': 10,
              'reg_lambda': 0.14522064555399145,
              'reg_alpha': 0.7113614331944631,
              'max_delta_step': 5,
              'grow_policy': 'depthwise',
              'max_leaves': 127,
              'max_bin': 256,
              'eval_metric': 'rmse'
}

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

config_1 ={
              'n_estimators': 261,
              'max_depth': 5,
              'learning_rate': 0.009855073801085426,
              'num_leaves': 68,
              'min_child_samples': 21,
              'min_child_weight': 0.01557636384683754,
              'feature_fraction': 0.6236883661645901,
              'bagging_fraction': 0.5921754125318668,
              'bagging_freq': 4,
              'lambda_l1': 0.004621624849205308,
              'lambda_l2': 0.42798791477155057,
              'min_split_gain': 0.050053849393189444,
              'path_smooth': 0.7940272295096928,
              'max_bin': 234,
              'extra_trees': False,
              'objective': 'regression',
              'metric': 'rmse',
              'verbose': -1
}

config_2 = {
              'n_estimators': 170,
              'max_depth': 10,
              'learning_rate': 0.012150658735032613,
              'num_leaves': 36,
              'min_child_samples': 33,
              'min_child_weight': 0.0445474321894643,
              'feature_fraction': 0.8091593625918829,
              'bagging_fraction': 0.9220046562291125,
              'bagging_freq': 3,
              'lambda_l1': 0.855143026001996,
              'lambda_l2': 0.13448370895430672,
              'min_split_gain': 0.13360981440465317,
              'path_smooth': 0.6366083660470108,
              'max_bin': 222,
              'extra_trees': False,
              'objective': 'regression',
              'metric': 'rmse',
              'verbose': -1
}

config_3 = {
              'n_estimators': 333,
              'max_depth': 8,
              'learning_rate': 0.0076194307392825796,
              'num_leaves': 21,
              'min_child_samples': 13,
              'min_child_weight': 0.08053934568837545,
              'feature_fraction': 0.742751464628032,
              'bagging_fraction': 0.696853930345524,
              'bagging_freq': 3,
              'lambda_l1': 0.519780554174363,
              'lambda_l2': 0.3395268913171968,
              'min_split_gain': 0.045219616892989214,
              'path_smooth': 0.794995521077607,
              'max_bin': 256,
              'extra_trees': False,
              'objective': 'regression',
              'metric': 'rmse',
              'verbose': -1
}

lgbm_rmse_params.append(config_1)
lgbm_rmse_params.append(config_2)
lgbm_rmse_params.append(config_3)


gb_params_list = [
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 
         'random_state': 42, 'subsample': 0.8},
        {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 7, 
         'random_state': 42, 'subsample': 0.9},
        {'n_estimators': 150, 'learning_rate': 0.2, 'max_depth': 3, 
         'random_state': 42, 'subsample': 0.7}
    ]

et_params_list = [
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 
         'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2, 
         'random_state': 42, 'n_jobs': -1}
    ]

hgb_params_list = [
        {'max_iter': 200, 'learning_rate': 0.1, 'max_depth': 10, 
         'random_state': 42, 'l2_regularization': 0.1},
        {'max_iter': 300, 'learning_rate': 0.05, 'max_depth': 15, 
         'random_state': 42, 'l2_regularization': 0.01}
    ]

ada_params_list = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
        {'n_estimators': 200, 'learning_rate': 0.05, 'random_state': 42}
    ]

bagging_params = [
        {'n_estimators': 20, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 50, 'random_state': 42, 'n_jobs': -1}
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


sub_28v


sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
sample['BeatsPerMinute'] = (test_pred + sub_28v['BeatsPerMinute']) / 2
sample.to_csv('submission.csv', index=False)
sample.head(10)

