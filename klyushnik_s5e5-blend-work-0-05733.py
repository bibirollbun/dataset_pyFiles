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

def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("y_true and y_pred must not contain negative values!")
    
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    squared_log_errors = (log_true - log_pred) ** 2
    mean_squared_log_error = np.mean(squared_log_errors)
    return np.sqrt(mean_squared_log_error)

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


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

train.shape, test.shape


calories = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv', index_col="User_ID")

calories.info()


calories = calories.rename(columns={"Gender": "Sex"})

train = pd.concat([train, calories])

train.shape, test.shape


test = test.drop(['id'], axis =1)
train = train.drop(['id'], axis =1)


train_df = train.copy()
test_df = test.copy()

train_df = train_df.drop(columns=['Calories', 'Sex'])
test_df = test_df.drop(columns=['Sex'])


train.info()


train.head(5)


train.describe().T


duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

train = train.drop_duplicates()


for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# train = optimize_memory_usage(train)
# test = optimize_memory_usage(test)


plot_numerical_features(train)


plot_numerical_boxplots(train)


plot_qq_plot(train)


plot_correlation_matrix(train)


X = sm.add_constant(train.select_dtypes(include=[np.number]).iloc [:, 1:])

VIFs = pd.DataFrame()
VIFs['Variable'] = X.columns
VIFs['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(VIFs)


plot_categorical_features(train)


categorize_variable(train, 'Age', ["young", "middle age", 'old'])
categorize_variable(train, 'Height', ["short", "middle", 'high'])
categorize_variable(train, 'Weight', ["thin", "normal", 'fat'])
categorize_variable(train, 'Duration', ["short", "mean", 'long'])
categorize_variable(train, 'Heart_Rate', ["low", "normal", 'high'])


categorize_variable(test, 'Age', ["young", "middle age", 'old'])
categorize_variable(test, 'Height', ["short", "middle", 'high'])
categorize_variable(test, 'Weight', ["thin", "normal", 'fat'])
categorize_variable(test, 'Duration', ["short", "mean", 'long'])
categorize_variable(test, 'Heart_Rate', ["low", "normal", 'high'])

train['BMI'] = train['Weight'] / (train['Height'] ** 2)
test['BMI'] = test['Weight'] / (test['Height'] ** 2)


train.shape, test.shape


# col = ['Age_group', 'Height_group', 'Weight_group', 'Duration_group',
#        'Heart_Rate_group', 'Sex']

# TE = MEstimateEncoder(cols=col, m=5.0)

# train[col] = TE.fit_transform(train[col], train['Calories'])
# test[col] = TE.transform(test[col])

# train.shape, test.shape

col = ['Age_group', 'Height_group', 'Weight_group', 'Duration_group',
       'Heart_Rate_group', 'Sex']

for c in col:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')


train.info()


transform = PowerTransformer(method='yeo-johnson')
transform = QuantileTransformer(n_quantiles=5, random_state=0)

for i in test_df.select_dtypes(include=[np.number]).columns:
    train_df[i+' +log'] = (train_df[i]+1).transform(np.log)
    test_df[i+' +log'] =(test_df[i]+1).transform(np.log)

    train_df[i+' +log1'] = (train_df[i]+1).transform(np.log1p)
    test_df[i+' +log1'] =(test_df[i]+1).transform(np.log1p)
    
    train_df[i+' +y_j'] = transform.fit_transform(train_df[[i]])
    test_df[i+' +y_j'] = transform.fit_transform(test_df[[i]])
    
    train_df[i+' +q_t'] = transform.fit_transform(train_df[[i]])
    test_df[i+' +q_t'] = transform.fit_transform(test_df[[i]])
    
    train_df[i+' +sqrt'] = (train_df[i]+1).transform(np.sqrt)
    test_df[i+' +sqrt'] =(test_df[i]+1).transform(np.sqrt)
    
train_df.shape, test_df.shape


train_df = PolynomialFeatures_labeled(train_df, 2)
test_df = PolynomialFeatures_labeled(test_df, 2)

train_df.shape, test_df.shape


train_df = variance_threshold(train_df,0.02)
list_name = (train_df.columns)
test_df = test_df[list_name]

train_df.shape, test_df.shape


n_components = 4

pca = PCA(n_components=n_components)

pca_components = pca.fit_transform(train_df)
pca_components_test = pca.transform(test_df)

pca_df = pd.DataFrame(pca_components, columns=[f'PCA_{i+1}' for i in range(n_components)])
train = pd.concat([train, pca_df], axis=1)

pca_df_test = pd.DataFrame(pca_components_test, columns=[f'PCA_{i+1}' for i in range(n_components)])
test = pd.concat([test, pca_df_test], axis=1)

train.shape, test.shape


total_missing = train.isnull().sum().sum()
print(total_missing)


train.dropna(inplace=True)
test.dropna(inplace=True)

train.shape, test.shape


X = train.drop(columns=['Calories'])
y = train['Calories']
print('before threshold:',X.shape, y.shape)

# X = variance_threshold(X,0.03)
# list_name = (X.columns)
# test = test[list_name]

# print('after threshold:',X.shape, y.shape, test.shape)


scaler = StandardScaler()

X[X.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(X[X.select_dtypes(include=[np.number]).columns])
test[X.select_dtypes(include=[np.number]).columns] = scaler.transform(test[X.select_dtypes(include=[np.number]).columns])

X.shape, y.shape, test.shape


catboost_params = [
    {
        'iterations': 3000,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'border_count': 32,
        'bagging_temperature': 1,
        'random_strength': 1,
        'task_type': 'GPU',
    },
    {
        'iterations': 4000,
        'depth': 8,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5,
        'loss_function': 'RMSE',
        'border_count': 64,
        'bagging_temperature': 0.5,
        'random_strength': 2,
        'task_type': 'GPU',
    },
    {
        'iterations': 4400,
        'depth': 5,
        'learning_rate': 0.1,
        'l2_leaf_reg': 10,
        'loss_function': 'RMSE',
        'border_count': 32,
        'bagging_temperature': 0.8,
        'random_strength': 1.5,
        'task_type': 'GPU',
    },
    {
        'iterations': 3200,
        'depth': 8,
        'learning_rate': 0.02,
        'l2_leaf_reg': 6,
        'loss_function': 'RMSE',
        'border_count': 24,
        'random_strength': 1.2,
        'bootstrap_type': 'Bernoulli',
        'task_type': 'GPU',
    },

]


xgb_params = [
    {
        'n_estimators': 3200,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'eval_metric': 'rmse',
        'colsample_bytree': 0.8,
        'gamma': 0,
        'min_child_weight': 1,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
    },
    {
        'n_estimators': 3500,
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.9,
        'eval_metric': 'rmse',
        'colsample_bytree': 0.9,
        'gamma': 0.1,
        'min_child_weight': 2,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
    },
    {
        'n_estimators': 2900,
        'max_depth': 10,
        'learning_rate': 0.02,
        'subsample': 0.75,
        'eval_metric': 'rmse',
        'colsample_bytree': 0.75,
        'gamma': 0.4,
        'min_child_weight': 2,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
    },
    {
        'n_estimators': 3100,
        'max_depth': 12,
        'learning_rate': 0.1,
        'subsample': 0.85,
        'eval_metric': 'rmse',
        'colsample_bytree': 0.85,
        'gamma': 0.5,
        'min_child_weight': 1,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
    },

]


lgbm_params = [
    {
        'n_estimators': 3200,
        'max_depth': 5,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'metric': 'rmse',
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'device': 'gpu',
    },
    {
        'n_estimators': 3500,
        'max_depth': 8,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'metric': 'rmse',
        'min_data_in_leaf': 15,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'device': 'gpu',
    },
    {
        'n_estimators': 3100,
        'max_depth': 4,
        'learning_rate': 0.05,
        'num_leaves': 15,
        'metric': 'rmse',
        'min_data_in_leaf': 10,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'device': 'gpu',
    },
    {
        'n_estimators': 4000,
        'max_depth': 7,
        'learning_rate': 0.01,
        'num_leaves': 95,
        'metric': 'rmse',
        'min_data_in_leaf': 18,
        'feature_fraction': 0.83,
        'bagging_fraction': 0.83,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'device': 'gpu',
    },
]



def create_ensemble(X, y, test, n_folds=5, use_log_transform=True):
    FOLDS = KFold(n_splits=5, shuffle=True, random_state=42)
    
    col = ['Age_group', 'Height_group', 'Weight_group', 'Duration_group',
       'Heart_Rate_group', 'Sex']
    
    if use_log_transform:
        y_transformed = np.log1p(y)
        print("Applied log1p transformation to target variable")
    else:
        y_transformed = y.copy()
    
    all_oof = {}
    all_predictions = {}
    models = []

    for i, params in enumerate(catboost_params, 1):
        models.append((f'cat_{i}', CatBoostRegressor(**params,cat_features = col, verbose=0, thread_count=1)))
    
    for i, params in enumerate(xgb_params, 1):
        models.append((f'xgb_{i}', xgb.XGBRegressor(**params, enable_categorical=True, n_jobs=1)))
    
    for i, params in enumerate(lgbm_params, 1):
        models.append((f'lgb_{i}', LGBMRegressor(**params, categorical_feature = col, verbose=-1, n_jobs=1)))
    
    for name, model in models:
        try:
            print(f"\nTraining {name}...")
            oof = np.zeros(len(X))
            pred = np.zeros(len(test))
            
            for fold, (trn_idx, val_idx) in enumerate(FOLDS.split(X, y_transformed)):
                X_train, y_train = X.iloc[trn_idx], y_transformed.iloc[trn_idx]
                X_val, y_val = X.iloc[val_idx], y_transformed.iloc[val_idx]
                
                model.fit(X_train, y_train)
                oof[val_idx] = model.predict(X_val)
                pred += model.predict(test) / FOLDS.n_splits
                
                if use_log_transform:
                    fold_rmsle = rmsle(np.expm1(y_val), np.expm1(oof[val_idx]))
                else:
                    fold_rmsle = rmsle(y_val, oof[val_idx])
                print(f'{name} - Fold {fold} RMSLE: {fold_rmsle:.4f}')
            
            all_oof[name] = oof
            all_predictions[name] = pred
            
            if use_log_transform:
                full_rmsle = rmsle(y, np.expm1(oof))
            else:
                full_rmsle = rmsle(y, oof)
            print(f'{name} - Full OOF RMSLE: {full_rmsle:.4f}')
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    oof_df = pd.DataFrame(all_oof)
    predictions_df = pd.DataFrame(all_predictions)
    
    if use_log_transform:
        oof_df['target'] = y.values
        oof_df['target_transformed'] = y_transformed.values
    else:
        oof_df['target'] = y.values
    
    model_info = {
        'model_names': [name for name, _ in models],
        'num_models': len(models),
        'features_used': list(X.columns),
        'used_log_transform': use_log_transform
    }
    
    if use_log_transform:
        for col in predictions_df.columns:
            predictions_df[col] = np.expm1(predictions_df[col])
    
    return oof_df, predictions_df, model_info


oof_results, test_predictions, model_info = create_ensemble(X, y, test)
    
oof_results.to_csv('oof_predictions.csv', index=False)
test_predictions.to_csv('test_predictions.csv', index=False)

print("\nModeling completed successfully!")
print(f"Trained {model_info['num_models']} models")
print("OOF predictions shape:", oof_results.shape)
print("Test predictions shape:", test_predictions.shape)



model_columns = [col for col in oof_results.columns if col not in ['target', 'target_transformed']]
model_scores = {name: rmsle(oof_results['target'], oof_results[name]) 
               for name in model_columns}

initial_weights = np.array([1/score for score in model_scores.values()])
initial_weights /= initial_weights.sum()

def objective(weights, alpha=0.01):
    combined = sum(w*oof_results[model] for w, model in zip(weights, model_scores.keys()))
    rmsle_val = rmsle(oof_results['target'], combined)
    penalty = alpha * np.sum(weights**2)  # L2 регуляризация
    return rmsle_val + penalty

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0,1)] * len(model_scores)

result = minimize(
    objective,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}
)

if not result.success:
    print("Optimization warning:", result.message)
    optimal_weights = initial_weights
else:
    optimal_weights = result.x

print("\nOptimized weights:")
for name, w in zip(model_scores.keys(), optimal_weights):
    print(f"{name}: {w:.4f} (RMSLE: {model_scores[name]:.4f})")

optimal_pred = np.clip(sum(w*test_predictions[model] for w, model in zip(optimal_weights, model_scores.keys())), 1, 314)


sample = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sample['Calories'] = optimal_pred
sample.to_csv('submission.csv', index=False)
sample.head(10)

