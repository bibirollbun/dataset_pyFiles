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


!pip install -q scikit-learn imbalanced-learn
!pip install -q scikeras
!pip install -q sdv


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mstats
from scipy.stats.mstats import winsorize
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, hstack
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import warnings

from imblearn.over_sampling import (
ADASYN, 
SMOTE, 
KMeansSMOTE)

from sklearn import preprocessing
from sklearn.decomposition import PCA
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
    chi2,
    SequentialFeatureSelector,
)
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    train_test_split,
    cross_validate,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
    classification_report
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    Activation,
)
from keras import backend as K
import keras_tuner
from keras_tuner import RandomSearch, Hyperband

from pathlib import Path
import logging
from functools import partial

import optuna
from optuna.samplers import CmaEsSampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis

from catboost import CatBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from category_encoders import TargetEncoder, MEstimateEncoder

import requests
import holidays
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import warnings
import re
import time
import logging
from functools import partial
from itertools import combinations
from IPython.display import Image

import logging

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


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


display(train.shape, test.shape)

display(train.info())
display(test.info())

test = test.drop(['id'], axis =1)
train = train.drop(['id'], axis =1)

display(train.head(5))
display(test.head(5))

display(train.describe().T)

duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

duplicates = test.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

for col in test.columns:
    pct_missing = np.mean(test[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

train = optimize_memory_usage(train)
test = optimize_memory_usage(test)

train_df = train.copy()
test_df = test.copy()


from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

metadata = Metadata.detect_from_dataframe(
    data=train,
    table_name='introvert')


synthesizer = CTGANSynthesizer(metadata=metadata)

synthesizer.fit(train)

synthetic_data = synthesizer.sample(num_rows=5000)

train = pd.concat([train, synthetic_data])

print(train.shape)


duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

train = train.drop_duplicates()

duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")


plot_numerical_features(train)


plot_numerical_boxplots(train)


plot_qq_plot(train)


plot_correlation_matrix(train)


plot_categorical_features(train)


num_cols = test.select_dtypes(include=['number']).columns
cat_cols = test.select_dtypes(include=['object', 'category']).columns


num_imputer = IterativeImputer(random_state=42)
train[num_cols] = num_imputer.fit_transform(train[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])

for col in cat_cols:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col] = test[col].fillna(mode_val)

train.shape, test.shape


for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

duplicates = test.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

for col in test.columns:
    pct_missing = np.mean(test[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


categorize_variable(train, 'Time_spent_Alone', ["min", "mean", "max"])
categorize_variable(test, 'Time_spent_Alone', ["min", "mean", "max"])
categorize_variable(train, 'Social_event_attendance', ["low", "medium", "high"])
categorize_variable(test, 'Social_event_attendance', ["low", "medium", "high"])
categorize_variable(train, 'Going_outside', ["No", "Maybe", "Yes"])
categorize_variable(test, 'Going_outside', ["No", "Maybe", "Yes"])
categorize_variable(train, 'Friends_circle_size', ["low", "medium", "high"])
categorize_variable(test, 'Friends_circle_size', ["low", "medium", "high"])
categorize_variable(train, 'Post_frequency', ["low", "medium", "high"])
categorize_variable(test, 'Post_frequency', ["low", "medium", "high"])

train.shape, test.shape


def create_features(df, train_stats=None):
        if train_stats is None:
            train_stats = {
                'post_median': df['Post_frequency'].median()
            }
        
        df['Social_Activity_Index'] = df['Social_event_attendance'] * df['Friends_circle_size']
        df['Isolation_Score'] = df['Time_spent_Alone'] / (df['Going_outside'] + 1)
        df['Social_Balance'] = df['Social_event_attendance'] - df['Time_spent_Alone']
        df['Log_Friends'] = np.log1p(df['Friends_circle_size'])
        
        if 'Post_frequency' in df.columns:
            df['High_Posting'] = (df['Post_frequency'] > train_stats['post_median']).astype(int)
        
        return df

create_features(train)
create_features(test)

train.shape, test.shape


col = ['Stage_fear', 'Drained_after_socializing']

for c in col:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')

le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])

X = train.drop(columns=['Personality'])
y = train['Personality']

display(X.shape, y.shape, test.shape)

numeric_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(exclude=['number']).columns

num_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
test[numeric_cols] = num_imputer.transform(test[numeric_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
test[categorical_cols] = cat_imputer.transform(test[categorical_cols])

display(X.info(), test.info())

cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

encoder = OneHotEncoder(drop='first', sparse_output=False)  

X_encoded = encoder.fit_transform(X[cat_cols])
X_encoded = pd.DataFrame(
    X_encoded,
    columns=encoder.get_feature_names_out(cat_cols)
)
X = pd.concat([
    X.drop(columns=cat_cols).reset_index(drop=True),  
    X_encoded                  
], axis=1)

test_encoded = encoder.fit_transform(test[cat_cols])
test_encoded = pd.DataFrame(
    test_encoded,
    columns=encoder.get_feature_names_out(cat_cols)
)
test = pd.concat([
    test.drop(columns=cat_cols),  
    test_encoded                  
], axis=1)

display(X.info(), test.info())

smote = KMeansSMOTE(
    kmeans_estimator=MiniBatchKMeans(n_init=3, random_state=0), random_state=42
)
X, y = smote.fit_resample(X, y)
display(X.shape, y.shape, test.shape)

X_pol = PolynomialFeatures_labeled(X, 2)
test_pol = PolynomialFeatures_labeled(test, 2)

n_components = 2

pca = PCA(n_components=n_components)

pca_components = pca.fit_transform(X_pol)
pca_components_test = pca.transform(test_pol)

pca_df = pd.DataFrame(pca_components, columns=[f'PCA_{i+1}' for i in range(n_components)])
X = pd.concat([X, pca_df], axis=1)

pca_df_test = pd.DataFrame(pca_components_test, columns=[f'PCA_{i+1}' for i in range(n_components)])
test = pd.concat([test, pca_df_test], axis=1)

X = variance_threshold(X,0.02)
list_name = (X.columns)
test = test[list_name]

display(X.shape, y.shape, test.shape)

scaler = RobustScaler()

X[X.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(X[X.select_dtypes(include=[np.number]).columns])
test[X.select_dtypes(include=[np.number]).columns] = scaler.transform(test[X.select_dtypes(include=[np.number]).columns])


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_model(X, y, model_type='xgb', n_trials=50):
    
    def objective(trial):
        try:
            if model_type == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'use_label_encoder': False,
                    'eval_metric': 'logloss',
                    'random_state': 42
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_type == 'lgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                }
                model = LGBMClassifier(**params, verbose=-1,)
                
            elif model_type == 'cat':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 3000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    'od_type': 'Iter',
                    'od_wait': 100,
                    'random_state': 42,
                }
                model = CatBoostClassifier(**params, verbose=0)
                
            elif model_type == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                    'max_depth': trial.suggest_categorical('max_depth', [None, *range(3, 16)]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                    'random_state': 42,
                    'n_jobs': -1  
                }
                model = RandomForestClassifier(**params)
            
            score = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()
            return score
        
        except Exception as e:
            logger.error(f"Error during trial: {str(e)}")
            return float('-inf')  
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best {model_type} params: {study.best_params}")
    logger.info(f"Best {model_type} accuracy: {study.best_value:.4f}")
    
    return study.best_params

models_params = {
    'xgb': optimize_model(X, y, 'xgb', n_trials=30),
    'lgb': optimize_model(X, y, 'lgb', n_trials=30),
    'cat': optimize_model(X, y, 'cat', n_trials=30),
    'rf': optimize_model(X, y, 'rf', n_trials=30)
}


def create_ensemble_binary(X, y, test, n_folds=5,
                         catboost_params=None, xgb_params=None, lgbm_params=None, 
                         rf_params=None): 
    """
    Create ensemble for binary classification with standard metrics.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Training features
    y : pd.Series (binary: 0/1)
        Training target
    test : pd.DataFrame
        Test features
    n_folds : int
        Number of cross-validation folds
    catboost_params : list of dict
        Parameters for CatBoost models
    xgb_params : list of dict
        Parameters for XGBoost models
    lgbm_params : list of dict
        Parameters for LightGBM models
    rf_params : list of dict  # Ğ�Ğ¾Ğ²Ñ‹Ğ¹ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€
        Parameters for RandomForest models
        
    Returns:
    --------
    tuple: (all_oof, all_test_preds, model_info)
    """
    FOLDS = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
            test[col] = test[col].astype('category')
    
    # Default parameters for binary classification
    if catboost_params is None:
        catboost_params = [{
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'border_count': 64,
            'verbose': 0,
            'loss_function': 'Logloss',  
            'random_state': 42
        }]
    
    if xgb_params is None:
        xgb_params = [{
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'min_child_weight': 3,
            'objective': 'binary:logistic',  
            'random_state': 42,
            'enable_categorical': True
        }]
    
    if lgbm_params is None:
        lgbm_params = [{
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': -1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary',  
            'random_state': 42
        }]
    
    if rf_params is None:
        rf_params = [{
            'n_estimators': 500,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': 42
        }]
    
    all_oof = {}
    all_test_preds = {}
    models_info = []
    
    # Create model instances
    models = []
    
    # CatBoost
    for i, params in enumerate(catboost_params, 1):
        model_name = f'cat_{i}'
        models.append((model_name, CatBoostClassifier(
            **params, verbose=0,
            cat_features=cat_cols if 'cat_features' in catboost_params[0] else None
        )))
        models_info.append({
            'name': model_name,
            'type': 'catboost',
            'params': params
        })
    
    # XGBoost
    for i, params in enumerate(xgb_params, 1):
        model_name = f'xgb_{i}'
        models.append((model_name, xgb.XGBClassifier(
            **params,
        )))
        models_info.append({
            'name': model_name,
            'type': 'xgboost',
            'params': params
        })
    
    # LightGBM
    for i, params in enumerate(lgbm_params, 1):
        model_name = f'lgb_{i}'
        models.append((model_name, LGBMClassifier(
            **params,verbose=-1,
            categorical_feature=cat_cols if 'categorical_feature' in lgbm_params[0] else None
        )))
        models_info.append({
            'name': model_name,
            'type': 'lightgbm',
            'params': params
        })
    
    for i, params in enumerate(rf_params, 1):
        model_name = f'rf_{i}'
        models.append((model_name, RandomForestClassifier(**params)))
        models_info.append({
            'name': model_name,
            'type': 'randomforest',
            'params': params
        })
    
    # Train models
    for name, model in models:
        print(f"\nTraining {name}...")
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(test))
        
        for fold, (trn_idx, val_idx) in enumerate(FOLDS.split(X, y)):
            X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            try:
                if name.startswith('rf_'):
                    X_train_fold = X_train.copy()
                    X_val_fold = X_val.copy()
                    test_fold = test.copy()
                    
                    for col in cat_cols:
                        if col in X_train_fold.columns:
                            X_train_fold[col] = X_train_fold[col].astype('category').cat.codes
                            X_val_fold[col] = X_val_fold[col].astype('category').cat.set_categories(
                                X_train_fold[col].cat.categories
                            ).cat.codes
                            test_fold[col] = test_fold[col].astype('category').cat.set_categories(
                                X_train_fold[col].cat.categories
                            ).cat.codes
                    
                    model.fit(X_train_fold, y_train)
                    oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
                    test_preds += model.predict_proba(test_fold)[:, 1] / FOLDS.n_splits
                
                else:
                    if name.startswith('cat_'):
                        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
                    else:
                        model.fit(X_train, y_train)
                    
                    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
                    test_preds += model.predict_proba(test)[:, 1] / FOLDS.n_splits
                
                pred_labels = (oof_preds[val_idx] > 0.5).astype(int)
                acc = accuracy_score(y_val, pred_labels)
                prec = precision_score(y_val, pred_labels)
                rec = recall_score(y_val, pred_labels)
                f1 = f1_score(y_val, pred_labels)
                
                print(f"{name} - Fold {fold+1}: "
                      f"Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
            
            except Exception as e:
                print(f"Error in {name} Fold {fold+1}: {str(e)}")
                continue
        
        all_oof[name] = oof_preds
        all_test_preds[name] = test_preds
        
        final_preds = (oof_preds > 0.5).astype(int)
        print(f"\n{name} - Final OOF Metrics:")
        print(f"Accuracy: {accuracy_score(y, final_preds):.4f}")
        print(f"Precision: {precision_score(y, final_preds):.4f}")
        print(f"Recall: {recall_score(y, final_preds):.4f}")
        print(f"F1-score: {f1_score(y, final_preds):.4f}")
        print("="*50)
    
    return all_oof, all_test_preds, models_info


optimized_cat_params = [models_params['cat']]
optimized_xgb_params = [models_params['xgb']]
optimized_lgbm_params = [models_params['lgb']]
optimized_rf_params = [models_params['rf']]  
oof_results, test_predictions, models_info = create_ensemble_binary(
    X, y, test,
    catboost_params=optimized_cat_params,
    xgb_params=optimized_xgb_params,
    lgbm_params=optimized_lgbm_params,
    rf_params=optimized_rf_params  
)


oof_results


test_predictions


def stack_predictions(oof_results, test_predictions, y_true):
    X_stack = np.zeros((len(y_true), len(oof_results)))
    for i, (name, preds) in enumerate(oof_results.items()):
        if preds.ndim == 1:
            X_stack[:, i] = preds
        else:
            X_stack[:, i] = preds[:, 1]
    
    X_test_stack = np.zeros((test_predictions[list(test_predictions.keys())[0]].shape[0], len(test_predictions)))
    for i, (name, preds) in enumerate(test_predictions.items()):
        if preds.ndim == 1:
            X_test_stack[:, i] = preds
        else:
            X_test_stack[:, i] = preds[:, 1]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_stack, y_true, test_size=0.2, random_state=42, stratify=y_true
    )
    
    meta_clf = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.01,
        depth=4,
        eval_metric='Accuracy',
        verbose=0,
        early_stopping_rounds=50,
        random_state=42
    )
    
    meta_clf.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    full_train_preds = meta_clf.predict_proba(X_stack)
    test_pred_proba = meta_clf.predict_proba(X_test_stack)
    
    val_pred = meta_clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"\nMeta-model Validation Accuracy: {val_acc:.4f}")
    
    return full_train_preds, test_pred_proba, meta_clf

train_meta_preds, test_meta_preds, meta_model = stack_predictions(oof_results, test_predictions, y)

if train_meta_preds.shape[1] == 2:
    train_meta_proba = train_meta_preds[:, 1]
    test_meta_proba = test_meta_preds[:, 1]

def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0.3, 0.7, 50)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (y_pred_proba > thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

optimal_threshold = find_optimal_threshold(y, train_meta_proba)
blended_test_labels = (test_meta_proba > optimal_threshold).astype(int)
print(f"Used optimal threshold: {optimal_threshold:.4f}")


sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
sample['Personality'] = le.inverse_transform(blended_test_labels)
sample.to_csv('submission.csv', index=False)
sample.head(10)

