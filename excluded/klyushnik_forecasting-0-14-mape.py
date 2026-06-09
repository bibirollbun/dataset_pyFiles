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
import re
import time
from functools import partial
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from IPython.display import Image

from scipy.optimize import minimize
from scipy.stats import mstats
from scipy import stats

from sklearn.linear_model import (SGDOneClassSVM, LinearRegression)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, mean_absolute_error, mean_absolute_percentage_error)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import (KNNImputer, SimpleImputer)
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, IsolationForest, BaggingRegressor,
                              RandomForestRegressor)
from sklearn.model_selection import (StratifiedKFold, KFold, StratifiedGroupKFold,
                                     RepeatedStratifiedKFold, RepeatedKFold, cross_validate,
                                     train_test_split, TimeSeriesSplit)
from sklearn.preprocessing import (LabelEncoder, QuantileTransformer, StandardScaler,
                                   PowerTransformer, MaxAbsScaler, MinMaxScaler,
                                   RobustScaler, PolynomialFeatures, OrdinalEncoder, 
                                    OneHotEncoder,FunctionTransformer)
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn import preprocessing
from sklearn.feature_selection import (VarianceThreshold, SequentialFeatureSelector, f_regression)
from sklearn.compose import ColumnTransformer

import requests
import holidays

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import optuna
from optuna.samplers import CmaEsSampler
from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor

warnings.filterwarnings('ignore')
%matplotlib inline


def plot_numerical_features(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.histplot(df[feature], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_numerical_boxplots(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.boxplot(x=df[feature], ax=axes[i])
        axes[i].set_title(f'Boxplot of {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_qq_plot(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        stats.probplot(df[feature], dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ Plot of {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, method='spearman'):
    num_df = df.select_dtypes(include=[np.number])
    
    corr = num_df.corr(method=method)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(f'Correlation Matrix ({method.capitalize()} Correlation)')
    plt.show()

def plot_pairplot(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    sns.pairplot(df[num_features])
    plt.show()

def plot_categorical_features(df, ncols=2, top_n=None):
    cat_features = df.select_dtypes(include=[object]).columns
    nrows = (len(cat_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(cat_features):
        if top_n is not None:
            top_categories = df[feature].value_counts().nlargest(top_n).index
            sns.countplot(data=df[df[feature].isin(top_categories)], y=feature, ax=axes[i])  
        else:
            sns.countplot(data=df, y=feature, ax=axes[i])  
        
        axes[i].set_title(f'Count of {feature}')
        axes[i].tick_params(axis='y', rotation=0)  

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
# Function optimizes memory usage in dataframe.

# Types for optimization.
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # Memory usage size before optimize (Mb).
    before_size = df.memory_usage().sum() / 1024**2    
    for column in df.columns:
        column_type = df[column].dtypes
        if column_type in numerics:
            column_min = df[column].min()
            column_max = df[column].max()
            if str(column_type).startswith('int'):
                if column_min > np.iinfo(np.int8).min and column_max < np.iinfo(np.int8).max:
                    df[column] = df[column].astype(np.int8)
                elif column_min > np.iinfo(np.int16).min and column_max < np.iinfo(np.int16).max:
                    df[column] = df[column].astype(np.int16)
                elif column_min > np.iinfo(np.int32).min and column_max < np.iinfo(np.int32).max:
                    df[column] = df[column].astype(np.int32)
                elif column_min > np.iinfo(np.int64).min and column_max < np.iinfo(np.int64).max:
                    df[column] = df[column].astype(np.int64)  
            else:
                if column_min > np.finfo(np.float32).min and column_max < np.finfo(np.float32).max:
                    df[column] = df[column].astype(np.float32)
                else:
                    df[column] = df[column].astype(np.float64)    
    # Memory usage size after optimize (Mb).
    after_size = df.memory_usage().sum() / 1024**2
    if print_size: print('Memory usage size: before {:5.4f} Mb - after {:5.4f} Mb ({:.1f}%).'.format(before_size, after_size, 100 * (before_size - after_size) / before_size))
    return df


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

train.shape, test.shape


train.head()


train = optimize_memory_usage(train)
test = optimize_memory_usage(test)


train.info()


train.describe().T


duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")


for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


test = test.drop(['id'], axis =1)
train = train.drop(['id'], axis =1)


train = train.dropna()
test = test.dropna()

train.shape, test.shape


train['date'] = pd.to_datetime(train['date'], infer_datetime_format=True)
test['date'] = pd.to_datetime(test['date'], infer_datetime_format=True)

product_ratio = train.groupby('product')['num_sold'].mean() / train['num_sold'].mean()
store_ratio = train.groupby('store')['num_sold'].mean() / train['num_sold'].mean()
country_ratio = train.groupby('country')['num_sold'].mean() / train['num_sold'].mean()
product_ratio_mean = product_ratio.mean()
store_ratio_mean = store_ratio.mean()
country_ratio_mean = country_ratio.mean()

def date_time(df):
        df["trainweekday_sv"] = df["date"].dt.strftime("%a").astype("str")
        df["weekday_num"] = df["date"].dt.dayofweek.astype("int")  
        df["day_of_month"] = df["date"].dt.day.astype("int")  
        df["month_name_sv"] = df["date"].dt.strftime("%b").astype("str")
        df["month_num"] = df["date"].dt.month.astype("int")  
        df["year_fv"] = df["date"].dt.year.astype("int")  
        df["day_number_year"] = df["date"].dt.dayofyear.astype("int")  
        df["week_number_year"] = df["date"].dt.isocalendar().week.astype("int")  
        df["quarter"] = df["date"].dt.quarter.astype("int")  
    
        df["country"] = df["country"].astype("str")
        df["store"] = df["store"].astype("str")
        df["product"] = df["product"].astype("str")

        df['country_store'] = df['country'] + "_" + df['store']
        df['country_product'] = df['country'] + "_" + df['product']
        df['store_product'] = df['store'] + "_" + df['product']
    
        df["year_sin"] = np.sin(2 * np.pi * df["year_fv"] / 7).astype("float")
        df["year_cos"] = np.cos(2 * np.pi * df["year_fv"] / 7).astype("float")
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12).astype("float")
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12).astype("float")
        
        df["days_in_month"] = df["date"].dt.days_in_month.astype("int")  
        df["day_of_week"] = df["date"].dt.dayofweek.astype("int") 

        df['product_ratio'] = df['product'].map(product_ratio)
        df['store_ratio'] = df['store'].map(store_ratio)
        df['country_ratio'] = df['country'].map(country_ratio)
        df['product_ratio'].fillna(product_ratio_mean, inplace=True)
        df['store_ratio'].fillna(store_ratio_mean, inplace=True)
        df['country_ratio'].fillna(country_ratio_mean, inplace=True)
        
        return df

date_time(train)
date_time(test)


train = train.drop(['date'], axis=1)
test = test.drop(['date'], axis=1)
train.shape, test.shape


plot_categorical_features(train)


plot_numerical_features(train)


plot_correlation_matrix(train)


ohe_columns = ['trainweekday_sv', 'month_name_sv', 'product', 'country', 'store','country_store','country_product','store_product']
ord_columns = []
num_columns = ['weekday_num', 'day_of_month', 'month_num', 'year_fv',
               'day_number_year', 'week_number_year', 'quarter', 'year_sin',
               'year_cos', 'month_sin', 'month_cos', 'days_in_month',
               'day_of_week', 'product_ratio', 'store_ratio', 'country_ratio', ]

ohe_pipe = Pipeline(
    [
        ('imputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ]
)

ord_pipe = Pipeline(
    [
        ('imputer_before', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),  
        ('simpleImputer_after', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
    ]
)

num_pipe = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='median')),  
        # ('scaler', StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    [
        ('ohe', ohe_pipe, ohe_columns),
        ('ord', ord_pipe, ord_columns),
        ('num', num_pipe, num_columns)
    ], 
    remainder='passthrough'
) 

print(preprocessor)


X = train.drop(columns=['num_sold'])
y = train['num_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_transformed = preprocessor.fit_transform(X, y)

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
test_transformed = preprocessor.transform(test)

X = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
X_train = pd.DataFrame(X_train_transformed, columns=preprocessor.get_feature_names_out())
X_test = pd.DataFrame(X_test_transformed, columns=preprocessor.get_feature_names_out())
test = pd.DataFrame(test_transformed, columns=preprocessor.get_feature_names_out())

X = variance_threshold(X,0.04)
list_name = (X.columns)
X_train = X_train[list_name]
X_test = X_test[list_name]
test = test[list_name]

X_train.shape, X_test.shape, y_train.shape, y_test.shape, test.shape


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'depth': trial.suggest_int('depth', 2, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'random_strength': trial.suggest_loguniform('random_strength', 1e-9, 10.0),  
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),  
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 1e-2, 10.0),  
        'border_count': trial.suggest_int('border_count', 32, 255),  
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),  
        'verbose': 0,
        # 'task_type': 'GPU',  
        # 'devices': '0' 
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)  
    mape = mean_absolute_percentage_error(y_test, preds)  

    return mape

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=150)  
cat_params = study.best_params
print("Best hyperparameters: ", study.best_params)
print("Best MAPE: ", study.best_value)


def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-3, 10.0),  
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),  
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),  
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),  
        'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 0.1, 10.0),  
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),  
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),  
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.5, 1.0),  
        'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.5, 1.0),  
        'verbosity': 0,
        # 'tree_method': 'gpu_hist'  
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],  
              early_stopping_rounds=100,  
              verbose=0)

    y_pred = model.predict(X_test)  
    mape = mean_absolute_percentage_error(y_test, y_pred)  

    return mape

study_xgboost = optuna.create_study(direction='minimize')
study_xgboost.optimize(objective_xgboost, n_trials=150)

xgb_param = study_xgboost.best_params
print("Best hyperparameters for XGBoost:", study_xgboost.best_params)
print("Best MAPE score for XGBoost:", study_xgboost.best_value)


def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 256),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-3, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'objective': 'regression',
        # 'metric': 'rmse',
        # 'device': 'gpu'  # Uncomment if GPU support is needed
    }

    model = LGBMRegressor(**params, early_stopping_rounds=50, verbose=-1)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],  
              eval_metric='mape')  

    y_pred = model.predict(X_test)  
    mape = mean_absolute_percentage_error(y_test, y_pred)  

    return mape

study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=150)

lgbm_param = study_lgbm.best_params
print("Best hyperparameters for LGBM:", study_lgbm.best_params)
print("Best MAPE score for LGBM:", study_lgbm.best_value)


catboost_mape = []
xgboost_mape = []
lightgbm_mape = []


model1 = CatBoostRegressor(**cat_params, verbose=0)
model1.fit(X, y)
y_pred_cat = model1.predict(X_test)
catboost_mape.append(mean_absolute_percentage_error(y_test, y_pred_cat))

model2 = xgb.XGBRegressor(**xgb_param)
model2.fit(X, y)
y_pred_xgb = model2.predict(X_test)
xgboost_mape.append(mean_absolute_percentage_error(y_test, y_pred_xgb))

model3 = LGBMRegressor(**lgbm_param, verbose=-1)
model3.fit(X, y)
y_pred_lgbm = model3.predict(X_test)
lightgbm_mape.append(mean_absolute_percentage_error(y_test, y_pred_lgbm))

print(f'CatBoost MAPE: {np.mean(catboost_mape) * 100:.2f}%')
print(f'XGBoost MAPE: {np.mean(xgboost_mape) * 100:.2f}%')
print(f'LightGBM MAPE: {np.mean(lightgbm_mape) * 100:.2f}%')

best_model = None
best_mape = float('inf')

best_model = None
best_mape = float('inf')

if np.mean(catboost_mape) < best_mape:
    best_mape = np.mean(catboost_mape)
    best_model = 'CatBoost'
    final_model = model1 

if np.mean(xgboost_mape) < best_mape:
    best_mape = np.mean(xgboost_mape)
    best_model = 'XGBoost'
    final_model = model2  

if np.mean(lightgbm_mape) < best_mape:
    best_mape = np.mean(lightgbm_mape)
    best_model = 'LightGBM'
    final_model = model3  

print(f'Best model: {best_model} с MAPE: {best_mape * 100:.2f}%')


y_pred_final = final_model.predict(test)

sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sub['num_sold'] = y_pred_final
sub.to_csv('submission.csv', index=False)
sub.head()

