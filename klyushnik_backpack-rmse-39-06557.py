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

from sklearn.linear_model import (SGDOneClassSVM, LinearRegression, Ridge, 
                                 Lasso, ElasticNet)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                            mean_absolute_percentage_error)
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
                                    OneHotEncoder,FunctionTransformer, KBinsDiscretizer)
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
from optuna.pruners import MedianPruner
import optuna.visualization as vis
from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor
# from category_encoders import TargetEncoder
from cuml.preprocessing import TargetEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
%matplotlib inline

sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")


def plot_numerical_features(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.histplot(df[feature], bins=30, kde=True, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {feature}', fontsize=16)
        axes[i].set_xlabel(feature, fontsize=14)
        axes[i].set_ylabel('Frequency', fontsize=14)
    
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
        axes[i].set_title(f'Boxplot of {feature}', fontsize=16)
        axes[i].set_xlabel(feature, fontsize=14)
    
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
        axes[i].set_title(f'QQ Plot of {feature}', fontsize=16)
        axes[i].set_xlabel('Theoretical Quantiles', fontsize=14)
        axes[i].set_ylabel('Sample Quantiles', fontsize=14)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, method='spearman'):
    num_df = df.select_dtypes(include=[np.number])
    
    corr = num_df.corr(method=method)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, linewidths=.5)
    plt.title(f'Correlation Matrix ({method.capitalize()} Correlation)', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def plot_pairplot(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    sns.pairplot(df[num_features], diag_kind='kde', plot_kws={'alpha': 0.6, 'edgecolor': 'k'}, height=2.5)
    plt.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=18)
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
        
        axes[i].set_title(f'Count of {feature}', fontsize=16)
        axes[i].set_xlabel('Count', fontsize=14)
        axes[i].set_ylabel(feature, fontsize=14)
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


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
original_data = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')

train.shape, test.shape, original_data.shape


original_data = original_data.dropna()
train = pd.concat([train, original_data], axis=0).reset_index(drop=True)

train.shape, test.shape


train = optimize_memory_usage(train)
test = optimize_memory_usage(test)


train.info()


train.describe().T


duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

train = train.drop_duplicates()


for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


test = test.drop(['id'], axis =1)
train = train.drop(['id'], axis =1)


plot_numerical_features(train)


plot_numerical_boxplots(train)


plot_correlation_matrix(train)


plot_pairplot(train)


plot_categorical_features(train)


dict_fen = {'Material':'NaN','Style':'NaN','Brand':'NaN','Size':'NaN','Waterproof':'NaN','Color':'NaN','Laptop Compartment':'NaN',}

def feh(df):
    
    df = df.fillna(dict_fen)

    map_size       = {'Small':    1.1,'Medium':  1.2,'Large':1.3,                                    'NaN':0}
    map_brand      = {'Jansport': 1.1,'Adidas':  1.2,'Nike': 1.3,'Puma':  1.4,'Under Armour':    1.5,'NaN':0}
    map_color      = {'Black':    1.1,'Green':   1.2,'Red':  1.3,'Blue':  1.4,'Gray':1.05,'Pink':1.5,'NaN':0}
    map_style      = {'Messenger':1.1,'Backpack':1.2,'Tote': 1.3,                                    'NaN':0}
    map_material   = {'Polyester':1.1,'Leather': 1.2,'Nylon':1.3,'Canvas':1.4,                       'NaN':0}
    map_waterproof = {'Yes':      1.1,'No':      1.0,                                                'NaN':0}
    map_laptop     = {'Yes':      1.1,'No':      1.0,                                                'NaN':0}
    
    df['Size_map']        = df['Size']              .map(map_size)
    df['Brand_map']       = df['Brand']             .map(map_brand)
    df['Color_map']       = df['Color']             .map(map_color)
    df['Style_map']       = df['Style']             .map(map_style)
    df['Material_map']    = df['Material']          .map(map_material)
    df['Waterproof_map']  = df['Waterproof']        .map(map_waterproof)
    df['Laptop_map']      = df['Laptop Compartment'].map(map_laptop)
    df['Compartments_map']= df['Compartments']      .apply(lambda x: x/1.1)
    
    df['_NaN_Material']   = df['Material']  .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Style']      = df['Style']     .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Brand']      = df['Brand']     .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Size']       = df['Size']      .apply(lambda x: 1 if x == 'NaN' else 0)                                    
    df['_NaN_Waterproof'] = df['Waterproof'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Color']      = df['Color']     .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Laptop']     = df['Laptop Compartment'].apply(lambda x: 1 if x == 'NaN' else 0)
    
    df['_7_NaNs'] = df['_NaN_Waterproof']+df['_NaN_Material']+df['_NaN_Laptop']+df['_NaN_Style']+df['_NaN_Brand']+df['_NaN_Size']+df['_NaN_Color']

    df = df.rename(columns={ 'Size_map':'x1', 'Brand_map':'x2', 'Color_map':'x3', 'Style_map':'x4', 'Material_map':'x5', 'Waterproof_map':'x6', 'Laptop_map':'x7', 'Compartments_map':'x8' } ) 

# feature construction inspired from @khsamaha notebook
    median_weight = df["Weight Capacity (kg)"].median()
    df["Weight Capacity (kg)"] = (
        df["Weight Capacity (kg)"].fillna(median_weight)
    )
    
    conditions = [
        (df["Weight Capacity (kg)"] <= 5),
        (df["Weight Capacity (kg)"]  > 5) & (df["Weight Capacity (kg)"] <= 15),
        (df["Weight Capacity (kg)"]  > 15) & (df["Weight Capacity (kg)"] <= 20),
        (df["Weight Capacity (kg)"]  > 20) & (df["Weight Capacity (kg)"] <= 25),
        (df["Weight Capacity (kg)"] > 25)
    ]
    choices = ['Light', 'Middle', 'Light_heavy', 'Middel_heavy','Heavy']
    df['Weight_Class'] = np.select(conditions, choices, default='')
    
    df["Weight Capacity (kg)"] = df["Weight Capacity (kg)"].astype("float64")
    df['Weight_Class'] = df['Weight_Class'].astype("category")

    return df



train = feh(train)
test = feh(test)

train.shape, test.shape


Compartments_bins = [-float('inf'), train['Compartments'].quantile(0.25), train['Compartments'].quantile(0.75), float('inf')]
Compartments_labels = [1, 2, 3]  
train['Compartments_group'] = pd.cut(train['Compartments'], bins=Compartments_bins, labels=Compartments_labels)

Weight_bins = [-float('inf'), train['Weight Capacity (kg)'].quantile(0.25), train['Weight Capacity (kg)'].quantile(0.75), float('inf')]
Weight_labels = [1, 2, 3]  
train['Weight Capacity (kg)_group'] = pd.cut(train['Weight Capacity (kg)'], bins=Weight_bins, labels=Weight_labels)

Compartments_bins = [-float('inf'), test['Compartments'].quantile(0.25), test['Compartments'].quantile(0.75), float('inf')]
Compartments_labels = [1, 2, 3]  
test['Compartments_group'] = pd.cut(test['Compartments'], bins=Compartments_bins, labels=Compartments_labels)

Weight_bins = [-float('inf'), test['Weight Capacity (kg)'].quantile(0.25), test['Weight Capacity (kg)'].quantile(0.75), float('inf')]
Weight_labels = [1, 2, 3]  
test['Weight Capacity (kg)_group'] = pd.cut(test['Weight Capacity (kg)'], bins=Weight_bins, labels=Weight_labels)

train.shape, test.shape


# num_f =  ['Compartments', 'Weight Capacity (kg)', 'x1', 'x2', 'x3', 'x4',
#        'x5', 'x6', 'x7', 'x8', '_NaN_Material', '_NaN_Style', '_NaN_Brand',
#        '_NaN_Size', '_NaN_Waterproof', '_NaN_Color', '_NaN_Laptop', '_7_NaNs']


# tmp = train[num_f].copy()
# tmp_t = test[num_f].copy()

# power_transformer = PowerTransformer(method='yeo-johnson')
# quantile_transformer = QuantileTransformer(n_quantiles=10, random_state=0)

# for i in num_f:
#     tmp[i + '+log'] = np.log(tmp[i] + 1)
#     tmp_t[i + '+log'] = np.log(tmp_t[i] + 1)
    
#     tmp[i + '+log1'] = np.log1p(tmp[i])
#     tmp_t[i + '+log1'] = np.log1p(tmp_t[i])
    
#     tmp[i + '+sqrt'] = np.sqrt(tmp[i] + 1)
#     tmp_t[i + '+sqrt'] = np.sqrt(tmp_t[i])

# new_cols = [col for col in tmp.columns if col.endswith(('+log', '+log1', '+sqrt'))]
# tmp[new_cols] = power_transformer.fit_transform(tmp[new_cols])
# tmp_t[new_cols] = power_transformer.transform(tmp_t[new_cols])

# tmp[new_cols] = quantile_transformer.fit_transform(tmp[new_cols])
# tmp_t[new_cols] = quantile_transformer.transform(tmp_t[new_cols])

# print(f"Transformed Train shape: {tmp.shape}, Transformed Test shape: {tmp_t.shape}")


# pca = PCA(n_components=1)
# train['pca_1'] = pca.fit_transform(tmp)
# test['pca_1'] = pca.transform(tmp_t)

# train.shape, test.shape


num_f = ['Compartments', 'Weight Capacity (kg)', 'x1', 'x2', 'x3', 'x4',
         'x5', 'x6', 'x7', 'x8', '_NaN_Material', '_NaN_Style', '_NaN_Brand',
         '_NaN_Size', '_NaN_Waterproof', '_NaN_Color', '_NaN_Laptop', '_7_NaNs',
         'Compartments_group', 'Weight Capacity (kg)_group', ]#'pca_1'
ohe_f = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Weight_Class']
ord_f = ['Waterproof', 'Laptop Compartment']

ohe_pipe = Pipeline([
    ('imputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
    ('imputer_ohe_after', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
])

ord_pipe = Pipeline([
    ('imputer_before', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('simpleImputer_after', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
])

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    [
        ('ohe', ohe_pipe, ohe_f),
        ('ord', ord_pipe, ord_f),
        ('num', num_pipe, num_f),
    ],
    remainder='passthrough'
)

print(preprocessor)


# cat_features = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Weight_Class',
#                 'Waterproof', 'Laptop Compartment', 'Compartments_group', 'Weight Capacity (kg)_group']

# TE = TargetEncoder(cols=cat_features)

# train[cat_features] = TE.fit_transform(train[cat_features], train['Price'])
# test[cat_features] = TE.transform(test[cat_features])


X = train.drop(columns=['Price'])
y = train['Price']

X_transformed = preprocessor.fit_transform(X, y)
test_transformed = preprocessor.transform(test)

X = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
test = pd.DataFrame(test_transformed, columns=preprocessor.get_feature_names_out())

X = variance_threshold(X,0.07)
list_name = (X.columns)
test = test[list_name]

TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

features = test.columns.tolist()

for col in features:
    TE.fit(X[col], y)
    X[col] = TE.transform(X[col])
    test[col] = TE.transform(test[col])

X.shape, y.shape, test.shape


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def objective(trial):
    depth = trial.suggest_int('depth', 1, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    iterations = trial.suggest_int('iterations', 100, 2000)
    l2_leaf_reg = trial.suggest_int('l2_leaf_reg', 1, 10)
    bagging_temperature = trial.suggest_uniform('bagging_temperature', 0, 1)
    border_count = trial.suggest_int('border_count', 1, 255)
    random_strength = trial.suggest_int('random_strength', 1, 10)
    early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 10, 50)

    model = CatBoostRegressor(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        bagging_temperature=bagging_temperature,
        border_count=border_count,
        random_strength=random_strength,
        early_stopping_rounds=early_stopping_rounds,
        silent=True
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        preds = model.predict(X_test)
        score = rmse(y_test, preds)
        scores.append(score)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=20)

cat_param = study.best_params

print("Best parameters found: ", cat_param)


def objective(trial):
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e2),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e2),
    }

    model = xgb.XGBRegressor(**params, silent=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        score = rmse(y_test, preds)
        scores.append(score)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=20)

xgb_param = study.best_params

print("Best parameters found: ", xgb_param)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', -1, 10),  
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e2),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e2),
    }

    model = LGBMRegressor(**params, verbose=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = model.predict(X_test)
        score = rmse(y_test, preds)
        scores.append(score)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=20)

lgb_param = study.best_params

print("Best parameters found: ", lgb_param)


xgb_p =  {'n_estimators': 1500, 'max_depth': 3, 'learning_rate': 0.017361813430665254, 'subsample': 0.8887776309294646,
          'colsample_bytree': 0.5612498118208982,'gamma': 0.6648708112620045, 'reg_alpha': 13.31662979961265,
          'reg_lambda': 0.30189200565420315}

lgb_p = {'n_estimators': 1500, 'max_depth': 3, 'learning_rate': 0.0478290225344182,
         'num_leaves': 59, 'min_child_samples': 28, 'subsample': 0.514524340076117,
         'colsample_bytree': 0.6145309145618922, 'reg_alpha': 0.00012178647606478384,
         'reg_lambda': 0.006638026244134063}

hgb_param = {'max_iter': 1479, 'max_depth': 4, 'learning_rate': 0.07316053785101517,
             'l2_regularization': 45.367042834411535, 'max_leaf_nodes': 40, 'min_samples_leaf': 57}

lgb_p1 =  {'n_estimators': 1934, 'max_depth': 1, 'learning_rate': 0.09591525064728601, 'num_leaves': 57,
           'min_child_samples': 47, 'subsample': 0.7890685769651394, 'colsample_bytree': 0.8086027949437358, 
           'reg_alpha': 20.024597361614145, 'reg_lambda': 14.28535930490372}

cat_p = {'depth': 7, 'learning_rate': 0.08795867332862672, 'iterations': 1500, 
         'l2_leaf_reg': 2, 'bagging_temperature': 0.13126062975562605, 'border_count': 85, 
         'random_strength': 7, 'early_stopping_rounds': 40}

cat_p1 = {'depth': 3, 'learning_rate': 0.09122054252630714, 'iterations': 682, 
          'l2_leaf_reg': 10, 'bagging_temperature': 0.5999535929973101, 'border_count': 250, 
          'random_strength': 7, 'early_stopping_rounds': 50}

xgb_p1 = {'n_estimators': 1297, 'max_depth': 2, 'learning_rate': 0.025372760278211344, 'subsample': 0.6311619395764025,
          'colsample_bytree': 0.5076445728470458, 'gamma': 4.137185146616119, 'reg_alpha': 0.0013306201153777924,
          'reg_lambda': 0.009333744281133519}

cat_p2 = {'depth': 3, 'learning_rate': 0.02902722386037917, 'iterations': 1961, 'l2_leaf_reg': 6,
          'bagging_temperature': 0.7046759124272901, 'border_count': 196, 'random_strength': 4, 
          'early_stopping_rounds': 43}


fold = 5
FOLDs = KFold(n_splits=fold, shuffle=True)

oof_cat, predictions_cat = np.zeros(len(X)), np.zeros(len(test))
oof_xgb, predictions_xgb = np.zeros(len(X)), np.zeros(len(test))
oof_lgb, predictions_lgb = np.zeros(len(X)), np.zeros(len(test))
oof_hgb, predictions_hgb = np.zeros(len(X)), np.zeros(len(test))
oof_ridge, predictions_ridge = np.zeros(len(X)), np.zeros(len(test)), 
oof_rf, predictions_rf = np.zeros(len(X)), np.zeros(len(test))
oof_lr,  predictions_lr  = np.zeros(len(X)), np.zeros(len(test))
oof_xgb1, predictions_xgb1 = np.zeros(len(X)), np.zeros(len(test))
oof_lgb1, predictions_lgb1 = np.zeros(len(X)), np.zeros(len(test))
oof_lgb2, predictions_lgb2 = np.zeros(len(X)), np.zeros(len(test))
oof_cat1, predictions_cat1 = np.zeros(len(X)), np.zeros(len(test))
oof_cat2, predictions_cat2 = np.zeros(len(X)), np.zeros(len(test))
oof_xgb2, predictions_xgb2 = np.zeros(len(X)), np.zeros(len(test))
oof_cat3, predictions_cat3 = np.zeros(len(X)), np.zeros(len(test))



for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(X, y)):
    X.iloc[trn_idx], y.iloc[trn_idx]
    X.iloc[val_idx], y.iloc[val_idx]

    # CatBoostRegressor
    cat_model = CatBoostRegressor(**cat_param, 
                                  random_state = 123,
                                  verbose=0)
    cat_model.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_cat[val_idx] = cat_model.predict(X.iloc[val_idx])
    predictions_cat += cat_model.predict(test) / FOLDs.n_splits
    cat_score = mean_squared_error(y.iloc[val_idx], oof_cat[val_idx], squared=False)
    print('Fold', fold_, ' CatBoostRegressor oof RMSE is ---', cat_score)

    # XGBRegressor
    xgb_model = xgb.XGBRegressor(**xgb_param,
                                random_state = 123)
    xgb_model.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_xgb[val_idx] = xgb_model.predict(X.iloc[val_idx])
    predictions_xgb += xgb_model.predict(test)/ FOLDs.n_splits
    xgb_score = mean_squared_error(y.iloc[val_idx], oof_xgb[val_idx], squared=False)
    print('Fold', fold_, ' XGBRegressor oof RMSE is ---', xgb_score)

    # LGBMRegressor
    lgb_model = LGBMRegressor(**lgb_param,
                              random_state = 123,
                              verbose=-1)
    lgb_model.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_lgb[val_idx] = lgb_model.predict(X.iloc[val_idx])
    predictions_lgb += lgb_model.predict(test) / FOLDs.n_splits
    lgb_score = mean_squared_error(y.iloc[val_idx], oof_lgb[val_idx], squared=False)
    print('Fold', fold_, ' LGBMRegressor oof RMSE is ---', lgb_score)

    # HistGradientBoostingRegressor
    hgb_model = HistGradientBoostingRegressor(**hgb_param,
                                              random_state=123)
    hgb_model.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_hgb[val_idx] = hgb_model.predict(X.iloc[val_idx])
    predictions_hgb += hgb_model.predict(test) / FOLDs.n_splits
    hgb_score = mean_squared_error(y.iloc[val_idx], oof_hgb[val_idx], squared=False)
    print('Fold', fold_, ' HistGradientBoostingRegressor oof RMSE is ---', hgb_score)

    # Ridge
    ridge_model = Ridge(alpha=1.0, fit_intercept=True, solver='auto')
    ridge_model.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_ridge[val_idx] = ridge_model.predict(X.iloc[val_idx])
    predictions_ridge += ridge_model.predict(test) / FOLDs.n_splits
    ridge_score = mean_squared_error(y.iloc[val_idx], oof_ridge[val_idx], squared=False)
    print('Fold', fold_, ' Ridge oof RMSE is ---', ridge_score)

    # RandomForestRegressor
    rf_model = RandomForestRegressor(   n_estimators=500,        
                                        max_depth=5,             
                                        min_samples_leaf=20,     
                                        random_state=123
                                    )
    rf_model.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_rf[val_idx] = rf_model.predict(X.iloc[val_idx])
    predictions_rf += rf_model.predict(test) / FOLDs.n_splits  
    rf_score = mean_squared_error(y.iloc[val_idx], oof_rf[val_idx], squared=False)
    print('Fold', fold_, ' RandomForestRegressor oof RMSE is ---', rf_score)

    # XGBRegressor1
    xgb_model1 = xgb.XGBRegressor(**xgb_p,
                                random_state = 123)
    xgb_model1.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_xgb1[val_idx] = xgb_model1.predict(X.iloc[val_idx])
    predictions_xgb1 += xgb_model1.predict(test)/ FOLDs.n_splits
    xgb_score1 = mean_squared_error(y.iloc[val_idx], oof_xgb1[val_idx], squared=False)
    print('Fold', fold_, ' XGBRegressor 1 oof RMSE is ---', xgb_score1)

    # LGBMRegressor1
    lgb_model1 = LGBMRegressor(**lgb_p,
                              random_state = 123,
                              verbose=-1)
    lgb_model1.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_lgb1[val_idx] = lgb_model1.predict(X.iloc[val_idx])
    predictions_lgb1 += lgb_model1.predict(test) / FOLDs.n_splits
    lgb_score1 = mean_squared_error(y.iloc[val_idx], oof_lgb1[val_idx], squared=False)
    print('Fold', fold_, ' LGBMRegressor 1 oof RMSE is ---', lgb_score1)

    # LGBMRegressor2
    lgb_model2 = LGBMRegressor(**lgb_p1,
                              random_state = 123,
                              verbose=-1)
    lgb_model2.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_lgb2[val_idx] = lgb_model2.predict(X.iloc[val_idx])
    predictions_lgb2 += lgb_model2.predict(test) / FOLDs.n_splits
    lgb_score2 = mean_squared_error(y.iloc[val_idx], oof_lgb2[val_idx], squared=False)
    print('Fold', fold_, ' LGBMRegressor 2 oof RMSE is ---', lgb_score2)

    # CatBoostRegressor1
    cat_model1 = CatBoostRegressor(**cat_p, 
                                  random_state = 123,
                                  verbose=0)
    cat_model1.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_cat1[val_idx] = cat_model1.predict(X.iloc[val_idx])
    predictions_cat1 += cat_model1.predict(test) / FOLDs.n_splits
    cat_score1 = mean_squared_error(y.iloc[val_idx], oof_cat1[val_idx], squared=False)
    print('Fold', fold_, ' CatBoostRegressor 1 oof RMSE is ---', cat_score1)

    # CatBoostRegressor2
    cat_model2 = CatBoostRegressor(**cat_p1, 
                                  random_state = 123,
                                  verbose=0)
    cat_model2.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_cat2[val_idx] = cat_model2.predict(X.iloc[val_idx])
    predictions_cat2 += cat_model2.predict(test) / FOLDs.n_splits
    cat_score2 = mean_squared_error(y.iloc[val_idx], oof_cat2[val_idx], squared=False)
    print('Fold', fold_, ' CatBoostRegressor 2 oof RMSE is ---', cat_score2)

    # XGBRegressor2
    xgb_model2 = xgb.XGBRegressor(**xgb_p1,
                                random_state = 123)
    xgb_model2.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_xgb2[val_idx] = xgb_model2.predict(X.iloc[val_idx])
    predictions_xgb2 += xgb_model2.predict(test)/ FOLDs.n_splits
    xgb_score2 = mean_squared_error(y.iloc[val_idx], oof_xgb2[val_idx], squared=False)
    print('Fold', fold_, ' XGBRegressor 2 oof RMSE is ---', xgb_score2)

    # CatBoostRegressor3
    cat_model3 = CatBoostRegressor(**cat_p2, 
                                  random_state = 123,
                                  verbose=0)
    cat_model3.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    oof_cat3[val_idx] = cat_model3.predict(X.iloc[val_idx])
    predictions_cat3 += cat_model3.predict(test) / FOLDs.n_splits
    cat_score3 = mean_squared_error(y.iloc[val_idx], oof_cat3[val_idx], squared=False)
    print('Fold', fold_, ' CatBoostRegressor 3 oof RMSE is ---', cat_score3)


blend_df = pd.DataFrame({'1': oof_cat,
                         '2': oof_xgb,
                         '3': oof_lgb,
                         '4': oof_hgb,
                         '5': oof_ridge,
                         '6': oof_rf,
                         '7': oof_xgb1,
                         '8': oof_lgb1,
                         '9': oof_lgb2,
                         '10': oof_cat1,
                         '11': oof_cat2,
                         '12': oof_xgb2,
                         '13': oof_cat3
                         })

blend_test_df = pd.DataFrame({  '1': predictions_cat,  
                                '2': predictions_xgb, 
                                '3': predictions_lgb, 
                                '4': predictions_hgb, 
                                '5': predictions_ridge, 
                                '6': predictions_rf,
                                '7': predictions_xgb1,
                                '8': predictions_lgb1,
                                '9': predictions_lgb2,
                                '10': predictions_cat1,
                                '11': predictions_cat2,
                                '12': predictions_xgb2,
                                '13': predictions_cat3
                        })

def calculate_rmse(weights, blend_df, y_):
    weighted_predictions = np.dot(blend_df, weights)
    return np.sqrt(mean_squared_error(y, weighted_predictions))

def constraint(weights):
    return np.sum(weights) - 1 

initial_weights = np.array([0.2] * blend_df.shape[1])  

constraints = {'type': 'eq', 'fun': constraint}
bounds = [(0, 1) for _ in range(blend_df.shape[1])]  

result = minimize(calculate_rmse, initial_weights, args=(blend_df, y), 
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
optimal_rmse = result.fun

print(f"Optimal weights: {optimal_weights}")
print(f"Best RMSE: {optimal_rmse:.4f}")


sample = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sample['Price'] = np.dot(blend_test_df, optimal_weights)
sample.to_csv('submission.csv', index=False)
sample.shape

