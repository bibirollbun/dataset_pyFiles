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

pd.set_option('display.max_colwidth', None)  
pd.set_option('display.max_rows', None)   
pd.set_option('display.max_columns', None)


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import seaborn as sns





df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_calories = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


# Helper function to convert all column names to lowercase
def lower_case_columns(df):
    df.columns = df.columns.str.lower()
    return df

# Main data wrangling function
def wrangle(df):
    df = lower_case_columns(df)

    # Calculate Body Mass Index (BMI)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Deviation from normal body temperature (assumed to be 37Â°C)
    df['temp_deviation'] = df['body_temp'] - 37

    # Combined cardiovascular load
    df['cardio_load'] = df['duration'] * df['heart_rate']

    # Ratio of body temperature to heart rate
    df['temp_hr_ratio'] = df['body_temp'] / (df['heart_rate'] + 1e-5) 

    # Heart rate squared (to capture non-linear effects)
    df['heart_rate_squared'] = df['heart_rate'] ** 2

    # Weight per unit duration (intensity proxy)
    df['weight_duration_ratio'] = df['weight'] / (df['duration'] + 1e-5)

    # Product of BMI and heart rate
    df['bmi_hr_product'] = df['bmi'] * df['heart_rate']

    # Heart rate normalized per minute of activity
    df['hr_per_minute'] = df['heart_rate'] / (df['duration'] + 1e-5)

    # Product of body temperature and activity duration
    df['temp_duration_product'] = df['body_temp'] * df['duration']

    # Age-weighted cardiovascular load
    df['age_cardio_load'] = df['age'] * df['cardio_load']

    # Statistical features (row-wise mean and standard deviation)
    original_features = ['age', 'height', 'weight', 'duration', 'heart_rate', 'body_temp']
    df['row_mean'] = df[original_features].mean(axis=1)
    df['row_std'] = df[original_features].std(axis=1)
    
    # Reverse Feature Engineering

    # Log-transformations
    cols = ['age_cardio_load', 'weight_duration_ratio']
    for col in cols:
        df[f'log_{col}'] = np.log1p(df[col])
    

    
    # Binning 
    df['temp_duration_product_bin'] = pd.cut(df['temp_duration_product'], bins=10, labels=[f'bin_{i}' for i in range(10)])
    df['temp_hr_ratio_bin'] = pd.qcut(df['temp_hr_ratio'], q=10, duplicates='drop', labels=[f'bin_{i}' for i in range(10)])
    df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=[f'bin_{i}' for i in range(6)])
  

    cols = ['sex', 'temp_duration_product_bin', 'temp_hr_ratio_bin', 'age_bin']
    for col in cols:
        df[col] = df[col].astype('category')

    # Drop ID column (not useful for modeling)
    df = df.drop(columns=['id'])

    # Log-transform the target for more normal distribution
    if 'calories' in df.columns:
        df['calories'] = np.log1p(df['calories'])

    return df

# Apply wrangling to train and calories datasets
df_train = wrangle(df_train).drop_duplicates()
df_calories = wrangle(df_calories.rename(columns={'Gender': 'Sex', 'User_ID': 'id'})).drop_duplicates()

# Concatenate both sets and drop duplicates
df_train = pd.concat([df_train, df_calories], axis=0).drop_duplicates()
df_test = wrangle(df_test)


X = df_train.drop(columns=['calories'])
y = df_train['calories']


# Create feature sets for each model type
drop_for_linear_models = ['heart_rate', 'height', 'body_temp', 'duration']
drop_for_other_models = ['temp_duration_product_bin', 'temp_hr_ratio_bin', 'age_bin']

linear_model_features = X.columns.difference(drop_for_linear_models)
other_model_features = X.columns.difference(drop_for_other_models)

lin_num = X[linear_model_features].select_dtypes(include='number').columns.tolist()
lin_cat = X[linear_model_features].select_dtypes(include='category').columns.tolist()

oth_num = X[other_model_features].select_dtypes(include='number').columns.tolist()
oth_cat = X[other_model_features].select_dtypes(include='category').columns.tolist()


model_names = ['ridge', 'extratrees', 'xgboost', 'lightgbm', 'catboost']

models = [Ridge, ExtraTreesRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor]

params = [
    {'alpha': 0.0002762284437551006, 'solver': 'sag', 'fit_intercept': True, 'tol': 0.00013031761354747614},
    {'n_estimators': 326, 'max_depth': 24, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'log2', 'bootstrap': False},
    {'n_estimators': 683, 'learning_rate': 0.05710400944032593, 'max_depth': 7, 'subsample': 0.8708501983892822, 'colsample_bytree': 0.5990353703327878, 'reg_alpha': 3.254066913534751, 'reg_lambda': 0.01593621652838458, 'min_child_weight': 2, 'gamma': 0.0009536580000644827, 'booster': 'gbtree', 'predictor': 'gpu_predictor', 'gpu_id': 0 },
    {'max_depth': 12, 'num_leaves': 99, 'n_estimators': 491, 'learning_rate': 0.05954360974397409, 'min_child_samples': 21, 'subsample': 0.5066852407414704, 'colsample_bytree': 0.8619994805735846, 'reg_alpha': 3.595735178870403, 'reg_lambda': 1.24462626575653,  'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False},
    {'bootstrap_type': 'Bernoulli', 'iterations': 406, 'learning_rate': 0.028763633853386924, 'depth': 10, 'l2_leaf_reg': 11.869905054427921, 'border_count': 125, 'random_strength': 7.610826130799793, 'grow_policy': 'Depthwise', 'verbose': False, 'task_type': 'GPU', 'devices': '0'}
]


linear_models = ['ridge', 'lasso', 'elasticnet', 'bayesianridge', 'huberregressor']
tree_models = ['randomforest', 'extratrees', 'gradientboosting', 'xgboost']

def create_pipeline(model_name, model, params):
    # Pipeline for linear models: scale selected numeric features
    if model_name in linear_models:
        pipeline = Pipeline(steps=[
            ('select', ColumnTransformer(transformers=[
                ('num', StandardScaler(), lin_num),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), lin_cat)
            ], remainder='drop')),
            (model_name, model(**params))
        ])

    # Pipeline for tree-based models: select relevant features directly (no scaling)
    elif model_name in tree_models:
        pipeline = Pipeline(steps=[
            ('select', ColumnTransformer(transformers=[
                ('num', 'passthrough', oth_num),
                ('cat', OneHotEncoder(drop='first'), oth_cat)
            ], remainder='drop')),
            (model_name, model(**params))
        ])

    elif model_name == 'lightgbm':
        pipeline = Pipeline(steps=[
            ('select', FunctionTransformer(lambda X: X[other_model_features], validate=False)),
            (model_name, model(**params))
        ])

    elif model_name == 'catboost':
        pipeline = pipeline = Pipeline(steps=[
            ('select', FunctionTransformer(lambda X: X[other_model_features], validate=False)),
            (model_name, model(**params, cat_features=oth_cat))
        ])
    else:
        raise ValueError(f"unknown model name: {model_name}")

    return pipeline


pipeline_dict = {}
# Create pipelines for all models and store in dictionary
for model_name, model, param in zip(model_names, models, params):
    pipeline = create_pipeline(model_name, model, param)
    pipeline_dict[model_name] = pipeline


params = {'alpha': 0.0007174822830545591, 'l1_ratio': 0.658029960276858, 'fit_intercept': True, 'tol': 0.00010062095710653478, 'max_iter': 8792, 'selection': 'cyclic'}

meta_model = ElasticNet(**params)

final_stack = StackingRegressor(
    estimators=[
        ('ridge', pipeline_dict['ridge']),
        ('extratrees', pipeline_dict['extratrees']),
        ('xgboost', pipeline_dict['xgboost']),
        ('catboost', pipeline_dict['catboost']),
        ('lightgbm', pipeline_dict['lightgbm']),
    ],
    final_estimator=meta_model,
    passthrough=False,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=1
)
final_stack.fit(X, y)
print("fitted model")


y_pred = final_stack.predict(df_test)
df_sub['Calories'] = np.expm1(y_pred)
df_sub.to_csv('submission.csv', index=False)

