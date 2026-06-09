from __future__ import print_function
import time
import re
from collections import defaultdict
import os

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder, MinMaxScaler, LabelBinarizer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Ансамбли

import xgboost as xgb
import lightgbm as lgb

%matplotlib inline
plt.rcParams["figure.figsize"] = (15, 8)
pd.options.display.float_format = '{:.2f}'.format


os.makedirs('/kaggle/working/data', exist_ok=True)

!7z x /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z -o/kaggle/working/data/
!7z x /kaggle/input/mercari-price-suggestion-challenge/test.tsv.7z -o/kaggle/working/data/
!7z x /kaggle/input/mercari-price-suggestion-challenge/sample_submission.csv.7z -o/kaggle/working/data/

!unzip /kaggle/input/mercari-price-suggestion-challenge/sample_submission_stg2.csv.zip -d /kaggle/working/data/
!unzip /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip -d /kaggle/working/data/


df_train = pd.read_csv('/kaggle/working/data/train.tsv', sep='\t')


display(df_train.sample(10))


df_train.info()


nan_values = df_train.isna().sum()


nan_table = pd.DataFrame({
    'column_name': df_train.columns,
    'nan_count': nan_values,
    'nan_percentage': ((nan_values / len(df_train)) * 100).round(5).astype(str)
})

nan_table = (
    nan_table
    .set_index('column_name')
    .query('nan_count > 0')
    .sort_values('nan_count', ascending=False)
)

display(nan_table)


fill_values = {
    'brand_name': 'Unknown Brand',
    'category_name':  df_train['category_name'].mode()[0],
    'item_description': 'No description yet'
}

for column, fill_value in fill_values.items():
    df_train[column] = df_train[column].fillna(fill_value)


print(df_train.isna().sum())


df_train = df_train.apply(lambda x: x.str.title() if x.dtype == 'object' else x)
object_columns = df_train.select_dtypes(include=['object']).columns
df_train[object_columns] = df_train[object_columns].astype('category')


def select_column_by_type(df, column_type):
    if column_type == 'numeric':
        columns = df.select_dtypes(include=['number']).columns.tolist()
        columns = [col for col in columns if 'id' not in col.lower()]
    elif column_type == 'categorical':
        columns = df.select_dtypes(include=['category']).columns.tolist()
    return columns


def plot_numeric_histograms(df):
    numeric_columns = select_column_by_type(df, 'numeric')
    
    n_cols = len(numeric_columns)
    n_rows = (n_cols + 1) // 2 
    fig, ax = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    
    for i, column in enumerate(numeric_columns):
        ax[i].hist(df[column], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax[i].set_title(f'Гистограмма: {column}', fontsize=12, fontweight='bold')
        ax[i].set_xlabel(column, fontsize=10)
        ax[i].set_ylabel('Частота', fontsize=10)
        ax[i].grid(True)   
        
    plt.tight_layout()
    plt.show()


def plot_categorical_bar(df):
    categorical_columns = select_column_by_type(df, 'categorical')

    n_cols = len(categorical_columns)
    n_rows = (n_cols + 1) // 2 
    fig, ax = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    ax = ax.flatten() 
    
    for i, column in enumerate(categorical_columns):
        value_counts = df[column].value_counts().head(20)
        ax[i].bar(value_counts.index, value_counts.values, color='skyblue', alpha=0.7, edgecolor='black')
        ax[i].set_title(f'Bar диаграмма: {column} с топ-20 значениями', fontsize=12, fontweight='bold')
        ax[i].set_ylabel('Частота', fontsize=10)
        ax[i].tick_params(axis='x', rotation=90)
        ax[i].grid(axis='y')
    
    plt.tight_layout()
    plt.show()


def explore_target(df, target_column):
    numeric_columns = select_column_by_type(df, 'numeric')
    numeric_columns = [col for col in numeric_columns if col != target_column]
    categorical_columns = select_column_by_type(df, 'categorical')
    for column in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='shipping', y='price', data=df)
        plt.xlabel(column)
        plt.ylabel('Price')
        plt.title(f'Распределение {target_column}({column})')
        plt.grid(True)
        plt.show()
    results = []

    for column in categorical_columns:
        top_categories = df[column].value_counts().head(5).index
        df_top = df[df[column].isin(top_categories)]
        df_pivot = df_top.pivot_table(
            index=column, 
            values=target_column, 
            aggfunc=['count', 'mean', 'median'],
            observed=False
        ).round(2)
        df_pivot.columns = ['count', 'mean', 'median']
        df_pivot = df_pivot.sort_values('count', ascending=False)
        print(f"Сводная таблица {column}({target_column}) для 5 наиболее популярных признаков")
        display(df_pivot.head(5))


plot_numeric_histograms(df_train)


plot_categorical_bar(df_train)


df_train['price'] = np.log1p(df_train['price'])


Q1 = np.percentile(df_train['price'], 25)
Q2 = np.percentile(df_train['price'], 50)
Q3 = np.percentile(df_train['price'], 75)

IQR = Q3 - Q1
upper_bound = min(df_train['price'].max(), Q3 + 1.5 * IQR)
lower_bound = max(df_train.query('price > 0')['price'].min(), Q1 - 1.5 * IQR)

stats_df = pd.DataFrame({
    'statistics': ['Q1', 'median', 'Q3', 
                   'IQR', 'upper limit emissions', 'lower limit emissions'],
    'value': [Q1, Q2, Q3, IQR, upper_bound, lower_bound]
}).set_index('statistics')
display(stats_df)

plt.figure(figsize=(8, 5))
box_plot = plt.boxplot(df_train['price'], vert=False, patch_artist=True, widths=0.7, whis=1.5)
colors = ['lightblue']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Распределение price')
plt.xlabel('price')
plt.xlim(max(0, lower_bound * 0.9), upper_bound * 1.1) 
plt.grid(axis='x') 
plt.axvline(x=Q2, color='red', linestyle='--', alpha=0.7, label=f'Медиана: {Q2:.2f}')
plt.axvline(x=upper_bound, color='orange', linestyle='--', alpha=0.7, label=f'Верхняя граница: {upper_bound:.2f}')
plt.axvline(x=lower_bound, color='orange', linestyle='--', alpha=0.7, label=f'Нижняя граница: {lower_bound:.2f}')
plt.legend()
plt.show()

outliers = df_train[df_train['price'] > upper_bound]
print(f"\nКоличество выбросов (выше верхней границы): {len(outliers)}")
print(f"Процент выбросов: {len(outliers)/len(df_train)*100:.2f}%")


df_train = df_train.query("1 < price < 5")


explore_target(df_train, 'price')


fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_train['price'], bins=10, color='skyblue', alpha=0.7, edgecolor='black')
ax.set_title('Гистограмма: price', fontsize=12, fontweight='bold')
ax.set_xlabel('price', fontsize=10)
ax.set_ylabel('Частота', fontsize=10)
ax.grid(True)
plt.tight_layout()
plt.show()


import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


df_test = pd.read_csv('/kaggle/working/data/test_stg2.tsv', sep='\t')


df_test.info()


df_test.head()


nan_values = df_test.isna().sum()


nan_table = pd.DataFrame({
    'column_name': df_test.columns,
    'nan_count': nan_values,
    'nan_percentage': ((nan_values / len(df_test)) * 100).round(5).astype(str)
})

nan_table = (
    nan_table
    .set_index('column_name')
    .query('nan_count > 0')
    .sort_values('nan_count', ascending=False)
)

display(nan_table)


fill_values = {
    'brand_name': 'Unknown Brand',
    'category_name':  df_test['category_name'].mode()[0],
}

for column, fill_value in fill_values.items():
    df_test[column] = df_test[column].fillna(fill_value)


print(df_test.isna().sum())


df_test = df_test.apply(lambda x: x.str.title() if x.dtype == 'object' else x)
object_columns = df_test.select_dtypes(include=['object']).columns
df_test[object_columns] = df_test[object_columns].astype('category')


df_test = df_test.drop(['test_id', 'item_condition_id'], axis=1)
df_train = df_train.drop(['train_id', 'item_condition_id'], axis=1)


X_test = df_test
X_train = df_train.drop('price', axis=1)  
y = df_train['price'] 
print(X_train.shape, X_test.shape)


X_merger = pd.concat([X_train, X_test], ignore_index=True)
print(X_merger.shape)


numeric_features = X_merger.select_dtypes(include=[np.number]).columns
categorical_features = X_merger.select_dtypes(include=[object]).columns


numeric_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

feature_preprocessor = ColumnTransformer(
    transformers=[
        ('numeric_pipeline', numeric_pipeline, numeric_features),
        ('categorical_pipeline', categorical_pipeline, categorical_features)
    ]
)

X_merger_encoded = feature_preprocessor.fit_transform(X_merger)
print(X_merger_encoded.shape)


X_test_encoded = X_merger_encoded[len(X_train):]
X_train_encoded = X_merger_encoded[:len(X_train)]
print(X_train_encoded.shape, X_test_encoded.shape)


X_train, X_val, y_train, y_val = train_test_split(X_train_encoded, y, test_size=0.2, random_state=42)


lgb = LGBMRegressor(
    learning_rate=0.5,
    objective='regression', 
    max_depth=3,
    num_leaves=60,
    verbosity=-1,
    metric='rmse', 
    random_state=1,  
    bagging_fraction=0.5,
    n_jobs=4, 
    n_estimators=8000,
)

lgb.fit(X_train, y_train)

y_valid_pred = lgb.predict(X_val)
r2_val = r2_score(y_val, y_valid_pred)
print(f"R² на валидационной выборке: {r2_val:.4f}")


y_test_pred = lgb.predict(X_test_encoded)
y_test_pred = np.expm1(y_test_pred)
print(y_test_pred)

submission = pd.DataFrame({
    'test_id': range(0, len(y_test_pred)),
    'price': y_test_pred
})

submission.to_csv('/kaggle/working/submission.csv', index=False)







