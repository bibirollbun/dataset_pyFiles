import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# change default colormap
plt.rcParams['image.cmap'] = 'Set3'

# Import the various sklear tools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error, mean_absolute_percentage_error

# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from sklearn.feature_selection import SequentialFeatureSelector as sk_sfs
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold)
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                              GradientBoostingRegressor, ExtraTreesRegressor, 
                              StackingRegressor, BaggingRegressor,VotingRegressor)
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, plot_importance, cv

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras import layers

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, minmax_scale,
                                   OneHotEncoder, FunctionTransformer, OrdinalEncoder)

import yellowbrick
from yellowbrick.classifier import ClassificationReport, DiscriminationThreshold, confusion_matrix
from yellowbrick.regressor import PredictionError
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from yellowbrick.regressor import ResidualsPlot

import optuna
from optuna.samplers import TPESampler
import plotly.express as px

# Set the color scheme 
# my_scheem = 'copper_r'
my_scheem = 'Dark2'
sns.set_palette(my_scheem)

pd.set_option('display.max_columns', 100)
# verify the versions
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'optuna version : {optuna.__version__}')
print(f'yellowbrick version: {yellowbrick.__version__}')


train_00 = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_00 = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_00.head(5)


train_00.info()


train_00.shape


train_00.isna().mean()*100


cat_feats = ['country', 'store', 'product']

for cat_feat in cat_feats:
    feat_count = train_00[cat_feat].value_counts().to_frame()
    display(feat_count)


# List of dates for public holidays in Kenya from 2010 to 2020
kenya_public_holidays = [
    '2010-01-01', '2010-04-02', '2010-04-05', '2010-05-01', '2010-06-01', '2010-10-20', '2010-12-12', '2010-12-25', '2010-12-26',
    '2011-01-01', '2011-04-22', '2011-04-25', '2011-05-01', '2011-06-01', '2011-10-20', '2011-12-12', '2011-12-25', '2011-12-26',
    '2012-01-01', '2012-04-06', '2012-04-09', '2012-05-01', '2012-06-01', '2012-10-20', '2012-12-12', '2012-12-25', '2012-12-26',
    '2013-01-01', '2013-04-19', '2013-04-22', '2013-05-01', '2013-06-01', '2013-10-20', '2013-12-12', '2013-12-25', '2013-12-26',
    '2014-01-01', '2014-04-18', '2014-04-21', '2014-05-01', '2014-06-01', '2014-10-20', '2014-12-12', '2014-12-25', '2014-12-26',
    '2015-01-01', '2015-04-03', '2015-04-06', '2015-05-01', '2015-06-01', '2015-10-20', '2015-12-12', '2015-12-25', '2015-12-26',
    '2016-01-01', '2016-03-25', '2016-03-28', '2016-05-01', '2016-06-01', '2016-10-20', '2016-12-12', '2016-12-25', '2016-12-26',
    '2017-01-01', '2017-04-14', '2017-04-17', '2017-05-01', '2017-06-01', '2017-10-20', '2017-12-12', '2017-12-25', '2017-12-26',
    '2018-01-01', '2018-04-19', '2018-04-22', '2018-05-01', '2018-06-01', '2018-10-20', '2018-12-12', '2018-12-25', '2018-12-26',
    '2019-01-01', '2019-04-19', '2019-04-22', '2019-05-01', '2019-06-01', '2019-10-20', '2019-12-12', '2019-12-25', '2019-12-26',
    '2020-01-01', '2020-04-10', '2020-04-13', '2020-05-01', '2020-06-01', '2020-10-20', '2020-12-12', '2020-12-25', '2020-12-26'
]

# List of dates for public holidays in Finland from 2010 to 2020
finland_public_holidays = [
    '2010-01-01', '2010-04-02', '2010-04-04', '2010-05-01', '2010-06-01', '2010-10-24', '2010-12-06', '2010-12-25', '2010-12-26',
    '2011-01-01', '2011-04-22', '2011-04-25', '2011-05-01', '2011-06-01', '2011-10-24', '2011-12-06', '2011-12-25', '2011-12-26',
    '2012-01-01', '2012-04-06', '2012-04-09', '2012-05-01', '2012-06-01', '2012-10-24', '2012-12-06', '2012-12-25', '2012-12-26',
    '2013-01-01', '2013-04-19', '2013-04-22', '2013-05-01', '2013-06-01', '2013-10-24', '2013-12-06', '2013-12-25', '2013-12-26',
    '2014-01-01', '2014-04-18', '2014-04-21', '2014-05-01', '2014-06-01', '2014-10-24', '2014-12-06', '2014-12-25', '2014-12-26',
    '2015-01-01', '2015-04-03', '2015-04-05', '2015-05-01', '2015-06-01', '2015-10-24', '2015-12-06', '2015-12-25', '2015-12-26',
    '2016-01-01', '2016-03-25', '2016-03-28', '2016-05-01', '2016-06-01', '2016-10-24', '2016-12-06', '2016-12-25', '2016-12-26',
    '2017-01-01', '2017-04-14', '2017-04-17', '2017-05-01', '2017-06-01', '2017-10-24', '2017-12-06', '2017-12-25', '2017-12-26',
    '2018-01-01', '2018-04-19', '2018-04-22', '2018-05-01', '2018-06-01', '2018-10-24', '2018-12-06', '2018-12-25', '2018-12-26',
    '2019-01-01', '2019-04-19', '2019-04-22', '2019-05-01', '2019-06-01', '2019-10-24', '2019-12-06', '2019-12-25', '2019-12-26',
    '2020-01-01', '2020-04-10', '2020-04-13', '2020-05-01', '2020-06-01', '2020-10-24', '2020-12-06', '2020-12-25', '2020-12-26'
]

# List of dates for public holidays in Italy from 2010 to 2020
italy_public_holidays = [
    '2010-01-01', '2010-04-04', '2010-04-05', '2010-05-01', '2010-06-02', '2010-08-15', '2010-11-01', '2010-12-08', '2010-12-25', '2010-12-26',
    '2011-01-01', '2011-04-24', '2011-04-25', '2011-05-01', '2011-06-02', '2011-08-15', '2011-11-01', '2011-12-08', '2011-12-25', '2011-12-26',
    '2012-01-01', '2012-04-08', '2012-04-09', '2012-05-01', '2012-06-02', '2012-08-15', '2012-11-01', '2012-12-08', '2012-12-25', '2012-12-26',
    '2013-01-01', '2013-03-31', '2013-04-01', '2013-05-01', '2013-06-02', '2013-08-15', '2013-11-01', '2013-12-08', '2013-12-25', '2013-12-26',
    '2014-01-01', '2014-04-20', '2014-04-21', '2014-05-01', '2014-06-02', '2014-08-15', '2014-11-01', '2014-12-08', '2014-12-25', '2014-12-26',
    '2015-01-01', '2015-04-05', '2015-04-06', '2015-05-01', '2015-06-02', '2015-08-15', '2015-11-01', '2015-12-08', '2015-12-25', '2015-12-26',
    '2016-01-01', '2016-03-27', '2016-03-28', '2016-05-01', '2016-06-02', '2016-08-15', '2016-11-01', '2016-12-08', '2016-12-25', '2016-12-26',
    '2017-01-01', '2017-04-16', '2017-04-17', '2017-05-01', '2017-06-02', '2017-08-15', '2017-11-01', '2017-12-08', '2017-12-25', '2017-12-26',
    '2018-01-01', '2018-04-01', '2018-04-02', '2018-05-01', '2018-06-02', '2018-08-15', '2018-11-01', '2018-12-08', '2018-12-25', '2018-12-26',
    '2019-01-01', '2019-04-21', '2019-04-22', '2019-05-01', '2019-06-02', '2019-08-15', '2019-11-01', '2019-12-08', '2019-12-25', '2019-12-26',
    '2020-01-01', '2020-04-12', '2020-04-13', '2020-05-01', '2020-06-02', '2020-08-15', '2020-11-01', '2020-12-08', '2020-12-25', '2020-12-26'
]

# List of dates for public holidays in Norway from 2010 to 2020
norway_public_holidays = [
    '2010-01-01', '2010-04-02', '2010-04-05', '2010-05-01', '2010-05-17', '2010-12-25', '2010-12-26',
    '2011-01-01', '2011-04-22', '2011-04-25', '2011-05-01', '2011-05-17', '2011-12-25', '2011-12-26',
    '2012-01-01', '2012-04-06', '2012-04-09', '2012-05-01', '2012-05-17', '2012-12-25', '2012-12-26',
    '2013-01-01', '2013-03-29', '2013-04-01', '2013-05-01', '2013-05-17', '2013-12-25', '2013-12-26',
    '2014-01-01', '2014-04-18', '2014-04-21', '2014-05-01', '2014-05-17', '2014-12-25', '2014-12-26',
    '2015-01-01', '2015-04-03', '2015-04-06', '2015-05-01', '2015-05-17', '2015-12-25', '2015-12-26',
    '2016-01-01', '2016-03-25', '2016-03-28', '2016-05-01', '2016-05-17', '2016-12-25', '2016-12-26',
    '2017-01-01', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-17', '2017-12-25', '2017-12-26',
    '2018-01-01', '2018-03-30', '2018-04-02', '2018-05-01', '2018-05-17', '2018-12-25', '2018-12-26',
    '2019-01-01', '2019-04-19', '2019-04-22', '2019-05-01', '2019-05-17', '2019-12-25', '2019-12-26',
    '2020-01-01', '2020-04-10', '2020-04-13', '2020-05-01', '2020-05-17', '2020-12-25', '2020-12-26'
]

# List of dates for public holidays in Singapore from 2010 to 2020
singapore_public_holidays = [
    '2010-01-01', '2010-02-14', '2010-02-15', '2010-04-02', '2010-05-01', '2010-08-09', '2010-11-05', '2010-12-25',
    '2011-01-01', '2011-02-03', '2011-02-04', '2011-04-22', '2011-05-01', '2011-08-09', '2011-11-26', '2011-12-25',
    '2012-01-01', '2012-01-23', '2012-01-24', '2012-04-06', '2012-05-01', '2012-08-09', '2012-11-13', '2012-12-25',
    '2013-01-01', '2013-02-10', '2013-02-11', '2013-03-29', '2013-05-01', '2013-08-09', '2013-11-03', '2013-12-25',
    '2014-01-01', '2014-01-31', '2014-02-01', '2014-04-18', '2014-05-01', '2014-08-09', '2014-11-23', '2014-12-25',
    '2015-01-01', '2015-02-19', '2015-02-20', '2015-04-03', '2015-05-01', '2015-08-09', '2015-11-10', '2015-12-25',
    '2016-01-01', '2016-02-08', '2016-02-09', '2016-03-25', '2016-05-01', '2016-08-09', '2016-10-29', '2016-12-25',
    '2017-01-01', '2017-01-28', '2017-01-29', '2017-04-14', '2017-05-01', '2017-08-09', '2017-10-18', '2017-12-25',
    '2018-01-01', '2018-02-16', '2018-02-17', '2018-03-30', '2018-05-01', '2018-08-09', '2018-11-06', '2018-12-25',
    '2019-01-01', '2019-02-05', '2019-02-06', '2019-04-19', '2019-05-01', '2019-08-09', '2019-10-27', '2019-12-25',
    '2020-01-01', '2020-01-25', '2020-01-26', '2020-04-10', '2020-05-01', '2020-08-09', '2020-10-18', '2020-12-25'
]

# List of dates for public holidays in Canada from 2010 to 2020
canada_public_holidays = [
    '2010-01-01', '2010-02-15', '2010-04-02', '2010-04-05', '2010-05-24', '2010-07-01', '2010-09-06', '2010-10-11', '2010-11-11', '2010-12-25', '2010-12-26',
    '2011-01-01', '2011-02-14', '2011-04-22', '2011-04-25', '2011-05-23', '2011-07-01', '2011-09-05', '2011-10-10', '2011-11-11', '2011-12-25', '2011-12-26',
    '2012-01-01', '2012-02-20', '2012-04-06', '2012-04-09', '2012-05-21', '2012-07-01', '2012-09-03', '2012-10-08', '2012-11-11', '2012-12-25', '2012-12-26',
    '2013-01-01', '2013-02-18', '2013-03-29', '2013-04-01', '2013-05-20', '2013-07-01', '2013-09-02', '2013-10-14', '2013-11-11', '2013-12-25', '2013-12-26',
    '2014-01-01', '2014-02-17', '2014-04-18', '2014-04-21', '2014-05-19', '2014-07-01', '2014-09-01', '2014-10-13', '2014-11-11', '2014-12-25', '2014-12-26',
    '2015-01-01', '2015-02-16', '2015-04-03', '2015-04-06', '2015-05-18', '2015-07-01', '2015-09-07', '2015-10-12', '2015-11-11', '2015-12-25', '2015-12-26',
    '2016-01-01', '2016-02-15', '2016-03-25', '2016-03-28', '2016-05-23', '2016-07-01', '2016-09-05', '2016-10-10', '2016-11-11', '2016-12-25', '2016-12-26',
    '2017-01-01', '2017-02-20', '2017-04-14', '2017-04-17', '2017-05-22', '2017-07-01', '2017-09-04', '2017-10-09', '2017-11-11', '2017-12-25', '2017-12-26',
    '2018-01-01', '2018-02-19', '2018-03-30', '2018-04-02', '2018-05-21', '2018-07-01', '2018-09-03', '2018-10-08', '2018-11-11', '2018-12-25', '2018-12-26',
    '2019-01-01', '2019-02-18', '2019-04-19', '2019-04-22', '2019-05-20', '2019-07-01', '2019-09-02', '2019-10-14', '2019-11-11', '2019-12-25', '2019-12-26',
    '2020-01-01', '2020-02-17', '2020-04-10', '2020-04-13', '2020-05-18', '2020-07-01', '2020-09-07', '2020-10-12', '2020-11-11', '2020-12-25', '2020-12-26'
]


train_00['set'] = 'train'
test_00['set'] = 'test'

df = pd.concat([train_00, test_00])

df_italy = df[df['country']=='Italy']
df_italy['public_holiday'] = df_italy['date'].apply(lambda x: 1 if x in italy_public_holidays else 0)
df_kenya = df[df['country']=='Kenya']
df_kenya['public_holiday'] = df_kenya['date'].apply(lambda x: 1 if x in kenya_public_holidays else 0)
df_canada = df[df['country']=='Canada']
df_canada['public_holiday'] = df_canada['date'].apply(lambda x: 1 if x in canada_public_holidays else 0)
df_norway = df[df['country']=='Norway']
df_norway['public_holiday'] = df_norway['date'].apply(lambda x: 1 if x in norway_public_holidays else 0)
df_finland = df[df['country']=='Finland']
df_finland['public_holiday'] = df_finland['date'].apply(lambda x: 1 if x in finland_public_holidays else 0)
df_singapore = df[df['country']=='Singapore']
df_singapore['public_holiday'] = df_singapore['date'].apply(lambda x: 1 if x in singapore_public_holidays else 0)


df = pd.concat([df_italy, df_kenya, df_canada, df_norway, df_finland, df_singapore])

train_01 = df[df['set']=='train']
train_01 = train_01.drop(columns='set').sort_values(by='id')
test_01 = df[df['set']=='test']
test_01 = test_01.drop(columns=['num_sold','set']).sort_values(by='id')
test_01


test_00


# drop the rows that are missing values

train_01 = train_01.dropna()


def date_processor(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year - 2009
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['weekend'] = df['day_of_week'].apply(lambda x: 'weekend' if x>=5 else 'weekday')
    df = df.drop(columns=['id'])
    return df


train_02 = date_processor(train_01)
test_02 = date_processor(test_01)
train_02.sample(5)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6)) # Adjusting the overall figure size

# First plot
plt.subplot(1, 2, 1)
sns.lineplot(train_01, x='date', y='num_sold', errorbar=('ci', .95))
plt.title('Sales trend: Years ', weight='bold', fontsize=14)

# Second plot
plt.subplot(1, 2, 2)
sns.boxplot(data=train_01, x='year', y='num_sold')
plt.tight_layout()

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6)) # Adjusting the overall figure size

# First plot
plt.subplot(1, 2, 1)
sns.lineplot(train_01, x='date', y='num_sold', hue='country', errorbar=('ci', 1))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Second plot
plt.subplot(1, 2, 2)
sns.boxplot(data=train_01, x='country', y='num_sold')
plt.tight_layout()

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8)) # Adjusting the overall figure size

# First plot
# plt.subplot(1, 2, 1)
for c, country in enumerate(train_01['country'].unique(), start=2):
    plt.subplot(4, 2, c)
    sns.lineplot(data=train_01[train_01['country'] == country], x='month', y='num_sold', palette='Dark2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(country, fontsize=12)
    if c < 6:
        plt.xlabel('')
    if c > 1:
        plt.legend([])

plt.suptitle('Sales by month for each country', fontsize=14)
plt.tight_layout()
# Second plot
plt.subplot(4, 2, 1)
sns.boxenplot(train_01, x='num_sold')

# third plot
plt.subplot(4, 2, 8)
sns.boxenplot(train_01, x='num_sold', y='country')
plt.tight_layout()

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6)) # Adjusting the overall figure size

# First plot
plt.subplot(1, 2, 1)
sns.lineplot(data=train_01, x='month', y='num_sold', hue='store')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Sales: months and countries', weight='bold', fontsize=14)

# Second plot
plt.subplot(1, 2, 2)
sns.boxplot(data=train_01, x='num_sold', y='store')
plt.tight_layout()

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) # Adjusting the overall figure size

# First plot
# plt.subplot(1, 2, 1)
for c, store in enumerate(train_01['store'].unique(), start=1):
    plt.subplot(2, 2, c)
    sns.lineplot(data=train_01[train_01['store'] == store], x='month', y='num_sold', hue='country', palette='Dark2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(store, fontsize=12)
    if c < 5:
        plt.xlabel('')
    if c > 1:
        plt.legend([])

plt.suptitle('Sales by month for each country', fontsize=14)
plt.tight_layout()

# Second plot
plt.subplot(2, 2, 4)
sns.boxplot(data=train_01, x='num_sold', y='store')
plt.tight_layout()

plt.show()



plt.figure(figsize=(12,8))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, height=3)
g.map(sns.boxplot, 'num_sold', 'country')
# Add titles and adjust layout
g.set_titles("{col_name}")
g.set_axis_labels("Date", "Value")
plt.tight_layout()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6)) # Adjusting the overall figure size

# First plot
plt.subplot(1, 2, 1)
sns.lineplot(data=train_01, x='month', y='num_sold', hue='product')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Sales: months and countries', weight='bold', fontsize=14)

# Second plot
plt.subplot(1, 2, 2)
sns.boxplot(data=train_01, x='num_sold', y='product')
plt.tight_layout()

plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6)) # Adjusting the overall figure size

# First plot
plt.subplot(1, 2, 1)
for c, prdt in enumerate(train_01['product'].unique(), start=1):
    plt.subplot(3,2, c)
    sns.lineplot(data=train_01[train_01['product'] == prdt], x='month', y='num_sold', palette='Dark2', hue='country')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(prdt, fontsize=12)
    if c < 5:
        plt.xlabel('')
    if c > 1:
        plt.legend([])

plt.suptitle('Sales trend by month for each product', fontsize=14)
plt.tight_layout()

# Second plot
plt.subplot(3, 2, 6)
sns.boxplot(data=train_01, x='num_sold', y='product')
plt.tight_layout()

plt.show()



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6)) # Adjusting the overall figure size

# First plot
plt.subplot(1, 2, 1)
sns.lineplot(data=train_01, x='month', y='num_sold', hue='year', palette='Dark2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='year')
plt.title('Sales: month vs year', fontsize=14, weight='bold')

# Second plot
plt.subplot(1, 2, 2)
sns.boxenplot(data=train_01, y='num_sold', x='month')
plt.tight_layout()

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6)) # Adjusting the overall figure size

# First plot
plt.subplot(1, 2, 1)
sns.lineplot(data=train_01, x='month', y='num_sold', hue='day_of_week', palette='Dark2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Sales: months and countries', weight='bold', fontsize=14)

# Second plot
plt.subplot(1, 2, 2)
sns.boxplot(data=train_01, y='num_sold', x='day_of_week')
plt.tight_layout()

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6)) # Adjusting the overall figure size

# First plot
plt.subplot(1, 3, 1)
sns.lineplot(data=train_01, x='month', y='num_sold', hue='weekend', palette='Dark2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Sales by month: weekend vs weekdays')

# Second plot
plt.subplot(1, 3, 2)
sns.histplot(data=train_01, x='num_sold', hue='weekend', bins=30)
plt.tight_layout()

# Second plot
plt.subplot(1, 3, 3)
sns.boxenplot(data=train_01, y='num_sold', x='weekend')
plt.tight_layout()

plt.show()


plt.figure(figsize=(10, 6))
for c, cntry in enumerate(train_01['country'].unique(), start=1):
    plt.subplot(4, 2, c)
    sns.lineplot(data=train_01[train_00['country']==cntry], x='month', y='num_sold', hue='weekend', palette='Dark2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(cntry, fontsize=12)
    if c < 5:
        plt.xlabel('')
    if c > 1:
        plt.legend([])
    plt.suptitle('Sales by month for each country', fontsize=14)
plt.tight_layout()

g = sns.FacetGrid(train_01, col="weekend", col_wrap=2, height=2.5, aspect=2)
g.map(sns.boxplot, 'num_sold', 'country')
# Add titles and adjust layout
g.set_titles("{col_name}")
g.set_axis_labels("Date", "Value")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 9))
for i, item in enumerate(train_01['store'].unique(), start=1):
    plt.subplot(3, 1, i)
    sns.lineplot(data=train_01[train_00['store']==item], x='month', y='num_sold', hue='product', palette='Dark2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(item, fontsize=12)
    if i < 3:
        plt.xlabel('')
    plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
for c, cntry in enumerate(train_01['country'].unique(), start=1):
    plt.subplot(3, 2, c)
    sns.lineplot(data=train_01[train_00['country']==cntry], x='month', y='num_sold', hue='store', palette='Dark2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if c < 5:
        plt.xlabel('')
    plt.title(cntry, fontsize=12)
    plt.tight_layout()
    plt.suptitle('Sales trend: by store in various countries', fontsize=14)
    if c > 1:
        plt.legend([])
plt.show()


g = sns.FacetGrid(train_01, col='public_holiday',col_wrap=2, height=3)
g.map(sns.histplot, 'num_sold', fill=True)
# Add titles and adjust layout
g.set_titles("public_holiday: {col_name}")
g.set_axis_labels("Date", "Value")
plt.show()

g = sns.FacetGrid(train_01, col='public_holiday', col_wrap=2, height=3)
g.map(sns.boxplot, 'num_sold', 'country')
# Add titles and adjust layout
g.set_titles("public_holiday: {col_name}")
g.set_axis_labels("Date", "Value")
plt.tight_layout()

plt.show()



ax = train_01.groupby(['country'])['num_sold'].mean().plot.barh(figsize=(6.2,3))
plt.xlabel('num_sold')
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6)) # Adjusting the overall figure size

# First plot
plt.subplot(1, 2, 1)
sns.kdeplot(train_01, x='num_sold', hue='store', fill=True, alpha=.8)
# Second plot
plt.subplot(1, 2, 2)
sns.histplot(train_01, x='num_sold', hue='store', fill=True, alpha=.8, bins=30)

plt.show()


plt.figure(figsize=(12,4))
# Create the facet grid
g = sns.FacetGrid(train_01, col="product", col_wrap=5, height=4, aspect=0.6)

# Map the plot to the facet grid
# g.map(sns.lineplot, 'date', 'value')
g.map(sns.boxplot, 'num_sold', 'country')

# Add titles and adjust layout
g.set_titles("{col_name}")
g.set_axis_labels("num_sold")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,4))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, height=3)

# Map the plot to the facet grid
g.map(sns.boxplot, 'num_sold', 'product')

# Add titles and adjust layout
g.set_titles("{col_name}")
g.set_axis_labels("num_sold", "Value")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,6))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, height=3)

# Map the plot to the facet grid
g.map(sns.kdeplot, 'num_sold', fill=True, alpha=0.8)

# Add titles and adjust layout
g.set_titles("{col_name}")
g.set_axis_labels("num_sold", "Value")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,6))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, hue='product', height=3)

# Map the plot to the facet grid
g.map(sns.kdeplot, 'num_sold', alpha=0.8, fill=True)

# Add titles and adjust layout
g.set_titles("{col_name}")
g.set_axis_labels("num_sold", "Value")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,4))
# Create the facet grid
g = sns.FacetGrid(train_01, col="country", hue='product', col_wrap=3, height=3, palette='Dark2')

# Map the plot to the facet grid
g.map(sns.histplot, 'num_sold')

# Add titles and adjust layout
g.set_titles("{col_name}")
g.set_axis_labels("num_sold", "Frequency")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,6))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, height=3)

# Map the plot to the facet grid
g.map(sns.boxenplot, 'num_sold', 'weekend')

# Add titles and adjust layout
g.set_titles("{col_name} category")
g.set_axis_labels("num_sold", "Frequency")
plt.tight_layout()
plt.show()


plt.figure(figsize=(16,8))
# Create the facet grid
g = sns.FacetGrid(train_01, col="store", col_wrap=3, height=3)

# Map the plot to the facet grid
g.map(sns.boxenplot, 'num_sold', 'country')

# Add titles and adjust layout
g.set_titles("{col_name}")
g.set_axis_labels("num_sold", "country")
plt.tight_layout()
plt.show()


X = train_02.copy()
y = X.pop('num_sold')

X.head(3)


cat_feats = test_02.select_dtypes(exclude='number').columns.tolist()
cat_feats.remove('date')
# cat_feats.append('year')


import category_encoders as ce

enc = 'ord'
if enc == 'cat':
    encoder = ce.CatBoostEncoder()
if enc == 'ord':
    encoder = OrdinalEncoder()

enc = 'cat'
# feat_to_scale = X_ts.select_dtypes(include='number').columns.tolist()
features_trans = make_column_transformer(
    ('drop', ['date']),
    (encoder, cat_feats),
    (StandardScaler(), ['month', 'day',	'day_of_year', 'year']),
    # (MinMaxScaler(), ['month', 'day',	'day_of_year', 'day_of_week', 'year']),
    remainder='passthrough', 
    sparse_threshold=0)


# pd.DataFrame(features_trans.fit_transform(X, y))


model = make_pipeline(features_trans, LGBMRegressor())
model


n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
kfold


def objective(trial):
    lgbm_params = {
        "objective": 'regression',
        # "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        # "learning_rate": 0.01,
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        'verbose': -1
    }
    
    cat = LGBMRegressor(**lgbm_params)
    model = make_pipeline(features_trans, cat)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_val)
    # mape = mean_absolute_percentage_error(y_val, y_pred)
    # return mape

    mapes = []
    for f, (tr_ind, ts_ind) in enumerate(kfold.split(X,y), start=1):
        X_tr, X_ts = X.iloc[tr_ind], X.iloc[ts_ind]
        y_tr, y_ts = y.iloc[tr_ind], y.iloc[ts_ind]
    
        model.fit(X_tr, y_tr)
        preds = model.predict(X_ts)
    
        f_mape = mean_absolute_percentage_error(y_ts, preds)
        mapes.append(f_mape)
        
    mape = np.mean(mapes)
    return mape

def run_optimizer(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
    else:
        best_params = {'lambda_l1': 2.303718659446125e-08, 
                       'lambda_l2': 1.3732797204342579e-06, 
                       'learning_rate': 0.07397527675224533, 
                       'num_leaves': 249, 
                       'max_depth': 16, 
                       'feature_fraction': 0.9991930316913752, 
                       'bagging_fraction': 0.7370536508496233, 
                       'bagging_freq': 7, 
                       'min_child_samples': 53}
    return best_params


best_params = run_optimizer(n_trials=1)


print(best_params)


model = make_pipeline(features_trans, LGBMRegressor(**best_params, verbose=-1))


X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=4)

model.fit(X_tr, y_tr)


visualizer = ResidualsPlot(model)

visualizer.fit(X_tr, y_tr)  # Fit the training data to the visualizer
visualizer.score(X_ts, y_ts)  # Evaluate the model on the test data
visualizer.poof()                 # Finalize and render the figure


visualizer = ResidualsPlot(model, hist=False, qqplot=True)

visualizer.fit(X_tr, y_tr)  # Fit the training data to the visualizer
visualizer.score(X_ts, y_ts)  # Evaluate the model on the test data
visualizer.poof()                  # Finalize and render the figure


viz = PredictionError(model, line_colors='green')
viz.fit(X_tr, y_tr)
viz.score(X_ts, y_ts)
viz.show()
plt.show()


for f, (tr_ind, ts_ind) in enumerate(kfold.split(X,y), start=1):
    X_tr, X_ts = X.iloc[tr_ind], X.iloc[ts_ind]
    y_tr, y_ts = y.iloc[tr_ind], y.iloc[ts_ind]

    model.fit(X_tr, y_tr)
    preds = model.predict(X_ts)

    mape = mean_absolute_percentage_error(y_ts, preds)
    print('fold_{}: {:.6f}'.format(f, mape))


preds = model.predict(test_02)

preds


pd.Series(preds).plot.kde(figsize=(6, 3))
plt.show()


pd.Series(preds).plot.hist(figsize=(6, 3), bins=50)
plt.show()


submission_00 = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission_00['num_sold'] = preds


submission_00.to_csv('submission.csv', index=False)


# # List of dates for public holidays in Canada from 2010 to 2020
# canada_public_holidays = [
#     ['2010-01-01', '2010-02-15', '2010-04-02', '2010-04-05', '2010-05-24', '2010-07-01', '2010-09-06', '2010-10-11', '2010-11-11', '2010-12-25', '2010-12-26'],
#     ['2011-01-01', '2011-02-14', '2011-04-22', '2011-04-25', '2011-05-23', '2011-07-01', '2011-09-05', '2011-10-10', '2011-11-11', '2011-12-25', '2011-12-26'],
#     ['2012-01-01', '2012-02-20', '2012-04-06', '2012-04-09', '2012-05-21', '2012-07-01', '2012-09-03', '2012-10-08', '2012-11-11', '2012-12-25', '2012-12-26'],
#     ['2013-01-01', '2013-02-18', '2013-03-29', '2013-04-01', '2013-05-20', '2013-07-01', '2013-09-02', '2013-10-14', '2013-11-11', '2013-12-25', '2013-12-26'],
#     ['2014-01-01', '2014-02-17', '2014-04-18', '2014-04-21', '2014-05-19', '2014-07-01', '2014-09-01', '2014-10-13', '2014-11-11', '2014-12-25', '2014-12-26'],
#     ['2015-01-01', '2015-02-16', '2015-04-03', '2015-04-06', '2015-05-18', '2015-07-01', '2015-09-07', '2015-10-12', '2015-11-11', '2015-12-25', '2015-12-26'],
#     ['2016-01-01', '2016-02-15', '2016-03-25', '2016-03-28', '2016-05-23', '2016-07-01', '2016-09-05', '2016-10-10', '2016-11-11', '2016-12-25', '2016-12-26'],
#     ['2017-01-01', '2017-02-20', '2017-04-14', '2017-04-17', '2017-05-22', '2017-07-01', '2017-09-04', '2017-10-09', '2017-11-11', '2017-12-25', '2017-12-26'],
#     ['2018-01-01', '2018-02-19', '2018-03-30', '2018-04-02', '2018-05-21', '2018-07-01', '2018-09-03', '2018-10-08', '2018-11-11', '2018-12-25', '2018-12-26'],
#     ['2019-01-01', '2019-02-18', '2019-04-19', '2019-04-22', '2019-05-20', '2019-07-01', '2019-09-02', '2019-10-14', '2019-11-11', '2019-12-25', '2019-12-26'],
#     ['2020-01-01', '2020-02-17', '2020-04-10', '2020-04-13', '2020-05-18', '2020-07-01', '2020-09-07', '2020-10-12', '2020-11-11', '2020-12-25', '2020-12-26']
# ]

# # Create a Pandas DataFrame
# df = pd.DataFrame(canada_public_holidays , columns=[
#     'New Year\'s Day', 'Family Day', 'Good Friday', 'Easter Monday', 
#     'Victoria Day', 'Canada Day', 'Labour Day', 'Thanksgiving', 
#     'Remembrance Day', 'Christmas', 'Boxing Day'])

# # Add the years as the index
# df.index = range(2010, 2021)

# df



# # List of dates for public holidays in Kenya from 2010 to 2020
# kenya_public_holidays = [
#     ['2010-01-01', '2010-04-02', '2010-04-05', '2010-05-01', '2010-06-01', '2010-10-20', '2010-12-12', '2010-12-25', '2010-12-26'],
#     ['2011-01-01', '2011-04-22', '2011-04-25', '2011-05-01', '2011-06-01', '2011-10-20', '2011-12-12', '2011-12-25', '2011-12-26'],
#     ['2012-01-01', '2012-04-06', '2012-04-09', '2012-05-01', '2012-06-01', '2012-10-20', '2012-12-12', '2012-12-25', '2012-12-26'],
#     ['2013-01-01', '2013-04-19', '2013-04-22', '2013-05-01', '2013-06-01', '2013-10-20', '2013-12-12', '2013-12-25', '2013-12-26'],
#     ['2014-01-01', '2014-04-18', '2014-04-21', '2014-05-01', '2014-06-01', '2014-10-20', '2014-12-12', '2014-12-25', '2014-12-26'],
#     ['2015-01-01', '2015-04-03', '2015-04-06', '2015-05-01', '2015-06-01', '2015-10-20', '2015-12-12', '2015-12-25', '2015-12-26'],
#     ['2016-01-01', '2016-03-25', '2016-03-28', '2016-05-01', '2016-06-01', '2016-10-20', '2016-12-12', '2016-12-25', '2016-12-26'],
#     ['2017-01-01', '2017-04-14', '2017-04-17', '2017-05-01', '2017-06-01', '2017-10-20', '2017-12-12', '2017-12-25', '2017-12-26'],
#     ['2018-01-01', '2018-04-19', '2018-04-22', '2018-05-01', '2018-06-01', '2018-10-20', '2018-12-12', '2018-12-25', '2018-12-26'],
#     ['2019-01-01', '2019-04-19', '2019-04-22', '2019-05-01', '2019-06-01', '2019-10-20', '2019-12-12', '2019-12-25', '2019-12-26'],
#     ['2020-01-01', '2020-04-10', '2020-04-13', '2020-05-01', '2020-06-01', '2020-10-20', '2020-12-12', '2020-12-25', '2020-12-26']
# ]

# # Create a Pandas DataFrame
# df = pd.DataFrame(kenya_public_holidays , columns=[
#     'New Year\'s Day', 'Good Friday', 'Easter Monday', 'Labour Day', 
#     'Madaraka Day', 'Mashujaa Day', 'Jamhuri Day', 'Christmas', 
#     'Boxing Day'])

# # Add the years as the index
# df.index = range(2010, 2021)

# print(df)



# import pandas as pd

# # List of dates for public holidays in Canada from 2010 to 2020
# canada_public_holidays = [
#     ['2010-01-01', '2010-02-15', '2010-04-02', '2010-04-05', '2010-05-24', '2010-07-01', '2010-09-06', '2010-10-11', '2010-11-11', '2010-12-25', '2010-12-26'],
#     ['2011-01-01', '2011-02-14', '2011-04-22', '2011-04-25', '2011-05-23', '2011-07-01', '2011-09-05', '2011-10-10', '2011-11-11', '2011-12-25', '2011-12-26'],
#     ['2012-01-01', '2012-02-20', '2012-04-06', '2012-04-09', '2012-05-21', '2012-07-01', '2012-09-03', '2012-10-08', '2012-11-11', '2012-12-25', '2012-12-26'],
#     ['2013-01-01', '2013-02-18', '2013-03-29', '2013-04-01', '2013-05-20', '2013-07-01', '2013-09-02', '2013-10-14', '2013-11-11', '2013-12-25', '2013-12-26'],
#     ['2014-01-01', '2014-02-17', '2014-04-18', '2014-04-21', '2014-05-19', '2014-07-01', '2014-09-01', '2014-10-13', '2014-11-11', '2014-12-25', '2014-12-26'],
#     ['2015-01-01', '2015-02-16', '2015-04-03', '2015-04-06', '2015-05-18', '2015-07-01', '2015-09-07', '2015-10-12', '2015-11-11', '2015-12-25', '2015-12-26'],
#     ['2016-01-01', '2016-02-15', '2016-03-25', '2016-03-28', '2016-05-23', '2016-07-01', '2016-09-05', '2016-10-10', '2016-11-11', '2016-12-25', '2016-12-26'],
#     ['2017-01-01', '2017-02-20', '2017-04-14', '2017-04-17', '2017-05-22', '2017-07-01', '2017-09-04', '2017-10-09', '2017-11-11', '2017-12-25', '2017-12-26'],
#     ['2018-01-01', '2018-02-19', '2018-03-30', '2018-04-02', '2018-05-21', '2018-07-01', '2018-09-03', '2018-10-08', '2018-11-11', '2018-12-25', '2018-12-26'],
#     ['2019-01-01', '2019-02-18', '2019-04-19', '2019-04-22', '2019-05-20', '2019-07-01', '2019-09-02', '2019-10-14', '2019-11-11', '2019-12-25', '2019-12-26'],
#     ['2020-01-01', '2020-02-17', '2020-04-10', '2020-04-13', '2020-05-18', '2020-07-01', '2020-09-07', '2020-10-12', '2020-11-11', '2020-12-25', '2020-12-26']
# ]

# # Create a Pandas DataFrame
# df_canada = pd.DataFrame(canada_public_holidays, columns=[
#     'New Year\'s Day', 'Family Day', 'Good Friday', 'Easter Monday', 
#     'Victoria Day', 'Canada Day', 'Labour Day', 'Thanksgiving', 
#     'Remembrance Day', 'Christmas', 'Boxing Day'])

# # Add the years as the index
# df_canada.index = range(2010, 2021)

# print(df_canada)



# # List of dates for public holidays in Finland from 2010 to 2020
# finland_public_holidays = [
#     ['2010-01-01', '2010-04-02', '2010-04-04', '2010-05-01', '2010-06-01', '2010-10-24', '2010-12-06', '2010-12-25', '2010-12-26'],
#     ['2011-01-01', '2011-04-22', '2011-04-25', '2011-05-01', '2011-06-01', '2011-10-24', '2011-12-06', '2011-12-25', '2011-12-26'],
#     ['2012-01-01', '2012-04-06', '2012-04-09', '2012-05-01', '2012-06-01', '2012-10-24', '2012-12-06', '2012-12-25', '2012-12-26'],
#     ['2013-01-01', '2013-04-19', '2013-04-22', '2013-05-01', '2013-06-01', '2013-10-24', '2013-12-06', '2013-12-25', '2013-12-26'],
#     ['2014-01-01', '2014-04-18', '2014-04-21', '2014-05-01', '2014-06-01', '2014-10-24', '2014-12-06', '2014-12-25', '2014-12-26'],
#     ['2015-01-01', '2015-04-03', '2015-04-05', '2015-05-01', '2015-06-01', '2015-10-24', '2015-12-06', '2015-12-25', '2015-12-26'],
#     ['2016-01-01', '2016-03-25', '2016-03-28', '2016-05-01', '2016-06-01', '2016-10-24', '2016-12-06', '2016-12-25', '2016-12-26'],
#     ['2017-01-01', '2017-04-14', '2017-04-17', '2017-05-01', '2017-06-01', '2017-10-24', '2017-12-06', '2017-12-25', '2017-12-26'],
#     ['2018-01-01', '2018-04-19', '2018-04-22', '2018-05-01', '2018-06-01', '2018-10-24', '2018-12-06', '2018-12-25', '2018-12-26'],
#     ['2019-01-01', '2019-04-19', '2019-04-22', '2019-05-01', '2019-06-01', '2019-10-24', '2019-12-06', '2019-12-25', '2019-12-26'],
#     ['2020-01-01', '2020-04-10', '2020-04-13', '2020-05-01', '2020-06-01', '2020-10-24', '2020-12-06', '2020-12-25', '2020-12-26']
# ]

# # Create a Pandas DataFrame
# df_finland = pd.DataFrame(finland_public_holidays, columns=[
#     'New Year\'s Day', 'Good Friday', 'Easter Monday', 'Labour Day', 
#     'Midsummer Day', 'Independence Day', 'All Saints\' Day', 'Christmas', 
#     'Boxing Day'])

# # Add the years as the index
# df_finland.index = range(2010, 2021)

# print(df_finland)



# # List of dates for public holidays in Italy from 2010 to 2020
# italy_public_holidays = [
#     ['2010-01-01', '2010-04-04', '2010-04-05', '2010-05-01', '2010-06-02', '2010-08-15', '2010-11-01', '2010-12-08', '2010-12-25', '2010-12-26'],
#     ['2011-01-01', '2011-04-24', '2011-04-25', '2011-05-01', '2011-06-02', '2011-08-15', '2011-11-01', '2011-12-08', '2011-12-25', '2011-12-26'],
#     ['2012-01-01', '2012-04-08', '2012-04-09', '2012-05-01', '2012-06-02', '2012-08-15', '2012-11-01', '2012-12-08', '2012-12-25', '2012-12-26'],
#     ['2013-01-01', '2013-03-31', '2013-04-01', '2013-05-01', '2013-06-02', '2013-08-15', '2013-11-01', '2013-12-08', '2013-12-25', '2013-12-26'],
#     ['2014-01-01', '2014-04-20', '2014-04-21', '2014-05-01', '2014-06-02', '2014-08-15', '2014-11-01', '2014-12-08', '2014-12-25', '2014-12-26'],
#     ['2015-01-01', '2015-04-05', '2015-04-06', '2015-05-01', '2015-06-02', '2015-08-15', '2015-11-01', '2015-12-08', '2015-12-25', '2015-12-26'],
#     ['2016-01-01', '2016-03-27', '2016-03-28', '2016-05-01', '2016-06-02', '2016-08-15', '2016-11-01', '2016-12-08', '2016-12-25', '2016-12-26'],
#     ['2017-01-01', '2017-04-16', '2017-04-17', '2017-05-01', '2017-06-02', '2017-08-15', '2017-11-01', '2017-12-08', '2017-12-25', '2017-12-26'],
#     ['2018-01-01', '2018-04-01', '2018-04-02', '2018-05-01', '2018-06-02', '2018-08-15', '2018-11-01', '2018-12-08', '2018-12-25', '2018-12-26'],
#     ['2019-01-01', '2019-04-21', '2019-04-22', '2019-05-01', '2019-06-02', '2019-08-15', '2019-11-01', '2019-12-08', '2019-12-25', '2019-12-26'],
#     ['2020-01-01', '2020-04-12', '2020-04-13', '2020-05-01', '2020-06-02', '2020-08-15', '2020-11-01', '2020-12-08', '2020-12-25', '2020-12-26']
# ]

# # Create a Pandas DataFrame
# df_italy = pd.DataFrame(italy_public_holidays, columns=[
#     'New Year\'s Day', 'Easter', 'Easter Monday', 'Labour Day', 
#     'Republic Day', 'Assumption Day', 'All Saints\' Day', 'Immaculate Conception', 
#     'Christmas', 'Boxing Day'])

# # Add the years as the index
# df_italy.index = range(2010, 2021)

# print(df_italy)



# # List of dates for public holidays in Norway from 2010 to 2020
# norway_public_holidays = [
#     ['2010-01-01', '2010-04-02', '2010-04-05', '2010-05-01', '2010-05-17', '2010-12-25', '2010-12-26'],
#     ['2011-01-01', '2011-04-22', '2011-04-25', '2011-05-01', '2011-05-17', '2011-12-25', '2011-12-26'],
#     ['2012-01-01', '2012-04-06', '2012-04-09', '2012-05-01', '2012-05-17', '2012-12-25', '2012-12-26'],
#     ['2013-01-01', '2013-03-29', '2013-04-01', '2013-05-01', '2013-05-17', '2013-12-25', '2013-12-26'],
#     ['2014-01-01', '2014-04-18', '2014-04-21', '2014-05-01', '2014-05-17', '2014-12-25', '2014-12-26'],
#     ['2015-01-01', '2015-04-03', '2015-04-06', '2015-05-01', '2015-05-17', '2015-12-25', '2015-12-26'],
#     ['2016-01-01', '2016-03-25', '2016-03-28', '2016-05-01', '2016-05-17', '2016-12-25', '2016-12-26'],
#     ['2017-01-01', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-17', '2017-12-25', '2017-12-26'],
#     ['2018-01-01', '2018-03-30', '2018-04-02', '2018-05-01', '2018-05-17', '2018-12-25', '2018-12-26'],
#     ['2019-01-01', '2019-04-19', '2019-04-22', '2019-05-01', '2019-05-17', '2019-12-25', '2019-12-26'],
#     ['2020-01-01', '2020-04-10', '2020-04-13', '2020-05-01', '2020-05-17', '2020-12-25', '2020-12-26']
# ]

# # Create a Pandas DataFrame
# df_norway = pd.DataFrame(norway_public_holidays, columns=[
#     'New Year\'s Day', 'Good Friday', 'Easter Monday', 'Labour Day', 
#     'Constitution Day', 'Christmas', 'Boxing Day'])

# # Add the years as the index
# df_norway.index = range(2010, 2021)

# print(df_norway)



# # List of dates for public holidays in Singapore from 2010 to 2020
# singapore_public_holidays = [
#     ['2010-01-01', '2010-02-14', '2010-02-15', '2010-04-02', '2010-05-01', '2010-08-09', '2010-11-05', '2010-12-25'],
#     ['2011-01-01', '2011-02-03', '2011-02-04', '2011-04-22', '2011-05-01', '2011-08-09', '2011-11-26', '2011-12-25'],
#     ['2012-01-01', '2012-01-23', '2012-01-24', '2012-04-06', '2012-05-01', '2012-08-09', '2012-11-13', '2012-12-25'],
#     ['2013-01-01', '2013-02-10', '2013-02-11', '2013-03-29', '2013-05-01', '2013-08-09', '2013-11-03', '2013-12-25'],
#     ['2014-01-01', '2014-01-31', '2014-02-01', '2014-04-18', '2014-05-01', '2014-08-09', '2014-11-23', '2014-12-25'],
#     ['2015-01-01', '2015-02-19', '2015-02-20', '2015-04-03', '2015-05-01', '2015-08-09', '2015-11-10', '2015-12-25'],
#     ['2016-01-01', '2016-02-08', '2016-02-09', '2016-03-25', '2016-05-01', '2016-08-09', '2016-10-29', '2016-12-25'],
#     ['2017-01-01', '2017-01-28', '2017-01-29', '2017-04-14', '2017-05-01', '2017-08-09', '2017-10-18', '2017-12-25'],
#     ['2018-01-01', '2018-02-16', '2018-02-17', '2018-03-30', '2018-05-01', '2018-08-09', '2018-11-06', '2018-12-25'],
#     ['2019-01-01', '2019-02-05', '2019-02-06', '2019-04-19', '2019-05-01', '2019-08-09', '2019-10-27', '2019-12-25'],
#     ['2020-01-01', '2020-01-25', '2020-01-26', '2020-04-10', '2020-05-01', '2020-08-09', '2020-10-18', '2020-12-25']
# ]

# # Create a Pandas DataFrame
# df_singapore = pd.DataFrame(singapore_public_holidays, columns=[
#     'New Year\'s Day', 'Chinese New Year (1st day)', 'Chinese New Year (2nd day)', 'Good Friday', 
#     'Labour Day', 'National Day', 'Deepavali', 'Christmas'])

# # Add the years as the index
# df_singapore.index = range(2010, 2021)

# print(df_singapore)

