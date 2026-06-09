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
import holidays

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Kaggle環境でPlotlyをオフラインで使用する設定
# pio.renderers.default = 'notebook'
# pio.renderers.default = 'notebook_connected'

from datetime import datetime, timedelta

import ipywidgets as widgets
from ipywidgets import interact, Layout
from IPython.display import HTML, display, clear_output
from IPython.display import IFrame

import matplotlib.pyplot as plt

from scipy import optimize

from xgboost import XGBRegressor,XGBClassifier, DMatrix
from lightgbm import LGBMRegressor, LGBMClassifier, log_evaluation, early_stopping
import lightgbm as lgb

from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from sklearn.metrics import matthews_corrcoef,roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import HistGradientBoostingRegressor

from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap', ['blue', 'white', 'red']
)

import random

from tqdm import tqdm

from gc import collect
from colorama import Fore, Style, init;

# import GPy

import optuna
import shap

from optuna.samplers import TPESampler

from scipy import optimize

# ignore wornings
import warnings
warnings.filterwarnings("ignore")

# Get the execution mode of the Kaggle environment
run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive')


# Load the training and test data from CSV files

df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', index_col=0)
# df_original = pd.read_csv('/kaggle/input/depression-surveydataset-for-analysis/final_depression_dataset_1.csv')

print(f'N_train = {len(df_train)}, N_test = {len(df_test)}')

# Assign a new column 'train_test' with the value 'train' to the training dataset, and 'test' to the test dataset respectively
df_train['train_test'] = 'train'
df_test['train_test'] = 'test'

# Create reduced versions of the training and test datasets by randomly sampling 1/20th of the rows
df_train_reduced = df_train.sample(len(df_train) // 20)
df_test_reduced = df_test.sample(len(df_test) // 20)
print(f'N_train_reduced = {len(df_train_reduced)}, N_test_reduced = {len(df_test_reduced)}')

# Specify the target column for the analysis or model
target_col = 'num_sold'

# Combine the training and test datasets into a single DataFrame for unified processing
df_all = pd.concat([
    df_train,
    df_test,
    # Uncomment the lines below to include the reduced datasets if needed
    # df_train_reduced,
    # df_test_reduced
])

# Convert the 'date' column to datetime format for easier manipulation of date-related features
df_all['date'] = pd.to_datetime(df_all['date'])


# display counts of each labels
counts = df_train.groupby(['country', 'store', 'product'])['num_sold'].count()
counts.reset_index().pivot(
    index = ['country', 'store'], columns='product', values='num_sold'
).style.background_gradient( cmap='Greens', vmin = 0)


import requests
def get_gdp_per_capita(alpha3, year):
    url='https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json'
    response = requests.get(url.format(alpha3,year)).json()
    return response[1][0]['value']

# df = df_all[['date', 'country']].copy()
alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']
df_all['alpha3'] = df_all['country'].map(dict(zip(
    np.sort(df_all['country'].unique()), alpha3s)))
years = np.sort(df_all['date'].dt.year.unique())
df_all['year'] = df_all['date'].dt.year
gdp = np.array([
    [get_gdp_per_capita(alpha3, year) for year in years]
    for alpha3 in alpha3s
])
gdp = pd.DataFrame(gdp, index=alpha3s, columns=years)
df_all['GDP'] = df_all.apply(lambda s: gdp.loc[s['alpha3'], s['year']], axis=1)
df_all.drop('alpha3', axis = 1, inplace=True)


# Number of days in a year
def num_of_days(year):
    if year%4 == 0: 
        return 366 # 366 if it is leap year
    else:
        return 365

# Extract the year, month, day, weekday from the 'date' column
df_all['year'] = df_all['date'].dt.year
df_all['month'] = df_all['date'].dt.month
df_all['day'] = df_all['date'].dt.day
df_all['weeknum'] = df_all['date'].dt.isocalendar().week.astype(int)
df_all['weekday'] = df_all['date'].dt.weekday
df_all['dayofyear'] = df_all['date'].dt.dayofyear
df_all['day_total'] = (df_all['date'] - pd.to_datetime(df_all['date']).min()).dt.days + 1

# Remove the 'date' column as it is no longer needed after extracting the features
# del df_all['date']

df_all['day_sin' ] = np.sin(2 * np.pi * df_all['dayofyear'] / df_all['year'].map(num_of_days))
df_all['day_cos' ] = np.cos(2 * np.pi * df_all['dayofyear'] / df_all['year'].map(num_of_days))
df_all['day_sin2'] = np.sin(2 * np.pi * df_all['dayofyear'] / df_all['year'].map(num_of_days) * 2)
df_all['day_cos2'] = np.cos(2 * np.pi * df_all['dayofyear'] / df_all['year'].map(num_of_days) * 2)
df_all['day_sin3'] = np.sin(2 * np.pi * df_all['dayofyear'] / df_all['year'].map(num_of_days) * 3)
df_all['day_cos3'] = np.cos(2 * np.pi * df_all['dayofyear'] / df_all['year'].map(num_of_days) * 3)
df_all['day_sin4'] = np.sin(2 * np.pi * df_all['dayofyear'] / df_all['year'].map(num_of_days) * 4)
df_all['day_cos4'] = np.cos(2 * np.pi * df_all['dayofyear'] / df_all['year'].map(num_of_days) * 4)

df_all['day_sin1/2'] = np.sin(2 * np.pi * df_all['day_total'] / df_all['year'].map(num_of_days) / 2)
df_all['day_cos1/2'] = np.cos(2 * np.pi * df_all['day_total'] / df_all['year'].map(num_of_days) / 2)
df_all['day_sin1/4'] = np.sin(2 * np.pi * df_all['day_total'] / df_all['year'].map(num_of_days) / 4)
df_all['day_cos1/4'] = np.cos(2 * np.pi * df_all['day_total'] / df_all['year'].map(num_of_days) / 4)
df_all['is_weekend'] = 0
df_all.loc[df_all['weekday'].isin([6]), 'is_weekend'] = 0.5
df_all.loc[df_all['weekday'].isin([6]), 'is_weekend'] = 1
df_all['is_yearend'] = np.round(2.0 ** (df_all['dayofyear'] - df_all['year'].map(num_of_days)),5)

df_all['holyday'] = 0
# get holidays for each country
for country in df_all['country'].unique():
    country_class = getattr(holidays, country)
    country_holidays = country_class(years=df_all['year'])  # get_holydays
    # print(f"{country}:", [holiday[0] for holiday in list(country_holidays.items())])
    holidays_date = [holiday[0] for holiday in list(country_holidays.items())]
    df_all.loc[(df_all['country']==country) & df_all['date'].isin(holidays_date), 'holyday'] = 1


# exclude Canada and Kenya because of missed values
df_tmp = df_all[~df_all.country.isin(['Canada', 'Kenya'])]

store_df = df_tmp.groupby(by='store').num_sold.mean().rename('store_factor').to_frame()
# df = df.drop('store_factor', axis=1, errors='ignore').join(store_df, on='store', how='left')
df_all = df_all.drop('store_factor', axis=1, errors='ignore').join(store_df, on='store', how='left')


from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor

# exclude Canada and Kenya because of missed values
df_tmp = df_all[~df_all.country.isin(('Canada', 'Kenya'))].copy()

total = df_tmp.groupby(by='date').num_sold.sum().rename('num_sold_total')
df_tmp = df_tmp.join(total, on='date', how='left')
df_tmp['num_sold_ratio'] = df_tmp['num_sold'] / df_tmp['num_sold_total']

plt.figure(figsize=(24, 6))
df_all['product_factor'] = None
for product in df_all['product'].unique():
    df_tmp_date = df_tmp[(df_tmp['product'] == product) & (df_tmp['train_test'] == 'train')].groupby(by='date')
    x = df_tmp_date[['day_sin','day_cos','day_sin1/2','day_cos1/2']].mean().to_numpy()
    y = df_tmp_date.num_sold_ratio.sum().to_numpy()

    # reg = Ridge()
    reg = GaussianProcessRegressor()
    reg.fit(x, y)
    p = reg.predict(x)
    df_all.loc[(df_all['product'] == product), 'product_factor'] = reg.predict(df_all.loc[(df_all['product'] == product), ['day_sin','day_cos','day_sin1/2','day_cos1/2']].to_numpy())
   
#     plt.plot(y, label=f'{product} true', alpha=0.5)
#     plt.plot(p, label=f'{product} predict', alpha=0.5)
# plt.legend()
# plt.show();


num_sold_per_week_country_weekday = df_all.groupby(['weeknum', 'country', 'weekday'])['num_sold'].sum().reset_index().pivot(index=['weeknum', 'country'], columns='weekday')
ratio_sold_per_week_country_weekday = num_sold_per_week_country_weekday.apply(lambda row: row/sum(row), axis=1).reset_index()

ratio_weekday = pd.DataFrame(data=[[0, ]*len(df_all['country'].unique())]*7, columns=df_all['country'].unique() )
for country in df_all['country'].unique():
    for d in range(7):
        dt = ratio_sold_per_week_country_weekday.loc[ratio_sold_per_week_country_weekday['country'] == country, ('num_sold', d)]#[:-60]
        ratio_weekday.loc[d, country] = dt.median()

ratio_weekday_mean = ratio_weekday.mean(axis=1)
ratio_weekday['mean'] = ratio_weekday_mean

df_all['weekday_factor'] = df_all.weekday.map(ratio_weekday_mean)

# The total ratio taking into account all factors
df_all['ratio'] = df_all['GDP'] * df_all['product_factor'] * df_all['store_factor'] * df_all['weekday_factor']

# The total sold items taking into account all factors
df_all['total'] = df_all['num_sold'] / df_all['ratio']


# Exclude holidays
holiday_response_len=10
df_holidays = df_all.copy()
df_holidays['holiday_response'] = 0
for country in df_all['country'].unique():
    country_class = getattr(holidays, country)
    country_holidays = country_class(years=df_all['year'])  # get_holydays
    holidays_date = [holiday[0] for holiday in list(country_holidays.items())]
    for holiday in holidays_date:
        df_holidays.loc[
            (df_holidays.country==country) &
            df_holidays.date.isin(pd.date_range(holiday, periods=holiday_response_len)),
            'holiday_response'
        ] = 1
        
fig = plt.figure(figsize=(24,6))
data = pd.DataFrame()
for n, country in enumerate(df_all['country'].unique()):
    dt = df_holidays[(df_holidays['country']==country) & (df_holidays['holiday_response'] == 0)].groupby(['dayofyear']).total.median()
    data[country]= dt
    date = pd.to_datetime(f"2020-01-01")+ pd.to_timedelta(dt.index, unit='days')
    plt.plot(date, dt, label=country)
data['median'] = data.median(axis=1)

# Linear regression on fourier series
x = data.index.to_numpy()
y = data['median'].to_numpy()
fourier = lambda t: np.array([np.sin(2*np.pi/365*t), np.cos(2*np.pi/365*t)])


# year_ratio = Ridge(alpha=0.01).fit(fourier(x).T, y.T).predict(fourier(np.arange(1, 366)).T)
year_ratio = GaussianProcessRegressor(alpha=0.01).fit(fourier(x).T, y.T).predict(fourier(np.arange(1, 366)).T)

year_ratio = np.append(year_ratio, year_ratio[-1])

df_all['dayofyear_factor'] = df_all.dayofyear.map(dict(zip(np.arange(1, 367), year_ratio)))

# The total ratio taking into account all factors
df_all['ratio'] = df_all['GDP'] * df_all['product_factor'] * df_all['store_factor'] * df_all['weekday_factor'] * df_all['dayofyear_factor']

# The total sold items taking into account all factors
df_all['total'] = df_all['num_sold'] / df_all['ratio']
pd.to_datetime(f"2020-01-01")+ pd.to_timedelta(dt - 1, unit='D')

plt.plot(pd.to_datetime(f"2020-01-01")+ pd.to_timedelta(range(len(year_ratio)), unit='D') ,year_ratio, 'k', linewidth=4)
plt.legend();


fig = plt.figure(figsize=(24,6))
data = pd.DataFrame()
for n, country in enumerate(df_all['country'].unique()):
    dt = df_holidays[
        (df_holidays['train_test']=='train') &
        (df_holidays['country']==country) &
        (df_holidays['holiday_response'] == 0)
    ].groupby(['date'])['total'].median()
    data[country]= dt
    plt.plot(dt, label=country)
plt.legend()
data['median'] = data.median(axis=1)


sincos_col = ['day_sin', 'day_cos', 'day_sin2', 'day_cos2', 'day_sin3', 'day_cos3', 'day_sin4', 'day_cos4', 'day_sin1/2', 'day_cos1/2']

# Linear regression on fourier series
df_sc = df_all[df_all['train_test']=='train'].groupby('date')[sincos_col].mean()#.to_numpy()
df_sc['median'] = data['median']

x = df_sc[~pd.isna(df_sc['median'])][sincos_col].to_numpy()
y = df_sc[~pd.isna(df_sc['median'])]['median'].to_numpy()

# reg = Ridge(alpha=0.01, fit_intercept=True)
reg = GaussianProcessRegressor()
reg.fit(x, y)

fig = plt.figure(figsize=(24,6))
plt.plot(y, 'k')
plt.plot(reg.predict(x), 'r')

# df_all['sincos_factor'] = reg.intercept_ + (df_all[sincos_col] * reg.coef_).sum(axis=1)
df_all['sincos_factor'] = reg.predict(df_all[sincos_col])


# The total ratio taking into account all factors
df_all['ratio'] = df_all['GDP'] * df_all['product_factor'] * df_all['store_factor'] * df_all['weekday_factor'] * df_all['sincos_factor']

# The total sold items taking into account all factors
df_all['total'] = df_all['num_sold'] / df_all['ratio']

fig = plt.figure(figsize=(24,6))
for country in df_all['country'].unique():
    df_p = df_all[(df_all['country'] == country) & (df_all['product'] == 'Kaggle')].groupby('date').total.sum().to_numpy()
    plt.plot(df_p, label=country)

plt.legend();


country_factor = df_all[(df_all['product'] == 'Kaggle')].groupby('country')['total'].sum().rename('country_factor')
country_factor = country_factor / country_factor.median()
df_all = df_all.join(country_factor, on='country', how='left')


df_all['ratio'] = df_all['GDP'] * df_all['product_factor'] * df_all['store_factor'] * df_all['weekday_factor'] * df_all['sincos_factor'] * df_all['country_factor']

# The total sold items taking into account all factors
df_all['total'] = df_all['num_sold'] / df_all['ratio']

fig = plt.figure(figsize=(24,6))
for country in df_all['country'].unique():
    df_all_p = df_all[(df_all['country'] == country) & (df_all['product'] == 'Kaggle')].groupby('date')['total'].sum().to_numpy()
    plt.plot(df_all_p, label=country)

plt.legend();


cya_factor = df_all.groupby(['country', 'dayofyear'])['total'].mean().rename('cya_factor')
df_all = df_all.join(cya_factor, on=['country', 'dayofyear'], how='left')


df_all['ratio'] = df_all['GDP'] * df_all['product_factor'] * df_all['store_factor'] * df_all['weekday_factor'] * df_all['sincos_factor'] * df_all['country_factor'] * df_all['cya_factor']

df_all['total'] = df_all['num_sold'] / df_all['ratio']
const_factor = df_all['total'].median() * 1.06

df_all['prediction'] = const_factor * df_all['ratio']
mape_train = mean_absolute_percentage_error(
    df_all[
        (df_all['train_test'] == 'train') &
        (~df_all['num_sold'].isna())
    ]['num_sold'],
    df_all[
        (df_all['train_test'] == 'train')&
        (~pd.isna(df_all.num_sold))
    ]['prediction']
)
mape_train


df_all_pivot = df_all.pivot(
    index = 'date',
    columns = ['store','product','country'],
    values = [target_col,'GDP'])

df_all.loc[(df_all['num_sold'].isna())&(df_all['train_test']=='train'), 'num_sold'] = df_all.loc[(df_all['num_sold'].isna())&(df_all['train_test']=='train'),'prediction']


# Define a custom format function to format float values for display
def custom_format(x):
    if isinstance(x, float):
        # Format float values to 3 decimal places and remove trailing zeros and decimal points
        return ('{0:.3f}'.format(x)).rstrip('0').rstrip('.')
    return x

# Display a shortened version of the DataFrame with samples from each 'train_test' category
def display_short(df, n):    
    pd.set_option('display.max_colwidth', 10)  # Temporarily set max column width for display
    pd.set_option('display.max_columns', None)  # Display all columns

    df_disp = []
    for i, train_test in enumerate(df['train_test'].unique()):
        # Sample 'n' rows for each unique value in the 'train_test' column
        tmp = df[df['train_test'] == train_test].sample(n)
        tmp.index = pd.MultiIndex.from_product([[train_test], tmp.index])  # Set multi-level index
        df_disp.append(tmp)
        
    df_disp = pd.concat(df_disp)  # Combine the sampled data
    pd.set_option('display.max_colwidth', None)  # Reset max column width
    pd.set_option('display.max_columns', 20)  # Reset max columns to 20

# Display detailed information about the DataFrame, including statistics and metadata
def display_dfinfo(df, train_only = False):
    df_disp = []
    for tt in df['train_test'].unique():
        # Generate descriptive statistics for numeric columns
        tmp = df.loc[df['train_test'] == tt].describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        ).drop(['Target'], axis=1, errors='ignore')

        # Add skewness and kurtosis for numeric columns
        tmp.loc['skew'] = df.loc[df['train_test'] == tt].select_dtypes(include=[int, float]).skew()
        tmp.loc['kurtosis'] = df.loc[df['train_test'] == tt].select_dtypes(include=[int, float]).kurtosis()

        # Add data type and NaN count for all columns
        tmp.loc['dtype'] = df.loc[df['train_test'] == tt].dtypes
        tmp.loc['NaN count'] = df.loc[df['train_test'] == tt].isna().sum(axis=0)

        # For non-numeric columns, add metadata such as count and unique values
        for col in df.select_dtypes(exclude=[int, float]).columns:
            tmp.loc[:, col] = 0
            tmp.loc['count', col] = df.loc[df['train_test'] == tt, col].count()
            tmp.loc['dtype', col] = df.loc[df['train_test'] == tt, col].dtype
            tmp.loc['NaN count', col] = df.loc[df['train_test'] == tt, col].isna().sum(axis=0)

        tmp.loc['N unique'] = df.loc[df['train_test'] == tt].nunique()
        tmp.columns = pd.MultiIndex.from_tuples(
            [list(col) + [tt] for col in tmp.columns])  # Add multi-level columns
        df_disp.append(tmp)

    df_disp = pd.concat(df_disp, axis=1)  # Combine statistics for all 'train_test' groups
    df_disp = df_disp[df_disp.columns.get_level_values(0).unique()]  # Remove duplicate columns

    # Reorganize the DataFrame and filter relevant statistics
    df_disp = df_disp.T
    df_disp = df_disp.loc[
        :, [
            'count', 'NaN count', 'N unique', 'dtype',
            'mean', 'min', '5%', '25%', '50%', '75%', '95%', 'max',
            'std', 'skew', 'kurtosis'
        ]
    ]
    df_disp.sort_index(inplace=True)
    if train_only:
        df_disp = df_disp[df_disp.index.get_level_values(3) == 'train']
    formatter = {}
    # Display the DataFrame with custom formatting and background gradients for numeric stats
    display(
        df_disp.style.format(formatter=custom_format).background_gradient(
            subset=['mean', 'min', '5%', '25%', '50%', '75%', '95%', 'max'],
            cmap='Reds', axis=0
        )
    )

# Display histogram data for numeric and categorical columns
def display_plotdata(num_col1, cat_col):
    if num_col1 == cat_col:
        df_plot = df[[num_col1]].copy()
    else:
        df_plot = df[[num_col1, cat_col]].copy()

    df_plot[cat_col] = df_plot[cat_col].astype('object')  # Convert categorical column to object type
    try:
        # Compute descriptive statistics grouped by 'train_test'
        df_description = df.groupby('train_test').describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        )[num_col1]
        df_description['skew'] = df.groupby('train_test')[num_col1].skew()
        df_description['count'] = df_description['count'].astype(int)
        df_description['nunique'] = df.groupby('train_test')[num_col1].nunique()

        # Display the statistics with custom formatting and background gradients
        display(
            df_description.loc[
                :,
                [
                    'count', 'nunique', 'mean',
                    'min', '5%', '25%', '50%', '75%', '95%', 'max',
                    'std', 'skew'
                ]
            ].style.format(
                formatter=custom_format
            ).background_gradient(
                subset=[
                    'mean', 'min', '5%', '25%', '50%', '75%', '95%', 'max'
                ],
                cmap='Reds', axis=1
            )
        )
    except:
        # Handle cases where the numeric column has issues
        display(
            df_all.groupby('train_test')[num_col1]
            .agg(['count', 'nunique']).loc[['train', 'test']]
            .style.format(formatter=custom_format)
            .background_gradient(cmap='Blues')
        )


# Function to plot a correlation matrix for numeric columns
def plot_correlation_matrix(df, num_cols, plottype='sns'):
    if plottype == 'plotly':
        fig = px.imshow(
            df[num_cols].corr(), zmax=1, zmin=-1, color_continuous_scale='rdbu_r',  # Red-blue color scale
            text_auto=".2f"
        )
        # Customize the layout of the Plotly figure
        fig.update_layout(
            width=max(min(len(num_cols) * 80, 600), 300),
            height=max(min(len(num_cols) * 80, 600), 300),
            title='Correlation matrix'
        )
        fig.show()
    elif plottype == 'sns':
        plt.figure(figsize=(min(len(df.columns) * 0.6, 12), min(len(df.columns) * 0.15, 12)))
        sns.heatmap(
            df[num_cols].corr(), annot=True, vmax=1, vmin=-1,
            cmap=custom_cmap, fmt='.2f'  # Use a custom color map and format values
        )
        plt.title('Correlation Matrix')
        plt.show()




# Function to plot a histogram for a given numeric column, optionally grouped by a categorical column
def hist_df(df, num_col1, cat_col, n_data, displaytype='density', plottype='sns'):
    print('hist_df')  # Print a message to indicate the function is called

    # Sample the data if a specific number of rows is specified, otherwise use the full DataFrame
    if np.isnan(n_data):
        df_plot = df.copy()
    else:
        df_plot = df.sample(n_data)
    
    # Use Plotly for visualization if specified
    if plottype == 'plotly':
        if num_col1 != cat_col:  # If a categorical column is provided, color the histogram by category
            fig = px.histogram(
                df_plot,
                x=num_col1, marginal='violin', color=cat_col,
                nbins=50, histnorm=displaytype,  # Normalize the histogram (e.g., density)
                barmode='relative', opacity=0.5
            )
        else:  # If no categorical column, create a plain histogram
            fig = px.histogram(
                df_plot,
                x=num_col1,
                histnorm=displaytype,
                barmode='relative',
                opacity=0.5
            )
        # Customize the layout of the Plotly figure
        fig.update_layout(
            width=900, height=350,
            margin=dict(l=0, r=0, b=0, t=20),  # Adjust margins
            xaxis=dict(title_font=dict(size=20)),
            yaxis=dict(title_font=dict(size=20)),
            legend=dict(font=dict(size=15)),
        )
        fig.show()  # Display the Plotly figure

    # Use Seaborn for visualization if specified
    elif plottype == 'sns':
        if num_col1 != cat_col:  # If a categorical column is provided, create a colored histogram
            fig = sns.histplot(
                df_plot,
                x=num_col1, hue=cat_col,  # Group by the categorical column
                edgecolor=None, alpha=0.5
            )
            plt.show()
        else:  # If no categorical column, create a plain histogram
            fig = sns.histplot(
                df_plot,
                x=num_col1,
                edgecolor=None, alpha=0.5
            )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0 )
            plt.show()


# Function to create a scatter plot for two numeric columns, optionally grouped by a categorical column
def scatter_df(df, num_col1, num_col2, cat_col, n_data=10000, plottype='sns'):
    # Sample the data if a specific number of rows is specified, otherwise use the full DataFrame
    if np.isnan(n_data):
        df_plot = df.copy()
    else:
        df_plot = df.sample(n_data)
    
    # Create a temporary DataFrame with the selected columns
    df_tmp = df_plot[[num_col1, num_col2, cat_col]]
    # Add spaces to column names to avoid conflicts with Plotly or Seaborn
    df_tmp.columns = [c + ' ' * i for i, c in enumerate(df_tmp.columns)]

    # Use Plotly for visualization if specified
    if plottype == 'plotly':
        fig = px.scatter(
            df_tmp,
            x=num_col1 + '', y=num_col2 + ' ', color=cat_col + '  ',  # Scatter plot with color grouping
            marginal_x='histogram', marginal_y='violin', opacity=0.2,  # Add marginal plots
            color_continuous_scale=px.colors.sequential.Rainbow,  # Use a rainbow color scale
            trendline='ols'  # Add a trendline
        )
        # Customize the layout of the Plotly figure
        fig.update_layout(
            width=750, height=550,
            margin=dict(l=0, r=0, b=0, t=0),  # Adjust margins
            xaxis1=dict(domain=[0.1, 0.75]),  # Specify the area of the x-axis for the main plot
            yaxis1=dict(domain=[0.1, 0.8]),  # Specify the area of the y-axis for the main plot
            xaxis2=dict(domain=[0.76, 1.0]),  # Specify the area of the x-axis for marginal plots
            yaxis2=dict(domain=[0.1, 0.8]),  # Specify the area of the y-axis for marginal plots
            xaxis3=dict(domain=[0.1, 0.75]),  # Specify the area of the x-axis for marginal plots
            yaxis3=dict(domain=[0.81, 1.0]),  # Specify the area of the y-axis for marginal plots
            xaxis=dict(title_font=dict(size=20)),
            yaxis=dict(title_font=dict(size=20)),
            legend=dict(font=dict(size=15)),
        )
        fig.show()  # Display the Plotly figure

    # Use Seaborn for visualization if specified
    elif plottype == 'sns':
        # If the categorical column is an object or has fewer than 10 unique values, use a JointGrid
        if (df_tmp[cat_col + '  '].dtype == object) or (df_tmp[cat_col + '  '].nunique() <= 10):
            g = sns.JointGrid(
                data=df_tmp,
                x=num_col1 + '', y=num_col2 + ' ', hue=cat_col + '  ',  # Scatter plot with color grouping
                palette='rainbow'
            )
            # Plot the scatterplot in the center
            g.plot_joint(sns.scatterplot, color="blue")
            # Plot KDE (Kernel Density Estimation) on the margins
            try:
                g.plot_marginals(sns.kdeplot, color="skyblue", alpha=0.4, fill=True)
            except: pass
            g.ax_joint.legend(bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0 )
            plt.tight_layout()
            plt.show()
        else:
            # Create a standard scatter plot for non-categorical grouping
            try:
                fig = sns.scatterplot(
                    data=df_tmp,
                    x=num_col1 + '', y=num_col2 + ' ', hue=cat_col + '  ',
                    palette='rainbow'
                )
            except:
                fig = sns.scatterplot(
                    data=df_tmp,
                    x=num_col1 + '', y=num_col2 + ' ',
                    palette='rainbow'
                )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0 )
            plt.show()

# Function to create a violin plot for two numeric columns, optionally grouped by a categorical column
def violin_df(df, num_col1, num_col2, cat_col, n_data=10000, plottype='sns'):
    # Sample the data if a specific number of rows is specified, otherwise use the full DataFrame
    if np.isnan(n_data):
        df_plot = df.copy()
    else:
        df_plot = df.sample(n_data)
    
    # Check if the categorical column is one of the numeric columns
    col_in_num = cat_col in [num_col1, num_col2]
    if col_in_num:
        df_tmp = df_plot[[num_col1, num_col2]]  # Exclude the categorical column if it's numeric
    else:
        df_tmp = df_plot[[num_col1, num_col2, cat_col]]
        df_tmp[cat_col] = df_tmp[cat_col].astype(object)  # Convert the categorical column to object type

    # Use Plotly for visualization if specified
    if plottype == 'plotly':
        if df_tmp[num_col1].dtype in [int, float]:
            bw = (df_tmp[num_col1].max() - df_tmp[num_col1].min())/50
            plot_height = min(1200, 120 * df_tmp[num_col1].nunique())
            plot_width = 450
        else:
            bw = (df_tmp[num_col2].max() - df_tmp[num_col2].min())/50
            plot_height = 450
            plot_width = min(1200, 120 * df_tmp[num_col1].nunique())
        
        if col_in_num or df_tmp[cat_col].nunique() > 10:  # Skip coloring if too many unique categories
            fig = px.violin(
                df_tmp, x=num_col1, y=num_col2,
            )
            
        else:  # Include coloring by the categorical column
            fig = px.violin(
                df_tmp, x=num_col1, y=num_col2, color=cat_col,violinmode = 'group'
            )
            plot_width+=120

        # Customize the layout of the Plotly figure
        fig.update_layout(
            width = plot_width,
            height = plot_height,
            margin = dict(l=0, r=0, b=0, t=20),  # Adjust margins
            xaxis = dict(title_font=dict(size=20)),
            yaxis = dict(title_font=dict(size=20)),
            legend = dict(font=dict(size=12)),
        )
        # unify the violin width
        fig.update_traces(
            scalemode='width',
            points=False,
            bandwidth=bw,    
            width=0.95
        )

        fig.show()

    # Use Seaborn for visualization if specified
    elif plottype == 'sns':
        fig = plt.figure(figsize=(max(min(df_tmp[num_col1].nunique() * df_tmp[cat_col].nunique() / 2,10),4),4))
        if col_in_num or df_tmp[cat_col].nunique() > 10:  # Skip coloring if too many unique categories
            sns.violinplot(
                df_tmp, x=num_col1, y=num_col2,
                linewidth=0.1
            )
        else:  # Include coloring by the categorical column
            sns.violinplot(
                df_tmp, x=num_col1, y=num_col2, hue=cat_col,
                linewidth=0.5
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0 )
        plt.show()


# Function to create a cross-tabulation or pivot table and visualize it
def cross_df(df, num_col2, num_col1, cat_col, n_data, plottype='sns'):
    if df[cat_col].dtype == object:  # If the categorical column is an object
        df_plot = pd.crosstab(df_all[num_col1], df_all[num_col2])  # Create a cross-tabulation
        label = cat_col
    else:  # If the categorical column is numeric
        df_plot = pd.pivot(
            df_all.groupby([num_col1, num_col2])[cat_col].mean().reset_index(),
            columns=num_col1, index=num_col2, values=cat_col
        )
        label = 'count'
    
    # Use Plotly for visualization if specified
    if plottype == 'plotly':
        fig = px.imshow(
            df_plot,
            color_continuous_scale='blues',  # Use a blue color scale
            labels={"color": cat_col}
        )
        # Customize the layout of the Plotly figure
        fig.update_layout(
            width=950, height=550,
            margin=dict(l=0, r=0, b=0, t=0),  # Adjust margins
            xaxis=dict(title_font=dict(size=20)),
            yaxis=dict(title_font=dict(size=20)),
            legend=dict(font=dict(size=12)),
        )
        fig.show()

    # Use Seaborn for visualization if specified
    elif plottype == 'sns':
        fig = sns.heatmap(
            df_plot,
            cmap='coolwarm',  # Use a cool-warm color scale
            annot=True,  # Annotate cells with data values
            fmt = 'd'
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0 )
        plt.show()

# Function to create an interactive plot with dropdowns for column selection
def plot_interact(df, num_col, cat_col, button_width=100, n_data=np.nan, plottype='sns'):
    # Create dropdowns for selecting numeric and categorical columns
    num_col_button1 = widgets.Dropdown(
        options=num_col,
        button_style='info',
        description='X'
    )
    num_col_button1.style.description_width = '10px'
    num_col_button1.style.font_size = '1px'

    num_col_button2 = widgets.Dropdown(
        options=num_col,
        value = target_col,
        button_style='primary',
        description='Y',
    )
    num_col_button2.style.button_width = f'{button_width}px'
    num_col_button2.style.description_width = '10px'

    cat_col_button = widgets.Dropdown(options=cat_col, button_style='warning', description='color')
    cat_col_button.style.button_width = '80px'
    cat_col_button.style.description_width = '50px'

    df.loc[:,df.nunique()<15] = df.loc[:,df.nunique()<15].astype(str)

    # Create an interactive function to update the plot based on user selections
    @interact(
        num_col1=num_col_button1, num_col2=num_col_button2,
        cat_col=cat_col_button
    )
    def plot_df(num_col1, num_col2, cat_col):
        clear_output()  # Clear the output to avoid clutter
        print(f'type({num_col1}):{df[num_col1].dtype}, type({num_col2}):{df[num_col2].dtype}')
        
        if num_col1 == num_col2:  # If the same column is selected for X and Y, plot a histogram
            hist_df(df, num_col1, cat_col, n_data, plottype=plottype)
        elif (df[num_col1].dtype != object) and (df[num_col2].dtype != object):  # Scatter plot for numeric columns
            scatter_df(df, num_col1, num_col2, cat_col, n_data, plottype=plottype)
        elif df[num_col1].dtype != object:  # Violin plot for numeric vs categorical
            violin_df(df, num_col1, num_col2, cat_col, n_data, plottype=plottype)
        elif df[num_col2].dtype != object:  # Violin plot for categorical vs numeric
            violin_df(df, num_col1, num_col2, cat_col, n_data, plottype=plottype)
        else:  # Cross-tabulation for categorical columns
            cross_df(df, num_col2, num_col1, cat_col, n_data, plottype=plottype)


# Function to calculate the date of the Monday for a given ISO year and week
def get_date_from_week(year, month, week):
    year = int(year)
    month = int(month)
    if week >= 40 and month == 1:
        year -= 1
    elif week <= 10 and month == 12:
        year += 1
    # January 4th is always in the first ISO week of the year
    jan4 = datetime(int(year), 1, 4)
    # Find the Monday of the first ISO week
    start_of_week = jan4 - timedelta(days=jan4.weekday())
    # Calculate the Monday of the specified week
    return start_of_week + timedelta(weeks=int(week) - 1)

def plot_timeseries(df, plottype='sns'):
    
    # Create dropdowns for selecting numeric and categorical columns
    col1_dropdown = widgets.Dropdown(
        options=cat_cols[:3],
        button_style='info',
        description='col1 : '
    )
    col1_dropdown.style.description_width = '60px'
    smooth = 'daily'
    @interact(col1=col1_dropdown)
    def plot_timeseries1(col1):
        value1_dropdown = widgets.Dropdown(
            options=df[col1].unique(),
            button_style='info',
            description='value1 : '
        )
        value1_dropdown.style.description_width = '100px'
        
        @interact(value1 = value1_dropdown)
        def plot_timeseries2(value1):
            col2_dropdown = widgets.Dropdown(
                options=[c for c in cat_cols[:3] if c != col1],
                button_style='info',
                description='col2 : '
            )
            col2_dropdown.style.description_width = '60px'
    
            @interact(col2 = col2_dropdown)
            def plot_timeseries3(col2):
                value2_dropdown = widgets.Dropdown(
                    options=df[col2].unique(),
                    button_style='info',
                    description='value2 : '
                )
                value2_dropdown.style.description_width = '100px'

                smooth_dropdown = widgets.Dropdown(
                    options=['daily', 'weekly', 'monthly', 'yearly'],
                    value= smooth,
                    button_style='info',
                    description='smoothing : '
                )
                smooth_dropdown.style.description_width = '150px'
    
                @interact(value2 = value2_dropdown, smooth = smooth_dropdown)
                def plot_timeseries4(value2, smooth):
                    col3 = [c for c in cat_cols[:3] if not c in [col1,col2]][0]

                    df_plot = df_all.loc[
                        (df_all[col1] == value1)&(df_all[col2]==value2),
                        ['date', col3, target_col,'year','month','weeknum','GDP']
                        ]
                    if smooth == 'yearly':
                        smooth_name= 'Year'
                        df_plot['time'] = pd.to_datetime(df_plot[['year']].assign(month = 1, day=1))
                    elif smooth == 'monthly':
                        smooth_name = 'Month'
                        df_plot['time'] = pd.to_datetime(df_plot[['year','month']].assign(day=1))
                    elif smooth == 'weekly':
                        smooth_name = 'Week'
                        df_plot['time'] = df_plot.apply(lambda row: get_date_from_week(row['year'], row['month'], row['weeknum']), axis=1)
                    elif smooth == 'daily':
                        smooth_name = 'Day'
                        df_plot['time'] = df_plot['date']

                    df_gdp = df_plot.pivot_table(index = 'time', columns = col3, values = 'GDP', aggfunc = 'mean')
                    df_plot = df_plot.pivot_table(index = 'time', columns = col3, values = 'num_sold', aggfunc = 'sum')
                    
                    for c in df_plot:
                        df_plot.loc[df_plot[c]==0,c] = np.nan
                    df_plot.dropna(how='all', inplace=True)
                    df_plot.index.name = 'Date'
                    if plottype == 'sns':
                        fig = plt.figure(figsize = (12,4))
                    
                        sns.lineplot(df_plot, lw = 1)
                        # plt.xlim(pd.Timestamp(2010,1,1), pd.Timestamp(2017,1,1));plt.grid()
                        plt.title(f'{col1} : {value1},  {col2}:{value2}, {smooth_name}')
                        plt.show()

                    else:
                        fig = px.line(df_plot)
                        fig.for_each_trace(lambda trace: trace.update(opacity=0.5))
                        # Customize the layout of the Plotly figure
                        fig.update_layout(
                            height=350,
                            margin=dict(l=0, r=0, b=0, t=20),  # Adjust margins
                            xaxis=dict(title_font=dict(size=20)),
                            yaxis=dict(title_font=dict(size=20)),
                            legend=dict(font=dict(size=15)),
                            title=dict(text=f'{col1} : {value1},  {col2}:{value2}, {smooth_name}',y=0.99,x= 0.4,xanchor='center')
                        )
                        # add gdp plots
                        if col3 == 'country':
                            fig.add_traces([
                                go.Scatter(
                                    x = df_gdp.index,
                                    y = df_gdp[c]/df_gdp.values.mean()*np.nanmean(df_plot.values),
                                    mode='lines',  # line
                                    line=dict(color='black', dash='dot', width=1),
                                    name=c  # label
                                ) for c in df_gdp.columns])

                        fig.show()


df_display = df_all_pivot['num_sold'].copy()
df_display.loc[df_display.index < pd.Timestamp(2017,1,1),'train_test'] = 'train'
df_display.loc[df_display.index >= pd.Timestamp(2017,1,1),'train_test'] = 'test'

display(df_display[df_display['train_test']=='train'].sample(3))
display_dfinfo(df_display, train_only = True)
num_cols = df_all.select_dtypes(include=[float, int]).columns
disp_cols = [c for c in df_all.columns if c!= 'date']
cat_cols = [c for c in df_all.columns if c!= 'date']
plot_correlation_matrix(df_all, num_cols,  plottype='plotly')


for col in df_all.columns:
    df_all[col] = df_all[col].astype(float, errors='ignore')
# ipywidgets　will run only in interactive mode
if run_type == 'Interactive':
    plot_interact(df_all, disp_cols, cat_cols, 107, n_data=5000, plottype='plotly')


# ipywidgets　will run only in interactive mode
if run_type == 'Interactive':
    plot_timeseries(df_all, plottype='plotly')


from statsmodels.graphics.tsaplots import plot_acf

# Aggregate data by day to create the time series
trend_data = df_all.dropna(subset = 'num_sold', axis = 0).groupby('date')['num_sold'].sum()

plt.figure(figsize=(10, 4))
plot_acf(trend_data.dropna(), lags=50)
plt.title('Autocorrelation Plot')
plt.show()


def safe_transform(encoder, labels):
    known_labels = set(encoder.classes_)
    return [
        encoder.transform([label])[0] if label in known_labels else -1 for label in labels
    ]

def target_encoder(df, input_col, target_col):
    tmp = df[[input_col, target_col]]
    means = df.groupby(input_col)[target_col].mean()
    for ind in means.index:
        tmp.loc[tmp[f'{input_col}']==ind, f'{input_col}_te'] = means[ind]

    return tmp[f'{input_col}_te'].values

def preprocessing(df, num_cols, cat_cols, target_col,train_test = 'train_test'):
    df_pp = df[num_cols].copy()
    for i, cat_col in enumerate(cat_cols):
        print(cat_col, end = ' / ')
        # target encoding
        df_pp[f'{cat_col}_te'] = target_encoder(df, cat_col, target_col)
        
    print()
    df_pp[target_col] = df[target_col]
    df_pp[train_test] = df[train_test]
    return df_pp

def adversarial_validation(df_adv):
#     Return a list of train data indistinguishable from test data
    xgb = XGBClassifier()
    X_adv = df_adv.drop('train_test',axis = 1)
    y_adv = df_adv['train_test'].map({'train':0,'original':0, 'test':1})
    
    xgb.fit(X_adv, y_adv)
    predict_adv = pd.DataFrame(
        xgb.predict_proba(X_adv.loc[y_adv==0])[:,0], columns=['train'],
        index = X_adv.index[y_adv==0]
    )
    predict_adv.sort_values(by='train',inplace = True)
    return predict_adv.index


df_all.select_dtypes(include=[int,float]).columns


df_all[target_col] = df_all[target_col].astype(float)
# cat_cols = list(df_all.select_dtypes(include='object').columns)
# cat_cols.remove('train_test')
cat_cols = ['country', 'store', 'product']
num_cols = ['num_sold', 'GDP','year', 'day', 'weeknum', 'dayofyear', 'day_total',
       'day_sin', 'day_cos', 'day_sin2', 'day_cos2', 'day_sin3', 'day_cos3',
       'day_sin4', 'day_cos4', 'day_sin1/2', 'day_cos1/2', 'day_sin1/4',
       'day_cos1/4', 'is_yearend', 'product_factor', 'ratio', 'total',
       'dayofyear_factor', 'sincos_factor', 'cya_factor', 'prediction']
print('Preprocessing start', end=' → ')
df_all_pp = preprocessing(
    df_all,
    num_cols, cat_cols,
    target_col
)


# Convert float64 to float32 and int64 to int32 for memory efficiency.
for col in df_all_pp.columns:
    if df_all_pp[col].dtype == object:
        try:
            df_all_pp[col] = df_all_pp[col].astype('float32')
        except:
            pass
    elif df_all_pp[col].isnull().any():
        df_all_pp[col].astype(float)
    elif df_all_pp[col].eq(df_all_pp[col].astype(int)).all():
        df_all_pp[col] = df_all_pp[col].astype('int32')
    else:
        df_all_pp[col] = df_all_pp[col].astype('float32')

print('cat_cols = ', df_all_pp.select_dtypes(include='object').columns)
print('num_cols = ', df_all_pp.select_dtypes(include=[int, float]).columns)


df_all_pp[target_col]=df_all_pp[target_col].astype(float)

input_coaggregatels = list(df_all_pp.columns)
# input_cols = list(df_all_pp.columns)
# input_cols.remove(target_col)
input_cols = [
    'year', 'GDP', 'day', 'weeknum', 'dayofyear', 'day_total',
    'is_yearend', 'product_factor', 'total',
   'dayofyear_factor', 'sincos_factor', 'cya_factor', 'prediction',
]

df_target = df_all_pp.loc[:, input_cols + [target_col, 'train_test']].copy()
df_target.drop(
    df_target.index[
        (df_target['train_test'] == 'train') &
        (df_target[target_col].isna())
    ], axis=0, inplace = True
)

train_data = df_target.loc[
    df_target['train_test'].isin(['train','original'])
].drop('train_test', axis=1)
test_data =  df_target.loc[
    df_target['train_test']=='test'
].drop('train_test', axis=1)

# valid_indices = adversarial_validation(df_target.drop(target_col, axis = 1))
# valid_indices = valid_indices[:round(len(valid_indices)*0.2)]

# X_train = train_data.drop(target_col,axis=1).drop(valid_indices)
# X_val = train_data.drop(target_col,axis=1).loc[valid_indices]
X_train = train_data[train_data['year'] < 2016].drop(target_col,axis=1)
X_val = train_data[train_data['year'] == 2016].drop(target_col,axis=1)
X_test = test_data.drop(target_col, axis = 1)

# y_train = train_data[target_col].drop(valid_indices)
# y_val = train_data[target_col].loc[valid_indices]
y_train = train_data.loc[train_data['year'] < 2016 , target_col]
y_val = train_data.loc[train_data['year'] == 2016, target_col]


def bayese_objective(X, y, Regressor, metric):
    def bayese_trial(trial):
        if Regressor == XGBRegressor:
            params = {
                'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                'gamma' : trial.suggest_float('gamma', 1e-9, 0.5),
                'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'max_depth': trial.suggest_int('max_depth', 0, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),
                
                'random_state': 42,
                'booster':'gbtree',
                'device':"cuda",
                'verbosity': 0,
                'tree_method':"hist",
                'eval_metric': metrics['XGB'],
            }
        elif Regressor == LGBMRegressor:
            params = {
                "n_estimators": trial.suggest_int('n_estimators', 50, 1000, step=10),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                "max_depth": trial.suggest_int('max_depth', 3, 15),
                "min_child_samples": trial.suggest_int('lgbm_min_child_samples', 1, 20),
                "subsample": trial.suggest_float('subsample', 0.5, 1.0),
                "colsample_bytree": trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                
                'random_state': 42,
                'verbose':-1,
                'metric': metrics['LGBM']
            }
        elif Regressor == CatBoostRegressor:
            params = {
                "iterations": trial.suggest_int('iterations', 50, 1000, step=10),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                "depth": trial.suggest_int('depth', 3, 15),
                "l2_leaf_reg": trial.suggest_float('l2_leaf_reg', 1e-3, 1),
                
                "random_state": 42,                
                "verbose": False,
                'eval_metric': metrics['CatBoost'], 
            }
        # cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

        # cv_splits = cv.split(X, y = y)
        cv_scores = list()

        for year in range(2012,2016):
            model = Regressor()
            model.set_params(**params)
            train_idx = X.loc[X['year']<year].index
            val_idx = X.loc[X['year']==year].index
            
            X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
            y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
            model.fit(X_train_fold, y_train_fold)
            
            # y_val_prob = model.predict_proba(X_val_fold)[:,1]
            # fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_prob)

            # score = auc(fpr, tpr)
            y_val_prob = model.predict(X_val_fold)
            score = mean_absolute_percentage_error(y_val_fold, np.abs(y_val_prob))

            cv_scores.append(score)
        return np.mean(cv_scores)
    return bayese_trial


%%time
best_params1 = []
best_scores1 = []
_optim = False

metrics = {
    'XGB': 'mape',
    'LGBM': 'mape',
    'CatBoost': 'MAPE'
}
models = {
    'XGB':XGBRegressor,
    'LGBM':LGBMRegressor,
    'Cat':CatBoostRegressor
}
if _optim:# run Bayese Optimization
    for key, model in models.items():
        print(f'{key}:')
        try:
            study = optuna.create_study(
                direction = 'minimize',
                sampler=optuna.samplers.TPESampler(seed=0),
                study_name=f"{key}_study", storage=f"sqlite:///{key}_study.db", load_if_exists=True
            )
                        
            study.optimize(
                bayese_objective(X_train, y_train, model, metrics),
                n_trials=300, timeout=3600 * 3, n_jobs = -1
            )

            best_params1.append(study.best_trial.params)
            best_scores1.append(study.best_trial.value)
            print(f'{key}:')
            print('best params:')
            print(best_params1[-1])
            print('best scores:')
            print(best_scores1[-1])
        except:
            print(f'{model} failed')

else:
    best_params1 = {
        'XGB':{
            'grow_policy': 'lossguide', 'n_estimators': 503, 'learning_rate': 0.02235775693867936, 'gamma': 0.4790660344297994, 'subsample': 0.8361979341137616, 'colsample_bytree': 0.9608415574978083, 'max_depth': 8, 'min_child_weight': 5, 'reg_lambda': 0.009833416500446338, 'reg_alpha': 0.011191899596666454,
            
            'random_state':42,
            'booster':'gbtree',
            'device':"cuda",
            'verbosity': 0,
            'tree_method':"hist",
            'eval_metric': metrics['XGB'],
        },'LGBM':{
            'n_estimators': 830, 'learning_rate': 0.012813594560112142, 'max_depth': 11, 'lgbm_min_child_samples': 6, 'subsample': 0.6721298148758514, 'colsample_bytree': 0.9782920568955418, 'num_leaves': 248,
            
            'random_state': 42,
            'verbose':-1,
            'metric': metrics['LGBM']
        },'CatBoost':{
            'iterations': 820, 'learning_rate': 0.02719359410776897, 'depth': 8, 'l2_leaf_reg': 0.003103253903191369,
            
            'random_state': 42,
            "verbose": False,
            'eval_metric': metrics['CatBoost'],
        }}

    print('Parameters are defined.')


%%time
models_best = [XGBRegressor(), LGBMRegressor(), CatBoostRegressor()]
# predict_cols = ['XGB','LGBM', 'CatBoost']
predict_cols = ['XGB','LGBM']

predict_trains = []
predict_vals = []
predict_tests = []
for i,model in enumerate(models_best[:2]):
    print(predict_cols[i], end = ' / ')
    model.set_params(**best_params1[predict_cols[i]])
    model.fit(X_train, y_train)
    predict_trains.append(model.predict(X_train).flatten())
    predict_vals.append(model.predict(X_val).flatten())
    predict_tests.append(model.predict(X_test).flatten())

predict_trains = pd.DataFrame(
    np.array(predict_trains).T, columns = predict_cols, index=X_train.index
)
predict_vals = pd.DataFrame(
    np.array(predict_vals).T, columns = predict_cols, index = X_val.index
)
predict_tests = pd.DataFrame(
    np.array(predict_tests).T, columns = predict_cols, index = X_test.index
)
print('Fitting finish')


ohes = {}
df_all_pp_LR = df_all.copy()

for col in ['country', 'store', 'product']:
    ohes[col] = OneHotEncoder(sparse=False)
    df_one_tmp = ohes[col].fit_transform(df_all_pp_LR[[col]])
    df_ohe_tmp = pd.DataFrame(df_one_tmp,index=df_all_pp_LR.index, columns=ohes[col].get_feature_names_out())
    df_all_pp_LR = pd.concat([df_all_pp_LR, df_ohe_tmp], axis=1)

# Convert float64 to float32 and int64 to int32 for memory efficiency.
for col in df_all_pp_LR.columns:
    if df_all_pp_LR[col].dtype == object:
        try:
            df_all_pp_LR[col] = df_all_pp_LR[col].astype('float32')
        except:
            pass
    elif df_all_pp_LR[col].isnull().any():
        df_all_pp_LR[col].astype(float)
    elif df_all_pp_LR[col].eq(df_all_pp_LR[col].astype(int)).all():
        df_all_pp_LR[col] = df_all_pp_LR[col].astype('int32')
    else:
        try:
            df_all_pp_LR[col] = df_all_pp_LR[col].astype('float32')
        except:pass

print('cat_cols = ', df_all_pp_LR.select_dtypes(include='object').columns)
print('num_cols = ', df_all_pp_LR.select_dtypes(include=[int, float]).columns)



df_all_pp_LR['bias'] = 1
df_all_pp_LR[target_col]=df_all_pp_LR[target_col].astype(float)

input_coaggregatels = list(df_all_pp.columns)
# input_cols_LR = list(df_all_pp.columns)
# input_cols_LR.remove(target_col)
input_cols_LR = [
    'country', 'store', 'product',
    'year', 'GDP', 'day', 'weeknum', 'dayofyear', 'day_total',
    'is_yearend', 'product_factor', 
   'dayofyear_factor', 'sincos_factor', 'cya_factor', 'prediction',
]

df_target_LR = df_all_pp_LR.loc[:, input_cols_LR + [target_col, 'train_test']].copy()
df_target_LR.drop(
    df_target_LR.index[
        (df_target_LR['train_test'] == 'train') &
        (df_target_LR[target_col].isna())
    ], axis=0, inplace = True
)

ss = StandardScaler()
tmp = df_target_LR.select_dtypes(include=[int, float]).drop(target_col, axis=1).copy()
tmp = pd.DataFrame(ss.fit_transform(tmp), index = tmp.index, columns = tmp.columns)
df_target_LR[tmp.columns] = tmp


train_data_LR = df_target_LR.loc[
    df_target_LR['train_test'].isin(['train','original'])
].drop('train_test', axis=1)
test_data_LR =  df_target_LR.loc[
    df_target_LR['train_test']=='test'
].drop('train_test', axis=1)

# valid_indices = adversarial_validation(df_target_LR.drop(target_col, axis = 1))
# valid_indices = valid_indices[:round(len(valid_indices)*0.2)]

# X_train_LR = train_data_LR.drop(target_col,axis=1).drop(valid_indices)
# X_val_LR = train_data_LR.drop(target_col,axis=1).loc[valid_indices]

X_train_LR = train_data_LR[train_data_LR['year'] < 0.504].drop(target_col,axis=1)
X_val_LR = train_data_LR[train_data_LR['year'] >= 0.504].drop(target_col,axis=1)
X_test_LR = test_data_LR.drop(target_col, axis = 1)

# y_train_LR = train_data_LR[target_col].drop(valid_indices)
# y_val_LR = train_data_LR[target_col].loc[valid_indices]
y_train_LR = train_data_LR.loc[train_data_LR['year'] < 0.504 , target_col]
y_val_LR = train_data_LR.loc[train_data_LR['year'] >= 0.504, target_col]


lr_models = {}
y_predict_train_LR = pd.DataFrame(index = y_train_LR.index, columns=[target_col])
y_predict_val_LR = pd.DataFrame(index = y_val_LR.index, columns=[target_col])
y_predict_test_LR = pd.DataFrame(index = X_test_LR.index, columns=[target_col])

for country in X_train_LR['country'].unique():
    for store in X_train_LR['store'].unique():
        for product in X_train_LR['product'].unique():
            csp_name = country+store+product
            lr_models[csp_name] = LinearRegression()
            pick_train = X_train_LR.index[(X_train_LR['country'] == country) & (X_train_LR['store'] == store) & (X_train_LR['product'] == product)]
            pick_val = X_val.index[(X_val_LR['country'] == country) & (X_val_LR['store'] == store) & (X_val_LR['product'] == product)]
            pick_test = X_test_LR.index[(X_test_LR['country'] == country) & (X_test_LR['store'] == store) & (X_test_LR['product'] == product)]
            X_pick = X_train_LR.loc[pick_train]
            y_pick = y_train_LR.loc[pick_train]
            try:
                # lr_models[csp_name].fit(X_pick.drop(['country','store','product'],axis=1), np.log(y_pick))
                # y_predict_train_LR.loc[pick_train, target_col] = np.exp(lr_models[csp_name].predict(X_pick.drop(['country','store','product'], axis=1)))
                # y_predict_val_LR.loc[pick_val,target_col] = np.exp(lr_models[csp_name].predict(X_val_LR.loc[pick_val].drop(['country','store','product'], axis=1)))
                # y_predict_test_LR.loc[pick_test,target_col] = np.exp(lr_models[csp_name].predict(X_test_LR.loc[pick_test].drop(['country','store','product'], axis=1)))
                lr_models[csp_name].fit(X_pick.drop(['country','store','product'],axis=1), y_pick)
                y_predict_train_LR.loc[pick_train, target_col] = lr_models[csp_name].predict(X_pick.drop(['country','store','product'], axis=1))
                y_predict_val_LR.loc[pick_val,target_col] = lr_models[csp_name].predict(X_val_LR.loc[pick_val].drop(['country','store','product'], axis=1))
                y_predict_test_LR.loc[pick_test,target_col] = lr_models[csp_name].predict(X_test_LR.loc[pick_test].drop(['country','store','product'], axis=1))
            except:
                print(country, ' / ', store, ' / failed')


predict_trains['LR'] = y_predict_train_LR
predict_vals['LR'] = y_predict_val_LR
predict_tests['LR'] = y_predict_test_LR
predict_cols += ['LR']


predict_cols = predict_cols + ['blend']

predict_trains['blend'] = predict_trains.mean(axis=1)
predict_vals['blend'] = predict_vals.mean(axis=1)

predict_trains['True'] = y_train
predict_trains['train_val'] = 'train'
predict_vals['True'] = y_val
predict_vals['train_val'] = 'val'


print('Plot results.')
target_max = max(predict_trains.max()[:-1].max(), predict_vals.max()[:-1].max())
target_min = min(np.abs(predict_trains.min()[:-1]).min(), np.abs(predict_vals.min()[:-1]).min())
fig, ax = plt.subplots(nrows = 2, ncols = len(predict_cols), figsize = (len(predict_cols)*3,6))
for i, col in enumerate(predict_cols):
    cc = mean_absolute_percentage_error(predict_trains['True'], np.abs(predict_trains.iloc[:,i]))
    ax[0][i].scatter(predict_trains['True'],predict_trains.iloc[:,i], s = 3, alpha=0.005)    
    ax[0][i].set_title(f'{col} Train MAPE={cc:.3}', fontdict={'size':12})
    ax[0][i].set_xlabel('True'); ax[0][i].set_ylabel('Predict');
    ax[0][i].set_xlim(target_min, target_max); ax[0][i].set_ylim(target_min, target_max)
    ax[0][i].set_aspect(1)
    ax[0][i].grid()
    ax[0][i].set_xscale('log'); ax[0][i].set_yscale('log')
    
    cc = mean_absolute_percentage_error(predict_vals['True'], np.abs(predict_vals.iloc[:,i]))
    ax[1][i].scatter(predict_vals['True'],predict_vals.iloc[:,i], s = 3, alpha=0.01)
    ax[1][i].set_title(f'{col} Valid MAPE={cc:.3}', fontdict={'size':12})
    ax[1][i].set_xlim(target_min, target_max); ax[1][i].set_ylim(target_min, target_max)
    ax[1][i].set_xlabel('True'); ax[1][i].set_ylabel('Predict');
    ax[1][i].set_aspect(1)
    ax[1][i].grid()
    ax[1][i].set_xscale('log'); ax[1][i].set_yscale('log');
    
plt.tight_layout()  


modelselect = 'blend'

# ipywidgets　will run only in interactive mode
if run_type == 'Interactive':
    df_result_val = pd.concat([df_all.loc[predict_vals.index,['date','country','store','product','GDP','num_sold']] , predict_vals], axis = 1)
    df_result_train = pd.concat([df_all.loc[predict_trains.index,['date','country','store','product','GDP','num_sold']] , predict_trains], axis = 1)
    
    country, store, product = 'Canada', 'Discount Stickers', 'Holographic Goose'
    country_dropdown = widgets.Dropdown( options=df_result_val['country'].unique(), button_style='info', description='country : ' )
    store_dropdown = widgets.Dropdown( options=df_result_val['store'].unique(), button_style='info', description='store : ' )
    product_dropdown = widgets.Dropdown( options=df_result_val['product'].unique(), button_style='info', description='product : ' )
    
    
    @interact(country = country_dropdown, store = store_dropdown, product = product_dropdown)
    def plot_result(country, store, product):
        df_result_val_tmp = df_result_val[(df_result_val['country'] == country) & (df_result_val['store'] == store) & (df_result_val['product'] == product)]
        df_result_train_tmp = df_result_train[(df_result_train['country'] == country) & (df_result_train['store'] == store) & (df_result_train['product'] == product)]
        A = 1 / df_result_val_tmp['GDP'].mean() * df_result_val_tmp['num_sold'].mean()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Plot", "residual"))
        fig.add_traces([
            go.Scatter(x = df_result_val_tmp['date'], y = df_result_val_tmp['num_sold'], mode='lines', name='true',line = dict(color='red'),opacity = 0.7),
            go.Scatter(x = df_result_val_tmp['date'], y = df_result_val_tmp[modelselect], mode='lines', name='predict',line = dict(color='blue'),opacity = 0.7),
            go.Scatter(x = df_result_train_tmp['date'], y = df_result_train_tmp['num_sold'], mode='lines', name='true',line = dict(color='red'),opacity = 0.3),
            go.Scatter(x = df_result_train_tmp['date'], y = df_result_train_tmp[modelselect], mode='lines', name='predict',line = dict(color='blue'),opacity = 0.3),
            
            go.Scatter(x = df_result_val_tmp['date'], 
                       y = df_result_val_tmp['GDP'] * A
                       , mode='lines', name='true',line = dict(color='black', dash='dot'),opacity = 0.7),
            go.Scatter(x = df_result_train_tmp['date'], y = df_result_train_tmp['GDP'] * A, mode='lines', name='predict',line = dict(color='black', dash='dot'),opacity = 0.3),
        ])
        fig.add_trace(
            go.Scatter(x = df_result_val_tmp['date'], y = df_result_val_tmp[modelselect] - df_result_val_tmp['num_sold'], mode='lines', name='predict-true',line = dict(color='green'),opacity = 0.7),
            row = 2, col = 1
        )
        fig.add_trace(
            go.Scatter(x = df_result_train_tmp['date'], y = df_result_train_tmp[modelselect] - df_result_train_tmp['num_sold'], mode='lines', name='predict-true',line = dict(color='green'),opacity = 0.3),
            row = 2, col = 1
        )
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, b=0, t=0),  # Adjust margins
            xaxis=dict(title_font=dict(size=20)),
            yaxis=dict(title_font=dict(size=20)),
            legend=dict(font=dict(size=15)),
        )
        fig.show()


# ipywidgets　will run only in interactive mode
if run_type == 'Interactive':
    shap.initjs()
    model_button = widgets.ToggleButtons(
        options = predict_cols[:-1],
        button_style='info',description = 'model:'
    )
    model_button.style.button_width = f'100px'
    model_button.style.description_width = '90px'
    
    type_button = widgets.ToggleButtons(
        options = ['dot','bar'],
        button_style='warning',description = 'type:'
    )
    type_button.style.button_width = f'100px'
    type_button.style.description_width = '90px'
    
    max_disp_slider = widgets.IntSlider(
        value=min(6,len(df_train.columns)), min=0, max=len(X_train.columns), step=1, 
        description='max_display:', orientation='horizontal'
    )
    max_disp_slider.style.button_width = f'100px'
    max_disp_slider.style.description_width = '90px'
    
    
    @interact(model_name = model_button, plot_type = type_button, max_display = max_disp_slider)
    def plot_re(model_name, plot_type, max_display):
        df_train = X_train.sample(500)
        i = list(predict_cols).index(model_name)
    
        model = models_best[i]
            
        explainer = shap.TreeExplainer(model=model, model_output='raw')
        shap_values = explainer.shap_values(X=df_train)
        shap.summary_plot(shap_values, df_train, plot_type=plot_type, max_display=max_display)


# models_final = [XGBRegressor(), LGBMRegressor(), CatBoostRegressor()]
models_final = [XGBRegressor(), LGBMRegressor()]
predict_tests = []
for i,model in enumerate(models_final):
    print(predict_cols[i], end = ' / ')
    model.set_params(**best_params1[predict_cols[i]])
    model.fit(train_data.drop(target_col,axis=1), train_data[target_col])
    predict_tests.append(model.predict(X_test).flatten())

predict_tests = pd.DataFrame(
    np.array(predict_tests).T, columns = predict_cols[:2], index = X_test.index
)

predict_tests['LR'] = y_predict_test_LR
predict_tests['blend'] = predict_tests.mean(axis=1)


df_all.loc[df_all['train_test']=='test','num_sold']=predict_tests['blend']

if run_type == 'Interactive':
    plot_timeseries(df_all, plottype='plotly')


y_test_predict = predict_tests['blend']
# y_test_predict = df_all['prediction']
df_submit = df_sample_submission.set_index('id').copy()
df_submit[target_col] = y_test_predict
df_submit.to_csv('submit.csv',index = True)

display(df_submit)




