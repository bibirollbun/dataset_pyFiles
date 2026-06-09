import numpy as np 
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from tqdm import tqdm
from colorama import Fore, Style, init


from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import *

from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')

from IPython.display import clear_output


# importing the data

train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


train_df.drop('id', axis = 1, inplace = True)
test_df.drop('id', axis = 1, inplace = True)


def display_info(train_df, test_df):
    '''Displays head, info, describe, missing values of both train_df and test_df'''
    for data, label in zip([train_df, test_df], ['Train', 'Test']):
        print(Style.BRIGHT + Fore.BLUE + f'\n{label} head \n')
        display(data.head())

        print(Style.BRIGHT + Fore.BLUE + f'\n{label} info \n' + Style.RESET_ALL)
        display(data.info())

        print(Style.BRIGHT + Fore.BLUE + f'\n{label} describe \n')
        display(data.describe().T)

        print(Style.BRIGHT + Fore.BLUE + f'\n{label} missing values \n' + Style.RESET_ALL)
        display(data.isnull().sum())
        print("------------------------------------------------------------------")

# display_info(train_df, test_df)


# Storing unique values...
uniques = {}
for col in ['country', 'store', 'product']:
    uniques[col] = train_df[col].unique().tolist()
    print(f'{col} : ', uniques[col])


# For starter analysis we drop nan containing rows
train_df.dropna(inplace = True)


def process_date(df : pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear

    df['week_of_month'] = df['date'].dt.day.apply(lambda x : (x-1)//7 + 1)
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # is Weekend
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5,6]).astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    date_features = [
        'year', 'month', 'day', 'quarter', 'day_of_week', 'day_of_year',
        'week_of_month', 'week_of_year', 'is_weekend', 'is_month_end'
    ]
    
    # df.drop('date',axis=1,inplace=True)
    
    return df, date_features

def preprocess(df : pd.DataFrame):
    
    # Convert columns to category
    df[ df.select_dtypes(include='object').columns ] = df.select_dtypes(include='object').astype("category")
    return process_date(df)
    


train_df, date_fe = preprocess(train_df)
test_df, _ = preprocess(test_df)


ncols = len(uniques['store']) 
fig, axs = plt.subplots(nrows = ncols, ncols = 1, figsize = (10, 10))

# First plot
grouped_data = train_df.groupby(['date', 'store'])['num_sold'].sum().reset_index()
for store in uniques['store']:
    store_data = grouped_data[grouped_data['store'] == store]
    axs[0].plot(store_data['date'], store_data['num_sold'], ".", label = store)
axs[0].legend()
axs[0].set_title('Aggregated Sales Over Time per Store')


# Second plot
for store in uniques['store']:
    store_data = grouped_data[grouped_data['store'] == store]
    axs[1].plot(store_data['date'], store_data['num_sold'] /
             store_data['num_sold'].sum(), ".", label=store)
axs[1].legend()
axs[1].set_title('Aggregated Sales after Normalization Over Time Per Store')


# Third plot
for store in uniques['store']:
    store_data = grouped_data[grouped_data['store'] == store]['num_sold']
    sum_store=store_data.sum()
    axs[2] = sns.kdeplot(store_data/sum_store, label=store, ax=axs[2])
axs[2].legend()
axs[2].set_title('Normalized Sales Distribution Across Stores')

plt.tight_layout()
plt.show()




fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# First plot
grouped_data_country = train_df.groupby(['date', 'country'])['num_sold'].sum().reset_index()

# Each entry of grouped_data_country means what is the total num_sold on a 
# particular date in a particular country

for country in uniques['country']:
    country_data = grouped_data_country[grouped_data_country['country'] == country]
    axs[0].plot(country_data['date'], country_data['num_sold'], ".", label = country)
axs[0].legend()
axs[0].set_title('Aggregated Sales Over Time Per Country')


# Second plot
for country in uniques['country']:
    country_data = grouped_data_country[grouped_data_country['country'] == country]
    axs[1].plot(country_data['date'], country_data['num_sold']/country_data['num_sold'].sum(), ".", label=country)
axs[1].legend()
axs[1].set_title('Aggregated Sales after Normalization Over Time Per Country')


# Third plot
for country in uniques['country']:
    country_data = grouped_data_country[grouped_data_country['country']== country]
    sum_country = country_data['num_sold'].sum()
    axs[2] = sns.kdeplot(country_data['num_sold']/sum_country, label=country)
axs[2].legend()
axs[2].set_title('Normalized Sales Distribution Across Countries')
plt.tight_layout()
plt.show()


import requests

def get_gdp_per_capita(country,year):
    alpha3 = {'Canada':'CAN','Finland' : 'FIN', 'Italy' : 'ITA', 'Kenya':'KEN','Norway':'NOR', 'Singapore' : 'SGP'}
    url="https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json".format(alpha3[country],year)
    response = requests.get(url).json()
    return response[1][0]['value']
    

X = train_df.copy()
X['year'] = X.date.apply(lambda d: d.year)

x_all = []
y_all = []

gdp = []

for country in uniques['country']:
    x = []
    for year in range(2010,2017):
        x.append(get_gdp_per_capita(country,year))
    gdp.append(x)
    y = X[X.country==country].groupby('year')['num_sold'].sum().to_list()
    x_all += x
    y_all += y
    plt.plot(x,y,'.',label=country)

gdp = np.array(gdp)
gdp /= np.sum(gdp) # To normalize the data

plt.legend(bbox_to_anchor=(1, 1))
plt.xlabel('GDP per capita')
plt.ylabel('Annual sales')
plt.show()


np.corrcoef(x_all, y_all)[0,1]


# Making gdp dataframe
rel_gdp_df = pd.DataFrame(gdp, index = uniques['country'], columns = range(2010, 2017))


df= train_df.groupby(['date', 'country'])[['num_sold']].sum().reset_index()
df['rel_gdp'] = df.apply(lambda s: rel_gdp_df.loc[s.country, s.date.year], axis=1)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

for country in uniques['country']:
    country_data = df[df['country'] == country]
    axs[0].plot(country_data['date'], country_data['num_sold']/country_data['rel_gdp'], ".", label=country)
axs[0].legend()
axs[0].set_title('Aggregated Sales divided by GDP Over Time Per Country')

# Third plot
for country in uniques['country']:
    country_data = df[df['country'] == country]
    axs[1] = sns.kdeplot(country_data['num_sold']/country_data['rel_gdp'], label=country)
axs[1].legend()
axs[1].set_title(
    'Normalized Sales Distribution  divided by GDP Across Countries')
plt.tight_layout()
plt.show()


# First plot

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

grouped_data = train_df.groupby(['date', 'country','product'])['num_sold'].sum().reset_index()
grouped_data['rel_gdp'] = grouped_data .apply(lambda s: rel_gdp_df.loc[s.country, s.date.year], axis=1)
grouped_data['num_sold'] = grouped_data['num_sold']/grouped_data['rel_gdp']
grouped_data = grouped_data.groupby(['date','product'])['num_sold'].sum().reset_index()
for product in uniques['product']:
    product_data = grouped_data[grouped_data['product'] == product]
    axs[0].plot(product_data['date'], product_data['num_sold'],label=product)
#axs[0].legend()
axs[0].set_title('Aggregated Sales Over Time Per Product')


grouped_data = train_df.groupby(['date', 'product'])['num_sold'].sum().reset_index()
for product in uniques['product']:
    product_data = grouped_data[grouped_data['product'] == product]
    axs[1].plot(product_data['date'], product_data['num_sold']/product_data['num_sold'].sum(),label=product)
axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1].set_title('Aggregated Sales after Normalization Over Time Per Product')


fig = plt.figure(figsize=(10, 4))
for w in [0,1,2,3]:
    w_data = train_df[train_df['day_of_week'] == w]['num_sold']
    sns.kdeplot(w_data, label=w)
plt.title("Distribution for Weekdays")
plt.legend()
plt.show()



fig = plt.figure(figsize=(10, 4))

for w in [0, 1, 2, 3,4,5,6]:
    w_data = train_df[train_df['day_of_week'] == w]['num_sold']
    sns.kdeplot(w_data, label=w)
plt.title("Distribution for Weekends")
plt.legend()
plt.show()










