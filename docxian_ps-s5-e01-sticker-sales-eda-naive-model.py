# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# configs

# show more/all columns:
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)

# random seed
my_random_seed = 180

# colors
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'


# load data
df_train = pd.read_csv('../input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('../input/playground-series-s5e1/test.csv')
df_sub = pd.read_csv('../input/playground-series-s5e1/sample_submission.csv')


# preview
df_train.head(10)


# convert dates
df_train.date = pd.to_datetime(df_train.date)
df_test.date = pd.to_datetime(df_test.date)


# structure training data
df_train.info()


# structure test data
df_test.info()


# additional features
df_train['year'] = df_train.date.dt.year
df_train['month'] = df_train.date.dt.month
df_train['day'] = df_train.date.dt.day

df_test['year'] = df_test.date.dt.year
df_test['month'] = df_test.date.dt.month
df_test['day'] = df_test.date.dt.day


# preview
df_train.head()


# basic stats - training
df_train.describe(include='all')


# basic stats - test
df_test.describe(include='all')


features_cat = ['country', 'store', 'product']
features_time = ['year', 'month', 'day']


# list of countries
countries = df_train['country'].value_counts().index.to_list()
print(countries)


# list of stores
stores = df_train['store'].value_counts().index.to_list()
print(stores)


# list of products
products = df_train['product'].value_counts().index.to_list()
print(products)


# visualize frequencies for categorical features
for f in features_cat:
    plt.figure(figsize=(10,3))
    df_train[f].value_counts().plot(kind='bar', color=default_color_1)
    plt.title(f + ' - training')
    plt.grid()
    plt.show()


# visualize frequencies for time-like features
for f in features_time:
    plt.figure(figsize=(10,3))
    df_train[f].value_counts().sort_index().plot(kind='bar', color=default_color_1)
    plt.title(f + ' - training')
    plt.grid()
    plt.show()


# visualize frequencies for categorical features
for f in features_cat:
    plt.figure(figsize=(10,3))
    df_test[f].value_counts().plot(kind='bar', color=default_color_2)
    plt.title(f + ' - test')
    plt.grid()
    plt.show()


# visualize frequencies for time-like features
for f in features_time:
    plt.figure(figsize=(10,3))
    df_test[f].value_counts().sort_index().plot(kind='bar', color=default_color_2)
    plt.title(f + ' - training')
    plt.grid()
    plt.show()


target = 'num_sold'


# missing values?
df_train[target].isna().sum()


# fill missings with 0
df_train[target] = df_train[target].fillna(0)


# plot histogram
plt.figure(figsize=(10,3))
plt.hist(df_train[target], bins=50, color=default_color_3)
plt.title(target)
plt.grid()
plt.show()


# what is the maximum target value?
max(df_train[target])


# set corresponding constant for plots
plot_x_max = 6000


for x in countries:
    df_temp = df_train[df_train['country'] == x]
    plt.figure(figsize=(10,3))
    plt.hist(df_temp[target], bins=50, color=default_color_3)
    plt.xlim(0, plot_x_max) # fix axis size for comparisons
    plt.title(target + ' - country = ' + x)
    plt.grid()
    plt.show()


for x in stores:
    df_temp = df_train[df_train['store'] == x]
    plt.figure(figsize=(10,3))
    plt.hist(df_temp[target], bins=50, color=default_color_3)
    plt.xlim(0, plot_x_max) # fix axis size for comparisons
    plt.title(target + ' - store = ' + x)
    plt.grid()
    plt.show()


for x in products:
    df_temp = df_train[df_train['product'] == x]
    plt.figure(figsize=(10,3))
    plt.hist(df_temp[target], bins=50, color=default_color_3)
    plt.xlim(0, plot_x_max) # fix axis size for comparisons
    plt.title(target + ' - product = ' + x)
    plt.grid()
    plt.show()


# full time series
plt.figure(figsize=(15,3))
plt.scatter(df_train.date, df_train[target], color=default_color_3, s=1)
plt.grid()
plt.show()


# full time series - colored by country
plt.figure(figsize=(15,6))
sns.scatterplot(data=df_train, x='date', y=target, 
                hue=df_train['country'], s=1)
plt.grid()
plt.show()


# full time series - colored by country - log scale
df_train['target_log'] = np.log(1 + df_train[target])
plt.figure(figsize=(15,6))
sns.scatterplot(data=df_train, x='date', y='target_log', 
                hue=df_train['country'], s=1)
plt.grid()
plt.show()


# list of years (sorted)
years_train = df_train.year.value_counts().sort_index().index.to_list()
print(years_train)


for y in years_train:
    df_temp = df_train[df_train['year']==y]
    plt.figure(figsize=(15,3))
    plt.scatter(df_temp.date, df_temp[target], color=default_color_3, s=1)
    plt.title('Year ' + str(y))
    plt.grid()
    plt.show()


# create combined key
df_train['combo_key'] = df_train['country'] + '|' + df_train['store'] + '|' + df_train['product']
df_test['combo_key'] = df_test['country'] + '|' + df_test['store'] + '|' + df_test['product']


# overall mean
df_train[target].mean()


# mean by year
stats_year = df_train.groupby(by=['year'])[target].mean()
stats_year


# plot target mean by year
plt.figure(figsize=(8,3))
plt.scatter(stats_year.index, stats_year, color=default_color_3)
plt.title('Mean of target by year')
plt.grid()
plt.show()


# mean by month
stats_month = df_train.groupby(by=['month'])[target].mean()
stats_month


# plot target mean by month
plt.figure(figsize=(8,3))
plt.scatter(stats_month.index, stats_month, color=default_color_3)
plt.title('Mean of target by month')
plt.grid()
plt.show()


# mean by country
df_train.groupby(by=['country'])[target].mean()


# mean by store
df_train.groupby(by=['store'])[target].mean()


# mean by product
df_train.groupby(by=['product'])[target].mean()


# mean by combination country|store|product
stats_combo = df_train.groupby(by=['combo_key'])[target].mean()
stats_combo


for c in countries:
    for s in stores:
        for p in products:
            my_key = c + '|' + s + '|' + p
            df_temp = df_train[df_train.combo_key == my_key]
            plt.figure(figsize=(15,3))
            plt.scatter(df_temp.date, df_temp[target], color=default_color_3, s=1)
            plt.title(my_key)
            plt.grid()
            plt.show()


# introduce key including month and day
df_train['combo_key_md'] = df_train['combo_key'] + '|' + df_train['month'].astype(str) + '|' + df_train['day'].astype(str)
df_test['combo_key_md'] = df_test['combo_key'] + '|' + df_test['month'].astype(str) + '|' + df_test['day'].astype(str)


# initial step; note that 2010 is not a leap year so our table will also exclude leap days!
predecessor_table = df_train[df_train['year']==2010]
predecessor_table = predecessor_table[['combo_key_md', target]]
predecessor_table.rename(columns={'num_sold' : 'num_2010'}, inplace=True)


# iteratively add the following years
for current_year in range(2011,2016+1):
    next_year = df_train[df_train['year']==current_year].reset_index()
    # remove leap day entries:
    next_year = next_year[~((next_year.month==2) & (next_year.day==29))].reset_index()
    next_year = next_year[['combo_key_md', target]]
    new_name = 'num_' + str(current_year)
    next_year.rename(columns={'num_sold' : new_name}, inplace=True)
    # add new column
    predecessor_table = pd.merge(predecessor_table, next_year, on='combo_key_md', how='left')


# preview of result
predecessor_table.head(10)


# add historical data to test set
df_test = pd.merge(left=df_test, right=predecessor_table, on='combo_key_md', how='left')


# preview
df_test.head()


# export extended test set
df_test.to_csv('test_set_extended.csv', index=False)


# create a prediction from the historical data for each day

# define weights
w_2010 = 0.1
w_2011 = 0.115
w_2012 = 0.125
w_2013 = 0.14
w_2014 = 0.155
w_2015 = 0.175
w_2016 = 0.19

# check
w_2010+w_2011+w_2012+w_2013+w_2014+w_2015+w_2016


# calc linear combination of historical values
df_test['predict'] = w_2010*df_test.num_2010 + w_2011*df_test.num_2011 + \
                     w_2012*df_test.num_2012 + w_2013*df_test.num_2013 + \
                     w_2014*df_test.num_2014 + w_2015*df_test.num_2015 + \
                     w_2016*df_test.num_2016


# stats
df_test['predict'].describe()


# plot histogram
plt.figure(figsize=(10,3))
plt.hist(df_test['predict'], bins=25, color=default_color_3)
plt.title('predict')
plt.grid()
plt.show()


# full time series - colored by country
plt.figure(figsize=(15,6))
sns.scatterplot(data=df_test, x='date', y='predict', 
                hue=df_train['country'], s=1)
plt.grid()
plt.show()


# prepare submission
df_sub.num_sold = df_test['predict']
df_sub.head(10)


# and save submission file
df_sub.to_csv('submission.csv', index=False)

