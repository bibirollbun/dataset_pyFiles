import datetime
import holidays
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


# Setting enviorment
pd.plotting.register_matplotlib_converters()
pd.options.display.float_format = '{:20.4f}'.format

warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline


# Reading all the filepaths in the dataset
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Loading Dataset URL
dataset_url_train = '/kaggle/input/playground-series-s5e1/train.csv'
dataset_url_test = '/kaggle/input/playground-series-s5e1/test.csv'


dataset = pd.read_csv(dataset_url_train, index_col=0)
dataset_test = pd.read_csv(dataset_url_test, index_col=0)

dataset.shape, dataset_test.shape


dataset.head()


dataset_test.head()


dataset.info(verbose=True)


# Converting date in dataset to 'datetime' from the original 'object' Dtype
# [Added to pipeline for future purposes]

dataset['date'] = pd.to_datetime(dataset.date, format='%Y-%m-%d')
dataset.info(verbose=True)


# Missing Data
missing_data = dataset.isnull().sum()
missing_data_percentage = (missing_data / len(dataset)) * 100

print(missing_data, '\n')
print(missing_data_percentage)


dataset.num_sold.hist(edgecolor='black', bins=100, figsize=(15,4))


# Cumulative look at the above hist, considering lot of values in the end are tailing off
dataset.num_sold.hist(edgecolor='black', bins=100, figsize=(15,4), cumulative=True)


dataset.boxplot()


dataset.describe(include='all')


plt.figure(figsize=(15, 4))
dataset.groupby('date')['num_sold'].sum().plot()
plt.grid()
plt.show()


missing_dataset = dataset[dataset.num_sold.isnull()]

missing_dataset.describe(include='all')


print(missing_dataset.country.value_counts(), '\n')
print(missing_dataset.store.value_counts(), '\n')
print(missing_dataset['product'].value_counts())


print(dataset.country.value_counts(), '\n')
print(dataset.store.value_counts(), '\n')
print(dataset['product'].value_counts())


fig, ax = plt.subplots(figsize=(15,4))
sns.histplot(data=dataset, x='date', ax=ax, label='Total')
sns.histplot(data=missing_dataset, x='date', ax=ax, color='r', label='Missing')
ax.legend()
plt.show()


dataset_noNa = dataset.dropna(axis=0)
assert dataset_noNa.shape[0] == dataset.shape[0] - missing_dataset.shape[0]

dataset_noNa.describe(include='all')


print(dataset_noNa.country.value_counts(), '\n')
print(dataset_noNa.store.value_counts(), '\n')
print(dataset_noNa['product'].value_counts())


# Converting Object to Datetime for this dataset
def convert_to_datetime(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
    return df


# Adding if particular date is a holiday or not
def add_holidays(df):
    country_holidays = {
        'Canada': holidays.CountryHoliday('CA'),
        'Finland': holidays.CountryHoliday('FI'),
        'Italy': holidays.CountryHoliday('IT'),
        'Kenya': holidays.CountryHoliday('KE'),
        'Norway': holidays.CountryHoliday('NO'),
        'Singapore': holidays.CountryHoliday('SG')
    }
    
    df = df.copy()
    df['is_holiday'] = df.apply(lambda row: 1 if row.date in country_holidays.get(row.country) else 0, axis = 1)
    return df
    

# Holidays Pipeline
# Based on EDA performed by another user on kaggle - this does not seem to have any impact
# It was also modeled by one more user who saw very little improvement to performance
def add_days_to_next_holiday(df):
    pass


def next_christmas_date(row):
    next_xmas = datetime.datetime(row.year, 12, 25)
    if next_xmas < row.date:
        next_xmas = datetime.datetime(row.year + 1, 12, 25)
    return next_xmas    


# Christmas Pipeline
def add_days_to_christmas(df):
    df = df.copy()   
    df['days_to_christmas'] = df.apply(lambda row: (next_christmas_date(row) - row.date).days, axis=1)
    return df


# Getting GDP value for country and year pair
def get_gdp(year, currency):
    url = f"https://api.worldbank.org/v2/country/{currency}/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
    response = requests.get(url).json()
    
    try:
        return response[1][0]['value']
    except (IndexError, TypeError):
        return None
        

# GDP Pipeline
def add_gdp_to_country(df):
    df = df.copy()

    min_year, max_year = df.year.min(), df.year.max() + 1

    gdps = {}
    currencies = {'Canada': 'CAN', 'Finland': 'FIN', 'Italy': 'ITA','Kenya': 'KEN', 'Norway': 'NOR', 'Singapore': 'SGP'}
    
    for year in range(min_year, max_year):
        for currency in currencies.values():
            gdps[(year, currency)] = get_gdp(year, currency)

    df['gdp'] = df.apply(lambda row: gdps[(row.year, currencies[row.country])], axis = 1)

    return df


# Day of the week pipeline
# Week starts on Monday, which is denoted by 0 and ends on Sunday which is denoted by 6.
# Letting the values be inbetween 0 to 6 as a measure of where the day is in comparison to Sunday
def add_day_of_the_week(df):
    df = df.copy()
    df['day_of_the_week'] = df.date.dt.weekday
    return df


# Applying entire data transformation and feature engineering pipeline
def apply_pipeline(df):
    df = df.copy()

    ############# Pipeline stuff
    df = convert_to_datetime(df)

    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month

    df = add_days_to_christmas(df)
    df = add_gdp_to_country(df)
    df = add_day_of_the_week(df)
    df = add_holidays(df)
    #############
    
    cardinality_columns = [col for col in df if df[col].dtype == "object"]
    df_cardinality_columns = pd.get_dummies(df.loc[:, cardinality_columns])

    df.drop(cardinality_columns, axis=1, inplace=True)
    df.drop('date', axis=1, inplace=True)

    df = pd.merge(df, df_cardinality_columns, left_index=True, right_index=True)
 
    return df


# Spliting the available data into a training and validation datasets

target = 'num_sold'

dataset_train = dataset_noNa.sample(frac = 0.8, random_state=200)
dataset_validation = dataset_noNa.drop(dataset_train.index)

y_train = dataset_train.num_sold
y_valid = dataset_validation.num_sold

X_train = dataset_train.drop(target, axis=1)
X_valid = dataset_validation.drop(target, axis=1)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


X_train = apply_pipeline(X_train)
X_valid = apply_pipeline(X_valid)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
assert X_train.shape[1] >= X_valid.shape[1]


# Checking correlation within the data if any after feature engineering
corrs = X_train.corr()

plt.figure(figsize=(15,15))

ax = sns.heatmap(data=corrs, square=True, annot=True, label='small', mask=np.triu(corrs))

for t in ax.texts:
    if abs(float(t.get_text())) >= 0.5:
        t.set_text(t.get_text())
    else:
        t.set_text("")

plt.show()


# Would dropping any feature impact error rate

model_summary = {}

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_valid)
y_pred = [0 if i <= 0 else i for i in y_pred]
model_summary['Base Model'] = mean_absolute_error(y_pred, y_valid)

for i in X_train.columns:
    x1 = X_train.drop(i, axis=1)
    x2 = X_valid.drop(i, axis=1)

    model_lr.fit(x1, y_train)
    y_pred = model_lr.predict(x2)
    y_pred = [0 if i <= 0 else i for i in y_pred]
    
    model_summary[i] = mean_absolute_error(y_pred, y_valid)

model_summary


# Modeling the base model
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_valid)
y_pred = [0 if i <= 0 else i for i in y_pred]

print(f'MA Error: {mean_absolute_error(y_pred, y_valid)}')
sns.scatterplot(x=y_pred, y=y_valid)


# Due to the outliers in target variable, implementing np.log1p
y_train_log = np.log1p(y_train)

model_lr.fit(X_train, y_train_log)
y_pred = model_lr.predict(X_valid)
y_pred = [0 if i <= 0 else i for i in y_pred] # No 0 were seen in the output, however there as a backup
y_pred = np.expm1(y_pred) # Normalizing the log output to compare with y_valid

print(f'MA Error: {mean_absolute_error(y_pred, y_valid)}')
sns.scatterplot(x=y_pred, y=y_valid)


# Using RandomForestRegressor

model_rfr = RandomForestRegressor(n_estimators=100, random_state=1234)

model_rfr.fit(X_train, y_train_log)
y_pred = model_rfr.predict(X_valid)
y_pred = [0 if i <= 0 else i for i in y_pred] # No 0 were seen in the output, however there as a backup
y_pred = np.expm1(y_pred) # Normalizing the log output to compare with y_valid

print(f'MA Error: {mean_absolute_error(y_pred, y_valid)}')
sns.scatterplot(x=y_pred, y=y_valid)


# Using XGBoost

xgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'random_state': 1234,
}

model_xgbr = XGBRegressor(**xgb_params)

model_xgbr.fit(X_train, y_train_log)
y_pred = model_xgbr.predict(X_valid)
y_pred = [0 if i <= 0 else i for i in y_pred] # No 0 were seen in the output, however there as a backup
y_pred = np.expm1(y_pred) # Normalizing the log output to compare with y_valid

print(f'MA Error: {mean_absolute_error(y_pred, y_valid)}')
sns.scatterplot(x=y_pred, y=y_valid)


dataset_test.isna().sum()


X_test = apply_pipeline(dataset_test)


# Fianl model using RandomForestRegressor trained previously

preds_test = model_xgbr.predict(X_test)
preds_test = [0 if i <= 0 else i for i in preds_test]
preds_test = np.expm1(preds_test)


# Saving test predictions to file
output = pd.DataFrame({'id': X_test.index,
                       'num_sold': preds_test})
output.to_csv('submission.csv', index=False)

