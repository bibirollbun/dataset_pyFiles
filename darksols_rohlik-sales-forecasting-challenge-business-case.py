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


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import openpyxl
import plotly.graph_objects as go
from matplotlib.sankey import Sankey


sales_test = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv"
sales_train = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv"
test_weights = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv"
calendar_file = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv"
inventory_file = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv"
solution_file = "/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv"


sales_test_df = pd.read_csv(sales_test)
print (sales_test_df.head())


sales_train_df = pd.read_csv(sales_train)
print(sales_train_df.head())


calendar_df = pd.read_csv(calendar_file)
print (calendar_df.head())


inventory_df = pd.read_csv(inventory_file)
print (inventory_df.head())


test_weights_df = pd.read_csv(test_weights)
print (test_weights_df.head())


print (sales_test_df.info())
## No sales or availability columns in testing dataset ##


print (sales_train_df.info())


print (inventory_df.info())


print (calendar_df.info())


print (test_weights_df.info())


print (sales_test_df.describe())


print (sales_train_df.describe())


print (inventory_df.describe())


print (calendar_df.describe())


print (test_weights_df.describe())


## Check for Duplicates Values ##
duplicates_sales_train = sales_train_df.duplicated()
duplicates_sales_test = sales_test_df.duplicated()
duplicates_inventory = inventory_df.duplicated()
duplicates_calendar = calendar_df.duplicated()
duplicates_test_weights = test_weights_df.duplicated()


print (duplicates_sales_train.sum())


print (duplicates_sales_test.sum())


print (duplicates_inventory.sum())


print (duplicates_calendar.sum())


print (duplicates_test_weights.sum())


sales_test_df.drop_duplicates()
print(f"After removing fully duplicated rows from sales_test_df: {sales_test_df.shape[0]} rows")
sales_train_df.drop_duplicates()
print(f"After removing fully duplicated rows from sales_train_df: {sales_train_df.shape[0]} rows")
inventory_df.drop_duplicates()
print(f"After removing fully duplicated rows from inventory_df: {inventory_df.shape[0]} rows")
test_weights_df.drop_duplicates()
print(f"After removing fully duplicated rows from test_weights_df: {test_weights_df.shape[0]} rows")
calendar_df.drop_duplicates()
print(f"After removing fully duplicated rows from calendar_df: {calendar_df.shape[0]} rows")


is_unique_1 = sales_train_df['unique_id'].is_unique
print(f"Are all 'unique_id' values unique? {is_unique_1}")
is_unique_2 = sales_test_df['unique_id'].is_unique
print(f"Are all 'unique_id' values unique? {is_unique_2}")
is_unique_3 = inventory_df['unique_id'].is_unique
print(f"Are all 'unique_id' values unique? {is_unique_3}")
is_unique_4 = test_weights_df['unique_id'].is_unique
print(f"Are all 'unique_id' values unique? {is_unique_4}")
is_unique_5 = calendar_df['date'].is_unique
print(f"Are all 'date' values unique? {is_unique_5}")


print (f"Date duplicates in calendar_df: {calendar_df.duplicated(subset=['date']).sum()}")


calendar_df = calendar_df.drop_duplicates(subset=['date'], keep='first')
print(f"After removing duplicates in calendar_df for date: {calendar_df.shape[0]} rows")


is_unique_5 = calendar_df['date'].is_unique
print(f"Are all 'date' values unique? {is_unique_5}")


print(sales_train_df.isnull().sum())


print(sales_test_df.isnull().sum())


print(calendar_df.isnull().sum())


print(inventory_df.isnull().sum())


print(test_weights_df.isnull().sum())


print (calendar_df.holiday_name)


# Define popular holidays for Brno_1 from 2020 to 2024
brno_1_holidays = {
    "2020": [
        ("2020-01-01", "New Year's Day"), ("2020-01-01", "Restoration Day of the Czech State"),
        ("2020-04-10", "Good Friday"), ("2020-04-13", "Easter Monday"), 
        ("2020-05-01", "Labour Day"), ("2020-05-08", "VE Day"), 
        ("2020-07-05", "Saint Cyril and Methodius Day"), ("2020-07-06", "Jan Hus Day"), 
        ("2020-09-28", "Saint Wencelas Day"), ("2020-10-28", "Independent Czechoslovak State Day"),
        ("2020-11-17", "Struggle for Freedom and Democracy Day"), ("2020-12-24", "Christmas Eve"), 
        ("2020-12-25", "Christmas Day")
    ],
    "2021": [
        ("2021-01-01", "New Year's Day"), ("2021-01-01", "Restoration Day of the Czech State"),
        ("2021-04-02", "Good Friday"), ("2021-04-05", "Easter Monday"), 
        ("2021-05-01", "Labour Day"), ("2021-05-08", "VE Day"), 
        ("2021-07-05", "Saint Cyril and Methodius Day"), ("2021-07-06", "Jan Hus Day"), 
        ("2021-09-28", "Saint Wencelas Day"), ("2021-10-28", "Independent Czechoslovak State Day"),
        ("2021-11-17", "Struggle for Freedom and Democracy Day"), ("2021-12-24", "Christmas Eve"), 
        ("2021-12-25", "Christmas Day")
    ],
    "2022": [
        ("2022-01-01", "New Year's Day"), ("2022-01-01", "Restoration Day of the Czech State"),
        ("2022-04-15", "Good Friday"), ("2022-04-18", "Easter Monday"), 
        ("2022-05-01", "Labour Day"), ("2022-05-08", "VE Day"), 
        ("2022-07-05", "Saint Cyril and Methodius Day"), ("2022-07-06", "Jan Hus Day"), 
        ("2022-09-28", "Saint Wencelas Day"), ("2022-10-28", "Independent Czechoslovak State Day"),
        ("2022-11-17", "Struggle for Freedom and Democracy Day"), ("2022-12-24", "Christmas Eve"), 
        ("2022-12-25", "Christmas Day")
    ],
    "2023": [
        ("2023-01-01", "New Year's Day"), ("2023-01-01", "Restoration Day of the Czech State"),
        ("2023-04-07", "Good Friday"), ("2023-04-10", "Easter Monday"), 
        ("2023-05-01", "Labour Day"), ("2023-05-08", "VE Day"), 
        ("2023-07-05", "Saint Cyril and Methodius Day"), ("2023-07-06", "Jan Hus Day"), 
        ("2023-09-28", "Saint Wencelas Day"), ("2023-10-28", "Independent Czechoslovak State Day"),
        ("2023-11-17", "Struggle for Freedom and Democracy Day"), ("2023-12-24", "Christmas Eve"), 
        ("2023-12-25", "Christmas Day")
    ],
    "2024": [
        ("2024-01-01", "New Year's Day"), ("2024-01-01", "Restoration Day of the Czech State"),
        ("2024-03-29", "Good Friday"), ("2024-04-01", "Easter Monday"), 
        ("2024-05-01", "Labour Day"), ("2024-05-08", "VE Day"), 
        ("2024-07-05", "Saint Cyril and Methodius Day"), ("2024-07-06", "Jan Hus Day"), 
        ("2024-09-28", "Saint Wencelas Day"), ("2024-10-28", "Independent Czechoslovak State Day"),
        ("2024-11-17", "Struggle for Freedom and Democracy Day"), ("2024-12-24", "Christmas Eve"), 
        ("2024-12-25", "Christmas Day")
    ]
}

# Convert to DataFrame
brno_1_holiday_list = []
for year, holidays in brno_1_holidays.items():
    for date, name in holidays:
        brno_1_holiday_list.append([date, name, "Czech Republic (Brno)", year])

# Create DataFrame
brno_1_holiday_df = pd.DataFrame(brno_1_holiday_list, columns=["date", "holiday_name", "Country", "Year"])


# Define popular holidays for Frankfurt_1 from 2020 to 2024
frankfurt_1_holidays = {
    "2020": [
        ("2020-01-01", "New Year's Day"), ("2020-04-10", "Good Friday"), 
        ("2020-04-13", "Easter Monday"), ("2020-05-01", "Labour Day"),
        ("2020-05-21", "Ascension Day"), ("2020-06-01", "Whit Monday"),
        ("2020-06-11", "Corpus Christi"), ("2020-10-03", "German Unity Day"),
        ("2020-12-25", "Christmas Day")
    ],
    "2021": [
        ("2021-01-01", "New Year's Day"), ("2021-04-02", "Good Friday"), 
        ("2021-04-05", "Easter Monday"), ("2021-05-01", "Labour Day"),
        ("2021-05-13", "Ascension Day"), ("2021-05-24", "Whit Monday"),
        ("2021-06-03", "Corpus Christi"), ("2021-10-03", "German Unity Day"),
        ("2021-12-25", "Christmas Day")
    ],
    "2022": [
        ("2022-01-01", "New Year's Day"), ("2022-04-15", "Good Friday"), 
        ("2022-04-18", "Easter Monday"), ("2022-05-01", "Labour Day"),
        ("2022-05-26", "Ascension Day"), ("2022-06-06", "Whit Monday"),
        ("2022-06-16", "Corpus Christi"), ("2022-10-03", "German Unity Day"),
        ("2022-12-25", "Christmas Day")
    ],
    "2023": [
        ("2023-01-01", "New Year's Day"), ("2023-04-07", "Good Friday"), 
        ("2023-04-10", "Easter Monday"), ("2023-05-01", "Labour Day"),
        ("2023-05-18", "Ascension Day"), ("2023-05-29", "Whit Monday"),
        ("2023-06-08", "Corpus Christi"), ("2023-10-03", "German Unity Day"),
        ("2023-12-25", "Christmas Day")
    ],
    "2024": [
        ("2024-01-01", "New Year's Day"), ("2024-03-29", "Good Friday"), 
        ("2024-04-01", "Easter Monday"), ("2024-05-01", "Labour Day"),
        ("2024-05-09", "Ascension Day"), ("2024-05-20", "Whit Monday"),
        ("2024-05-30", "Corpus Christi"), ("2024-10-03", "German Unity Day"),
        ("2024-12-25", "Christmas Day")
    ]
}

# Convert to DataFrame
frankfurt_1_holiday_list = []
for year, holidays in frankfurt_1_holidays.items():
    for date, name in holidays:
        frankfurt_1_holiday_list.append([date, name, "Germany (Frankfurt)", year])

# Create DataFrame
frankfurt_1_holiday_df = pd.DataFrame(frankfurt_1_holiday_list, columns=["date", "holiday_name", "Country", "Year"])


# Define popular holidays for Budapest_1 from 2020 to 2024
budapest_1_holidays = {
    "2020": [
        ("2020-01-01", "New Year's Day"), ("2020-03-15", "1848 Revolution Memorial Day"),
        ("2020-04-10", "Good Friday"), ("2020-04-12", "Easter Sunday"),
        ("2020-04-13", "Easter Monday"), ("2020-05-01", "Labour Day"),
        ("2020-10-23", "1956 Memorial Day"), ("2020-11-01", "All Saints' Day"),
        ("2020-12-25", "Christmas Day"), ("2020-12-26", "Second Day of Christmas")
    ],
    "2021": [
        ("2021-01-01", "New Year's Day"), ("2021-03-15", "1848 Revolution Memorial Day"),
        ("2021-04-02", "Good Friday"), ("2021-04-04", "Easter Sunday"),
        ("2021-04-05", "Easter Monday"), ("2021-05-01", "Labour Day"),
        ("2021-10-23", "1956 Memorial Day"), ("2021-11-01", "All Saints' Day"),
        ("2021-12-25", "Christmas Day"), ("2021-12-26", "Second Day of Christmas")
    ],
    "2022": [
        ("2022-01-01", "New Year's Day"), ("2022-03-14", "1848 Revolution Memorial Day (Extra Holiday)"),
        ("2022-03-15", "1848 Revolution Memorial Day"), ("2022-04-15", "Good Friday"),
        ("2022-04-17", "Easter Sunday"), ("2022-04-18", "Easter Monday"),
        ("2022-05-01", "Labour Day"), ("2022-10-23", "1956 Memorial Day"),
        ("2022-11-01", "All Saints' Day"), ("2022-12-25", "Christmas Day"),
        ("2022-12-26", "Second Day of Christmas")
    ],
    "2023": [
        ("2023-01-01", "New Year's Day"), ("2023-03-15", "1848 Revolution Memorial Day"),
        ("2023-04-07", "Good Friday"), ("2023-04-09", "Easter Sunday"),
        ("2023-04-10", "Easter Monday"), ("2023-05-01", "Labour Day"),
        ("2023-10-23", "1956 Memorial Day"), ("2023-11-01", "All Saints' Day"),
        ("2023-12-25", "Christmas Day"), ("2023-12-26", "Second Day of Christmas")
    ],
    "2024": [
        ("2024-01-01", "New Year's Day"), ("2024-03-15", "1848 Revolution Memorial Day"),
        ("2024-03-29", "Good Friday"), ("2024-03-31", "Easter Sunday"),
        ("2024-04-01", "Easter Monday"), ("2024-05-01", "Labour Day"),
        ("2024-10-23", "1956 Memorial Day"), ("2024-11-01", "All Saints' Day"),
        ("2024-12-25", "Christmas Day"), ("2024-12-26", "Second Day of Christmas")
    ]
}

# Convert to DataFrame
budapest_1_holiday_list = []
for year, holidays in budapest_1_holidays.items():
    for date, name in holidays:
        budapest_1_holiday_list.append([date, name, "Hungary (Budapest)", year])

# Create DataFrame
budapest_1_holiday_df = pd.DataFrame(budapest_1_holiday_list, columns=["date", "holiday_name", "Country", "Year"])


# Define popular holidays for Munich_1 from 2020 to 2024
munich_1_holidays = {
    "2020": [
        ("2020-01-01", "New Year's Day"), ("2020-05-01", "Labour Day"),
        ("2020-05-21", "Ascension Day"), ("2020-08-15", "Assumption of Mary"),
        ("2020-01-06", "Epiphany"), ("2020-10-03", "German Unity Day"),
        ("2020-04-10", "Good Friday"), ("2020-06-01", "Whit Monday"),
        ("2020-11-01", "All Saints' Day"), ("2020-04-13", "Easter Monday"),
        ("2020-06-11", "Corpus Christi"), ("2020-12-25", "Christmas Day")
    ],
    "2021": [
        ("2021-01-01", "New Year's Day"), ("2021-05-01", "Labour Day"),
        ("2021-05-13", "Ascension Day"), ("2021-08-15", "Assumption of Mary"),
        ("2021-01-06", "Epiphany"), ("2021-10-03", "German Unity Day"),
        ("2021-04-02", "Good Friday"), ("2021-05-24", "Whit Monday"),
        ("2021-11-01", "All Saints' Day"), ("2021-04-05", "Easter Monday"),
        ("2021-06-03", "Corpus Christi"), ("2021-12-25", "Christmas Day")
    ],
    "2022": [
        ("2022-01-01", "New Year's Day"), ("2022-05-01", "Labour Day"),
        ("2022-05-26", "Ascension Day"), ("2022-08-15", "Assumption of Mary"),
        ("2022-01-06", "Epiphany"), ("2022-10-03", "German Unity Day"),
        ("2022-04-15", "Good Friday"), ("2022-06-06", "Whit Monday"),
        ("2022-11-01", "All Saints' Day"), ("2022-04-18", "Easter Monday"),
        ("2022-06-16", "Corpus Christi"), ("2022-12-25", "Christmas Day")
    ],
    "2023": [
        ("2023-01-01", "New Year's Day"), ("2023-05-01", "Labour Day"),
        ("2023-05-18", "Ascension Day"), ("2023-08-15", "Assumption of Mary"),
        ("2023-01-06", "Epiphany"), ("2023-10-03", "German Unity Day"),
        ("2023-04-07", "Good Friday"), ("2023-05-29", "Whit Monday"),
        ("2023-11-01", "All Saints' Day"), ("2023-04-10", "Easter Monday"),
        ("2023-06-08", "Corpus Christi"), ("2023-12-25", "Christmas Day")
    ],
    "2024": [
        ("2024-01-01", "New Year's Day"), ("2024-05-01", "Labour Day"),
        ("2024-05-09", "Ascension Day"), ("2024-08-15", "Assumption of Mary"),
        ("2024-01-06", "Epiphany"), ("2024-10-03", "German Unity Day"),
        ("2024-03-29", "Good Friday"), ("2024-05-20", "Whit Monday"),
        ("2024-11-01", "All Saints' Day"), ("2024-04-01", "Easter Monday"),
        ("2024-05-30", "Corpus Christi"), ("2024-12-25", "Christmas Day")
    ]
}

# Convert to DataFrame
munich_1_holiday_list = []
for year, holidays in munich_1_holidays.items():
    for date, name in holidays:
        munich_1_holiday_list.append([date, name, "Germany (Munich)", year])

# Create DataFrame
munich_1_holiday_df = pd.DataFrame(munich_1_holiday_list, columns=["date", "holiday_name", "Country", "Year"])


# Define popular holidays for Prague_1 from 2020 to 2024
prague_1_holidays = {
    "2020": [
        ("2020-01-01", "New Year's Day"), ("2020-01-01", "Restoration Day of the Czech State"),
        ("2020-04-10", "Good Friday"), ("2020-04-13", "Easter Monday"), 
        ("2020-05-01", "Labour Day"), ("2020-05-08", "VE Day"), 
        ("2020-07-05", "Saint Cyril and Methodius Day"), ("2020-07-06", "Jan Hus Day"), 
        ("2020-09-28", "Saint Wencelas Day"), ("2020-10-28", "Independent Czechoslovak State Day"),
        ("2020-11-17", "Struggle for Freedom and Democracy Day"), ("2020-12-24", "Christmas Eve"), 
        ("2020-12-25", "Christmas Day")
    ],
    "2021": [
        ("2021-01-01", "New Year's Day"), ("2021-01-01", "Restoration Day of the Czech State"),
        ("2021-04-02", "Good Friday"), ("2021-04-05", "Easter Monday"), 
        ("2021-05-01", "Labour Day"), ("2021-05-08", "VE Day"), 
        ("2021-07-05", "Saint Cyril and Methodius Day"), ("2021-07-06", "Jan Hus Day"), 
        ("2021-09-28", "Saint Wencelas Day"), ("2021-10-28", "Independent Czechoslovak State Day"),
        ("2021-11-17", "Struggle for Freedom and Democracy Day"), ("2021-12-24", "Christmas Eve"), 
        ("2021-12-25", "Christmas Day")
    ],
    "2022": [
        ("2022-01-01", "New Year's Day"), ("2022-01-01", "Restoration Day of the Czech State"),
        ("2022-04-15", "Good Friday"), ("2022-04-18", "Easter Monday"), 
        ("2022-05-01", "Labour Day"), ("2022-05-08", "VE Day"), 
        ("2022-07-05", "Saint Cyril and Methodius Day"), ("2022-07-06", "Jan Hus Day"), 
        ("2022-09-28", "Saint Wencelas Day"), ("2022-10-28", "Independent Czechoslovak State Day"),
        ("2022-11-17", "Struggle for Freedom and Democracy Day"), ("2022-12-24", "Christmas Eve"), 
        ("2022-12-25", "Christmas Day")
    ],
    "2023": [
        ("2023-01-01", "New Year's Day"), ("2023-01-01", "Restoration Day of the Czech State"),
        ("2023-04-07", "Good Friday"), ("2023-04-10", "Easter Monday"), 
        ("2023-05-01", "Labour Day"), ("2023-05-08", "VE Day"), 
        ("2023-07-05", "Saint Cyril and Methodius Day"), ("2023-07-06", "Jan Hus Day"), 
        ("2023-09-28", "Saint Wencelas Day"), ("2023-10-28", "Independent Czechoslovak State Day"),
        ("2023-11-17", "Struggle for Freedom and Democracy Day"), ("2023-12-24", "Christmas Eve"), 
        ("2023-12-25", "Christmas Day")
    ],
    "2024": [
        ("2024-01-01", "New Year's Day"), ("2024-01-01", "Restoration Day of the Czech State"),
        ("2024-03-29", "Good Friday"), ("2024-04-01", "Easter Monday"), 
        ("2024-05-01", "Labour Day"), ("2024-05-08", "VE Day"), 
        ("2024-07-05", "Saint Cyril and Methodius Day"), ("2024-07-06", "Jan Hus Day"), 
        ("2024-09-28", "Saint Wencelas Day"), ("2024-10-28", "Independent Czechoslovak State Day"),
        ("2024-11-17", "Struggle for Freedom and Democracy Day"), ("2024-12-24", "Christmas Eve"), 
        ("2024-12-25", "Christmas Day")
    ]
}

# Convert to DataFrame
prague_1_holiday_list = []
for year, holidays in prague_1_holidays.items():
    for date, name in holidays:
        prague_1_holiday_list.append([date, name, "Czech Republic (Prague)", year])

# Create DataFrame
prague_1_holiday_df = pd.DataFrame(prague_1_holiday_list, columns=["date", "holiday_name", "Country", "Year"])


# Define popular holidays for Prague_1 from 2020 to 2024
prague_2_holidays = {
    "2020": [
        ("2020-01-01", "New Year's Day"), ("2020-01-01", "Restoration Day of the Czech State"),
        ("2020-04-10", "Good Friday"), ("2020-04-13", "Easter Monday"), 
        ("2020-05-01", "Labour Day"), ("2020-05-08", "VE Day"), 
        ("2020-07-05", "Saint Cyril and Methodius Day"), ("2020-07-06", "Jan Hus Day"), 
        ("2020-09-28", "Saint Wencelas Day"), ("2020-10-28", "Independent Czechoslovak State Day"),
        ("2020-11-17", "Struggle for Freedom and Democracy Day"), ("2020-12-24", "Christmas Eve"), 
        ("2020-12-25", "Christmas Day")
    ],
    "2021": [
        ("2021-01-01", "New Year's Day"), ("2021-01-01", "Restoration Day of the Czech State"),
        ("2021-04-02", "Good Friday"), ("2021-04-05", "Easter Monday"), 
        ("2021-05-01", "Labour Day"), ("2021-05-08", "VE Day"), 
        ("2021-07-05", "Saint Cyril and Methodius Day"), ("2021-07-06", "Jan Hus Day"), 
        ("2021-09-28", "Saint Wencelas Day"), ("2021-10-28", "Independent Czechoslovak State Day"),
        ("2021-11-17", "Struggle for Freedom and Democracy Day"), ("2021-12-24", "Christmas Eve"), 
        ("2021-12-25", "Christmas Day")
    ],
    "2022": [
        ("2022-01-01", "New Year's Day"), ("2022-01-01", "Restoration Day of the Czech State"),
        ("2022-04-15", "Good Friday"), ("2022-04-18", "Easter Monday"), 
        ("2022-05-01", "Labour Day"), ("2022-05-08", "VE Day"), 
        ("2022-07-05", "Saint Cyril and Methodius Day"), ("2022-07-06", "Jan Hus Day"), 
        ("2022-09-28", "Saint Wencelas Day"), ("2022-10-28", "Independent Czechoslovak State Day"),
        ("2022-11-17", "Struggle for Freedom and Democracy Day"), ("2022-12-24", "Christmas Eve"), 
        ("2022-12-25", "Christmas Day")
    ],
    "2023": [
        ("2023-01-01", "New Year's Day"), ("2023-01-01", "Restoration Day of the Czech State"),
        ("2023-04-07", "Good Friday"), ("2023-04-10", "Easter Monday"), 
        ("2023-05-01", "Labour Day"), ("2023-05-08", "VE Day"), 
        ("2023-07-05", "Saint Cyril and Methodius Day"), ("2023-07-06", "Jan Hus Day"), 
        ("2023-09-28", "Saint Wencelas Day"), ("2023-10-28", "Independent Czechoslovak State Day"),
        ("2023-11-17", "Struggle for Freedom and Democracy Day"), ("2023-12-24", "Christmas Eve"), 
        ("2023-12-25", "Christmas Day")
    ],
    "2024": [
        ("2024-01-01", "New Year's Day"), ("2024-01-01", "Restoration Day of the Czech State"),
        ("2024-03-29", "Good Friday"), ("2024-04-01", "Easter Monday"), 
        ("2024-05-01", "Labour Day"), ("2024-05-08", "VE Day"), 
        ("2024-07-05", "Saint Cyril and Methodius Day"), ("2024-07-06", "Jan Hus Day"), 
        ("2024-09-28", "Saint Wencelas Day"), ("2024-10-28", "Independent Czechoslovak State Day"),
        ("2024-11-17", "Struggle for Freedom and Democracy Day"), ("2024-12-24", "Christmas Eve"), 
        ("2024-12-25", "Christmas Day")
    ]
}

# Convert to DataFrame
prague_2_holiday_list = []
for year, holidays in prague_2_holidays.items():
    for date, name in holidays:
        prague_2_holiday_list.append([date, name, "Czech Republic (Prague)", year])

# Create DataFrame
prague_2_holiday_df = pd.DataFrame(prague_2_holiday_list, columns=["date", "holiday_name", "Country", "Year"])


# Define popular holidays for Prague_1 from 2020 to 2024
prague_3_holidays = {
    "2020": [
        ("2020-01-01", "New Year's Day"), ("2020-01-01", "Restoration Day of the Czech State"),
        ("2020-04-10", "Good Friday"), ("2020-04-13", "Easter Monday"), 
        ("2020-05-01", "Labour Day"), ("2020-05-08", "VE Day"), 
        ("2020-07-05", "Saint Cyril and Methodius Day"), ("2020-07-06", "Jan Hus Day"), 
        ("2020-09-28", "Saint Wencelas Day"), ("2020-10-28", "Independent Czechoslovak State Day"),
        ("2020-11-17", "Struggle for Freedom and Democracy Day"), ("2020-12-24", "Christmas Eve"), 
        ("2020-12-25", "Christmas Day")
    ],
    "2021": [
        ("2021-01-01", "New Year's Day"), ("2021-01-01", "Restoration Day of the Czech State"),
        ("2021-04-02", "Good Friday"), ("2021-04-05", "Easter Monday"), 
        ("2021-05-01", "Labour Day"), ("2021-05-08", "VE Day"), 
        ("2021-07-05", "Saint Cyril and Methodius Day"), ("2021-07-06", "Jan Hus Day"), 
        ("2021-09-28", "Saint Wencelas Day"), ("2021-10-28", "Independent Czechoslovak State Day"),
        ("2021-11-17", "Struggle for Freedom and Democracy Day"), ("2021-12-24", "Christmas Eve"), 
        ("2021-12-25", "Christmas Day")
    ],
    "2022": [
        ("2022-01-01", "New Year's Day"), ("2022-01-01", "Restoration Day of the Czech State"),
        ("2022-04-15", "Good Friday"), ("2022-04-18", "Easter Monday"), 
        ("2022-05-01", "Labour Day"), ("2022-05-08", "VE Day"), 
        ("2022-07-05", "Saint Cyril and Methodius Day"), ("2022-07-06", "Jan Hus Day"), 
        ("2022-09-28", "Saint Wencelas Day"), ("2022-10-28", "Independent Czechoslovak State Day"),
        ("2022-11-17", "Struggle for Freedom and Democracy Day"), ("2022-12-24", "Christmas Eve"), 
        ("2022-12-25", "Christmas Day")
    ],
    "2023": [
        ("2023-01-01", "New Year's Day"), ("2023-01-01", "Restoration Day of the Czech State"),
        ("2023-04-07", "Good Friday"), ("2023-04-10", "Easter Monday"), 
        ("2023-05-01", "Labour Day"), ("2023-05-08", "VE Day"), 
        ("2023-07-05", "Saint Cyril and Methodius Day"), ("2023-07-06", "Jan Hus Day"), 
        ("2023-09-28", "Saint Wencelas Day"), ("2023-10-28", "Independent Czechoslovak State Day"),
        ("2023-11-17", "Struggle for Freedom and Democracy Day"), ("2023-12-24", "Christmas Eve"), 
        ("2023-12-25", "Christmas Day")
    ],
    "2024": [
        ("2024-01-01", "New Year's Day"), ("2024-01-01", "Restoration Day of the Czech State"),
        ("2024-03-29", "Good Friday"), ("2024-04-01", "Easter Monday"), 
        ("2024-05-01", "Labour Day"), ("2024-05-08", "VE Day"), 
        ("2024-07-05", "Saint Cyril and Methodius Day"), ("2024-07-06", "Jan Hus Day"), 
        ("2024-09-28", "Saint Wencelas Day"), ("2024-10-28", "Independent Czechoslovak State Day"),
        ("2024-11-17", "Struggle for Freedom and Democracy Day"), ("2024-12-24", "Christmas Eve"), 
        ("2024-12-25", "Christmas Day")
    ]
}

# Convert to DataFrame
prague_3_holiday_list = []
for year, holidays in prague_3_holidays.items():
    for date, name in holidays:
        prague_3_holiday_list.append([date, name, "Czech Republic (Prague)", year])

# Create DataFrame
prague_3_holiday_df = pd.DataFrame(prague_3_holiday_list, columns=["date", "holiday_name", "Country", "Year"])


# List of all holiday DataFrames (make sure these variables are defined)
holiday_dfs = [frankfurt_1_holiday_df, munich_1_holiday_df, prague_1_holiday_df, prague_2_holiday_df, prague_3_holiday_df, brno_1_holiday_df, budapest_1_holiday_df]

# Concatenate all holiday data into one DataFrame
holiday_name_df = pd.concat(holiday_dfs, ignore_index=True)


print (holiday_name_df)


# Merge calendar_df with holiday_name_df on 'date' to bring in holiday names
calendar_df = calendar_df.merge(holiday_name_df[['date', 'holiday_name']], on='date', how='left')

# Fill missing values in 'holiday_name' column in calendar_df
calendar_df['holiday_name'] = calendar_df['holiday_name_x'].fillna(calendar_df['holiday_name_y'])

# Drop the duplicate columns created from merging
calendar_df.drop(columns=['holiday_name_x', 'holiday_name_y'], inplace=True)


print (calendar_df)


print (calendar_df.isnull().sum())


calendar_df['holiday_name'] = calendar_df['holiday_name'].fillna("N/A")


print (calendar_df.isnull().sum())


# Define impact scores for holidays in each region
holiday_impact_by_region = {
    "Brno_1": {
        "New Year's Day": 3, "Labour Day": 3, "Jan Hus Day": 2, "Christmas Eve": 3,
        "Restoration Day of the Czech State": 3, "Victory Day": 2, "Saint Wencelas Day": 2,
        "Good Friday": 2, "VE Day": 2, "Christmas Day": 3, "Easter Monday": 3,
        "Saint Cyril and Methodius Day": 2, "Struggle for Freedom and Democracy Day": 2
    },
    
    "Budapest_1": {
        "New Year's Day": 3, "Easter Monday": 3, "State Foundation Day": 3, "Christmas Day": 3,
        "Memorial Day of 1848 Revolution": 2, "Labour Day": 3, "1956 Memorial Day": 2,
        "Good Friday": 2, "All Saint's Day": 2, "Easter": 3, "Whit Monday": 3
    },
    
    "Munich_1": {
        "New Year's Day": 3, "Labour Day": 3, "Ascension Day": 2, "Assumption of Mary": 2, 
        "Epiphany": 2, "German Unity Day": 3, "Good Friday": 2, "Whit Monday": 3, 
        "All Saint's Day": 2, "Easter Monday": 3, "Corpus Christi": 2,
        "Christmas Day": 3
    },

    "Prague_1": {
        "New Year's Day": 3, "Labour Day": 3, "Jan Hus Day": 2, "Christmas Eve": 3,
        "Restoration Day of the Czech State": 3, "Victory Day": 2, "Saint Wencelas Day": 2,
        "Good Friday": 2, "VE Day": 2, "Christmas Day": 3, "Easter Monday": 3,
        "Saint Cyril and Methodius Day": 2, "Struggle for Freedom and Democracy Day": 2
    },

    "Prague_2": {
        "New Year's Day": 3, "Labour Day": 3, "Jan Hus Day": 2, "Christmas Eve": 3,
        "Restoration Day of the Czech State": 3, "Victory Day": 2, "Saint Wencelas Day": 2,
        "Good Friday": 2, "VE Day": 2, "Christmas Day": 3, "Easter Monday": 3,
        "Saint Cyril and Methodius Day": 2, "Struggle for Freedom and Democracy Day": 2
    },

    "Prague_3": {
        "New Year's Day": 3, "Labour Day": 3, "Jan Hus Day": 2, "Christmas Eve": 3,
        "Restoration Day of the Czech State": 3, "Victory Day": 2, "Saint Wencelas Day": 2,
        "Good Friday": 2, "VE Day": 2, "Christmas Day": 3, "Easter Monday": 3,
        "Saint Cyril and Methodius Day": 2, "Struggle for Freedom and Democracy Day": 2
    },

    "Frankfurt_1": {
        "New Year's Day": 3, "Good Friday": 2, "Ascension Day": 2, "Christmas Day": 3,
        "Whit Monday": 3, "Easter Monday": 3, "Labour Day": 3, "Corpus Christi": 2,
        "German Unity Day": 3
    }
}

## We will use a scoring system:

# High Impact (3) → Public holidays where most businesses and schools are closed.
# Moderate Impact (2) → Holidays where some businesses close but not all.
# Low Impact (1) → Observances that may not impact business hours much.


# Function to get holiday impact score based on region
def get_holiday_impact(row):
    region = row["warehouse"]  # Assuming region info is stored in this column
    holiday = row["holiday_name"]
    
    if region in holiday_impact_by_region and holiday in holiday_impact_by_region[region]:
        return holiday_impact_by_region[region][holiday]
    else:
        return 0  # Default impact if holiday is not in the predefined list

# Apply function to calendar_df
calendar_df["holiday_impact"] = calendar_df.apply(get_holiday_impact, axis=1)

# Ensure the column is integer type
calendar_df["holiday_impact"] = calendar_df["holiday_impact"].astype(int)


df_1 = sales_test_df
df_2 = sales_train_df
df_3 = inventory_df
df_4 = calendar_df 
df_5 = test_weights_df


print (df_1.info())


df_1.sort_values(by='date', inplace=True)
df_filtered = df_1[(df_1['date'] >= '2024-06-03') & (df_1['date'] <= '2024-06-15')]

# Sample every 20th row (adjust sampling rate as needed)

plt.figure(figsize=(15, 9))  # Adjust width and height

df_sampled = df_1.iloc[::300, :] ## Reduce Data Size for more clearer visualization

df_sampled.plot(
    kind='line', 
    x='date', 
    y='total_orders', 
    linewidth=2,
    color='blue'
)

plt.title('Total Orders over Time (2024-06-03 to 2024-06-15)')
plt.xlabel('Date')
plt.ylabel('Total Orders')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)  # Add dashed gridlines with transparency
plt.show()


df_2.sort_values(by='date', inplace=True)
df_filtered = df_2[(df_2['date'] >= '2020-08-01') & (df_2['date'] <= '2024-06-02')]

# Sample every 40th row (adjust sampling rate as needed)

plt.figure(figsize=(15, 9))  # Adjust width and height

df_sampled = df_2.iloc[::30000, :] ## Reduce Data Size for more clearer visualization

df_sampled.plot(
    kind='line', 
    x='date', 
    y='total_orders', 
    linewidth=2,
    color='blue'
)

plt.title('Total Orders over Time (2020-08-01 to 2024-06-02)')
plt.xlabel('Date')
plt.ylabel('Total Orders')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)  # Add dashed gridlines with transparency
plt.show()


print (df_2['date'].value_counts())


print (df_2['date'].max())


df_warehouse_3 = df_3['warehouse'].value_counts().reset_index()
df_warehouse_3.columns = ['Warehouse', 'Count']
print (df_warehouse_3)


import plotly.io as pio
pio.renderers.default = 'notebook'


df_warehouse_3 = pd.DataFrame({
    'Warehouse': ['Budapest_1', 'Prague_3', 'Prague_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Frankfurt_1'],
    'Count': [949, 867, 863, 860, 763, 643, 487]
})
figure_3 = px.treemap(
    df_warehouse_3,
    path=['Warehouse'],  # Hierarchy: each rectangle represents a warehouse
    values='Count',      # Size of each rectangle based on warehouse frequency
    color='Count',       # Color based on frequency
    color_continuous_scale='Blues'  # Blue gradient for clarity
)

figure_3.update_layout(
    title='Frequency of Warehouse Usage for Storage from Inventory',
    font=dict(size=14),
    margin=dict(t=50, l=25, r=25, b=25)
)

figure_3.show()


df_warehouse_2 = df_2['warehouse'].value_counts().reset_index()
df_warehouse_2.columns = ['Warehouse', 'Count']
print (df_warehouse_2)


df_warehouse_2 = pd.DataFrame({
    'Warehouse': ['Prague_1', 'Prague_3', 'Prague_2', 'Brno_1', 'Budapest_1', 'Munich_1', 'Frankfurt_1'],
    'Count': [780566, 779655, 770709, 643637, 574582, 259333, 198937]
})
figure_2 = px.treemap(
    df_warehouse_2,
    path=['Warehouse'],  # Hierarchy: each rectangle represents a warehouse
    values='Count',      # Size of each rectangle based on warehouse frequency
    color='Count',       # Color based on frequency
    color_continuous_scale='Blues'  # Blue gradient for clarity
)

figure_2.update_layout(
    title='Frequency of Warehouse Usage for Storage from Sales Train',
    font=dict(size=14),
    margin=dict(t=50, l=25, r=25, b=25)
)

figure_2.show()


df_warehouse_1 = df_1['warehouse'].value_counts().reset_index()
df_warehouse_1.columns = ['Warehouse', 'Count']
print (df_warehouse_1)


df_warehouse_1 = pd.DataFrame({
    'Warehouse': ['Prague_3', 'Prague_1', 'Prague_2', 'Brno_1', 'Budapest_1', 'Munich_1', 'Frankfurt_1'],
    'Count': [8751, 8714, 8655, 7196, 6576, 3907, 3222]
})
figure_1 = px.treemap(
    df_warehouse_1,
    path=['Warehouse'],  # Hierarchy: each rectangle represents a warehouse
    values='Count',      # Size of each rectangle based on warehouse frequency
    color='Count',       # Color based on frequency
    color_continuous_scale='Blues'  # Blue gradient for clarity
)

figure_1.update_layout(
    title='Frequency of Warehouse Usage for Storage from Sales Train',
    font=dict(size=14),
    margin=dict(t=50, l=25, r=25, b=25)
)

figure_1.show()


df_Lcategory = df_3[['L1_category_name_en', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']].value_counts().reset_index()
df_Lcategory.columns = ['L1_category_name_en', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en', 'Count']

# Display the result
print(df_Lcategory)


df_Lcategory_1 = df_3[['L1_category_name_en']].value_counts().reset_index()
df_Lcategory_1.columns = ['L1_category_name_en', 'Count']

# Display the result
print(df_Lcategory_1)


df_Lcategory_2 = df_3[['L2_category_name_en']].value_counts().reset_index()
df_Lcategory_2.columns = ['L2_category_name_en', 'Count']

# Display the result
print(df_Lcategory_2)


df_Lcategory_3 = df_3[['L3_category_name_en']].value_counts().reset_index()
df_Lcategory_3.columns = ['L3_category_name_en', 'Count']

# Display the result
print(df_Lcategory_3)


df_Lcategory_4 = df_3[['L4_category_name_en']].value_counts().reset_index()
df_Lcategory_4.columns = ['L4_category_name_en', 'Count']

# Display the result
print(df_Lcategory_4)


df_Lcategory_1.set_index('L1_category_name_en')['Count'].plot(
    kind='pie', 
    autopct='%1.1f%%', 
    figsize=(8, 8),
    startangle=90,    # Start angle for first slice
    explode=[0, 0, 0.1],  # Explode the slices
    title='Category Type Breakdown from L1_category'
)

plt.ylabel('')  # Remove the y-axis label
plt.show()





# Merge sales_train with calendar_df
sales_train_df = sales_train_df.merge(calendar_df, on='date', how='left')

# Merge sales_test with calendar_df
sales_test_df = sales_test_df.merge(calendar_df, on='date', how='left')


# Convert 'date' column to datetime format in both DataFrames
calendar_df['date'] = pd.to_datetime(calendar_df['date'])
sales_train_df['date'] = pd.to_datetime(sales_train_df['date'])
sales_test_df['date'] = pd.to_datetime(sales_test_df['date'])

# Merge sales_train with inventory_df
sales_train_df = sales_train_df.merge(inventory_df, on='unique_id', how='left')

# Merge sales_test with inventory_df
sales_test_df = sales_test_df.merge(inventory_df, on='unique_id', how='left')


# Merge sales_test with test_weights
sales_test_df = sales_test_df.merge(test_weights_df, on='unique_id', how='left')


#Merge sales_train with test_weights
sales_train_df = sales_train_df.merge(test_weights_df, on='unique_id', how='left')


print (sales_test_df.info())


print (sales_train_df.info())


print (sales_train_df.duplicated())


merged_sales_test_df =  sales_test_df
merged_sales_train_df = sales_train_df


# Total and average sales per category
category_sales = sales_train_df.groupby('L1_category_name_en')['sales'].agg(['sum', 'mean']).reset_index()
category_sales.rename(columns={'sum': 'category_sales_sum', 'mean': 'category_sales_avg'}, inplace=True)

# Merge into training and test datasets
sales_train_df = sales_train_df.merge(category_sales, on='L1_category_name_en', how='left')
sales_test_df = sales_test_df.merge(category_sales, on='L1_category_name_en', how='left')


# Add product-level sales or availability patterns
product_sales = sales_train_df.groupby('product_unique_id')['sales'].sum().reset_index()
product_sales.rename(columns={'sales': 'product_sales_sum'}, inplace=True)

# Merge with train and test
sales_train_df = sales_train_df.merge(product_sales, on='product_unique_id', how='left')
sales_test_df = sales_test_df.merge(product_sales, on='product_unique_id', how='left')


# Convert date to datetime and extract components
for df in [sales_train_df, sales_test_df]:
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)


# A feature for holiday impact
sales_train_df["final_holiday_impact"] = (
    sales_train_df["holiday_impact"] + sales_train_df["shops_closed"] + sales_train_df["school_holidays"]
)
sales_test_df["final_holiday_impact"] = (
    sales_test_df["holiday_impact"] + sales_test_df["shops_closed"] + sales_test_df["school_holidays"]
)


# Calculate the total discount applied in Sales_Train
discount_columns = [f'type_{i}_discount' for i in range(7)]
sales_train_df['total_discount'] = sales_train_df[discount_columns].sum(axis=1)

# Check the correlation between discount and sales in Sales_Train
discount_sales_corr = sales_train_df[['total_discount', 'sales']].corr()
print(discount_sales_corr)

# Inventory-to-sales ratio in Sales_Train Only
sales_train_df['inventory_to_sales'] = sales_train_df['availability'] / (sales_train_df['sales'] + 1)


# Calculate the total discount applied in Sales_Test
discount_columns_test = [f'type_{i}_discount' for i in range(7)]
sales_test_df['total_discount'] = sales_train_df[discount_columns_test].sum(axis=1)

# Check the correlation between discount and sales in Sales_Test
discount_sales_corr_test = sales_train_df[['total_discount', 'total_orders']].corr()
print(discount_sales_corr_test)


# Use weight as a feature for WAME
sales_test_df['weighted_price'] = sales_test_df['sell_price_main'] * sales_test_df['weight']


# Interaction features
for df in [sales_train_df, sales_test_df]:
    df['discount_price_interaction'] = df['total_discount'] * df['sell_price_main']
    df['category_holiday_interaction'] = df['holiday_impact'] * df['category_sales_avg']


# Food Feature 

# Define holiday-related high-demand categories

# Define holiday-related high-demand categories (only using L1_category_name_en values)
budapest_holiday_demand = {
    "New Year's Day": ["Meat and Fish", "Bakery"],
    "Easter Monday": ["Meat and Fish", "Bakery"],
    "State Foundation Day": ["Bakery"],
    "Christmas Day": ["Meat and Fish", "Bakery"],
    "Memorial Day of 1848 Revolution": ["Meat and Fish", "Bakery"],
    "Labour Day": ["Bakery"],
    "1956 Memorial Day": ["Meat and Fish", "Bakery"],
    "2nd Christmas Day": ["Meat and Fish", "Bakery"],
    "Good Friday": ["Meat and Fish"],
    "Whitsun": ["Bakery"],
    "All Saints' Day": ["Bakery"],
    "Easter": ["Meat and Fish", "Bakery"],
    "Whit Monday": ["Bakery"]
}

frankfurt_holiday_demand = {
    "New Year's Day": ["Meat and Fish", "Bakery"],
    "Good Friday": ["Meat and Fish"],
    "Ascension Day": ["Bakery"],
    "Christmas Day": ["Meat and Fish", "Bakery"],
    "Whit Monday": ["Bakery"],
    "Easter Monday": ["Meat and Fish", "Bakery"],
    "Labour Day": ["Bakery"],
    "Corpus Christi": ["Bakery"],
    "German Unity Day": ["Bakery"]
}

munich_holiday_demand = {
    "New Year's Day": ["Meat and Fish", "Bakery"],
    "Labour Day": ["Bakery"],
    "Ascension Day": ["Bakery"],
    "2nd Day of Christmas": ["Meat and Fish", "Bakery"],
    "Assumption of Mary": ["Bakery"],
    "Epiphany": ["Bakery"],
    "German Unity Day": ["Bakery"],
    "Good Friday": ["Meat and Fish"],
    "Whit Monday": ["Bakery"],
    "All Saints' Day": ["Bakery"],
    "Easter Monday": ["Meat and Fish", "Bakery"],
    "Corpus Christi": ["Bakery"],
    "Christmas Day": ["Meat and Fish", "Bakery"]
}

prague_holiday_demand = {
    "New Year's Day": ["Meat and Fish", "Bakery"],
    "Labour Day": ["Bakery"],
    "Jan Hus Day": ["Bakery"],
    "Christmas Eve": ["Meat and Fish", "Bakery"],
    "Restoration Day of the Czech State": ["Bakery"],
    "Victory Day": ["Meat and Fish"],
    "Saint Wenceslas Day": ["Bakery"],
    "Good Friday": ["Meat and Fish"],
    "VE Day": ["Meat and Fish"],
    "Christmas Day": ["Meat and Fish", "Bakery"],
    "Easter Monday": ["Meat and Fish", "Bakery"],
    "Saint Cyril and Methodius Day": ["Bakery"],
    "Struggle for Freedom and Democracy Day": ["Bakery"]
}

brno_holiday_demand = {
    "New Year's Day": ["Meat and Fish", "Bakery"],
    "Labour Day": ["Bakery"],
    "Jan Hus Day": ["Bakery"],
    "Christmas Eve": ["Meat and Fish", "Bakery"],
    "Restoration Day of the Czech State": ["Bakery"],
    "Victory Day": ["Meat and Fish"],
    "Saint Wenceslas Day": ["Bakery"],
    "Good Friday": ["Meat and Fish"],
    "VE Day": ["Meat and Fish"],
    "Christmas Day": ["Meat and Fish", "Bakery"],
    "Easter Monday": ["Meat and Fish", "Bakery"],
    "Saint Cyril and Methodius Day": ["Bakery"],
    "Struggle for Freedom and Democracy Day": ["Bakery"]
}


# Function to assign high-demand L1 categories for each warehouse
def assign_high_demand_l1(row):
    warehouse = row['warehouse']  # Assuming 'warehouse' column exists
    holiday = row['holiday_name']
    
    if warehouse == "Budapest_1":
        return ', '.join(budapest_holiday_demand.get(holiday, ["None"]))
    elif warehouse == "Frankfurt_1":
        return ', '.join(frankfurt_holiday_demand.get(holiday, ["None"]))
    elif warehouse == "Munich_1":
        return ', '.join(munich_holiday_demand.get(holiday, ["None"]))
    elif warehouse == "Prague_1":
        return ', '.join(prague_holiday_demand.get(holiday, ["None"]))
    elif warehouse == "Brno_1":
        return ', '.join(brno_holiday_demand.get(holiday, ["None"]))
    return "None"

# Apply the function to create the feature in both train and test sets
sales_train_df['high_demand_l1_category'] = sales_train_df.apply(assign_high_demand_l1, axis=1)
sales_test_df['high_demand_l1_category'] = sales_test_df.apply(assign_high_demand_l1, axis=1)

# One-hot encode high-demand categories
sales_train_df = sales_train_df.join(
    sales_train_df['high_demand_l1_category'].str.get_dummies(sep=', ')
)
sales_test_df = sales_test_df.join(
    sales_test_df['high_demand_l1_category'].str.get_dummies(sep=', ')
)

# Drop the original string column
sales_train_df.drop(columns=['high_demand_l1_category'], inplace=True)
sales_test_df.drop(columns=['high_demand_l1_category'], inplace=True)


# Define target variable
target = 'sales'  # In train dataset only

# Define feature columns (exclude target and irrelevant columns)
feature_columns = [col for col in sales_train_df.columns if col not in ['sales', 'date', 'unique_id', 'name']]

# Separate features and target
X_train = sales_train_df[feature_columns]
y_train = sales_train_df[target]


print (sales_train_df.info())


print (sales_test_df.info())


# Check column data types
print(X_train.dtypes)

# Identify non-numeric columns
non_numeric_cols = X_train.select_dtypes(include=['object']).columns
print(f"Non-numeric columns: {list(non_numeric_cols)}")


X_train = pd.get_dummies(X_train, columns=non_numeric_cols, drop_first=True)


X_train.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)


## Calculate WMAE
def calculate_wmae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


# Convert float columns to float32 and int columns to int32
for col in X_train.columns:
    if X_train[col].dtype == 'float64':
        X_train[col] = X_train[col].astype('float32')
    elif X_train[col].dtype == 'int64':
        X_train[col] = X_train[col].astype('int32')


from xgboost import XGBRegressor

# Initialize XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

X_train_sample = X_train.sample(frac=0.5, random_state=42)  # Use 50% of the data
y_train_sample = y_train.loc[X_train_sample.index]  # Get corresponding target values

xgb_model.fit(X_train_sample, y_train_sample)  # Train on the reduced dataset

# Predict on train data
y_pred_xgb_sample = xgb_model.predict(X_train_sample)

# Compute WMAE for XGBoost
wmae_xgb_sample = calculate_wmae(y_train_sample, y_pred_xgb_sample, sales_train_df['weight'])
print(f'XGBoost Training WMAE: {wmae_xgb_sample}')


if 'availability' not in sales_test_df.columns:
    sales_test_df['availability'] = sales_test_df['total_orders'] / (sales_test_df['sell_price_main'] + 1)
if 'inventory_to_sales' not in sales_test_df.columns:
    median_inventory_to_sales = sales_train_df['inventory_to_sales'].median()
    sales_test_df['inventory_to_sales'] = median_inventory_to_sales


# Reapply one-hot encoding ensuring missing columns are handled
l1_categories = ["Bakery", "Meat and Fish", "Fruit and Vegetable"]

# Ensure all categories exist in both train and test
for category in l1_categories:
    if category not in sales_test_df.columns:
        sales_test_df[category] = 0  # Add missing categories with default value 0
    if category not in sales_train_df.columns:
        sales_train_df[category] = 0

# Re-check columns
print(sales_test_df.columns)


X_test = sales_test_df[feature_columns]


missing_features = [col for col in feature_columns if col not in sales_test_df.columns]
print(f"Missing in test set: {missing_features}")


non_numeric_cols_test = X_test.select_dtypes(include=['object']).columns
print(f"Non-numeric columns: {list(non_numeric_cols_test)}")


X_test = pd.get_dummies(X_test, columns=non_numeric_cols_test, drop_first=True)


print(f"Non-numeric columns: {list(non_numeric_cols_test)}")


# Check for missing columns in test set
missing_cols_in_test = set(X_train.columns) - set(X_test.columns)
print(f"Missing in test: {missing_cols_in_test}")


# Fill missing holiday names with "N/A" to ensure consistency
sales_train_df['holiday_name'] = sales_train_df['holiday_name'].fillna("N/A")
sales_test_df['holiday_name'] = sales_test_df['holiday_name'].fillna("N/A")

# Ensure holiday names in train and test are from the same set
common_holidays = set(sales_train_df['holiday_name'].unique()).intersection(set(sales_test_df['holiday_name'].unique()))

# Replace unknown holidays with "N/A" in train and test
sales_train_df['holiday_name'] = sales_train_df['holiday_name'].apply(lambda x: x if x in common_holidays else "N/A")
sales_test_df['holiday_name'] = sales_test_df['holiday_name'].apply(lambda x: x if x in common_holidays else "N/A")


for col in missing_cols_in_test:
    X_test[col] = 0  # Assign a default value

X_test = X_test[X_train.columns]  # Ensure column order matches


# Check for missing columns in test set again
missing_cols_in_test = set(X_train.columns) - set(X_test.columns)
print(f"Missing in test: {missing_cols_in_test}")


# Check for extra columns in test set
extra_in_test = set(X_test.columns) - set(X_train.columns)
print(f"Extra in test: {extra_in_test}")


sales_test_df['sales_hat'] = xgb_model.predict(X_test)


# Ensure 'id' column is correctly formatted
sales_test_df['id'] = sales_test_df['unique_id'].astype(str) + "_" + sales_test_df['date'].astype(str)

# Keep only the required columns for submission
submission_df = sales_test_df[['id', 'sales_hat']]

submission_df = submission_df.drop_duplicates()

# Display the first few rows
print(submission_df)


submission_df.to_csv("submission.csv", index=False)


submission_df

