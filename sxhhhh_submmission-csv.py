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


# Read the datasets into pandas DataFrames
macro_economic_df = pd.read_excel('/kaggle/input/walmart-sales-prediction-hyd-nov-2023/macro_economic.xlsx')
events_holidays_df = pd.read_excel('/kaggle/input/walmart-sales-prediction-hyd-nov-2023/Events_HolidaysData.xlsx')
weather_df = pd.read_excel('/kaggle/input/walmart-sales-prediction-hyd-nov-2023/WeatherData.xlsx')
train_data_df = pd.read_csv('/kaggle/input/walmart-sales-prediction-hyd-nov-2023/train.csv')


# Display the first few rows of each DataFrame to verify
print("Macro Economic Dataset:")
print(macro_economic_df.head())

print("\nEvents and Holidays Dataset:")
print(events_holidays_df.head())

print("\nWeather Data Set:")
print(weather_df.head())

print("\nTrain Data:")
print(train_data_df.head())


# Check the data types and missing values in each DataFrame
print("Macro Economic Dataset Info:")
print(macro_economic_df.info())
print("\nMacro Economic Dataset Missing Values:")
print(macro_economic_df.isnull().sum())

print("\nEvents and Holidays Dataset Info:")
print(events_holidays_df.info())
print("\nEvents and Holidays Dataset Missing Values:")
print(events_holidays_df.isnull().sum())

print("\nWeather Data Set Info:")
print(weather_df.info())
print("\nWeather Data Set Missing Values:")
print(weather_df.isnull().sum())

print("\nTrain Data Info:")
print(train_data_df.info())
print("\nTrain Data Missing Values:")
print(train_data_df.isnull().sum())

# Display basic statistics for numerical columns
print("\nMacro Economic Dataset Statistics:")
print(macro_economic_df.describe())

print("\nEvents and Holidays Dataset Statistics:")
print(events_holidays_df.describe(include='all'))

print("\nWeather Data Set Statistics:")
print(weather_df.describe())

print("\nTrain Data Statistics:")
print(train_data_df.describe())


print("\nMacro Economic Dataset Data Types:")
print(macro_economic_df.dtypes)


print("\nEvents and Holidays Dataset Data Types:")
print(events_holidays_df.dtypes)


print("\nWeather Data Set Data Types:")
print(weather_df.dtypes)


print("\nTrain Data Data Types:")
print(train_data_df.dtypes)


from sklearn.preprocessing import LabelEncoder


# Check the column names in the DataFrame
print("Columns in Macro Economic Dataset:")
print(macro_economic_df.columns)


if 'Year-Month' in macro_economic_df.columns:
    # Specify the date format, assuming the format is 'YYYY - MMM'
    date_format = '%Y - %b'

    # Convert "Year-Month" to separate "Year" and "Month" columns
    macro_economic_df['Year'] = pd.to_datetime(macro_economic_df['Year-Month'], format=date_format).dt.year
    macro_economic_df['Month'] = pd.to_datetime(macro_economic_df['Year-Month'], format=date_format).dt.month
    macro_economic_df.drop('Year-Month', axis=1, inplace=True)
else:
    print("Column 'Year-Month' does not exist. Please check the column names and format.")

# Proceed with other transformations if 'Year-Month' exists
if 'Year' in macro_economic_df.columns and 'Month' in macro_economic_df.columns:
    # Convert "PartyInPower" to numerical using Label Encoding
    le_macro_party = LabelEncoder()
    macro_economic_df['PartyInPower'] = le_macro_party.fit_transform(macro_economic_df['PartyInPower'])

    # 方案2：直接使用 pd.to_numeric（推荐）
    macro_economic_df['AdvertisingExpenses (in Thousand Dollars)'] = pd.to_numeric(
        macro_economic_df['AdvertisingExpenses (in Thousand Dollars)'], 
        errors='coerce'
    )

    # Print the updated data types
    print("\nUpdated Macro Economic Dataset Data Types:")
    print(macro_economic_df.dtypes)
else:
    print("Year and Month columns were not created. Please check the 'Year-Month' conversion step.")


# Initialize LabelEncoder
le = LabelEncoder()

# Convert 'Event' column to numeric
events_holidays_df['Event'] = le.fit_transform(events_holidays_df['Event'])

# Convert 'DayCategory' column to numeric
events_holidays_df['DayCategory'] = le.fit_transform(events_holidays_df['DayCategory'])

# Print the updated data types
print("\nUpdated Events and Holidays Dataset Data Types:")
print(events_holidays_df.dtypes)

# Print a sample of the transformed DataFrame
print("\nTransformed Events and Holidays DataFrame Sample:")
print(events_holidays_df.head())


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Convert 'Month' column to numeric
weather_df['Month'] = le.fit_transform(weather_df['Month'])

# Convert 'WeatherEvent' column to numeric
weather_df['WeatherEvent'] = le.fit_transform(weather_df['WeatherEvent'])

# Debugging: Print out column names in weather_df
print("Column Names in weather_df:")
print(weather_df.columns)

# Convert 'Wind (km/h) low', 'Wind (km/h) avg', 'Wind (km/h) high', and 'Precip. (mm) sum' to numeric
columns_to_convert = ['Wind (km/h) low', 'Wind (km/h) avg', 'Wind (km/h) high', 'Precip. (mm) sum']
for column in columns_to_convert:
    # Debugging: Print the column being processed
    print(f"Processing column: {column}")
    
    # Check if the column exists in weather_df
    if column in weather_df.columns:
        weather_df[column] = pd.to_numeric(weather_df[column], errors='coerce')
    else:
        print(f"Column '{column}' not found in weather_df.")

# Print the updated data types
print("\nUpdated Weather Data Set Data Types:")
print(weather_df.dtypes)

# Print a sample of the transformed DataFrame
print("\nTransformed Weather DataFrame Sample:")
print(weather_df.head())


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Convert 'ProductCategory' column to numerical
train_data_df['ProductCategory'] = le.fit_transform(train_data_df['ProductCategory'])

# Print the updated data types
print("\nUpdated Train Data Data Types:")
print(train_data_df.dtypes)

