import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


train_df = pd.read_csv("/kaggle/input/playground-series-s3e20/train.csv")
test_df = pd.read_csv ("/kaggle/input/playground-series-s3e20/test.csv")


train_df


train_df.info()


drop_column = [57, 58, 59, 60, 61, 62, 63]

train_df = train_df.drop(train_df.columns[drop_column], axis=1)

train_df


train_df.info(0)


def data_cleaning(df, start_column=5):
    df = df.copy()

    for col in df.columns[start_column:]:
        
        # Convert to numeric safely (strings → NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Skip columns that are completely NaN
        if df[col].isnull().all():
            continue
        
        # Fill NaN with column mean
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)

    return df



train_df = data_cleaning(train_df, start_column=5)
train_df.info()


train_df.describe()


train_df.head(10)


avg_yearly_emission = (train_df.groupby('year', as_index=False)['emission'].mean())

avg_yearly_emission = avg_yearly_emission.round(2)




print(avg_yearly_emission)


def plot_yearly_trend(df, year_col, value_col, agg='mean'):
        df = df.copy()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    
        yearly_data = (
            df
            .groupby(year_col, as_index=False)[value_col]
            .agg(agg)
            .sort_values(year_col)
        )

    
        plt.figure(figsize=(8, 5))
        plt.plot(
            yearly_data[year_col],
            yearly_data[value_col],
            marker='o',
            linestyle='-'
        )
    
        plt.xlabel(year_col.capitalize())
        plt.ylabel(f"{agg.capitalize()} {value_col}")
        plt.title(f"{agg.capitalize()} {value_col} by {year_col}")
        plt.grid(True)
        plt.show()


plot_yearly_trend(
    train_df,
    year_col='year',
    value_col='emission',
    agg='mean'
)


plot_yearly_trend(
    train_df,
    year_col='year',
    value_col='CarbonMonoxide_CO_column_number_density',
    agg='mean'
)


plot_yearly_trend(
    train_df,
    year_col='year',
    value_col='UvAerosolIndex_absorbing_aerosol_index',
    agg='mean'
)


plot_yearly_trend(
    train_df,
    year_col='year',
    value_col='Ozone_O3_column_number_density',
    agg='mean'
)


plot_yearly_trend(
    train_df,
    year_col='year',
    value_col='Cloud_cloud_base_pressure',
    agg='mean'
)


train_df['geography'] = train_df['longitude'].astype(str) + "_" + train_df ['latitude'].astype(str)
train_df.head()


len(train_df['geography'].unique())


print(list(train_df['geography'].unique()))


# Approximate month from week number (0-51)
train_df['month'] = ((train_df['week_no']) // 4 + 1).clip(upper=12)


train_df.tail(10)


month_avg_by_area = (
    train_df
    .groupby(['year','month', 'geography'], as_index=False)['emission']
    .mean()
)
month_avg_by_area = month_avg_by_area.round(2)


print(len(month_avg_by_area))


def plot_year_month_for_geography(df, value_col, geography_value, plot=True):
   
    df = df.copy()
    
    # Filter by geography
    df_geo = df[df['geography'] == geography_value]
    
    if df_geo.empty:
        print(f"No data found for geography: {geography_value}")
        return None
    
    # Ensure column is numeric
    df_geo[value_col] = pd.to_numeric(df_geo[value_col], errors='coerce')
    
    # Group by year and month
    month_avg = (
        df_geo.groupby(['year', 'month'], as_index=False)[value_col]
        .mean()
        .round(2)
    )
    
    # Plot line chart
    if plot:
        plt.figure(figsize=(10,6))
        
        # X-axis: continuous month over multiple years
        x = month_avg['month'] + (month_avg['year'] - month_avg['year'].min())*12
        plt.plot(x, month_avg[value_col], marker='o', linestyle='-', color='blue')
        
        plt.xlabel('Month (continuous over years)')
        plt.ylabel(f'Average {value_col}')
        plt.title(f'Year-Month-wise Average {value_col} for {geography_value}')
        plt.grid(True)
        plt.show()
    
   



# Example: Plot emission for a specific geography
result = plot_year_month_for_geography(
    train_df, 
    value_col='emission', 
    geography_value='29.29_-0.51'
)

print(result)



# Example: Plot emission for a specific geography
result = plot_year_month_for_geography(
    train_df, 
    value_col='emission', 
    geography_value='29.826_-1.074'
)

print(result)



test_df.head()


test_df.info()


test_df = test_df.drop(test_df.columns[drop_column], axis=1)

test_df.info()


test_df = data_cleaning(test_df, start_column=5)
test_df.info()


# Select numeric columns
numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns

# Compute absolute correlation with 'emission'
top20_corrs = abs(train_df[numeric_cols].corr()['emission']).sort_values(ascending=False).head(20)

# Display top 20 correlated features
print(top20_corrs)


