# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


traindf = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=True)
traindf.head()


traindf.describe()


traindf.dtypes


# 1. Import the dataset into a pandas dataframe. Make sure that the date column is in pandas date time format.

traindf['datetime'] = pd.to_datetime(traindf['datetime'])
traindf.dtypes


# 2. Check the data type of each column. How many rows are there in the dataset ? Does the dataset contain any missing values ?
traindf.isna().sum()


# 3. Using the date column, create new columns for: year, month, day of the week and hour of the day.
traindf['year'] = traindf['datetime'].dt.year
traindf['month'] = traindf['datetime'].dt.month
traindf['day'] = traindf['datetime'].dt.day
traindf['dayofweek'] = traindf['datetime'].dt.dayofweek
traindf['hour'] = traindf['datetime'].dt.hour
traindf.head()


# 4. Rename the values in the season column to spring, summer, fall and winter.
def seasonIntToString(season):
    return {
        1: "spring",
        2: "summer",
        3: "fall",
        4: "winter"
    }[season]
traindf['season'] = traindf['season'].apply(seasonIntToString)


traindf.head()


# 5. Calculate the total number of casual and registered bikes rented in the years 2011 and 2012.

traindf[['casual','registered']].sum()


# 6. Calculate the mean of the hourly total rentals count by season. Which season has the highest mean ?

mean_rentals_by_season = traindf.groupby("season")['count'].mean()
mean_rentals_by_season


highest_mean_season = mean_rentals_by_season.idxmax()
highest_mean_value = mean_rentals_by_season.max()
print( "highest_mean_season: ", highest_mean_season, ", highest_mean_value: ", highest_mean_value)


# 7. Are more bikes rented by registered users on working or non-working days ? Does the answer differ for non-registered users ? Is the answer the same for both years ?

mean_rentals_by_workingday = traindf.groupby("workingday")['count'].mean()
mean_rentals_by_workingday


if (mean_rentals_by_workingday[1] > mean_rentals_by_workingday[0]):
    print("more bikes rented on working days")
else:
    print("more bikes rented on non-working days")



mean_rentals_by_workingday_registered = traindf.groupby("workingday")['registered'].mean()
mean_rentals_by_workingday_registered


if (mean_rentals_by_workingday_registered[1] > mean_rentals_by_workingday_registered[0]):
    print("more bikes rented on working days by registered")
else:
    print("more bikes rented on non-working days by registered")



traindf['non_register'] = traindf['count'] - traindf['registered']
mean_rentals_by_workingday_nonregistered = traindf.groupby("workingday")['non_register'].mean()
mean_rentals_by_workingday_nonregistered


if (mean_rentals_by_workingday_nonregistered[1] > mean_rentals_by_workingday_nonregistered[0]):
    print("more bikes rented on working days by non registered")
else:
    print("more bikes rented on non-working days by non registered")



traindf['year'].unique()


mean_rentals_by_workingday_year = traindf.groupby(["workingday","year"])['count'].mean()
mean_rentals_by_workingday_year


mean_rentals_by_workingday_year_registered = traindf.groupby(["workingday","year"])['registered'].mean()
mean_rentals_by_workingday_year_registered


mean_rentals_by_workingday_year_nonregistered = traindf.groupby(["workingday","year"])['non_register'].mean()
mean_rentals_by_workingday_year_nonregistered


#8. Which months in the year 2011 have the highest and the lowest total number of bikes rented ? Repeat for the year 2012.

total_rentals_by_month_year = traindf.groupby(["year","month"])['count'].sum()
total_rentals_by_month_year


maxmonth2011 = total_rentals_by_month_year[2011].idxmax()
minmonth2011 = total_rentals_by_month_year[2011].idxmin()

print (" Max and min month in 2011: ", maxmonth2011, minmonth2011)


maxmonth2012 = total_rentals_by_month_year[2012].idxmax()
minmonth2012 = total_rentals_by_month_year[2012].idxmin()

print (" Max and min month in 2012: ", maxmonth2012, minmonth2012)


# 9. Which type of weather have the highest and lowest mean of the hourly total rentals count ?
mean_rentals_by_weather = traindf.groupby(["weather"])['count'].mean()
mean_rentals_by_weather


max_rentals_by_weather = mean_rentals_by_weather.idxmax()
min_rentals_by_weather = mean_rentals_by_weather.idxmin()

print (" Max and min weather: ", max_rentals_by_weather, min_rentals_by_weather)

weather_map = {
    1: 'Clear, Few clouds, Partly cloudy, Partly cloudy',
    2: 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',
    3: 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',
    4: 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog'
}

print (" Max and min weather: ", weather_map[max_rentals_by_weather], weather_map[min_rentals_by_weather])




#10. Calculate the correlation between the hourly total rentals count and all the numerical columns in the dataset. Which column has the highest correlation with the total rentals count ?

traindf_numerical = traindf[["temp",	"atemp",	"humidity",	"windspeed","count"]]
correlation = traindf_numerical.corr()
correlation


correlation_with_count = correlation['count'].drop('count')
correlation_with_count


print("highest correlation with count: ", correlation_with_count.idxmax())


# 11. Create a new categorical column called day_period, which can take four
# possible values: night, morning, afternoon and evening. These values
# correspond to the following binning of the hour column: 0-6: night, 6-12: morning,
# 12-6: afternoon, 6-24:evening.

def get_day_period(hour):
    if 0 <= hour < 6:  # 0, 1, 2, 3, 4, 5
        return 'night'
    elif 6 <= hour < 12: # 6, 7, 8, 9, 10, 11
        return 'morning'
    elif 12 <= hour < 18: # 12, 13, 14, 15, 16, 17
        return 'afternoon'
    else:  # 18, 19, 20, 21, 22, 23
        return 'evening'
traindf['day_period']=traindf['hour'].apply(get_day_period)
traindf


# 12. Generate a pivot table for the mean of the hourly total rentals count, with the
# index set to the day period and the column set to the working day column. What
# can you observe from the table ?

mean_rentals_by_dayperiod = traindf.groupby(["day_period"])['count'].mean()
mean_rentals_by_dayperiod


df_mean_rentals_by_dayperiod = pd.DataFrame(mean_rentals_by_dayperiod).reset_index()
df_mean_rentals_by_dayperiod


pivot_table_rentals = pd.pivot_table(
    traindf,
    values='count',
    index='day_period',
    columns='workingday',
    aggfunc='mean'
)
pivot_table_rentals


# 1. Plot the distributions of all the numerical columns in the dataset using histograms.

for col in traindf_numerical.columns:
    ax = traindf_numerical[col].hist()
    plt.title(col)
    plt.show()


# 2. Plot the distributions of all the numerical columns in the dataset using box plots.
traindf_numerical.boxplot()



traindf_numerical.drop("count", axis=1).boxplot()


traindf_numerical[["count"]].boxplot()


# 3. Plot the the mean of the hourly total rentals count for working and non-working days.
pd.DataFrame(mean_rentals_by_workingday).plot(kind="bar", title = "mean of the hourly total rentals count for working and non-working days")


# 4. Plot the the mean of the hourly total rentals count for the different months for both years combined.

mean_rentals_by_month_year = traindf.groupby(["year","month"])['count'].mean()
mean_rentals_by_month_year


# 4. Plot the the mean of the hourly total rentals count for the different months for both years combined.

mean_rentals_by_month = traindf.groupby(["month"])['count'].mean()
mean_rentals_by_month


mean_rentals_by_month.plot(kind="bar", title="mean of the hourly total rentals count for the different months for both years combined.")


# 5. Plot the the mean of the hourly total rentals count for the different months for both years separately in a multi-panel figure.

mean_rentals_by_month_year[2011].plot(kind="bar", title="mean of the hourly total rentals count for the different months for 2011")
plt.show()


mean_rentals_by_month_year[2012].plot(kind="bar", title="mean of the hourly total rentals count for the different months for 2012")
plt.show()


# 6. Plot the the mean and the 95% confidence interval of the hourly total rentals count for the four different weather categories. What can you observe ?

mean_rentals_by_weather = traindf.groupby(["weather"])['count'].mean()
mean_rentals_by_weather.plot(kind="bar", title="mean of the hourly total rentals count for the four different weather ")
plt.show()




sns.barplot(x='weather', y='count', data=traindf, errorbar=('ci', 95), palette='viridis')
plt.title('Mean Hourly Total Rentals by Weather Category (with 95% CI)')
plt.xlabel('Weather Category')
plt.ylabel('Mean Total Rentals Count')
plt.xticks(rotation=15) # Rotate labels slightly for readability
plt.tight_layout()
plt.show()


# 7. Plot the the mean of the hourly total rentals count versus the hour of the day. Which hours of the day have the highest rentals count ?


mean_rentals_by_hour = traindf.groupby(["hour"])['count'].mean()
mean_rentals_by_hour


mean_rentals_by_hour.plot(kind="bar", title = "the mean of the hourly total rentals count versus the hour of the day")


print("hour of the day with highest rent : ", mean_rentals_by_hour.idxmax())


# 8. Repeat the plot in 7 for different days of the week. What patterns can you observe ?
mean_rentals_by_dayofweek = traindf.groupby(["dayofweek"])['count'].mean()
mean_rentals_by_dayofweek


mean_rentals_by_dayofweek.plot(kind="bar", title = "the mean of the hourly total rentals count versus the day of the week")


#9. Repeat the plot in 8 for the four seasons using a multi-panel figure. What patterns can you observe ?

def plot_season(season):
    mean_rentals_by_dayofweek_byseason = traindf[traindf['season']==season].groupby(["dayofweek"])['count'].mean()
    title = "the mean of the hourly total rentals count versus the day of the week for season " + season
    mean_rentals_by_dayofweek_byseason.plot(kind="bar", title = title)
    plt.show()

for season in traindf['season'].unique():
    plot_season(season)


# 10. Plot the the mean and the 95% confidence interval of the hourly total rentals
# count versus the period of the day column, which you created in the first part of
# the assignment. Which period of the day has the highest rentals count ? Does
# this peak period differ for working and non-working days ?

mean_rentals_by_dayperiod.plot(kind="bar")


pivot_table_rentals


pivot_table_rentals[0].plot(kind="bar", title="On non-working day")


pivot_table_rentals[1].plot(kind="bar", title="On working day")


#  11. Plot a heatmap for the correlation matrix of the dataset numerical variables. What observations can you make ?

sns.heatmap(correlation, annot=True)

