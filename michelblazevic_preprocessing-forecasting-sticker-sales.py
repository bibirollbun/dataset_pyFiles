import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import json


dtrain = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
# Transform column date as datetime to extract informations
dtrain['date'] = pd.to_datetime(dtrain['date'])
df_gpd_per_capitagrowth = pd.read_csv('/kaggle/input/daily-time-series-global-public-and-school-holiday/GPDperCapitaGrowth.csv')
df_revenue_excluding_grants = pd.read_csv('/kaggle/input/daily-time-series-global-public-and-school-holiday/Revenueexcludinggrants.csv')
df_unesco_demo = pd.read_csv('/kaggle/input/daily-time-series-global-public-and-school-holiday/unesco_csv_prep.csv')
df_holiday_per_country = pd.read_csv('/kaggle/input/daily-time-series-global-public-and-school-holiday/day_public_and_school_holidays_2010_2019.csv')
df_location = pd.read_csv('/kaggle/input/daily-time-series-global-public-and-school-holiday/countries_location.csv')


dtrain['is_weekend'] = dtrain['date'].dt.weekday >= 5  # 5 and 6 are Saturday and Sunday
dtrain['day'] = dtrain['date'].dt.day # begin or end of the month could be interesting 
dtrain['year'] = dtrain['date'].dt.year
dtrain['month'] = dtrain['date'].dt.month


dtrain['latitude'] = pd.merge(dtrain[['date','country']],df_location[['latitude','name']],left_on='country',right_on='name',how='left').latitude


def get_season(month,latitude):
    """Function taking month and latitude to return a ordinal value of the season with -2 for winter up to summer at 2"""
    if month in [12, 1, 2]:
        return -2 if latitude>0 else 2
    elif month in [3, 4, 5]:
        return 1 if latitude>0 else -1
    elif month in [6, 7, 8]:
        return 2 if latitude>0 else -2
    elif month in [9, 10, 11]:
        return -1 if latitude>0 else 1


dtrain.isna().mean()


dtrain['season'] = dtrain.apply(lambda x : get_season(x['month'],x['latitude']),axis=1)
list_of_country = set(dtrain.country.unique())
dholiday = df_holiday_per_country[df_holiday_per_country['ADM_name'].apply(lambda x:x in list_of_country)][['ADM_name','Year','Month','Day','holiday','school','hl_sch','all_break']]
dholiday.columns = ['country','year','month','day','holiday',"school",'hl_sch','all_break']
dtrain = pd.merge(dtrain,dholiday,on=['country','year','month','day'],how='left')


dtrain.iloc[:,-3:].mean()


dtrain.isna().mean()


df_gpd_per_capitagrowth.columns = ['country','year','GPD_per_cap_growth_percent']
dtrain = pd.merge(dtrain,df_gpd_per_capitagrowth,on=['country','year'],how='left')
# df_revenue_excluding_grants.columns = ['country','year','Revenue_excluding_grants_percent_gpd']
# dtrain = pd.merge(dtrain,df_revenue_excluding_grants,on=['country','year'],how='left')


dtrain[dtrain.country=='Kenya']

