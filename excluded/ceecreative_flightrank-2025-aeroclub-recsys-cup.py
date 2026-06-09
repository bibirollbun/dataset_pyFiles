# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


import dask.dataframe as dd


train = dd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')


train.head()


train.shape[0].compute() #check rows


train.shape[1] #check columns


def divide_df(df,chunk_size,threshold_ratio):
    #get total number of rows
    total_rows = df.shape[0].compute()

    #create threshold
    threshold = total_rows * threshold_ratio

    #create number of times/chunks 
    chunk_number = math.ceil(len(df.columns)/chunk_size)

    #create 3 lists to store each columns
    cols_1 = []  #to store df with no null cvalues
    cols_2 = []  #to store cols with less than 50% null values
    cols_3 = []  #store cols with  more than 50% null values
 
    for i in range(chunk_number):
        #select null columns in each column chunks
        null_cols = df.iloc[:,i*chunk_size:(i+1)*chunk_size].isnull().sum().compute()
        
        list1 = null_cols[null_cols == 0].index.tolist()
        list2 = null_cols[(null_cols > 0) & (null_cols <= threshold)].index.tolist()
        list3 = null_cols[null_cols > threshold].index.tolist()
        
        cols_1.extend(list1)
        cols_2.extend(list2)
        cols_3.extend(list3)


    return cols_1,cols_2,cols_3


no_nulls,less_50,more_50 = divide_df(train,30,0.5)


df1 = train[no_nulls]  #df with no null values
df2 = train[less_50]  #df with null values less than 50%
df3 = train[more_50]  #df with null values more fhan 50%


train_copy = train.copy()


df2['has_leg1'] = df2['legs1_arrivalAt'].notnull().astype(int)


new_df2 = df2[['legs0_segments0_aircraft_code','legs0_segments0_arrivalTo_airport_city_iata',
     'legs0_segments0_arrivalTo_airport_iata',
     'legs0_segments0_baggageAllowance_quantity',
     'legs0_segments0_baggageAllowance_weightMeasurementType',
     'legs0_segments0_departureFrom_airport_iata',
     'miniRules0_monetaryAmount','miniRules0_statusInfos',
     'miniRules1_monetaryAmount','miniRules1_statusInfos','pricingInfo_isAccessTP',
     'has_leg1']]


df3['has_leg0_segment1'] = df3[
'legs0_segments1_aircraft_code'].isnull().astype(int)


df3['has_leg0_segment2'] = df3[
'legs0_segments2_aircraft_code'].isnull().astype(int)


df3['has_leg0_segment3'] = df3[
'legs0_segments3_aircraft_code'].isnull().astype(int)


df3['has_leg1_segment1'] = df3[
'legs1_segments1_aircraft_code'].isnull().astype(int)


df3['has_leg1_segment2'] = df3[
'legs1_segments2_aircraft_code'].isnull().astype(int)


df3['has_leg1_segment3'] = df3[
'legs1_segments3_aircraft_code'].isnull().astype(int)


new_df3 = df3[['has_leg0_segment1','has_leg0_segment2',
               'has_leg0_segment3','has_leg1_segment1',
               'has_leg1_segment2','has_leg1_segment3']]


train_copy = dd.concat([df1,new_df2,new_df3],axis=1)


train_copy['miniRules0_statusInfos'] = train_copy['miniRules0_statusInfos'].where(
    train_copy['miniRules0_monetaryAmount'] != 0, 
    0.0
)


train_copy['miniRules1_statusInfos'] = train_copy['miniRules1_statusInfos'].where(
    train_copy['miniRules1_monetaryAmount'] != 0, 
    0.0
)


train_copy = train_copy.drop('legs0_segments0_arrivalTo_airport_city_iata',axis=1)


train_copy = train_copy.dropna(subset = [
    'legs0_segments0_arrivalTo_airport_iata',
'legs0_segments0_departureFrom_airport_iata',
'legs0_segments0_aircraft_code']) 


train_copy['pricingInfo_passengerCount'].value_counts().compute()


train_copy = train_copy.drop('pricingInfo_passengerCount',axis=1)


mask = {True:1,False:0}

train_copy['bySelf'] = train_copy['bySelf'].replace(mask).astype(int)
train_copy['isAccess3D'] = train_copy['isAccess3D'].replace(mask).astype(int)
train_copy['isVip'] = train_copy['isVip'].replace(mask).astype(int)


train_copy['legs0_segments0_flightNumber'] = train_copy[
'legs0_segments0_flightNumber'].astype(int)


train_copy = train_copy.drop('sex',axis=1)


#df.groupby('departure_iata')['selected'].mean().sort_values()


#from sklearn.feature_selection import mutual_info_classif
#from sklearn.preprocessing import LabelEncoder

#X = df[['departure_iata']].astype(str)  # categorical
#y = df['selected']

#le = LabelEncoder()
#X_encoded = le.fit_transform(X['departure_iata']).reshape(-1,1)

#mi = mutual_info_classif(X_encoded, y, discrete_features=True)
#print("Mutual Information:", mi[0])


#import pandas as pd
#from scipy.stats import chi2_contingency

#contingency = pd.crosstab(df['departure_iata'], df['selected'])
#chi2, p, _, _ = chi2_contingency(contingency)

#print("Chi-square p-value:", p)


#import category_encoders as ce

#encoder = ce.TargetEncoder(cols=['legs0_segments0_departureFrom_airport_iata'])
#df_encoded = encoder.fit_transform(df, df['selected'])


#freq_map = df['legs0_segments0_departureFrom_airport_iata'].value_counts().to_dict()
#df['iata_freq'] = df['legs0_segments0_departureFrom_airport_iata'].map(freq_map)


number_df = train_copy.select_dtypes(include='number')
object_df = train_copy.select_dtypes(exclude='number') #not just categories but datetimes as well


(train_copy['selected'].value_counts()/len(train_copy['selected'])).compute()


iata_rates = train_copy.groupby(
    'legs0_segments0_arrivalTo_airport_iata')[
    'selected'].agg(['mean','count']).compute()


iata_rates.sort_values(by='count',ascending=False).head()


depart_iata_rates = train_copy.groupby(
'legs0_segments0_departureFrom_airport_iata')[
    'selected'].agg(['mean','count']).compute()

depart_iata_rates.sort_values(by='count',ascending=False).head()


#from sklearn.feature_selection import mutual_info_classif

# Example: categorical column (IATA code) + target
#X = train_copy[['legs0_segments0_departureFrom_airport_iata']]
#y = train_copy['selected']

# Convert categorical to numbers (label encoding, not one-hot)
#X_encoded = X.astype('category').apply(lambda col: col.cat.codes)

# Compute MI
#mi = mutual_info_classif(X_encoded, y, discrete_features=True)

#print("Mutual Information:", mi[0])


#features = ['legs0_segments0_departureFrom_airport_iata', 
#            'legs0_segments0_arrivalTo_airport_iata',
#            'sex', 'isVip', 'isAccess3D']

#X = train_copy[features].astype('category').apply(lambda col: col.cat.codes)
#y = train_copy['selected']

#mi_scores = mutual_info_classif(X, y, discrete_features=True)

#pd.Series(mi_scores, index=features).sort_values(ascending=False)


#from sklearn.feature_selection import mutual_info_classif

#cat_features = ['legs0_segments0_marketingCarrier_code',
#               'legs0_segments0_operatingCarrier_code',
#               'legs0_segments0_aircraft_code',
#                'legs0_segments0_arrivalTo_airport_iata',
#                'legs0_segments0_departureFrom_airport_iata']

#X = train_copy[cat_features].map_partitions(
#    lambda df: df.apply(lambda col: col.astype('category').cat.codes,axis=1)
#)
                                            
#y = train_copy['selected']

#mi_scores = mutual_info_classif(X, y, discrete_features=True)


import datetime as dt


datetime_cols = ['legs0_arrivalAt', 'legs0_departureAt', 
                 'legs0_duration', 'legs0_segments0_duration', 
                 'requestDate']

for col in datetime_cols:
    train_copy[col] = dd.to_datetime(train_copy[col], errors="coerce")


#pd.to_timedelta("2 days 3 hours 5 minutes 10 seconds").total_seconds()


#td = pd.to_timedelta("2 days 3 hours 5 minutes 10 seconds")
#td.seconds


train_copy['legs0_duration(hr)'] = ((train_copy['legs0_arrivalAt'] - train_copy[
 'legs0_departureAt']).dt.total_seconds()/3600).compute()


train_copy['day_of_departure'] = train_copy['legs0_departureAt'].dt.day.compute()


train_copy['month_of_departure'] = train_copy[
 'legs0_departureAt'].dt.month.compute()


def is_departure_weekend(element):
    if element == 5 or element == 6:
        return 1
    else:
        return 0
        

train_copy['is_departure_weekend'] = (train_copy[
 'legs0_departureAt'].dt.dayofweek).apply(is_departure_weekend)


train_copy['day_of_arrival'] = train_copy[
 'legs0_arrivalAt'].dt.day.compute()


train_copy['month_of_arrival'] = train_copy[
 'legs0_arrivalAt'].dt.month.compute()


def is_arrival_weekend(element):
    if element == 5 or element == 6:
        return 1
    else:
        return 0
        

train_copy['is_arrival_weekend'] = (train_copy[
 'legs0_arrivalAt'].dt.dayofweek).apply(is_arrival_weekend)


train_copy['requestDate'] = dd.to_datetime(train_copy['requestDate'])


train_copy['month_of_request'] = train_copy['requestDate'].dt.month.compute()


train_copy['day_of_request'] = train_copy['requestDate'].dt.day.compute()


train_copy['hour_request_made'] = train_copy['requestDate'].dt.hour.compute()


def is_weekend(row):
    if row == 5 or row == 6:
        return 1
    else:
        return 0

train_copy['is_requestdate_weekend'] = (train_copy[
                            'requestDate'].dt.dayofweek).apply(is_weekend).compute()


features = train_copy.drop(['legs0_arrivalAt', 'legs0_departureAt', 'legs0_duration',
       'legs0_segments0_duration', 'legs0_segments0_marketingCarrier_code',
       'legs0_segments0_operatingCarrier_code', 'ranker_id', 'requestDate',
       'searchRoute', 'legs0_segments0_aircraft_code',
       'legs0_segments0_arrivalTo_airport_iata',
       'legs0_segments0_departureFrom_airport_iata'], axis=1)

target = train_copy['selected']


features.isnull().sum().compute()
















