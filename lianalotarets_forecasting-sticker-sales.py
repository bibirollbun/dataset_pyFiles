import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import date
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


df


df.isnull().sum()


for country in df['country'].unique():
    for store in df['store'].unique():
        for product in df['product'].unique():
            condition = (df['country'] == country) & (df['store'] == store) & (df['product'] == product)
            df.loc[condition, 'num_sold'] = df.loc[condition, 'num_sold'].interpolate()

condition_Canada_DS_HG = (df['country'] == 'Canada') & (df['store'] == 'Discount Stickers') & (df['product'] == 'Holographic Goose')
df.loc[condition_Canada_DS_HG, 'num_sold'] = df.loc[condition_Canada_DS_HG, 'num_sold'].fillna(0)
# df = df[~condition_Canada_DS_HG]

condition_Kenya_DS_HG = (df['country'] == 'Kenya') & (df['store'] == 'Discount Stickers') & (df['product'] == 'Holographic Goose')
df.loc[condition_Kenya_DS_HG, 'num_sold'] = df.loc[condition_Kenya_DS_HG, 'num_sold'].fillna(0)
# df = df[~condition_Kenya_DS_HG]


df.isnull().sum()


df['date'] = pd.to_datetime(df['date'])
df_test['date'] = pd.to_datetime(df_test['date'])


# country	store	product
print(df['country'].unique())
print(df['store'].unique())
print(df['product'].unique())


df = df.rename(columns={'date': 'ds', 'num_sold': 'y'})


df


df_dict = {}
for country in df['country'].unique():
    for store in df['store'].unique():
        for product in df['product'].unique():
            condition = (df['country'] == country) & (df['store'] == store) & (df['product'] == product)
            df_dict.update({(country, store, product): df[['ds','y']][condition]})


df_dict[('Canada','Discount Stickers','Kaggle')]


forecast_dict = {}

for key, df in df_dict.items():
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)

    # 3 роки = 3 * 365 = 1095 днів
    future = model.make_future_dataframe(periods=1095)
    forecast = model.predict(future)

    # forecast_dict[key] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_dict[key] = forecast[['ds', 'yhat']][
        (forecast['ds'] >= '2017-01-01') & 
        (forecast['ds'] <= '2019-12-31')
    ]


forecast_dict[('Canada','Discount Stickers','Kaggle')]


all_forecasts = []

for (country, store, product), forecast in forecast_dict.items():
    df = forecast[['ds', 'yhat']].copy()
    df['country'] = country
    df['store'] = store
    df['product'] = product
    all_forecasts.append(df)

# Обʼєднуємо всі в один великий датафрейм
forecast_df = pd.concat(all_forecasts, ignore_index=True)


forecast_df = forecast_df.sort_values(by=['ds', 'country', 'store', 'product'])


forecast_df


df_test


df_sample_submission['num_sold'] = forecast_df['yhat'].values


df_sample_submission


df_sample_submission.to_csv('submission.csv', index=False)

