import pandas as pd
import numpy as np
import plotly.express as px



sumb=pd.read_csv(r'/kaggle/input/playground-series-s5e1/sample_submission.csv')
sumb.head()


df=pd.read_csv(r'/kaggle/input/playground-series-s5e1/train.csv').drop(columns=['id'])
df=df.dropna()
df.info()


df.describe()


df.date=pd.to_datetime(df.date,errors='coerce')
df['years']=df.date.dt.year
df['month']=df.date.dt.month
df['day']=df.date.dt.day
df.columns
df=df[['years', 'month','day', 'country', 'store', 'product', 'num_sold']]
df.head()


px.box(data_frame=df,x='years',y='num_sold',title='Box Plot TO Check About Outlier Every Years')


px.box(data_frame=df,x='num_sold',title='Box Plot TO Check About Outlier All Years')


px.histogram(data_frame=df,x='num_sold')


px.bar(data_frame=df.country.value_counts(),title='The proportion of each countrys counts in the data')


px.pie(data_frame=df,names='country',title='The proportion of each countrys presence in the data')


px.bar(data_frame=df.store.value_counts(),title='The proportion of each stores counts in the data')


px.pie(df,names='store')


px.bar(data_frame=df['product'].value_counts(),title='The proportion of each products counts in the data')


df.head()


px.bar(df.groupby('years')['num_sold'].sum(),title='Total sales each year')


all_years=[2010,2011,2012,2013,2014,2015,2016]
ever_month={}
for i in all_years:
    y=df[df['years']==i]
    ever_month[str(i)]=y.groupby('month')['num_sold'].sum()

data=pd.DataFrame(data=ever_month)



data


px.bar(data_frame=data['2010'])


px.bar(data_frame=data['2011'])


px.bar(data_frame=data['2012'])


px.bar(data_frame=data['2013'])


px.bar(data_frame=data['2014'])


px.bar(data_frame=data['2015'])


px.bar(data_frame=data['2016'])


px.pie(df,names='years')


df.head()


px.bar(df.groupby('country')['num_sold'].sum(),title='To find out which country achieved the highest sales')


px.bar(df.groupby('store')['num_sold'].sum(),title='To find out which store achieved the highest sales')


px.bar(df.groupby('product')['num_sold'].sum(),title='To find out which product achieved the highest sales')


countery=set(df.country.values)
d={}
for i in countery:
    ST_CO=df[df['country']==i]
    d[i]=ST_CO.groupby('store')['num_sold'].sum()

store_country=pd.DataFrame(data=d)


store_country


px.bar(data_frame=store_country,title='Sales of every store in every country')



proudects=set(df['product'].values)
prod={}
for i in proudects:
    df_pro=df[df['product']==i]
    prod[i]=df_pro.groupby('store')['num_sold'].sum()

store_prod=pd.DataFrame(data=prod)
store_prod


px.bar(data_frame=store_prod,title='Sales of every proudect in every store')


proudects=set(df['product'].values)
prod={}
for i in proudects:
    df_pro=df[df['product']==i]
    prod[i]=df_pro.groupby('country')['num_sold'].sum()

country_prod=pd.DataFrame(data=prod)
country_prod


px.bar(data_frame=country_prod,title='Sales of every proudect in every country')

