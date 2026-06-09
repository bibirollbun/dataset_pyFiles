# lets imports tools 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px


import os
print(os.listdir('/kaggle/input/'))



train_data=pd.read_csv('/kaggle/input/manually-loading-forecasting-sticker-sales-dataset/train.csv',parse_dates=['date'])
train_data



test_data =pd.read_csv('/kaggle/input/manually-loading-forecasting-sticker-sales-dataset/test.csv',parse_dates=['date'])
test_data


train_data.isnull().sum()


train_data.shape


train_data.info()


train_data.describe()


# check value counts for nymeric columns

for label,content in train_data.items():
  if pd.api.types.is_numeric_dtype(content):
    print(f'Value Counts For Column Numeric:{label}')
    print(train_data[label].value_counts())
    print("--"*40)


# check value counts for object columns
for label,content in train_data.items():
  if pd.api.types.is_object_dtype(content):
    print(f'The Value Count For Objects:{label}')
    print(train_data[label].value_counts())
    print("--"*40)


# make a copy of our dataframe
df=train_data.copy()


avg_sales=train_data.groupby('country')['num_sold'].mean()
avg_sales


avg_sales.plot(kind='bar',
               figsize=(10,6),
               xlabel='Country',
               ylabel='Avg Sales Per Day By Country wise',
               color='purple')
plt.show()


total_sales=train_data.groupby('country')['num_sold'].sum()
total_sales


total_sales_df = total_sales.reset_index()
total_sales_df.columns = ["Country", "num_sold"]
total_sales_df


import plotly.express as px
fig1 = px.choropleth(total_sales_df,
                    locations="Country",
                    locationmode='country names',
                    color="num_sold",
                    hover_name="Country",
                    title="Sales Distribution by Country (in Units Sold)",
                    color_continuous_scale=px.colors.sequential.Plasma)

fig1.update_layout(
    coloraxis_colorbar=dict(
        title="Total Units Sold (in millions)"
    ),
    title_font_size=20
)
fig1.show(renderer='iframe')


store = train_data.groupby('store')['num_sold'].sum().reset_index()
store.columns = ["Store", "num_sold"]
store



store.plot(x='Store', y='num_sold',
           kind='barh',
           color='firebrick',
           xlabel='Store',
           ylabel='Num Sold',
           title='Sales Distribution as per Stores')
plt.show()


df_viz = train_data.groupby(['store', 'product'])['num_sold'].sum().reset_index()
df_viz.columns = ['Store', 'Product', 'num_sold']
df_viz = df_viz.sort_values('Store')

# Create a bar chart
fig = px.bar(df_viz,x='Store',y='num_sold', color='Product',title="Sales Distribution By Products And Stores")
fig.update_traces(textposition='auto',textfont_size=20)
fig.update_layout(barmode='group')
fig.show(renderer='iframe')


train_data['Month']=train_data.date.dt.month
train_data['Year']=train_data.date.dt.year
train_data['Quarter']=train_data.date.dt.quarter
train_data['Day']=train_data.date.dt.day
train_data['Week']=train_data.date.dt.dayofweek
# separating date, year, week and month from data column
train_data.drop('date',axis=1,inplace=True)


train_data.dropna(subset=['num_sold'], axis=0, inplace=True)



train_data


test_data['Month']=test_data.date.dt.month
test_data['Year']=test_data.date.dt.year
test_data['Quarter']=test_data.date.dt.quarter
test_data['Day']=test_data.date.dt.day
test_data['Week']=test_data.date.dt.dayofweek
# separating date, year, week and month from data column
test_data.drop('date',axis=1,inplace=True)


test_data


train_data.isnull().sum()


test_data.isnull().sum()


#Thank you 




