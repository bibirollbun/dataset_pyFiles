import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker 
from datetime import datetime
from tqdm import tqdm
import folium
from random import randint



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split





from sklearn.linear_model import LinearRegression


!pip install haversine


from haversine import haversine


tqdm.pandas()


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        paths.append(path)
        print(path)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import zipfile

for file in paths:
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall('kaggle/working')


df = pd.read_csv("kaggle/working/train.csv")


df.head()


df.info()


df["dropoff_latitude"].min(), df["dropoff_latitude"].max()


df["dropoff_longitude"].min(), df["dropoff_longitude"].max()


import folium

m = folium.Map(location=[32.1811408996582, 121.9333038330078], tiles='CartoDB Positron', zoom_start=12)
m


df["dropoff_latitude"].hist()


num_columns = [col for col in df.columns if df[col].dtype != 'object']
num_columns


df[num_columns]


sns.heatmap(df[num_columns].corr(), annot=True)


df = df.drop(columns=["dropoff_datetime"])
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])


fig, ax = plt.subplots(figsize=(15, 7))
plt.hist(df['trip_duration'].clip(None, 10000), bins=100)
plt.show()


def vec_haversine(row):
    return haversine((row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']))


df['haversine'] = df.progress_apply(vec_haversine, axis=1)


df.info()


df.head()


def show_circles_on_map(data, latitude_column, longitude_column, color):
    """
    The function draws map with circle on it.
    The center of the map is the mean of the coordinates passed in the data. 
    data: Dataframe that contains columns latitude_column and longitude_column
    latitude_col: string, the name of the col for latitude cordinate
    longitude_col: string, the name of the col for the longitude cordinate
    color: string, the color of the circles to be drawn
    """

    location = (data[latitude_column].mean(), data[longitude_column].mean())
    m = folium.Map(location=location, zoom_start=12)

    for _, row in data.iterrows():
        folium.Circle(
            radius=100,
            location=(row[latitude_column], row[longitude_column]),
            color=color,
            fill_color=color,
            fill=True,
        ).add_to(m)

    return m


df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_month'] = df['pickup_datetime'].dt.month


df.info()


show_circles_on_map(df.sample(10000), 'pickup_latitude', 'pickup_longitude', 'blue')



best_constant = df['trip_duration'].mean()
best_constant


mean_squared_error(df['trip_duration'], [best_constant] * len(df['trip_duration'])) ** 0.5





train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)


selected_col = ['pickup_weekday', 'pickup_hour', 'pickup_month', 'passenger_count', 'vendor_id', 'haversine', 'trip_duration']



x_train = train[selected_col].drop(columns='trip_duration')
y_train = train['trip_duration']

x_test = test[selected_col].drop(columns='trip_duration')
y_test = test['trip_duration']


model = LinearRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
mean_squared_error(y_pred, y_test) ** 0.5


from sklearn.linear_model import Ridge

model_1 = Ridge(alpha=4)
model_1.fit(x_train, y_train)


y_pred = model_1.predict(x_test)


mean_squared_error(y_pred, y_test) ** 0.5


from sklearn.model_selection import cross_val_score


cross_val_score?




