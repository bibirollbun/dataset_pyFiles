import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", sep=',')

data.head()


print(data.shape)

data.isna().sum()


data.dropna(subset=['num_sold'], inplace=True)


import plotly.express as px

data['num_sold'] = data['num_sold'].astype(int)

aggregated_data = data.groupby('country')['num_sold'].sum().reset_index()

df = px.data.gapminder()  # Example dataset from Plotly
fig = px.choropleth( aggregated_data,  locations="country", color="num_sold",  locationmode="country names", 
    title="# of sales accross countris", color_continuous_scale="plasma", scope="world")

fig.show()


aggregated_data = data.groupby(['date', 'country'])['num_sold'].sum().reset_index()
print(aggregated_data['country'].unique())
fig = px.line(aggregated_data, x='date', y='num_sold', color ='country', title="Time Series")
fig.show()


from prophet import Prophet


# Initialize the model
model = Prophet()

# Rename columns for Prophet
data.rename(columns={"date": "ds", "num_sold": "y"}, inplace=True)
# Fit the model
model.fit(data)


# Create future dates for prediction (e.g., next 90 days)
future = model.make_future_dataframe(periods=90)

# Predict future values
forecast = model.predict(future)

# Display the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
model.plot(forecast).show()

