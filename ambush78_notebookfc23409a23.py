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


train = pd.read_csv('/kaggle/input/time-series-forcasting/train.csv')
test = pd.read_csv('/kaggle/input/time-series-forcasting/test.csv')
sample_submission = pd.read_csv('/kaggle/input/time-series-forcasting/sample_submission.csv')
train


#train = train*1000
train


test


sample_submission


plt.scatter(x='x', y = 'y', data = train)


plt.scatter(x='x', y = 'z', data = train)


plt.scatter(x='y', y = 'z', data = train)


import plotly.graph_objs as go
trace = go.Scatter3d(
    x=train['x'], y=train['y'], z=train['z'],
    mode='lines',
    line=dict(color='blue')
)

layout = go.Layout(
    title='Scrollable 3D x, y, z Plot',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()


# from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(train[['t']],train[['x', 'y','z']], test_size = 0.1, random_state=12, shuffle = True )
# rf_model =RandomForestRegressor()
# rf_model.fit(X_train, y_train)
# y_pred = rf_model.predict(X_test)





from sklearn.metrics import r2_score, mean_squared_error
# print(r2_score(y_pred, y_test))
# print(mean_squared_error(y_pred, y_test)*1000)


# sample_submission


# pred = pd.DataFrame(rf_model.predict(test[['t']]), columns = ["x", 'y', 'z'])
# pred


# sample_submission['x'] = pred['x']
# sample_submission['y'] = pred['y']
# sample_submission['z'] = pred['z']


sample_submission.head(5)





from statsmodels.tsa.seasonal import seasonal_decompose

decompose = seasonal_decompose(train.x, model='additive', extrapolate_trend='freq', period=12)

decompose.plot().show()


train.plot(x = 't', y = 'x' )


train.plot(x = 't', y = 'y' )





train.plot(x = 't', y = 'z' )


!pip install prophet
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



plot_acf(train['x'], lags=1000)
plot_pacf(train['x'], lags=20)
plt.show()


plot_acf(train['y'], lags=1000)
plot_pacf(train['y'], lags=10)

plt.show()


plot_acf(train['z'], lags=1000)
plot_pacf(train['z'], lags=10)

plt.show()





# train['x_lag'] =  train['x'].shift(1)
# train['y_lag'] =  train['y'].shift(1)
# train['z_lag'] =  train['z'].shift(1)
# train.dropna(inplace = True)
# train


t1 = train.loc[:7500, :]
t2 = train.loc[7500:,:]
t1


t2


x_model = Prophet()
x_model.fit(t1[['t', 'x' ]].rename(columns={'t':'ds', 'x':'y'}))
forecast_prophet = x_model.predict(t2[['t']].rename(columns={'t':'ds'}))
forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round().tail()


mean_squared_error(t2[['x']], forecast_prophet['yhat'])


y_model = Prophet()
y_model.fit(t1[['t', 'y']].rename(columns={'t':'ds', 'y':'y'}))
forecast_prophet = y_model.predict(t2[['t']].rename(columns={'t':'ds'}))
forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round().tail()


mean_squared_error(t2[['y']], forecast_prophet['yhat'])


z_model = Prophet()
z_model.fit(t1[['t', 'z']].rename(columns={'t':'ds', 'z':'y'}))
forecast_prophet = z_model.predict(t2[['t']].rename(columns={'t':'ds'}))
forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round().tail()


mean_squared_error(t2[['z']], forecast_prophet['yhat'])


test


sample_submission['x'] = x_model.predict(test[['t']].rename(columns={'t':'ds'}))['yhat']
sample_submission['y'] = y_model.predict(test[['t']].rename(columns={'t':'ds'}))['yhat']
sample_submission['z'] = z_model.predict(test[['t']].rename(columns={'t':'ds'}))['yhat']


sample_submission


sample_submission.to_csv('submission.csv', index = False)
sample_submission.head(5)

