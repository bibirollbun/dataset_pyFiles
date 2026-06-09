!pip install mapclassify --quiet


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd # 
import folium
import matplotlib as mpl
import matplotlib.pyplot as plt
import mapclassify
import seaborn as sns  
from datetime import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data_path = '/kaggle/input/bpl-ai4good-wheat-price-forecasting/'
sample_submission_filename = 'sample_submission.csv'
train_data = 'train.csv'
test_data = 'test.csv'


df = pd.read_csv(f'{data_path}{train_data}')
df = df.dropna()
df


markets = df[['market', 'latitude', 'longitude']]
markets = markets.drop_duplicates()
markets


gdf = gpd.GeoDataFrame(markets, geometry=gpd.points_from_xy(markets.longitude, markets.latitude), crs="EPSG:4326")

gdf.explore()


df['year'] = df['date'].str.extract('(\d+)', expand=False)


df.head()


df_subset = df[['year','month','price_usd','igc_wheat','igc_maize','igc_rice','igc_barley']]


df_subset.head()


df_u=df[df['admin0']=='Ukraine']


#= fix python path if working locally
#from utils import fix_pythonpath_if_working_locally

#fix_pythonpath_if_working_locally()

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from darts import concatenate
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset, ElectricityDataset, MonthlyMilkDataset
from darts.metrics import mae, mape
from darts.models import (
    VARIMA,
    BlockRNNModel,
    NBEATSModel,
    RNNModel,
)
from darts.utils.callbacks import TFMProgressBar
from darts.utils.timeseries_generation import (
    datetime_attribute_timeseries,
    sine_timeseries,
)

logging.disable(logging.CRITICAL)

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)


def generate_torch_kwargs():
    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }


scaler_u = Scaler()
series_u_scaled = scaler_u.fit_transform(ukraine)

#series_air_scaled.plot(label="ukraine")
#plt.legend();


train = df_u['price_usd']


pip install darts


from darts import TimeSeries

df_u['date'] = pd.to_datetime(df_u['date'])  # ensure proper datetime
df_u = df_u.set_index('date')                # set datetime index

series_u = TimeSeries.from_dataframe(df_u, value_cols='price_usd')


from darts import TimeSeries
import pandas as pd

# Ensure datetime format
df_u['date'] = pd.to_datetime(df_u['date'])

# Set datetime index (required by darts)
df_u = df_u.set_index('date')

# Convert to TimeSeries
series_u = TimeSeries.from_dataframe(df_u, value_cols='price_usd')


model = NBEATSModel(input_chunk_length=12, output_chunk_length=6)
model.fit(train)
forecast = model.predict(36)














#plot local currency for afghanistan
plt.figure(figsize=(12, 6))
df_af=df[df['admin0']=='Afghanistan']
#plt.plot(df_af['formatted_date'],df_af['price_localcurrency'],'.')
plt.plot(df_af['price_localcurrency'])
plt.show()


plt.figure(figsize=(12, 6))
df_u=df[df['admin0']=='Ukraine']
plt.plot(df_u['price_localcurrency'])
plt.show()


df['admin0'].unique()


from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse



multiplicative_decomposition = seasonal_decompose(df_af['price_usd'], model='multiplicative', period=30)
additive_decomposition = seasonal_decompose(df_af['price_usd'], model='additive', period=30)

# Plot
plt.rcParams.update({'figure.figsize': (16,12)})
multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()


rand_numbers = np.random.randn(1000)
pd.Series(rand_numbers).plot(title='Random White Noise', color='b')


from scipy import signal
plt.rcParams.update({'figure.figsize': (12,4)})
detrended = signal.detrend(df_af['price_usd'].values)
plt.plot(detrended)


from statsmodels.tsa.seasonal import seasonal_decompose
result_mul = seasonal_decompose(df_af['price_usd'], model='multiplicative', period=30)
detrended = df_af['price_usd'].values - result_mul.trend
plt.plot(detrended)


# Time Series Decomposition
result_mul = seasonal_decompose(df_af['price_usd'], model='multiplicative', period=30)

# Deseasonalize
deseasonalized = df_af['price_usd'].values / result_mul.seasonal

# Plot
plt.plot(deseasonalized)
plt.plot()


# Test for seasonality
from pandas.plotting import autocorrelation_plot

# Draw Plot
plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':120})
autocorrelation_plot(df_af['price_usd'].tolist())


from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df_af['price_usd'].tolist(), lags=50, ax=axes[0])
plot_pacf(df_af['price_usd'].tolist(), lags=50, ax=axes[1])


from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(df_af['price_usd'], lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))
  
plt.show()


from statsmodels.tsa.stattools import grangercausalitytests
data_af = df_af
data_af['date'] = pd.to_datetime(data_af['date'])
data_af['month'] = data_af.date.dt.month
grangercausalitytests(data_af[['price_usd', 'month']], maxlag=2)




