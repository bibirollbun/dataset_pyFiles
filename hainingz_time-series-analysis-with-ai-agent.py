# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Time Series Analysis with a Google Agent (ADK / Vertex AI)
# **What this notebook does**
# - Performs standard time-series EDA & forecasting (Prophet + ARIMA).
# - Wraps core analysis functions into "tools" that an ADK / Vertex agent can call.
# - Shows how to register the tools into a simple ADK agent (local dev example).


# Cell 0 — prerequisites / install (run in Colab or local venv)
# If you're using Colab, uncomment the pip installs. In a managed environment,
# install these into your project's environment.
!pip install pandas numpy matplotlib seaborn prophet statsmodels scikit-learn google-cloud-aiplatform google-adk-python



# Cell 1 — imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet                # forecasting
import statsmodels.api as sm               # decomposition, ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Vertex / ADK imports (ADK may be in google.adk.* or google_adk.* depending on package)
# We'll import the ADK app template; adjust import names if package paths differ.
try:
    from google.adk.app import AdkApp     # ADK-style app base (example path)
except Exception:
    # fallback placeholder import; in your environment, install adk package and use the docs.
    AdkApp = None

# Vertex AI for model deployment / logging (optional)
from google.cloud import aiplatform



# Cell 2 — auth/setup
PROJECT_ID = os.getenv("GCP_PROJECT") or "your-gcp-project-id"
REGION = os.getenv("GCP_REGION") or "us-central1"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

# If using a service account JSON:
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/service-account.json"

aiplatform.init(project=PROJECT_ID, location=REGION)
print("Vertex initialized for project:", PROJECT_ID)



# Cell 3 — load sample data (AirPassengers)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=['Month'])
df.columns = ['ds', 'y']   # Prophet expects 'ds' (date) and 'y' (value)
df = df.set_index('ds').asfreq('MS').reset_index()
df.head()






# Cell 4 — EDA plots
plt.figure(figsize=(12,4))
plt.plot(df['ds'], df['y'], marker='o')
plt.title('AirPassengers (monthly)')
plt.xlabel('Date'); plt.ylabel('Passengers')
plt.grid(True)
plt.show()

# Decomposition (additive + multiplicative)
ts = df.set_index('ds')['y']

res_add = sm.tsa.seasonal_decompose(ts, model='additive', period=12)
res_mul = sm.tsa.seasonal_decompose(ts, model='multiplicative', period=12)

fig = res_add.plot()
fig.set_size_inches(10,8)
plt.suptitle('Additive decomposition')
plt.show()

fig = res_mul.plot()
fig.set_size_inches(10,8)
plt.suptitle('Multiplicative decomposition')
plt.show()



# Cell 5 — Prophet model
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(df[['ds','y']])

future = m.make_future_dataframe(periods=24, freq='MS')
forecast = m.predict(future)

fig = m.plot(forecast)
plt.title('Prophet forecast')
plt.show()

# performance on historical window (last 12 months)
hist = forecast.merge(df[['ds','y']], on='ds', how='left')
train_eval = hist.dropna(subset=['y']).copy()
mae = mean_absolute_error(train_eval['y'][-12:], train_eval['yhat'][-12:])
rmse = mean_squared_error(train_eval['y'][-12:], train_eval['yhat'][-12:], squared=False)
print(f"Last-12-month MAE: {mae:.3f}, RMSE: {rmse:.3f}")



# Cell 5 — Prophet model
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(df[['ds','y']])

future = m.make_future_dataframe(periods=24, freq='MS')
forecast = m.predict(future)

fig = m.plot(forecast)
plt.title('Prophet forecast')
plt.show()

# performance on historical window (last 12 months)
hist = forecast.merge(df[['ds','y']], on='ds', how='left')
train_eval = hist.dropna(subset=['y']).copy()
mae = mean_absolute_error(train_eval['y'][-12:], train_eval['yhat'][-12:])
rmse = mean_squared_error(train_eval['y'][-12:], train_eval['yhat'][-12:], squared=False)
print(f"Last-12-month MAE: {mae:.3f}, RMSE: {rmse:.3f}")



# Cell 6 — ARIMA baseline
# We'll pick a seasonal order (p,d,q)(P,D,Q,12) manually for demo.
from statsmodels.tsa.statespace.sarimax import SARIMAX

train = ts
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
res = model.fit(disp=False)
pred = res.get_forecast(steps=24)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

plt.figure(figsize=(10,4))
plt.plot(ts.index, ts, label='observed')
plt.plot(pred_mean.index, pred_mean, label='forecast')
plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], alpha=0.2)
plt.legend(); plt.title('SARIMA Forecast')
plt.show()



# Cell 7 — tool functions
def describe_series(df):
    ts = df.set_index('ds')['y']
    info = {
        "start": str(ts.index.min().date()),
        "end": str(ts.index.max().date()),
        "n_points": len(ts),
        "freq": str(ts.index.inferred_freq)
    }
    stats = ts.describe().to_dict()
    info.update({f"stat_{k}": float(v) for k,v in stats.items()})
    return info

def decompose_series(df, model='multiplicative', period=12):
    ts = df.set_index('ds')['y']
    res = sm.tsa.seasonal_decompose(ts, model=model, period=period)
    return {
        "trend": res.trend.dropna().to_json(date_format='iso'),
        "seasonal": res.seasonal.head(36).to_json(date_format='iso'),  # sampled
        "resid": res.resid.dropna().to_json(date_format='iso')
    }

def forecast_prophet(df, periods=12):
    model = Prophet(yearly_seasonality=True)
    model.fit(df[['ds','y']])
    future = model.make_future_dataframe(periods=periods, freq='MS')
    fc = model.predict(future)
    return fc[['ds','yhat','yhat_lower','yhat_upper']].to_json(date_format='iso', orient='records')

def forecast_sarima(df, steps=12, order=(1,1,1), seasonal_order=(1,1,1,12)):
    ts = df.set_index('ds')['y']
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    pred_df = pd.DataFrame({
        'ds': pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS'),
        'yhat': pred.predicted_mean.values,
        'lower': pred.conf_int().iloc[:,0].values,
        'upper': pred.conf_int().iloc[:,1].values
    })
    return pred_df.to_json(date_format='iso', orient='records')

def evaluate_forecast(true_df, pred_df, on='ds'):
    t = true_df.set_index(on)['y']
    p = pred_df.set_index(on)['yhat']
    # align indexes
    common = t.index.intersection(p.index)
    mae = mean_absolute_error(t.loc[common], p.loc[common])
    rmse = mean_squared_error(t.loc[common], p.loc[common], squared=False)
    return {"mae": float(mae), "rmse": float(rmse)}



# Cell 7 — tool functions
def describe_series(df):
    ts = df.set_index('ds')['y']
    info = {
        "start": str(ts.index.min().date()),
        "end": str(ts.index.max().date()),
        "n_points": len(ts),
        "freq": str(ts.index.inferred_freq)
    }
    stats = ts.describe().to_dict()
    info.update({f"stat_{k}": float(v) for k,v in stats.items()})
    return info

def decompose_series(df, model='multiplicative', period=12):
    ts = df.set_index('ds')['y']
    res = sm.tsa.seasonal_decompose(ts, model=model, period=period)
    return {
        "trend": res.trend.dropna().to_json(date_format='iso'),
        "seasonal": res.seasonal.head(36).to_json(date_format='iso'),  # sampled
        "resid": res.resid.dropna().to_json(date_format='iso')
    }

def forecast_prophet(df, periods=12):
    model = Prophet(yearly_seasonality=True)
    model.fit(df[['ds','y']])
    future = model.make_future_dataframe(periods=periods, freq='MS')
    fc = model.predict(future)
    return fc[['ds','yhat','yhat_lower','yhat_upper']].to_json(date_format='iso', orient='records')

def forecast_sarima(df, steps=12, order=(1,1,1), seasonal_order=(1,1,1,12)):
    ts = df.set_index('ds')['y']
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    pred_df = pd.DataFrame({
        'ds': pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS'),
        'yhat': pred.predicted_mean.values,
        'lower': pred.conf_int().iloc[:,0].values,
        'upper': pred.conf_int().iloc[:,1].values
    })
    return pred_df.to_json(date_format='iso', orient='records')

def evaluate_forecast(true_df, pred_df, on='ds'):
    t = true_df.set_index(on)['y']
    p = pred_df.set_index(on)['yhat']
    # align indexes
    common = t.index.intersection(p.index)
    mae = mean_absolute_error(t.loc[common], p.loc[common])
    rmse = mean_squared_error(t.loc[common], p.loc[common], squared=False)
    return {"mae": float(mae), "rmse": float(rmse)}



# Cell 8 — minimal ADK-style agent registration (pseudo-code)
# NOTE: ADK/Agent import paths vary by package version. This is a conceptual example.

tools = {
    "describe_series": describe_series,
    "decompose_series": decompose_series,
    "forecast_prophet": forecast_prophet,
    "forecast_sarima": forecast_sarima,
    "evaluate_forecast": evaluate_forecast
}

# PSEUDO: Example AdkApp usage (refer to ADK docs for exact API)
if AdkApp is not None:
    class TimeSeriesAgent(AdkApp):
        def __init__(self, tools):
            super().__init__(name="time_series_agent")
            # register each tool with the ADK toolkit
            for tname, func in tools.items():
                self.register_tool(name=tname, func=func)

    agent = TimeSeriesAgent(tools=tools)
    print("Agent initialized (local demo). You can call agent.invoke('describe_series', payload) in local dev.")
else:
    print("ADK package not detected; skip agent instantiation. Install google-adk-python and check docs.")



# Cell 9 — simulate agent invoking tools
print("Describe series:")
print(describe_series(df))

print("\nProphet 12-month forecast (sample):")
print(forecast_prophet(df, periods=12)[:500])  # truncated json





