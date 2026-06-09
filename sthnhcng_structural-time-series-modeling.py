#@title Import and set ups{ display-mode: "form" }

from matplotlib import pylab as plt
import matplotlib.dates as mdates
import seaborn as sns

import collections

import numpy as np
import tensorflow.compat.v2 as tf
import tf_keras
import tensorflow_probability as tfp

from tensorflow_probability import sts


if tf.test.gpu_device_name() != '/device:GPU:0':
  print('WARNING: GPU device not found.')
else:
  print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sns.set_context("notebook", font_scale=1.)
sns.set_style("whitegrid")
%config InlineBackend.figure_format = 'retina'


def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None):
  """Plot a forecast distribution against the 'true' time series."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]
  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1, 1, 1)

  num_steps = len(y)
  num_steps_forecast = forecast_mean.shape[-1]
  num_steps_train = num_steps - num_steps_forecast


  ax.plot(x, y, lw=2, color=c1, label='ground truth')

  forecast_steps = np.arange(
      x[num_steps_train],
      x[num_steps_train]+num_steps_forecast,
      dtype=x.dtype)

  ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

  ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
           label='forecast')
  ax.fill_between(forecast_steps,
                   forecast_mean-2*forecast_scale,
                   forecast_mean+2*forecast_scale, color=c2, alpha=0.2)

  ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
  yrange = ymax-ymin
  ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
  ax.set_title("{}".format(title))
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()

  return fig, ax


def plot_components(dates,
                    component_means_dict,
                    component_stddevs_dict,
                    x_locator=None,
                    x_formatter=None):
  """Plot the contributions of posterior components in a single figure."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  axes_dict = collections.OrderedDict()
  num_components = len(component_means_dict)
  fig = plt.figure(figsize=(12, 2.5 * num_components))
  for i, component_name in enumerate(component_means_dict.keys()):
    component_mean = component_means_dict[component_name]
    component_stddev = component_stddevs_dict[component_name]

    ax = fig.add_subplot(num_components,1,1+i)
    ax.plot(dates, component_mean, lw=2)
    ax.fill_between(dates,
                     component_mean-2*component_stddev,
                     component_mean+2*component_stddev,
                     color=c2, alpha=0.5)
    ax.set_title(component_name)
    if x_locator is not None:
      ax.xaxis.set_major_locator(x_locator)
      ax.xaxis.set_major_formatter(x_formatter)
    axes_dict[component_name] = ax
  fig.autofmt_xdate()
  fig.tight_layout()
  return fig, axes_dict


def plot_one_step_predictive(dates, observed_time_series,
                             one_step_mean, one_step_scale,
                             x_locator=None, x_formatter=None):
  """Plot a time series against a model's one-step predictions."""

  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  fig=plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1,1,1)
  ax.plot(dates, observed_time_series, label="observed time series", color=c1)
  ax.plot(dates, one_step_mean, label="one-step prediction", color=c2)
  ax.fill_between(dates,
                  one_step_mean - one_step_scale,
                  one_step_mean + one_step_scale,
                  alpha=0.1, color=c2)
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()
  fig.tight_layout()
  return fig, ax


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').drop(['warehouse','product_unique_id'],axis=1)
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
submit=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv').set_index('id')
test_id = test.unique_id.unique()
test_id


id_ = 3272
train = train[train['unique_id']==id_]
test = test[test['unique_id']==id_]
train = train.drop(columns=['availability'])
train = train.sort_values(by='date').reset_index(drop=True)

train = train[~train['sales'].isna()]
train['id'] = train['unique_id'].astype('str') + '_' + train['date'].astype('str')
train.set_index('id',inplace=True)
test['id'] = test['unique_id'].astype('str') + '_' + test['date'].astype('str')
test.set_index('id',inplace=True)

all_data = pd.concat([train,test])
all_data['max_discount'] = all_data[['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']].max(axis=1)
    
all_data['num_sales_days_28D'] = pd.MultiIndex.from_frame(all_data[['unique_id','date']]).map(all_data.sort_values('date').groupby('unique_id').rolling(
    window='28D', on='date', closed='left')['date'].count().fillna(0))
all_data['sell_price_main'] = np.log(all_data['sell_price_main']) 
all_data['sales_shift_15'] = all_data['sales'].shift(15)  
all_data['total_orders_shift_15'] = all_data['total_orders'].shift(1)  
all_data['sell_price_main_diff_7'] = all_data['sell_price_main'].diff(7)  
cols = ['date', 'total_orders', 'sales',
       'sell_price_main', 'max_discount',
       'num_sales_days_28D', 'sales_shift_15', 'total_orders_shift_15',
       'sell_price_main_diff_7']
train = all_data.loc[train.index][cols]
test = all_data.loc[test.index][cols]


train.info()


# Victoria electricity demand dataset, as presented at
# https://otexts.com/fpp2/scatterplots.html
# and downloaded from https://github.com/robjhyndman/fpp2-package/blob/master/data/elecdaily.rda
# This series contains the first eight weeks (starting Jan 1). The original
# dataset was half-hourly data; here we've downsampled to hourly data by taking
# every other timestep.
demand_dates =train['date'].to_numpy()
demand_loc = mdates.WeekdayLocator(byweekday=mdates.WE)
demand_fmt = mdates.DateFormatter('%a %b %d')

demand = train['sales'].to_numpy()
num_forecast_steps = 14 # Two weeks.
demand_training_data = demand[:-num_forecast_steps]


colors = sns.color_palette()
c1, c2 = colors[0], colors[1]

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(2, 1, 1)
ax.plot(demand_dates[:-num_forecast_steps],
        demand[:-num_forecast_steps], lw=2, label="training data")
ax.set_ylabel("Hourly demand (GW)")


# ax.xaxis.set_major_locator(demand_loc)
ax.xaxis.set_major_formatter(demand_fmt)
fig.suptitle("Sales",
             fontsize=15)
fig.autofmt_xdate()



def build_model(observed_time_series):
  day_of_year_effect = sts.Seasonal(
      num_seasons=365,
      observed_time_series=observed_time_series,
      name='hour_of_day_effect')
  day_of_week_effect = sts.Seasonal(
      num_seasons=7, num_steps_per_season=1,
      observed_time_series=observed_time_series,
      name='day_of_week_effect')
  # temperature_effect = sts.LinearRegression(
  #     design_matrix=tf.reshape(temperature - np.mean(temperature),
  #                              (-1, 1)), name='temperature_effect')
  autoregressive = sts.Autoregressive(
      order=1,
      observed_time_series=observed_time_series,
      name='autoregressive')
  model = sts.Sum([day_of_year_effect,
                   day_of_week_effect,
                   # temperature_effect,
                   autoregressive],
                   observed_time_series=observed_time_series)
  return model


demand_model = build_model(demand_training_data)

# Build the variational surrogate posteriors `qs`.
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
    model=demand_model)


#@title Minimize the variational loss.

# Allow external control of optimization to reduce test runtimes.
num_variational_steps = 200 # @param { isTemplate: true}
num_variational_steps = int(num_variational_steps)

# Build and optimize the variational loss function.
elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=demand_model.joint_distribution(
        observed_time_series=demand_training_data).log_prob,
    surrogate_posterior=variational_posteriors,
    optimizer=tf_keras.optimizers.Adam(learning_rate=0.1),
    num_steps=num_variational_steps,
    jit_compile=True)
plt.plot(elbo_loss_curve)
plt.show()

# Draw samples from the variational posterior.
q_samples_demand_ = variational_posteriors.sample(50)


print("Inferred parameters:")
for param in demand_model.parameters:
  print("{}: {} +- {}".format(param.name,
                              np.mean(q_samples_demand_[param.name], axis=0),
                              np.std(q_samples_demand_[param.name], axis=0)))


demand_forecast_dist = tfp.sts.forecast(
    model=demand_model,
    observed_time_series=demand_training_data,
    parameter_samples=q_samples_demand_,
    num_steps_forecast=num_forecast_steps)


num_samples=10

(
    demand_forecast_mean,
    demand_forecast_scale,
    demand_forecast_samples
) = (
    demand_forecast_dist.mean().numpy()[..., 0],
    demand_forecast_dist.stddev().numpy()[..., 0],
    demand_forecast_dist.sample(num_samples).numpy()[..., 0]
    )


demand_dates


fig, ax = plot_forecast(demand_dates, demand,
                        demand_forecast_mean,
                        demand_forecast_scale,
                        demand_forecast_samples,
                        title="sales 3272")
fig.tight_layout()

