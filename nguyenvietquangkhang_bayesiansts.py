%matplotlib inline
from matplotlib import pylab as plt
import matplotlib.dates as mdates
import seaborn as sns

import collections

import numpy as np
import tensorflow.compat.v2 as tf
import tf_keras
import tensorflow_probability as tfp
import pandas as pd
from tensorflow_probability import sts
from datetime import timedelta
from sklearn.metrics import mean_absolute_error


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

  step = x[1] - x[0]  # Ước lượng bước nhảy (datetime.timedelta)
  forecast_steps = [x[num_steps_train] + i * step for i in range(num_steps_forecast)]

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


train=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
calendar=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])


train.shape


train_merge = train.merge(calendar, left_on=['date','warehouse'],right_on=['date','warehouse'])
train_merge.drop(['availability'],axis=1)


train_merge['discount'] = train_merge[['type_0_discount', 'type_1_discount', 'type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']].max(axis=1)
train_merge = train_merge.drop(['type_0_discount', 'type_1_discount', 'type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount'],axis=1)
train_merge.head()


train_data = train_merge[['unique_id','date','sales','holiday','discount']].copy()


i=416
df_1id=train_data[train_data['unique_id']==i]
df_1id=df_1id.sort_values(by='date')
date_range = pd.date_range(start=df_1id['date'].min(), end=df_1id['date'].max())
date_range_forecast = pd.date_range(start=df_1id['date'].max()-timedelta(days=60), end=df_1id['date'].max())
df_1id = df_1id.set_index('date').reindex(date_range)
df_1id['unique_id'] = df_1id['unique_id'].fillna(i)
#df_1id.reset_index()
#df_1id['date'] = df_1id.index
#df_1id=df_1id.set_index('date',drop=True)

df_1id.isna().sum()


df_1id.shape


df_1id['sales'] = df_1id['sales'].interpolate()
df_1id['holiday'] = df_1id['holiday'].fillna(0)
df_1id['discount'] = df_1id['discount'].fillna(0)


fig = plt.figure(figsize=(12, 6))
df_1id['sales'].plot()
plt.show()


sns.displot(df_1id['sales'], height=6, aspect=1.5)


resampled_dft = df_1id[['sales']].copy()

resampled_dft['date'] = resampled_dft.index
resampled_dft['indexi'] = range(1, len(resampled_dft) + 1)
resampled_dft = resampled_dft.reset_index(drop=True)
resampled_dft = resampled_dft.drop(['indexi'],axis=1)
#resampled_df = resampled_df[['date','sales']].resample('M', on='date').mean().reset_index(drop=False)
fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(16,12))
resampled_df = resampled_dft[['date','sales']].resample('W', on='date').mean().reset_index(drop=False)
sns.lineplot(x=resampled_df['date'], y=resampled_df['sales'], color='dodgerblue',ax=ax[0])
ax[0].set_title('weekly')
resampled_df = resampled_dft[['date','sales']].resample('M', on='date').mean().reset_index(drop=False)
sns.lineplot(x=resampled_df['date'], y=resampled_df['sales'], color='dodgerblue',ax=ax[1])
ax[1].set_title('monthly')
resampled_df = resampled_dft[['date','sales']].resample('Q', on='date').mean().reset_index(drop=False)
sns.lineplot(x=resampled_df['date'], y=resampled_df['sales'], color='dodgerblue',ax=ax[2])
ax[2].set_title('quarter')
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
def decompose(df, column_name):
    """
    A function that returns the trend, seasonality and residual captured by applying both multiplicative and
    additive model.
    df -> DataFrame
    column_name -> column_name for which trend, seasonality is to be captured
    """
    result_add = seasonal_decompose(df[column_name], model = 'additive', extrapolate_trend='freq')

    plt.rcParams.update({'figure.figsize': (20, 10)})
    result_add.plot().suptitle('Additive Decompose', fontsize=30)
    plt.show()
    
    return  result_add
decompose(df_1id,'sales')


from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df_1id['sales'])


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
fig, ax = plt.subplots(nrows=2, ncols=1)
plot_acf(df_1id['sales'], lags=100,ax=ax[0])
plot_pacf(df_1id['sales'], lags=30,ax=ax[1])
plt.tight_layout()
plt.show()


def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


df_1id_train = df_1id['sales'].iloc[:-14].to_numpy()
sales=df_1id['sales'].iloc[-61:].to_numpy()
y_true=df_1id['sales'].iloc[-14:].to_numpy()


def predict_n_plot(model,ylim):

    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
    model=model)
    num_variational_steps = 200 # @param { isTemplate: true}
    num_variational_steps = int(num_variational_steps)
# Build and optimize the variational loss function.
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_distribution(
        observed_time_series=df_1id_train).log_prob,
        surrogate_posterior=variational_posteriors,
        optimizer=tf_keras.optimizers.Adam(learning_rate=0.1),
        num_steps=num_variational_steps,
        jit_compile=True)
#    plt.plot(elbo_loss_curve)
#    plt.show()

    # Draw samples from the variational posterior.
    q_samples_demand_ = variational_posteriors.sample(50)
    forecast_dist = tfp.sts.forecast(
    model=model,
    observed_time_series=df_1id_train,
    parameter_samples=q_samples_demand_,
    num_steps_forecast=14)
    num_samples=10

    (
        forecast_mean,
        forecast_scale,
        forecast_samples
    ) = (
        forecast_dist.mean().numpy()[..., 0],
        forecast_dist.stddev().numpy()[..., 0],
        forecast_dist.sample(num_samples).numpy()[..., 0]
        )
    mae=mean_absolute_error(y_true, forecast_mean)
    print('VI-MAE: ', mae)
    fig, ax = plot_forecast(date_range_forecast, sales,
                        forecast_mean,
                        forecast_scale,
                        forecast_samples,
                        title="sales forecast"
                        )
    ax.set_ylim([0, ylim])
    fig.tight_layout()
    plt.show()
    
    
     


def predict_n_plot_MCMC(model,ylim,num_results=300,warmup=100,leapfrog=15,vari_step=100):

    
##############################
    samples, kernel_results = tfp.sts.fit_with_hmc(
        model=model,
        observed_time_series=df_1id_train,
        num_results=num_results,
        num_warmup_steps=warmup,
        num_leapfrog_steps=leapfrog,
        initial_state=None,
        initial_step_size=None,
        chain_batch_shape=(),
        num_variational_steps=vari_step,
        variational_optimizer=None,
        variational_sample_size=5,
        seed=None,
        name=None
    )
    num_samples=50
    forecast_dist = tfp.sts.forecast(model, df_1id_train,
                                   parameter_samples=samples,
                                   num_steps_forecast=14)

    forecast_mean = forecast_dist.mean()[..., 0]
    forecast_scale = forecast_dist.stddev()[..., 0] 
    forecast_samples = forecast_dist.sample(num_samples)[..., 0] 

    mae=mean_absolute_error(y_true, forecast_mean)
    print('MCMC-MAE: ', mae)



def build_model(observed_time_series):
  day_of_week_effect = sts.Seasonal(
      num_seasons=7, num_steps_per_season=1,
      observed_time_series=observed_time_series,
      name='day_of_week_effect')
  residual = sts.Autoregressive(
      order=1,
      observed_time_series=observed_time_series,
      name='residual')
  model = sts.Sum([day_of_week_effect,
                   residual],
                   observed_time_series=observed_time_series)
  return model


model=build_model(df_1id_train)
ylim = df_1id['sales'].max()


predict_n_plot(model,ylim*1.5)


print('---------10,2,5,1-------------')
model2=build_model(df_1id_train)
predict_n_plot_MCMC(model2,ylim*1.5,10,2,5,1)


print('---------20,5,5,20-------------')
model3=build_model(df_1id_train)
predict_n_plot_MCMC(model3,ylim*1.5,20,5,5,20)


print('---------default-------------')
model4=build_model(df_1id_train)
predict_n_plot_MCMC(model4,ylim*1.5)




