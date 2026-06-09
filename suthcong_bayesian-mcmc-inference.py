import numpy as np
import pandas as pd



inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').drop(['warehouse','product_unique_id'],axis=1)
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])


train = train.sort_values(by='date')
train.head(10)


import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset (assuming real data is similar)
data = {
    'unique_id': [4845, 4845, 4845, 4845, 4845],
    'sales': [16.17, 3.85, 38.97, 23.81, 34.34]
}
df = pd.DataFrame(data)

# Creating time index (if available)
df['time'] = np.arange(len(df))  # Assuming sequential time



train[train['unique_id']==4845][['unique_id', 'sales']]


import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns


id_list = train.sample(5).unique_id.unique()

df=train[train['unique_id'].isin(id_list)][['unique_id', 'date','sales']]
df


# Setting unique_id and date as indices
df.set_index(["unique_id", "date"], inplace=True)

# Prepare data by unique_id for modeling
unique_ids = df.index.get_level_values("unique_id").unique()
sales_data = {unique_id: df.loc[unique_id, "sales"].values for unique_id in unique_ids}

# Bayesian State-Space Model with Gaussian Random Walk for each unique_id
with pm.Model() as model:
    # Hyperparameters for each unique_id (common prior)
    tau = pm.HalfNormal("tau", sigma=1, shape=len(unique_ids))  # Random Walk scale for each unique_id
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1, shape=len(unique_ids))  # Observation noise for each series

    # Create a list to hold the state-space models for each unique_id
    trends = {}
    for i, unique_id in enumerate(unique_ids):
        trend = pm.GaussianRandomWalk(
            f"trend_{unique_id}", sigma=tau[i], shape=len(sales_data[unique_id])
        )
        trends[unique_id] = trend

        # Likelihood for observed sales
        pm.Normal(
            f"y_{unique_id}",
            mu=trends[unique_id],
            sigma=sigma_obs[i],
            observed=sales_data[unique_id],
        )

    # Sample from the posterior distribution using MCMC
    trace = pm.sample(2000, 
                      # tune=50,
                      target_accept=0.9)


# Visualize trace plots for the parameters
az.plot_trace(trace, var_names=["tau", "sigma_obs"], figsize=(10, 6))
plt.show()


# Forecasting for the next 7 days for each unique_id
future_days = 7
new_dates = pd.date_range(df.index.get_level_values('date')[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")

# Simulate future values for each unique_id from the posterior samples
future_trends = {}
for i, unique_id in enumerate(unique_ids):
    # Simulate future sales trend using the posterior samples of the trend
    trend_samples = trace[f"trend_{unique_id}"]
    future_trends[unique_id] = trend_samples[:, -1:] + np.random.normal(0, np.std(trend_samples, axis=1), (len(trend_samples), future_days))

# Plot the forecasted values with credible intervals
plt.figure(figsize=(12, 6))


for unique_id in unique_ids:
    # Get the forecasted trend and credible intervals
    forecast = future_trends[unique_id]
    lower, upper = np.percentile(forecast, 2.5, axis=0), np.percentile(forecast, 97.5, axis=0)

    # Plot observed sales data
    plt.plot(df.loc[unique_id].index, df.loc[unique_id, 'sales'], label=f"Observed {unique_id}")

    # Plot forecast and credible intervals
    plt.fill_between(new_dates, lower.flatten(), upper.flatten(), alpha=0.3, label=f"Forecast CI {unique_id}")
    plt.plot(new_dates, np.mean(future_trends[unique_id], axis=0), label=f"Forecast Mean {unique_id}")

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Bayesian Forecast for Sales with 94% CI (Next 7 Days)")
plt.legend()
plt.xticks(rotation=45)
plt.show()


with pm.Model() as model:
    # Priors for intercept and slope
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Linear model
    mu = alpha + beta
    
    # Likelihood (observed sales)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=df["sales"].values)

    # Sampling
    trace = pm.sample(2000, return_inferencedata=True)
    
    # Posterior Predictive
    posterior_pred = pm.sample_posterior_predictive(trace)



az.plot_posterior(trace, var_names=["alpha", "beta"])
plt.show()



# Extract posterior predictive samples
posterior_samples = posterior_pred.posterior_predictive["y_obs"].stack(samples=("chain", "draw")).values
mean_pred = posterior_samples.mean(axis=1)
std_pred = posterior_samples.std(axis=1)

# Plot actual vs predicted with uncertainty
plt.figure(figsize=(10, 6))
plt.scatter(df["date"], df["sales"], label="Actual Sales", alpha=0.6)
plt.plot(df["date"], mean_pred, label="Predicted Sales", linewidth=2)
plt.fill_between(df["date"], 
                 mean_pred - 1.96 * std_pred, 
                 mean_pred + 1.96 * std_pred, 
                 alpha=0.3, color='orange', label="95% Confidence Interval")

plt.xlabel("Time")
plt.ylabel("Sales")
plt.title("Bayesian Regression Predictions for Sales")
plt.legend()
plt.show()






import pymc as pm
import numpy as np

# Define a Bayesian State-Space model using PyMC
with pm.Model() as model:
    # Hyperparameters
    tau = pm.HalfNormal("tau", sigma=1)  # Random Walk scale
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)  # Observation noise
    
    # State-space model: Random walk
    trend = pm.GaussianRandomWalk("trend", sigma=tau, shape=len(data))
    
    # Likelihood (data model)
    y = pm.Normal("y", mu=trend, sigma=sigma_obs, observed=data)
    
    # Sample from the posterior distribution using MCMC
    trace = pm.sample(2000, tune=1000)

# Posterior predictive checks
posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["trend"], samples=500)

# Visualize the posterior predictive
import arviz as az
az.plot_posterior(posterior_predictive)



df


import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# Example: Multi-Series Bayesian Regression
np.random.seed(42)
n_series = 3
n_obs = 100

# Generate synthetic data
X = np.linspace(0, 10, n_obs)
y_series = [2 * X + np.random.normal(scale=2, size=n_obs) for _ in range(n_series)]

# Convert to DataFrame
df = pd.DataFrame({'X': np.tile(X, n_series), 'y': np.concatenate(y_series), 'series': np.repeat(range(n_series), n_obs)})

# Bayesian regression model
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10, shape=n_series)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=n_series)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Model equation
    mu = alpha[df['series'].values] + beta[df['series'].values] * df['X'].values

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=df['y'].values)

    # Sampling
    trace = pm.sample(2000, return_inferencedata=True)

# Plot results
az.plot_posterior(trace)



trace.posterior


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic multi-series data
n_series = 3
n_obs = 100
X = np.linspace(0, 10, n_obs)

# Different trends for each series
true_slopes = [2, -1.5, 3]  
true_intercepts = [5, 10, -5]
noise = np.random.normal(scale=2, size=(n_series, n_obs))

y_series = [true_intercepts[i] + true_slopes[i] * X + noise[i] for i in range(n_series)]

# Convert to DataFrame
df = pd.DataFrame({'X': np.tile(X, n_series), 
                   'y': np.concatenate(y_series), 
                   'series': np.repeat(range(n_series), n_obs)})

# Plot raw data
plt.figure(figsize=(10, 5))
for i in range(n_series):
    plt.scatter(df[df['series'] == i]['X'], df[df['series'] == i]['y'], label=f'Series {i}', alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Multi-Series Data')
plt.legend()
plt.show()



with pm.Model() as model:
    # Priors for each series (separate parameters)
    alpha = pm.Normal("alpha", mu=0, sigma=10, shape=n_series)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=n_series)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    # Model equation
    mu = alpha[df['series'].values] + beta[df['series'].values] * df['X'].values
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=df['y'].values)
    
    # Sampling
    trace = pm.sample(2000, return_inferencedata=True)
    
    # Generate posterior predictions
    posterior_pred = pm.sample_posterior_predictive(trace)



az.plot_posterior(trace, var_names=['alpha', 'beta'])
plt.show()


# Extract posterior predictive samples
posterior_samples = posterior_pred.posterior_predictive["y_obs"].stack(samples=("chain", "draw")).values
mean_pred = posterior_samples.mean(axis=1)
std_pred = posterior_samples.std(axis=1)

# Plot actual vs predicted with uncertainty
plt.figure(figsize=(10, 6))
for series_id in range(n_series):
    mask = df['series'] == series_id
    plt.scatter(df['X'][mask], df['y'][mask], label=f"Series {series_id} - Actual", alpha=0.6)
    plt.plot(df['X'][mask], mean_pred[mask], label=f"Series {series_id} - Predicted", linewidth=2)
    plt.fill_between(df['X'][mask], 
                     mean_pred[mask] - 1.96 * std_pred[mask], 
                     mean_pred[mask] + 1.96 * std_pred[mask], 
                     alpha=0.3)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Bayesian Multi-Series Regression Predictions")
plt.legend()
plt.show()


!pip install -U arviz pymc -q


import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Define date range
dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

# Define hidden state trends (random walk)
np.random.seed(42)
true_trend_1 = np.cumsum(np.random.normal(0, 0.5, 100))  # Latent state 1
true_trend_2 = np.cumsum(np.random.normal(0, 0.3, 100))  # Latent state 2

# Observations with noise
series_1 = true_trend_1 + np.random.normal(0, 1, 100)
series_2 = true_trend_2 + np.random.normal(0, 1, 100)

# Create DataFrame
df = pd.DataFrame({"date": dates, "series_1": series_1, "series_2": series_2})
df.set_index("date", inplace=True)

df.head()



import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Generate Multi-Series Time Series Data
dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

np.random.seed(42)
true_trend_1 = np.cumsum(np.random.normal(0, 0.5, 100))  
true_trend_2 = np.cumsum(np.random.normal(0, 0.3, 100))  

series_1 = true_trend_1 + np.random.normal(0, 1, 100)
series_2 = true_trend_2 + np.random.normal(0, 1, 100)

df = pd.DataFrame({"date": dates, "series_1": series_1, "series_2": series_2})
df.set_index("date", inplace=True)

# Bayesian State-Space Model
with pm.Model() as ssm_model:
    tau = pm.HalfNormal("tau", sigma=1, shape=2)
    trend_1 = pm.GaussianRandomWalk("trend_1", sigma=tau[0], shape=100)
    trend_2 = pm.GaussianRandomWalk("trend_2", sigma=tau[1], shape=100)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1, shape=2)

    y1 = pm.Normal("y1", mu=trend_1, sigma=sigma_obs[0], observed=series_1)
    y2 = pm.Normal("y2", mu=trend_2, sigma=sigma_obs[1], observed=series_2)

    trace = pm.sample(2000, tune=1000, target_accept=0.9)





idata = az.plot_trace(trace, compact=False);


# ArviZ Visualizations
az.plot_trace(trace, var_names=["tau", "sigma_obs"], figsize=(10, 6))
plt.show()

az.plot_posterior(trace, var_names=["tau", "sigma_obs"], hdi_prob=0.95)
plt.show()


import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Generate Multi-Series Time Series Data (as before)
dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

np.random.seed(42)
true_trend_1 = np.cumsum(np.random.normal(0, 0.5, 100))  
true_trend_2 = np.cumsum(np.random.normal(0, 0.3, 100))  

series_1 = true_trend_1 + np.random.normal(0, 1, 100)
series_2 = true_trend_2 + np.random.normal(0, 1, 100)

df = pd.DataFrame({"date": dates, "series_1": series_1, "series_2": series_2})
df.set_index("date", inplace=True)

# Bayesian State-Space Model (as before)
with pm.Model() as ssm_model:
    tau = pm.HalfNormal("tau", sigma=1, shape=2)
    trend_1 = pm.GaussianRandomWalk("trend_1", sigma=tau[0], shape=100)
    trend_2 = pm.GaussianRandomWalk("trend_2", sigma=tau[1], shape=100)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1, shape=2)

    y1 = pm.Normal("y1", mu=trend_1, sigma=sigma_obs[0], observed=series_1)
    y2 = pm.Normal("y2", mu=trend_2, sigma=sigma_obs[1], observed=series_2)

    trace = pm.sample(2000, tune=1000, target_accept=0.9)




df


# Extend forecast by 7 days (for next 7 days)
future_days = 7
new_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")
new_dates


trend_samples_1


# Extract posterior samples
posterior_samples = az.extract(trace, num_samples=500)
trend_samples_1, trend_samples_2 = posterior_samples["trend_1"], posterior_samples["trend_2"]


# Create future forecast samples
# Create a random walk for the next 7 days based on the last posterior sample
future_trends_1 = np.zeros((500, future_days))
future_trends_2 = np.zeros((500, future_days))

# Add random walk noise to extend the trends
for i in range(100):
    future_trends_1[i] = np.cumsum(np.random.normal(0, np.std(trend_samples_1[i]), future_days))
    future_trends_2[i] = np.cumsum(np.random.normal(0, np.std(trend_samples_2[i]), future_days))

# Concatenate observed and forecasted trends
extended_trend_1 = np.concatenate([trend_samples_1, future_trends_1], axis=1)
extended_trend_2 = np.concatenate([trend_samples_2, future_trends_2], axis=1)

# Compute the 94% HDI for the extended period (next 7 days)
hdi_low_1, hdi_high_1 = az.hdi(extended_trend_1, hdi_prob=0.94).T
hdi_low_2, hdi_high_2 = az.hdi(extended_trend_2, hdi_prob=0.94).T

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot observed data and forecast for Series 1
plt.plot(dates, series_1, label="Observed Series 1", color="blue")
plt.fill_between(new_dates, hdi_low_1[-1], hdi_high_1[-1], alpha=0.3, color="red", label="94% CI Forecast 1")

# Plot observed data and forecast for Series 2
plt.plot(dates, series_2, label="Observed Series 2", color="green")
plt.fill_between(new_dates, hdi_low_2[-1], hdi_high_2[-1], alpha=0.3, color="orange", label="94% CI Forecast 2")

plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Bayesian State-Space Model: Forecast for Next 7 Days with HDI")
plt.legend()
plt.xticks(rotation=45)
plt.show()





import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas_datareader.data import DataReader
cpi = DataReader('CPIAUCNS', 'fred', start='1971-01', end='2016-12')
cpi.index = pd.DatetimeIndex(cpi.index, freq='MS')
inf = np.log(cpi).resample('QS').mean().diff()[1:] * 400

nile = sm.datasets.nile.load_pandas().data['volume']
nile.index = pd.date_range('1871', '1970', freq='AS')
nile.plot()


from pandas_datareader.data import DataReader
start = '1984-01'
end = '2016-09'
labor = DataReader('HOANBS', 'fred',start=start, end=end).resample('QS').first()
cons = DataReader('PCECC96', 'fred', start=start, end=end).resample('QS').first()
inv = DataReader('GPDIC1', 'fred', start=start, end=end).resample('QS').first()
pop = DataReader('CNP16OV', 'fred', start=start, end=end)
pop = pop.resample('QS').mean()  # Convert pop from monthly to quarterly observations
recessions = DataReader('USRECQ', 'fred', start=start, end=end)
recessions = recessions.resample('QS').last()['USRECQ'].iloc[1:]

# Get in per-capita terms
N = labor['HOANBS'] * 6e4 / pop['CNP16OV']
C = (cons['PCECC96'] * 1e6 / pop['CNP16OV']) / 4
I = (inv['GPDIC1'] * 1e6 / pop['CNP16OV']) / 4
Y = C + I

# Log, detrend
y = np.log(Y).diff()[1:]
c = np.log(C).diff()[1:]
n = np.log(N).diff()[1:]
i = np.log(I).diff()[1:]
rbc_data = pd.concat((y, n, c), axis=1)
rbc_data.columns = ['output', 'labor', 'consumption']
rbc_data.plot()


# This is the class definition. Object oriented programming has the concept
# of inheritance, whereby classes may be "children" of other classes. The
# parent class is specified in the parentheses. When defining a class with
# no parent, the base class `object` is specified instead.
class Point(object):

    # The __init__ function is a special method that is run whenever an
    # object is created. In this case, the initial coordinates are set to
    # the origin. `self` is a variable which refers to the object instance
    # itself.
    def __init__(self):
        self.x = 0
        self.y = 0

    def change_x(self, dx):
        self.x = self.x + dx

    def change_y(self, dy):
        self.y = self.y + dy


# An object of class Point is created
point_object = Point()

# The object exposes it's attributes
print(point_object.x)  # 0

# And we can call the object's methods
# Notice that although `self` is the first argument of the class method,
# it is automatically populated, and we need only specify the other
# argument, `dx`.
point_object.change_x(-2)
print(point_object.x)  # -2


# This is the new class definition. Here, the parent class, `Point`, is in
# the parentheses.
class Vector(Point):

    def __init__(self, x, y):
        # Call the `Point.__init__` method to initialize the coordinates
        # to the origin
        super(Vector, self).__init__()

        # Now change to coordinates to those provided as arguments, using
        # the methods defined in the parent class.
        self.change_x(x)
        self.change_y(y)

    def length(self):
        # Notice that in Python the exponentiation operator is a double
        # asterisk, "**"
        return (self.x**2 + self.y**2)**0.5

# An object of class Vector is created
vector_object = Vector(1, 1)
print(vector_object.length())  # 1.41421356237


# Create a new class with parent sm.tsa.statespace.MLEModel
class LocalLevel(sm.tsa.statespace.MLEModel):

    # Define the initial parameter vector; see update() below for a note
    # on the required order of parameter values in the vector
    start_params = [1.0, 1.0]

    # Recall that the constructor (the __init__ method) is
    # always evaluated at the point of object instantiation
    # Here we require a single instantiation argument, the
    # observed dataset, called `endog` here.
    def __init__(self, endog):
        super(LocalLevel, self).__init__(endog, k_states=1)

        # Specify the fixed elements of the state space matrices
        self['design', 0, 0] = 1.0
        self['transition', 0, 0] = 1.0
        self['selection', 0, 0] = 1.0

        # Initialize as approximate diffuse, and "burn" the first
        # loglikelihood value
        self.initialize_approximate_diffuse()
        self.loglikelihood_burn = 1

    # Here we define how to update the state space matrices with the
    # parameters. Note that we must include the **kwargs argument
    def update(self, params, **kwargs):
        # Using the parameters in a specific order in the update method
        # implicitly defines the required order of parameters
        self['obs_cov', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]

# Instantiate a new object
nile_model_1 = LocalLevel(nile)


# Compute the loglikelihood at values specific to the nile model
print(nile_model_1.loglike([15099.0, 1469.1]))  # -632.537695048

# Try computing the loglikelihood with a different set of values; notice that it is different
print(nile_model_1.loglike([10000.0, 1.0]))  # -687.5456216


# Retrieve filtering output
nile_filtered_1 = nile_model_1.filter([15099.0, 1469.1])
# print the filtered estimate of the unobserved level
print(nile_filtered_1.filtered_state[0])         # [ 1103.34065938  ... 798.37029261 ]
print(nile_filtered_1.filtered_state_cov[0, 0])  # [ 14874.41126432  ... 4032.15794181 ]


# BEFORE: Perform some simulations with the original parameters
nile_simsmoother_1 = nile_model_1.simulation_smoother()
nile_model_1.update([15099.0, 1469.1])
nile_simsmoother_1.simulate()
# ...

# AFTER: Perform some new simulations with new parameters
nile_model_1.update([10000.0, 1.0])
nile_simsmoother_1.simulate()


# Load the generic minimization function from scipy
from scipy.optimize import minimize

# Create a new function to return the negative of the loglikelihood
nile_model_2 = LocalLevel(nile)
def neg_loglike(params):
    return -nile_model_2.loglike(params)

# Perform numerical optimization
output = minimize(neg_loglike, nile_model_2.start_params, method='Nelder-Mead')

print(output.x)  # [ 15108.31   1463.55]
print(nile_model_2.loglike(output.x))  # -632.537685587


class FirstMLELocalLevel(sm.tsa.statespace.MLEModel):
    start_params = [1.0, 1.0]
    param_names = ['obs.var', 'level.var']

    def __init__(self, endog):
        super(FirstMLELocalLevel, self).__init__(endog, k_states=1)

        self['design', 0, 0] = 1.0
        self['transition', 0, 0] = 1.0
        self['selection', 0, 0] = 1.0

        self.initialize_approximate_diffuse()
        self.loglikelihood_burn = 1

    def update(self, params, **kwargs):
        # Transform the parameters if they are not yet transformed
        params = super(FirstMLELocalLevel, self).update(params, **kwargs)

        self['obs_cov', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]


nile_mlemodel_1 = FirstMLELocalLevel(nile)

print(nile_mlemodel_1.loglike([15099.0, 1469.1]))  # -632.537695048

# Again we use Nelder-Mead; now specified as method='nm'
nile_mleresults_1 = nile_mlemodel_1.fit(method='nm', maxiter=1000)
print(nile_mleresults_1.summary())


nile_mleresults_1.plot_diagnostics()


inf_model = FirstMLELocalLevel(inf)
inf_results = inf_model.fit()

inf_forecast = inf_results.get_prediction(start='2005-01-01', end='2020-01-01')
print(inf_forecast.predicted_mean)  # [2005-01-01   2.439005 ...
print(inf_forecast.conf_int()) 


import matplotlib.pyplot as plt
import pandas as pd

# Assuming inf_forecast is your forecast result
# Extract predicted mean and confidence intervals
predicted_mean = inf_forecast.predicted_mean
conf_int = inf_forecast.conf_int()

# Plot the predicted mean and confidence intervals
plt.figure(figsize=(10, 6))

# Plot the predicted values (mean)
plt.plot(predicted_mean, label='Predicted Mean', color='blue')

# Plot the confidence intervals (upper and lower bounds)
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.2, label='95% Confidence Interval')

# Add labels and title
plt.title('ARMA(1,1) Forecast with 95% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')

# Show the plot
plt.show()



class MLELocalLevel(sm.tsa.statespace.MLEModel):
    start_params = [1.0, 1.0]
    param_names = ['obs.var', 'level.var']

    def __init__(self, endog):
        super(MLELocalLevel, self).__init__(endog, k_states=1)

        self['design', 0, 0] = 1.0
        self['transition', 0, 0] = 1.0
        self['selection', 0, 0] = 1.0

        self.initialize_approximate_diffuse()
        self.loglikelihood_burn = 1

    def transform_params(self, params):
        return params**2

    def untransform_params(self, params):
        return params**0.5

    def update(self, params, **kwargs):
        # Transform the parameters if they are not yet transformed
        params = super(MLELocalLevel, self).update(params, **kwargs)

        self['obs_cov', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]
        
from scipy.stats import multivariate_normal, invgamma, uniform

# Create the model for likelihood evaluation
model = MLELocalLevel(nile)

# Specify priors
prior_obs = invgamma(3, scale=300)
prior_level = invgamma(3, scale=120)

# Specify the random walk proposal
rw_proposal = multivariate_normal(cov=np.eye(2)*10)


# Create storage arrays for the traces
n_iterations = 10000
trace = np.zeros((n_iterations + 1, 2))
trace_accepts = np.zeros(n_iterations)
trace[0] = [120, 30]  # Initial values

# Iterations
for s in range(1, n_iterations + 1):
    proposed = trace[s-1] + rw_proposal.rvs()

    acceptance_probability = np.exp(
        model.loglike(proposed**2) - model.loglike(trace[s-1]**2) +
        prior_obs.logpdf(proposed[0]) + prior_level.logpdf(proposed[1]) -
        prior_obs.logpdf(trace[s-1, 0]) - prior_level.logpdf(trace[s-1, 1]))

    if acceptance_probability > uniform.rvs():
        trace[s] = proposed
        trace_accepts[s-1] = 1
    else:
        trace[s] = trace[s-1]


model.plot_diagnostics()


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace import UnobservedComponents

# Example dataframe
data = {
    'unique_id': [2992, 5274, 5274, 5274, 2992],
    'date': ['2020-08-01', '2020-08-01', '2020-08-02', '2020-08-03', '2020-08-03'],
    'sales': [18.85, 16.41, 7.86, 3.02, 24.72]
}

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# Pivot data to have sales for each unique_id in separate columns
df_pivoted = df.pivot(index='date', columns='unique_id', values='sales')

# Define the Unobserved Components model
# We assume here we have two components: trend and seasonal.
model = UnobservedComponents(df_pivoted, level='local linear trend', seasonal=7)

# Fit the model
results = model.fit()

# Print the summary of the model fitting
print(results.summary())

# Make predictions
forecast = results.get_forecast(steps=5)  # Forecast next 5 days
print(forecast.predicted_mean)

# You can also plot the predictions if needed
import matplotlib.pyplot as plt
df_pivoted.plot(label='Original', color='blue')
forecast.predicted_mean.plot(label='Forecast', color='red')
plt.legend()
plt.show()



import arviz as az
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)

xdata = np.linspace(0, 50, 100)
b0, b1, sigma = -2, 1, 3
ydata = rng.normal(loc=b1 * xdata + b0, scale=sigma)
plt.plot(xdata, ydata);


with pm.Model() as linreg_model:
    # optional: add coords to "time" dimension
    linreg_model.add_coord("time", np.arange(len(xdata)), mutable=True)

    x = pm.MutableData("x", xdata, dims="time")
    y_obs = pm.MutableData("y_obs", ydata, dims="time")

    b0 = pm.Normal("b0", 0, 10)
    b1 = pm.Normal("b1", 0, 10)
    sigma_e = pm.HalfNormal("sigma_e", 10)

    pm.Normal("y", b0 + b1 * x, sigma_e, observed=y_obs, dims="time")


sample_kwargs = {"chains": 4, "draws": 500, "log_likelihood": True}
with linreg_model:
    idata = pm.sample(**sample_kwargs)


idata





from scipy import stats
from xarray_einstats.stats import XrContinuousRV


class PyMCLinRegWrapper(az.PyMCSamplingWrapper):
    def sample(self, modified_observed_data):
        with self.model:
            # if the model had coords the dim needs to be updated before
            # modifying the data in the model with set_data
            # otherwise, we don't need to overwrite the sample method
            n__i = len(modified_observed_data["x"])
            self.model.set_dim("time", n__i, coord_values=np.arange(n__i))

            pm.set_data(modified_observed_data)
            idata = pm.sample(
                **self.sample_kwargs,
            )
        return idata

    def log_likelihood__i(self, excluded_observed_data, idata__i):
        post = idata__i.posterior
        dist = XrContinuousRV(
            stats.norm,
            post["b0"] + post["b1"] * excluded_observed_data["x"],
            post["sigma_e"],
        )
        return dist.logpdf(excluded_observed_data["y_obs"])

    def sel_observations(self, idx):
        xdata = self.idata_orig["constant_data"]["x"]
        ydata = self.idata_orig["observed_data"]["y"]
        mask = np.isin(np.arange(len(xdata)), idx)
        data_dict = {"x": xdata, "y_obs": ydata}
        data__i = {key: value.values[~mask] for key, value in data_dict.items()}
        data_ex = {key: value.isel(time=idx) for key, value in data_dict.items()}
        return data__i, data_ex


loo_orig = az.loo(idata, pointwise=True)
loo_orig


import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Simulate some data (for illustration)
np.random.seed(42)
n_samples = 100
X = np.linspace(0, 10, n_samples)
y_true = 2.5 * X + 5 + np.random.normal(0, 1, size=n_samples)

# Define the PyMC model
with pm.Model() as model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Likelihood (sampling distribution) of observations
    mu = alpha + beta * X
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_true)
    
    # Inference: draw posterior samples
    trace = pm.sample(2000, return_inferencedata=True)

# Use ArviZ to analyze and visualize the results
az.plot_trace(trace)
plt.show()

az.summary(trace)



posterior_predictive.observed_data['Y_obs']


# Posterior predictive samples
posterior_predictive = pm.sample_posterior_predictive(trace, model=model)

# Compute the posterior predictive mean
posterior_predictive_mean =posterior_predictive.observed_data['Y_obs'].mean(axis=0)

# Plot HDI and true data
ax = az.plot_hdi(X, posterior_predictive_mean, plot_kwargs={"ls": "--"})
ax.plot(X, y_true, label='True Data', color='red')
plt.legend()
plt.show()


# Posterior predictive mean
posterior_predictive_mean = posterior_predictive.observed_data['Y_obs']

# Plot the observed data and the posterior predictive line
plt.plot(X, y_true, label="Observed Data")
plt.plot(X, posterior_predictive_mean, label="Posterior Predictive", color='red')
plt.legend()
plt.show()






import numpy as np
import scipy.stats as st

# generate observed data
X = st.norm(loc=3, scale=1).rvs(size=1000)


def guassian_posterior(X, theta):
    # returns the unnormalized log posterior
    loglik = np.sum(np.log(st.norm(loc=theta, scale=1).pdf(X)))
    logprior = np.log(st.norm(loc=0, scale=1).pdf(theta))
    
    return loglik + logprior
    
def guassian_proposal(theta_curr):
    # proposal based on Gaussian
    theta_new = st.norm(loc=theta_curr, scale=0.2).rvs()
    return theta_new

def guassian_proposal_prob(x1, x2):
    # calculate proposal probability q(x2|x1), based on Gaussian
    q = st.norm(loc=x1, scale=1).pdf(x2)
    return q

def mcmc_mh_posterior(X, theta_init, func, proposal_func, proposal_func_prob, n_iter=1000):
    # Metropolis-Hastings to estimate posterior
    thetas = []
    theta_curr = theta_init
    accept_rates = []
    accept_cum = 0
    
    for i in range(1, n_iter+1):
        theta_new = proposal_func(theta_curr)
        
        prob_curr = func(X, theta_curr)
        prob_new = func(X, theta_new)
        
        # we calculate the prob=exp(x) only when prob<1 so the exp(x) will not overflow for large x
        if prob_new > prob_curr:
            acceptance_ratio = 1
        else:
            qr = proposal_func_prob(theta_curr, theta_new)/proposal_func_prob(theta_curr, theta_new)
            acceptance_ratio = np.exp(prob_new - prob_curr) * qr
        acceptance_prob = min(1, acceptance_ratio)
        
        if acceptance_prob > st.uniform(0,1).rvs():
            theta_curr = theta_new
            accept_cum = accept_cum+1
            thetas.append(theta_new)
        else:
            thetas.append(theta_curr)
            
        accept_rates.append(accept_cum/i)
        
    return thetas, accept_rates

# run MCMC
thetas, accept_rates = mcmc_mh_posterior(X, 1, 
                                         guassian_posterior, guassian_proposal, guassian_proposal_prob, 
                                         n_iter=8000)


from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import seaborn as sns

def plot_res(xs, burn_in, x_name):
    # plot trace (based on xs), distribution, and autocorrelation

    xs_kept = xs[burn_in:]
    
    # plot trace full
    fig, ax = plt.subplots(2,2, figsize=(15,5))
    ax[0,0].plot(xs)
    ax[0,0].set_title('Trace, full')
    
    # plot trace, after burn-in
    ax[0,1].plot(xs_kept)
    ax[0,1].set_title('Trace, after discarding burn-in')

    # plot distribution, after burn-in
    sns.histplot(xs_kept, ax=ax[1,0])
    ax[1,0].set_xlabel(f'{x_name} (after burn-in)')
    
    # plot autocorrelation, after burn-in
    plot_acf(np.array(xs_kept), lags=100, ax=ax[1,1], title='')
    ax[1,1].set_xlabel('Lag (after burn-in)')
    ax[1,1].set_ylabel('Autocorrelation')

plot_res(thetas, 500, 'theta')
print(f"Mean acceptance rate: {np.mean(accept_rates[500:]): .3f}")


import pymc as pm

with pm.Model() as model:

    prior = pm.Normal('mu', mu=0, sigma=1)  # prior
    obs = pm.Normal('obs', mu=prior, sigma=1, observed=X)  # likelihood
    step = pm.Metropolis()

    # sample with 3 independent Markov chains
    trace = pm.sample(draws=50000, chains=3, step=step, return_inferencedata=True)  

pm.plot_trace(trace)
pm.plot_posterior(trace)


# Basic scientific libraries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Statsmodels is the primarily library used here
# for Bayesian inference of time series models
import statsmodels.api as sm

# Output from Markov chain Monte Carlo (MCMC)
# simulations is naturally multidimensional and
# tends to be quite large. The xarray and ArviZ
# libraries make it easy to work with.
import xarray as xr
import arviz as az

# Formatting settings for the notebook
np.set_printoptions(suppress=True, linewidth=120)


from pandas_datareader.data import DataReader
# Load data and assign the correct date index
y = DataReader('IPB50001N', 'fred', start='1980', end='2022-04')['IPB50001N'].asfreq('MS')
y.index.name = 'date'

# Plot the data
y.plot(title='U.S. Industrial production, monthly', figsize=(15, 5));


# Construct an object that contains our dataset, `y`,
# and specifies the model, here a random walk
model_rw = sm.tsa.UnobservedComponents(y, 'random walk')

# Check the parameters that the model depends on only
# include a variance term:
print(model_rw.param_names)


# Prior distribution
prior = stats.uniform(0.0001, 100)

# Plot the density
X = np.linspace(0.0001, 100, num=1000)
fig, ax = plt.subplots()
ax.plot(X, prior.pdf(X))
ax.set_title(r'Prior density for $\sigma^2$');


# Perturbation distribution
perturb = stats.norm(scale=0.1)


def draw_ig_gs_step(equation, ix, mod, states, prior):
    """
    Sample a variance parameter in Gibbs sampling iteration
    
    Assumes a state space model and an inverse-Gamma prior.
    
    Parameters
    ----------
    equation : {'observation', 'state'}
        State space system equation in which the variance
        parameter occurs.
    ix : int or iterable of int
        Index of the variance parameter or parameters to be
        sampled within the given equation. For example, if
        `equation="observation"` and `ix=[0, 1]`, then this
        function samples the variance associated with the
        first column of the data.
    mod : sm.tsa.statespace.mlemodel.MLEModel
        State space model object (for example, in the two
        applications above this would have been an instance
        of sm.tsa.UnobservedComponents)
    states : array_like
        Current values of the latent state vector in the
        Markov chain.
    prior : stats.invgamma object or list of stats.invgamma
        Prior distribution(s) for the variance parameter(s).
        
    Returns
    -------
    draw : float or array_like
        Sampled parameter(s) at index `ix` in equation
        `equation`.
    """
    # Retrieve the prior hyperparameters
    if isinstance(prior, (list, tuple)):
        prior_shape = np.array([p.args[0] for p in prior])
        prior_scale = np.array([p.kwds['scale'] for p in prior])
    else:
        prior_shape = prior.args[0]
        prior_scale = prior.kwds['scale']

    # Compute the appropriate residual
    if equation == 'observation':
        d = mod['obs_intercept', ix]
        Z = mod['design', ix, :]
        resid = mod.endog[:, ix] - d - states @ Z.T
    elif equation == 'state':
        c = mod['state_intercept', ix]
        T = mod['transition', ix, :]
        resid = states[1:, ix] - c - states[:-1] @ T.T

    squeeze = False
    if resid.ndim == 1:
        squeeze = True
        resid = resid[:, np.newaxis]

    # Draw from the conditional posterior
    post_shape = np.sum(~np.isnan(resid), axis=0) / 2 + prior_shape
    post_scale = np.nansum(resid**2, axis=0) / 2 + prior_scale
    draw = np.array([stats.invgamma(post_shape[i], scale=post_scale[i]).rvs()
                     for i in range(resid.shape[1])])

    if squeeze:
        draw = draw[0]

    return draw


# Construct the model object...
mod = sm.tsa.UnobservedComponents(y, 'lltrend', seasonal=12, use_exact_diffuse=True)

# ... and simulation smoother object
sim = mod.simulation_smoother(simulate_state=True)

# Print the parameters
print(mod.param_names)



# Priors
priors = [stats.invgamma(0.001, scale=0.001)] * 4


# Forecast
n_fcast = 100
ix = pd.date_range(y.index[0], periods=len(y) + n_fcast, freq='MS')
fcast_ix = ix[-n_fcast:]


# Storage
niter = 100000
params = xr.DataArray(np.zeros((niter + 1, mod.k_params)),
                      dims=['draw', 'param'],
                      coords={'param': mod.param_names},
                      name='params')
states = xr.DataArray(np.zeros((niter + 1, mod.nobs, mod.k_states)),
                      dims=['draw', 'date', 'state'],
                      coords={'date': mod._index.rename('date'),
                              'state': mod.state_names},
                      name='states')
fcast = xr.DataArray(np.zeros((niter + 1, n_fcast, mod.k_endog)),
                     dims=['draw', 'date', 'variable'],
                     coords={'date': fcast_ix,
                             'variable': [mod.endog_names]},
                     name='forecast')


niter = 5000


# Initial values
params[0] = mod.fit(disp=False).params

# Iterations
for i in range(1, niter + 1):
    if (i % max(200, niter // 20)) == 0:
        print(i)
        
    # Sample from state posterior
    mod.update(params[i - 1])
    sim.simulate()
    states[i] = sim.simulated_state.T

    # Predictive simulation from sample end
    fcast[i] = mod.simulate(params[i - 1], n_fcast,
                            initial_state=states[i, -1]).to_frame()

    # Gibbs sampling
    params[i, 0] = draw_ig_gs_step(
        'observation', 0, mod, sim.simulated_state.T, priors[0])
    params[i, 1:4] = draw_ig_gs_step(
        'state', [0, 1, 2], mod, sim.simulated_state.T, priors[1:4])


np.s_[nburn + 1::nthin]


params.sel(draw=np.s_[nburn + 1::nthin])


nburn = 5000
nthin = 10
az.plot_posterior(params.sel(draw=np.s_[nburn + 1::nthin]),
                  point_estimate='median')


fig, ax = plt.subplots()

# Plot the observed data
ax = y.loc['2009':].rename('Industrial production').plot(ax=ax)

# Plot the median forecast, through December 2023
fcast_plot = fcast.sel(draw=np.s_[nburn + 1::nthin], date=np.s_[:'2023-12':])
fcast_plot.median(axis=0).to_pandas()['IPB50001N'].rename('Forecast').plot(ax=ax)

# Plot the 90/10 credible interval
ax.fill_between(fcast_plot.coords['date'].to_index(),
                fcast_plot.quantile(0.1, dim='draw')[:, 0],
                fcast_plot.quantile(0.9, dim='draw')[:, 0],
                color='C1', alpha=0.2)

# Finalize the plot
ax.legend(loc='upper left')
ax.set_xlabel(None)
ax.yaxis.set_label_coords(1, 1)
ax.yaxis.set_tick_params(left=False, right=False, labelleft=False, labelright=True)
ax.yaxis.grid()
[ax.spines[s].set_visible(False) for s in ['left', 'top', 'right']]
fig.tight_layout()


fig, axes = plt.subplots(3, figsize=(6, 6));

states_plot = states.sel(draw=np.s_[nburn + 1::nthin])
level = states_plot.sel(state='level')
trend = states_plot.sel(state='trend')
seasonal = states_plot.sel(state='seasonal')

# Plot the estimated level, over time
level.median(axis=0).to_pandas().rename('Estimated level').plot(ax=axes[0])
axes[0].fill_between(level.coords['date'].to_index(),
                     level.quantile(0.1, dim='draw'),
                     level.quantile(0.9, dim='draw'),
                     color='C0', alpha=0.2)
axes[0].legend(loc=(0, 0.80))
axes[0].set_xlabel(None)
axes[0].yaxis.set_label_coords(1, 1)
axes[0].yaxis.set_tick_params(left=False, right=False, labelleft=False, labelright=True)
axes[0].yaxis.grid()
[axes[0].spines[s].set_visible(False) for s in ['left', 'top', 'right']]

# Plot the estimated trend, over time
trend.median(axis=0).to_pandas().rename('Estimated trend').plot(ax=axes[1])
axes[1].fill_between(trend.coords['date'].to_index(),
                     trend.quantile(0.1, dim='draw'),
                     trend.quantile(0.9, dim='draw'),
                     color='C0', alpha=0.2)
axes[1].axhline(0, color='k', linestyle='--', linewidth=1, zorder=0)
axes[1].legend(loc=(0, 0.85))
axes[1].set_xlabel(None)
axes[1].yaxis.set_label_coords(1, 1)
axes[1].yaxis.set_tick_params(left=False, right=False, labelleft=False, labelright=True)
axes[1].yaxis.grid()
axes[1].axhline(0, color='#555', linewidth=1)
[axes[1].spines[s].set_visible(False) for s in ['left', 'top', 'right']]

# Plot the estimated seasonal components, aggregated over time
x = seasonal.T.to_pandas()
x.index = x.index.month.rename('month')
x = x.T.melt()
g = x.groupby('month')
gm = g.median()['value']
gm.index = np.arange(12)
gm.rename('Estimated seasonal effect').plot(ax=axes[2], marker='s', markersize=5)
axes[2].legend(loc=(0, 0.85))
axes[2].fill_between(np.arange(12),
                     g.quantile(0.1)['value'],
                     g.quantile(0.9)['value'],
                     color='C0', alpha=0.2)
axes[2].xaxis.set_ticks(np.arange(0, 12))
axes[2].xaxis.set_ticklabels(pd.period_range('2020', periods=12, freq='M').strftime('%b'))
axes[2].set_xlabel('')
axes[2].axhline(0, color='k', linewidth=1)
axes[2].yaxis.set_tick_params(left=False, right=False, labelleft=False, labelright=True)
axes[2].yaxis.grid(alpha=0.5)
axes[2].xaxis.set_tick_params(bottom=False)
[axes[2].spines[s].set_visible(False) for s in ['left', 'top', 'right', 'bottom']]

fig.suptitle('')
fig.tight_layout()


https://github.com/ChadFulton/fulton_statsmodels_2017/blob/master/notebooks/Local%20Level%20-%20Nile.ipynb

