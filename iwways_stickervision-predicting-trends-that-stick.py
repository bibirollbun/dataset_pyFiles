import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



# this code specifies a path for files, since I'm working on my local machine and pushing notebooks to kaggle
BASE_PATH = ""
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    BASE_PATH = '/kaggle/input/playground-series-s5e1/'
else:
    BASE_PATH = 'kaggle/input/playground-series-s5e1/'



X_train = pd.read_csv(f'{BASE_PATH}train.csv')
X_test = pd.read_csv(f'{BASE_PATH}test.csv')




print(f'Train dataset contains {X_train.shape[0]} rows and {X_train.shape[1]} columns.')
X_train.head()


X_train.info()


# Save 'id' column for submission
test_ids = X_test['id']

# Define the target column
target_column = 'num_sold'

# Select categorical and numerical columns (initial)
categorical_columns = X_train.select_dtypes(include=['object']).columns
numerical_columns = X_train.select_dtypes(exclude=['object']).columns

# Print out column information
print("Target Column:", target_column)
print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


X_train.describe()


for column in categorical_columns:
    num_unique = X_train[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")



# Print top 10 unique value counts for each categorical column
for column in categorical_columns:
    print(f"\nTop value counts in '{column}':\n{X_train[column].value_counts().head(10)}")


print("The mean of columns:")
print(X_train[numerical_columns].mean())

print("\nThe std dev of columns:")
print(X_train[numerical_columns].std())

print("\nThe skewness of columns:")
print(X_train[numerical_columns].skew())


plt.figure(figsize=(15,9))
plt.title("Visualizing Missing Values")
sns.heatmap(X_train.isnull(), cbar=False, cmap=sns.color_palette('magma'), yticklabels=False)
plt.show()


filtered_columns = [col for col in categorical_columns if col != 'date']

fig, axes = plt.subplots(len(filtered_columns), 2, figsize=(15, 5 * len(filtered_columns)))

for i, column in enumerate(filtered_columns):
    sns.countplot(data=X_train, x=column, ax=axes[i, 0], palette='tab10')
    axes[i, 0].set_title(f'Distribution of {column}', fontsize=14)
    axes[i, 0].set_xlabel(column, fontsize=12)
    axes[i, 0].set_ylabel('Count', fontsize=12)
    sns.despine(ax=axes[i, 0])

    sns.boxplot(data=X_train, x=column, y=target_column, ax=axes[i, 1], palette='tab10')
    axes[i, 1].set_title(f'{column} vs {target_column}', fontsize=14)
    axes[i, 1].set_xlabel(column, fontsize=12)
    axes[i, 1].set_ylabel(target_column, fontsize=12)
    sns.despine(ax=axes[i, 1])

plt.tight_layout()   
plt.show()


sns.kdeplot(X_train['num_sold'])


missing_num_sold_df = X_train[X_train['num_sold'].isna()]
missing_num_sold_df


for c in ['country', 'store', 'product']:
    print(missing_num_sold_df[c].value_counts())
    print()


X_train = X_train.dropna(axis=0)


X_train.isna().sum()


X_train.date = X_train.date.astype('datetime64[ns]')


plt.figure(figsize=(28, 6))
X_train.groupby('date')['num_sold'].sum().plot(title='Total Sales Over Time', xlabel='Date', ylabel='Number of Products Sold')
plt.grid()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(28, 16))

df = X_train.copy()

df = df.drop('id', axis=1)

df = df.set_index('date')

grouped = df.groupby(['date', 'product'])[['num_sold']].sum()

sns.lineplot(data=grouped, x=grouped.index.get_level_values('date'), y='num_sold', hue='product')

plt.title('Sales Over Time by Product')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(28, 16))

df = X_train.copy()

df = df.drop('id', axis=1)

df = df.set_index('date')

grouped = df.groupby(['date', 'country'])[['num_sold']].sum()

sns.lineplot(data=grouped, x=grouped.index.get_level_values('date'), y='num_sold', hue='country')

plt.title('Sales Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(28, 16))

df = X_train.copy()

df = df.drop('id', axis=1)

df = df.set_index('date')

grouped = df.groupby(['date', 'store'])[['num_sold']].sum()

sns.lineplot(data=grouped, x=grouped.index.get_level_values('date'), y='num_sold', hue='store')

plt.title('Sales Over Time by Product')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



X_data_index = X_train.set_index('date')
from statsmodels.tsa.seasonal import seasonal_decompose

monthly_data = X_data_index.resample('D').sum()

monthly_data['year'] = monthly_data.index.year
monthly_data['month'] = monthly_data.index.month
trend_data = []
for year in monthly_data['year'].unique():
    yearly_data = monthly_data[monthly_data['year'] == year]
    
    decomposition = seasonal_decompose(yearly_data['num_sold'], model='additive', period=12)
    
    trend = pd.DataFrame({
        "trend": decomposition.trend,
        "month": yearly_data['month'],
        "year": year
    })
    trend_data.append(trend)

trend_df = pd.concat(trend_data)

plt.figure(figsize=(20, 10))
sns.lineplot(data=trend_df, x='month', y='trend', hue='year', palette='tab10')
plt.title("Trend Components Over Years")
plt.xlabel("Month")
plt.ylabel("Trend Component")
plt.grid(True)
plt.show()




seasonal_data = []
for year in monthly_data['year'].unique():
    yearly_data = monthly_data[monthly_data['year'] == year]
    

    decomposition = seasonal_decompose(yearly_data['num_sold'], model='additive', period=12)
    
    seasonal = pd.DataFrame({
        "seasonal": decomposition.seasonal,
        "month": yearly_data['month'],
        "year": year
    })

    seasonal_data.append(seasonal)
    
seasonal_df = pd.concat(seasonal_data)
plt.figure(figsize=(20, 10))
sns.lineplot(data=seasonal_df, x='month', y='seasonal', hue='year', palette='tab10')
plt.title("Seasonal Components Over Years")
plt.xlabel("Month")
plt.ylabel("Seasonal Component")
plt.grid(True)
plt.show()
 


resid_data = []
for year in monthly_data['year'].unique():
    yearly_data = monthly_data[monthly_data['year'] == year]
    

    decomposition = seasonal_decompose(yearly_data['num_sold'], model='additive', period=12)
    
    resid = pd.DataFrame({
        "resid": decomposition.resid,
        "month": yearly_data['month'],
        "year": year
    })

    resid_data.append(resid)
    
resid_df = pd.concat(resid_data)
plt.figure(figsize=(20, 10))
sns.lineplot(data=resid_df, x='month', y='resid', hue='year', palette='tab10')
plt.title("Resid Components Over Years")
plt.xlabel("Month")
plt.ylabel("Resid Component")
plt.grid(True)
plt.show()


from statsmodels.tsa.stattools import adfuller

def test_stationarity(series, title=''):
    print(f"Results of ADF Test on {title}:")
    result = adfuller(series, autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] > 0.05:
        print("Non-stationary")
    else:
        print("stationary")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value}")
    print("\n")

test_stationarity(X_train['num_sold'], 'Sold')


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(X_train['num_sold'], lags=50)
plot_pacf(X_train['num_sold'], lags=50)
plt.show()


train_end = '2015-01-01'
val_end = '2017-01-01'

train_data = X_train[X_train['date'] < train_end]
val_data = X_train[(X_train['date'] >= train_end) & (X_train['date'] < val_end)]

train_data = train_data.groupby('date')['num_sold'].sum()
val_data = val_data.groupby('date')['num_sold'].sum()

train_data = train_data.asfreq('D', method='pad') 
val_data = val_data.asfreq('D', method='pad')     

print(f"Train data: {train_data.shape}")
print(f"Validation data: {val_data.shape}")

print(train_data.head())
print(val_data.head())



# import pmdarima as pm
# auto_arima = pm.auto_arima(train_data, stepwise=False, seasonal=True)
# print(auto_arima)


# from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.arima.model import ARIMA


# model_train = ARIMA(train_data, order=(4,1,1))
# model_train_fit = ''



# history_endog = list(train_data.copy(deep=True))
# y_true = []
# y_pred = []

# for obs in val_data: 
#     model = ARIMA(endog=history_endog, order=(2, 1, 1))
#     model_train_fit = model.fit()
#     forecast = model_train_fit.forecast()[0]

#     y_true.append(obs)
#     y_pred.append(forecast)
#     history_endog.append(obs)







# test_forecast = model_train_fit.get_forecast(steps=len(val_data))
# test_forecast_series = pd.Series(y_pred, index=val_data.index)



# plt.figure(figsize=(14,7))
# plt.plot(train_data, label='Training Data')
# plt.plot(val_data, label='Actual Data', color='orange')
# plt.plot(test_forecast_series, label='Forecasted Data', color='green')
# plt.title('ARIMA Model Evaluation')
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.legend()
# plt.show()




# # Predict specific points in x_test
# x_test_forecasts = []

# for i in range(X_test.shape[0]):
#     # Fit ARIMA model on the updated history
#     model = ARIMA(endog=history_endog, order=(2, 1, 1))
#     model_fit = model.fit()

#     # Forecast the next point (use the ARIMA model)
#     forecast = model_fit.forecast()[0]
#     x_test_forecasts.append(forecast)

#     # (Optional) Add the predicted value to the history for rolling prediction
#     history_endog.append(forecast)

# # Convert x_test_forecasts to a numpy array
# x_test_forecasts = np.array(x_test_forecasts)

# # Print x_test forecasts
# print(f"x_test Forecasts: {x_test_forecasts}")


# subm = pd.DataFrame({
#     "id": test_ids,
#     "num_sold": x_test_forecasts
# })

# subm.to_csv('arima_sum')


# from statsmodels.tsa.statespace.sarimax import SARIMAX
# model_train = SARIMAX(
#     train_data, 
#     order=(1, 1, 2),       
#     seasonal_order=(1, 1, 1, 7), 
#     enforce_stationarity=False,
#     enforce_invertibility=False
# )

# model_train_fit = model_train.fit(disp=False)

# print(model_train_fit.summary())

# test_forecast = model_train_fit.get_forecast(steps=len(val_data))

# forecast_series = test_forecast.predicted_mean


# # Create a plot to compare the forecast with the actual test data
# plt.figure(figsize=(14,7))
# plt.plot(train_data, label='Training Data')
# plt.plot(val_data, label='Actual Data', color='orange')
# plt.plot(forecast_series, label='Forecasted Data', color='green')
# plt.fill_between(val_data.index, 
#                  test_forecast.conf_int().iloc[:, 0], 
#                  test_forecast.conf_int().iloc[:, 1], 
#                  color='k', alpha=.15)
# plt.title('SARIMA Model Evaluation')
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.legend()
# plt.show()



X_train = pd.read_csv(f'{BASE_PATH}train.csv')
X_test = pd.read_csv(f'{BASE_PATH}test.csv')


def time_features(df):
    col = 'date'
    df['day'] = df.date.dt.day
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year
    df['quarter'] = df.date.dt.quarter
    df['dayofyear'] = df.date.dt.dayofyear
    df['weekday'] = df.date.dt.weekday
    df['sine_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['sine_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sine_year'] = np.sin(2 * np.pi * df['year'])
    df['cos_year'] = np.cos(2 * np.pi * df['year'])
    df['sine_quarter'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['cos_quarter'] = np.cos(2 * np.pi * df['quarter'] / 4)
    df['sine_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 366)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 366)
    df['sine_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    #Add group feature (for time-based grouping)
    df[f'{col}_Group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    

    return df


# def time_features(df):
#     col = 'date'
#    # Extract temporal features
#     df[f'{col}_year'] = df[col].dt.year.astype('float64')
#     df[f'{col}_quarter'] = df[col].dt.quarter.astype('float64')
#     df[f'{col}_month'] = df[col].dt.month.astype('float64')
#     df[f'{col}_day'] = df[col].dt.day.astype('float64')
#     df[f'{col}_day_of_week'] = df[col].dt.dayofweek.astype('float64')
#     df[f'{col}_week_of_year'] = df[col].dt.isocalendar().week.astype('float64')
#     df[f'{col}_hour'] = df[col].dt.hour.astype('float64')
#     df[f'{col}_minute'] = df[col].dt.minute.astype('float64')
    
#     # Add cyclical encodings
#     df[f'{col}_day_sin'] = np.sin(2 * np.pi * df[f'{col}_day'] / 365.0)
#     df[f'{col}_day_cos'] = np.cos(2 * np.pi * df[f'{col}_day'] / 365.0)
#     df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12.0)
#     df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12.0)
#     df[f'{col}_year_sin'] = np.sin(2 * np.pi * df[f'{col}_year'] / 7.0)
#     df[f'{col}_year_cos'] = np.cos(2 * np.pi * df[f'{col}_year'] / 7.0)
    
#     # Add group feature (for time-based grouping)
#     df[f'{col}_Group'] = (df[f'{col}_year'] - 2010) * 48 + df[f'{col}_month'] * 4 + df[f'{col}_day'] // 7
    

#     return df


import holidays
def holiday_features(df):
    df['is_holiday'] = 0
    country_codes = {
        "Finland": "FI",
        "Italy": "IT",
        "Norway": "NO",
        "Singapore": "SG",
        "Canada": "CA",
        "Kenya": "KE"
    }
    
    for i in range(df.shape[0]):
        ct = country_codes[df.iloc[i].country]
        ct_holidays = holidays.country_holidays(ct)

        if df.iloc[i].date in ct_holidays:
            df['is_holiday']= 1

    return df


X_train.date = X_train.date.astype('datetime64[ns]')
X_test.date = X_test.date.astype('datetime64[ns]')


train_data_ft = time_features(X_train)
X_test = time_features(X_test)


train_data_ft = holiday_features(train_data_ft.copy())
X_test = holiday_features(X_test.copy())


import requests

def get_gdp_per_capita(alpha3, year):
    url='https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json'
    response = requests.get(url.format(alpha3,year)).json()
    return response[1][0]['value']

def add_gdp_column(df_in):
    df = df_in[['date','country']].copy()
    alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']
    df['alpha3'] = df['country'].map(dict(zip(
        np.sort(df['country'].unique()), alpha3s)))
    years = np.sort(df.date.dt.year.unique())
    df['year'] = df.date.dt.year
    gdp = np.array([
        [get_gdp_per_capita(alpha3, year) for year in years]
        for alpha3 in alpha3s
    ])
    gdp = pd.DataFrame(gdp/gdp.sum(axis=0), index=alpha3s, columns=years)
    df['GDP'] = df.apply(lambda s: gdp.loc[s['alpha3'], s['year']], axis=1)

    df_in['GDP'] = df['GDP']
    return df_in

train_data_ft = add_gdp_column(train_data_ft)
X_test = add_gdp_column(X_test)
    
# _, ax = plt.subplots(figsize=(8,10))
# decompose(train, 'country', ax)
# for country in df['country'].unique():
#     mask = df['country']==country
#     ax.plot(df[mask].index,df[mask]['GDP'],'k--')
# plt.show()






from sklearn.model_selection import train_test_split


# train_end = '2015-01-01'
# val_end = '2017-01-01'

# train_data = train_data_ft[train_data_ft['date'] < train_end]
# val_data = train_data_ft[(train_data_ft['date'] >= train_end) & (train_data_ft['date'] < val_end)]

# # train_data = train_data.groupby('date')['num_sold'].sum()
# # val_data = val_data.groupby('date')['num_sold'].sum()

# print(f"Train data: {train_data.shape}")
# print(f"Validation data: {val_data.shape}")

train_data_ft.dropna(inplace=True)
y = train_data_ft['num_sold']
train_data_ft = train_data_ft.drop('num_sold', axis=1)
train_data, val_data, y_train, y_val = train_test_split(train_data_ft, y, test_size=0.2, random_state=42)




train_data.dropna(inplace=True)
val_data.dropna(inplace=True)




# y_train = train_data['num_sold']
# y_val = val_data['num_sold']

# train_data = train_data.drop(['num_sold', 'id'], axis=1)
# val_data = val_data.drop(['num_sold', 'id'], axis=1)

# test_ids = X_test['id']
# X_test.drop(['id'], axis=1, inplace=True)


val_data


train_data.set_index('date', inplace=True)
val_data.set_index('date', inplace=True)

X_test.set_index('date', inplace=True)


# Save 'id' column for submission

# Define the target column
target_column = 'num_sold'

# Select categorical and numerical columns (initial)
categorical_columns = train_data.select_dtypes(include=['object']).columns
numerical_columns = train_data.select_dtypes(exclude=['object']).columns

# Print out column information
print("Target Column:", target_column)
print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


import optuna
from sklearn.metrics import mean_absolute_percentage_error as mape


from sklearn.preprocessing import OneHotEncoder

def one_hot_sklearn(df, categorical):
    encoder = OneHotEncoder(sparse=False, drop=None)  # sparse=False Ð¿Ð¾Ð²ÐµÑ€Ñ‚Ð°Ñ” dense-matrix
    
    encoded_data = encoder.fit_transform(df[categorical])
    
    encoded_df = pd.DataFrame(
        encoded_data, 
        columns=encoder.get_feature_names_out(categorical), 
        index=df.index
    )
    
    df = df.drop(columns=categorical)
    
    return pd.concat([df, encoded_df], axis=1)



columns=['country', 'product', 'store']
train_data_enc = one_hot_sklearn(train_data.copy(), columns)
val_data_enc = one_hot_sklearn(val_data.copy(), columns)
test_data_enc = one_hot_sklearn(X_test.copy(), columns)


y_val


y_train


from xgboost import XGBRegressor


train_data_enc.shape


y_train.shape


xgb_model = XGBRegressor(eval_metric='mape')
xgb_model.fit(train_data_enc, (y_train), eval_set=[(val_data_enc, (y_val))])


xgb_model.predict(val_data_enc)


y_val


mape(y_val, xgb_model.predict(val_data_enc))


from catboost import CatBoostRegressor


def objective_cat(trial):
    params = {
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 100, 300),
        'iterations': trial.suggest_int('iterations', 1000, 10000),
        'depth': trial.suggest_int('depth', 1, 16),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
    }

    cat_model = CatBoostRegressor(task_type = 'GPU', has_time=True,
        **params
    )

    cat_model.fit(train_data_enc, y_train, eval_set=(val_data_enc, y_val), verbose=0)

    val_pred = (cat_model.predict(val_data_enc))

    return mape(y_val, val_pred)

# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective_cat, n_trials=100)

# Best trial
trial = study.best_trial
print("Best Parameters:", trial.params)



cat_params = {'n_estimators': 10000,
    'learning_rate': 0.05, 
    'verbose': False, 
    'allow_writing_files': False}

cat_model = CatBoostRegressor(eval_metric='MAPE', task_type='GPU', **trial.params)
cat_model.fit(train_data_enc, (y_train), eval_set=[(val_data_enc, (y_val))])


pred = cat_model.predict(val_data_enc)


pred


mape(y_val, (pred))


y_val


np.exp(pred)





sns.lineplot(x=train_data.index, y=y_train, ci=False, label='Train')
sns.lineplot(x=val_data.index, y=y_val, ci=False, label='Val')
sns.lineplot(x=val_data.index, y=np.exp(cat_model.predict(val_data_enc)), ci=False, label='Predicted Val')
sns.lineplot(x=X_test.index, y=np.exp(cat_model.predict((test_data_enc))), ci=False, label='Test Predicted')
plt.legend()
plt.show()


feature_importance = cat_model.get_feature_importance()
feature_names = train_data_enc.columns

# Display feature importance
for name, importance in zip(feature_names, feature_importance):
    print(f"Feature: {name}, Importance: {importance:.2f}")


plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


submission = pd.DataFrame({'id': test_ids, 'num_sold': np.ceil(np.exp(cat_model.predict(test_data_enc)))})
submission.to_csv('cat_boost.csv', index=False)


submission




