!pip install --upgrade --quiet prophet


import prophet
prophet.__version__


import pandas as pd
import os
import numpy as np
from datetime import datetime

comprep = './'
datarep = '/kaggle/input/playground-series-s5e1'

trainfile = 'train.csv'
testfile = 'test.csv'
train_df = pd.read_csv(os.path.join(datarep,trainfile)) #.fillna(0)
test_df = pd.read_csv(os.path.join(datarep,testfile))

data_df = pd.concat((train_df,test_df),axis=0)

for df in [train_df,test_df]:
    df['date'] = pd.to_datetime(df['date']).dt.date

# To plot
figsize = (12,5)

first_train_date = np.min(train_df['date'])
last_train_date = np.max(train_df['date'])
print(f"Train set {train_df.shape} from {first_train_date} to {last_train_date} ")

first_pred_date = np.min(test_df['date'])
last_pred_date = np.max(test_df['date'])
print(f"Test  set {test_df.shape} from {first_pred_date} to {last_pred_date} ")

print('Colums: ',test_df.columns)

days_to_predict = last_pred_date - first_pred_date
days_to_predict = days_to_predict.days+1
print('Number of days to predict: ',days_to_predict,'days between',first_pred_date,'and',last_pred_date)

del train_df
del test_df


metric = 'mape'


countries = data_df.country.unique()
stores = data_df.store.unique()
products = data_df['product'].unique()

print('List of countries:',countries)
print('List of stores:',stores)
print('List of products',products)

print('Nb of cases:',len(countries)*len(stores)*len(products))


prophet_country_names = {
        'Canada' : 'CA', 
    'Finland' : 'FI', 
    'Italy' : 'IT', 
    'Kenya' : 'KE', 
    'Norway' : 'NO',
    'Singapore': 'SG'
}


gdpfile = '/kaggle/input/gdp-per-capita-world-bank/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5607100.csv'
gdp_df = pd.read_csv(gdpfile,header=2)

col_years = [str(year) for year in range(2009,2021) ]

gdp_df = gdp_df[['Country Name',] + col_years]

gdp_df = gdp_df[gdp_df['Country Name'].isin(countries)]
gdp_df


import pandas as pd
import numpy as np

# Converting year columns to float for interpolation
gdp_melted_df = gdp_df.melt(id_vars=['Country Name'], var_name='Year', value_name='GDP')
gdp_melted_df['Year'] = pd.to_datetime(gdp_melted_df['Year'], format='%Y') + pd.DateOffset(days=182)  # Mid-year

# Creating a daily range for interpolation
start_date = gdp_melted_df['Year'].min()
end_date = gdp_melted_df['Year'].max()
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Creating the final dataframe with the interpolated values
result = []
for country in gdp_df['Country Name']:
    country_data = gdp_melted_df[gdp_melted_df['Country Name'] == country].set_index('Year').reindex(all_dates)
    country_data['Country Name'] = country
    country_data['GDP'] = country_data['GDP'].interpolate()  # Linear interpolation
    country_data.reset_index(inplace=True)
    country_data.rename(columns={'index': 'Date'}, inplace=True)
    result.append(country_data)

# Concatenation of results for each country
gdp_day_df = pd.concat(result, ignore_index=True)
gdp_day_df.columns = ['date','country','gdp']

gdp_day_df



# Adding gdp column in dataset
data_df['date'] = pd.to_datetime(data_df['date'])
data_df = pd.merge(data_df, gdp_day_df, on=['country', 'date'], how='inner')
data_df


def getSerie(df,country,store,product,last_train_date=last_train_date):
    df1 = df.copy()
    if country is not None:
        df1 = df1[df1['country']==country]
        del df1['country']
    if store is not None:
        df1 = df1[df1['store']==store]
        del df1['store']
    if product is not None:
        df1 = df1[df1['product']==product]
        del df1['product']
    del df1['id']
    df1.rename(columns={"date": "ds", "num_sold": "y"},inplace=True)

    df1['ds'] = pd.to_datetime(df1['ds'])

    # Missing
    if df1['y'].isna().all():
        df1.loc[df1.ds.dt.date <= last_train_date,'y'] = 0
    else:
        pass
        #df1.loc[df1.ds.dt.date <= last_train_date,'y'] = df1.loc[df1.ds.dt.date <= last_train_date,'y'].interpolate(method='linear',axis=0)

    return df1

# To check with only Nan for y
getSerie(data_df,country = 'Canada', store = 'Discount Stickers', product = 'Holographic Goose')





import matplotlib.pyplot as plt

def plot_serie(tserie,title):

    fig, ax1 = plt.subplots(figsize=(10, 6))  

    # target axis
    ax1.plot(tserie['ds'], tserie['y'], color='b', label='num_sold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('num_sold', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # gdp axis
    ax2 = ax1.twinx()  # Crée un second axe y partagé
    ax2.plot(tserie['ds'], tserie['gdp'], color='r', label='GDP')
    ax2.set_ylabel('GDP', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(title)
    fig.tight_layout()  
plt.show()



store = stores[1]
product = products[2]
print('Num_sold and gdp correlation per country for {store} and {product}:')
for country in countries:
    tserie = getSerie(data_df,country,store,product)
    print(country,'r = ',tserie[['y', 'gdp']].corr()['y']['gdp'])


store = stores[1]
product = products[2]


for country in countries:
    tserie = getSerie(data_df,country,store,product)
    plot_serie(tserie, title=country)


country = countries[3]
store = stores[1]
product = products[2]

tserie = getSerie(data_df,country,store,product)
print('Country:',country)

# Known data
tserie[:-days_to_predict]


# Unknown data
tserie[-days_to_predict:]


from prophet import Prophet
import logging

# Only critical message for the 'cmdstanpy' are logged
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

m = Prophet(growth='linear') #,changepoint_prior_scale=0.5)

# Adding holiday
m.add_country_holidays(country_name=prophet_country_names[country])

# Adding the covariate
m.add_regressor('gdp',)

_ = m.fit(tserie[:-days_to_predict])

future = m.make_future_dataframe(
    periods = days_to_predict, # Range for the prediction.
    freq = 'D', # Frequence. Here: a prediction every day
    )


# make_future_dataframe generates a dataframe with ds column only.
# We need to add 'gdp' column to future dataframe
# in order to use it as a regressor
future['gdp'] = tserie['gdp'].values

forecast = m.predict(future)
_=m.plot_components(forecast,figsize=(12,10) )


# Prediction end of 2019
forecast.tail(5)


# See: https://github.com/facebook/prophet/blob/main/python/prophet/plot.py
print(country)
_ = m.plot(forecast, figsize=figsize,include_legend=True)


from prophet.diagnostics import cross_validation, performance_metrics

cutoffs = pd.to_datetime(['2015-12-31'])


df_cv = cross_validation(m, cutoffs=cutoffs,  
                         horizon = '365 days',  # forecast horizon
                         disable_tqdm=True
                        )
df_cv


df_p = performance_metrics(df_cv, metrics=[metric], rolling_window=1)
df_p


from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric=metric)


from prophet import Prophet
import tqdm

pred_df = pd.DataFrame()

cpf =[]
for country in countries:
    for store in stores:
        for product in products:
            cpf.append([country,store,product])

for country, store, product in tqdm.tqdm(cpf):
    tserie = getSerie(data_df,country,store,product)
    m = Prophet()
    m.add_country_holidays(country_name=prophet_country_names[country])
    
    # Adding the covariate
    m.add_regressor('gdp')        
    m.fit(tserie[:-days_to_predict])
    
    future = m.make_future_dataframe(periods = days_to_predict, freq = 'D')
    future['gdp'] = tserie['gdp'].values
    
    forecast = m.predict(future)
         
    forecast['country']=country
    forecast['store']=store
    forecast['product']=product
            
    forecast.rename(columns={"ds": "date", "yhat": "num_sold"},inplace=True)
    pred_df = pd.concat((pred_df,forecast[['date','country','store','product','num_sold']]))
            
    del m
    del forecast
    del tserie
            
pred_df


test_df = pd.read_csv(os.path.join(datarep,testfile))
test_df['date'] = pd.to_datetime(test_df['date'] )  # For merge        
    
# we combine the forecast with the test set
test_df = pd.merge(left = test_df, right = pred_df, on = ['date','country','store','product'], how = 'left')
test_df


test_df[['id','num_sold']].to_csv('submission.csv',index=False)


import itertools

param_grid = {  
    # tuning those parameters can potentially improve the performance of our model

    # changepoint_prior_scale:
    # 0.01 --> The trend will be smoothed. Model will favor a general trend without reacting strongly to sudden changes.
    # 0.05 --> Default
    # 0.10 --> Model will become more flexible, it will be able to capture important trend changes
    'changepoint_prior_scale': [  0.01, 0.05, 0.1 ], 
    
    'seasonality_prior_scale': [ 1.0, 10],
    'holidays_prior_scale': [ 1.0, 10],

    # growth:
    # flat --> If your time series does not exhibit a strong long-term trend (e.g. fluctuations around a constant mean) 
    # or if focus on regressors: If you believe that the dynamics of the series depends mainly on regressors or seasonality
    # linear --> Default
    'growth' : ['linear','flat',],

    # gdp_mode:
    # None --> No GDP regressor, default
    # multiplicative or additive
    'gdp_mode' : [ None,'multiplicative','additive',], 

    # seasonality_mode:
    # additive --> Default
    # multiplicative 
    'seasonality_mode': ['additive', 'multiplicative',], 
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

# Quick peek at what our combinations look like
print('Number of combinations:',len(all_params))
#all_params


cpf = []
config = {}
for country in countries:
    config[country] = {}
    for store in stores:
        config[country][store] = {}
        for product in products:
            config[country][store][product] = {}
            cpf.append([country,store,product])

for country, store, product in tqdm.tqdm(cpf):
    best_mapes = 10_000
    for params in all_params:
        tserie = getSerie(data_df,country,store,product)

        #prophet_config = { k:v for k,v in params.items() 
        m = Prophet(
            seasonality_mode = params['seasonality_mode'],
            growth = params['growth'],
            changepoint_prior_scale = params['changepoint_prior_scale']
            )
        # Holiday
        m.add_country_holidays(country_name=prophet_country_names[country] )
        # Adding the covariate
        if params['gdp_mode'] is not None:
            m.add_regressor('gdp',mode = params['gdp_mode'])
        m.fit(tserie[:-days_to_predict])
    
        df_cv = cross_validation(m, cutoffs = cutoffs, horizon='365 days',disable_tqdm=True)
        df_p = performance_metrics(df_cv, metrics=[metric],rolling_window=1)
        if df_p is None:
            mapes = 0
            config[country][store][product]['prophet'] = params
            continue

        if df_p[metric].values[0] < best_mapes:
            best_mapes = df_p[metric].values[0]
            config[country][store][product]['prophet'] = params
            
    config[country][store][product]['mape'] = best_mapes
    
          


for country in countries:
    print(f'----- {country}-----')
    for k,v in config[country].items():
        print(f'{k}: {v}')



import json

with open('prophet_config.json', 'w') as outfile:
    json.dump(config, outfile)


from prophet import Prophet
import tqdm

pred_df = pd.DataFrame()

cpf =[]
for country in countries:
    for store in stores:
        for product in products:
            cpf.append([country,store,product])

for country, store, product in tqdm.tqdm(cpf):
    tserie = getSerie(data_df,country,store,product)

    
    m = Prophet(
        seasonality_mode = config[country][store][product]['prophet']['seasonality_mode'],
        growth=config[country][store][product]['prophet']['growth'],
        changepoint_prior_scale = config[country][store][product]['prophet']['changepoint_prior_scale']
               )
    
    m.add_country_holidays(country_name=prophet_country_names[country])

    # Adding the covariate
    if config[country][store][product]['prophet']['gdp_mode'] is not None:
        m.add_regressor('gdp',mode = config[country][store][product]['prophet']['gdp_mode'])
       
    m.fit(tserie[:-days_to_predict])
    
    future = m.make_future_dataframe(periods = days_to_predict, freq = 'D')
    if config[country][store][product]['prophet']['gdp_mode'] is not None:
        future['gdp'] = tserie['gdp'].values
    
    forecast = m.predict(future)
         
    forecast['country']=country
    forecast['store']=store
    forecast['product']=product
            
    forecast.rename(columns={"ds": "date", "yhat": "num_sold"},inplace=True)
    pred_df = pd.concat((pred_df,forecast[['date','country','store','product','num_sold']]))
            
    del m
    del forecast
    del tserie
            
pred_df


 config[country][store]


test_df = pd.read_csv(os.path.join(datarep,testfile))
test_df['date'] = pd.to_datetime(test_df['date'] )  # For merge        
    
# we combine the forecast with the test set
test_df = pd.merge(left = test_df, right = pred_df, on = ['date','country','store','product'], how = 'left')
test_df


test_df[['id','num_sold']].to_csv('submission_after_tunning.csv',index=False)

