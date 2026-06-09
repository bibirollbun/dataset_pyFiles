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


import pandas as pd
import numpy as np
import seaborn as sns
import holidays
import requests 
import h2o
from h2o.frame import H2OFrame
from h2o.automl import H2OAutoML
from h2o.automl import get_leaderboard
from h2o.estimators import H2OXGBoostEstimator, H2OGradientBoostingEstimator, H2OStackedEnsembleEstimator

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

train.drop(columns=['id'],inplace=True)
test.drop(columns=['id'],inplace=True)



def get_gdp_per_capita(country, year):
    alpha3 = {
        'Canada': 'CAN', 'Finland': 'FIN', 'Italy': 'ITA',
        'Kenya': 'KEN', 'Norway': 'NOR', 'Singapore': 'SGP'
    }
    url = f"https://api.worldbank.org/v2/country/{alpha3[country]}/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
    response = requests.get(url).json()
    try:
        return response[1][0]['value']
    except (IndexError, TypeError):
        return None

countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
years = range(2010, 2020)
gdp_data = {}

for country in countries:
    for year in years:
        gdp_data[(country, year)] = get_gdp_per_capita(country, year)

def add_gdp_feature(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year  
    df['gdp'] = df.apply(lambda row: gdp_data.get((row['country'], row['year']), None), axis=1)
    return df

train = add_gdp_feature(train)
test = add_gdp_feature(test)


combined = pd.concat([train[['year', 'gdp']], test[['year', 'gdp']]])
gdp_total = combined.groupby('year')['gdp'].sum().reset_index().rename(columns={'gdp': 'total_gdp'})

train = train.merge(gdp_total, on='year', how='left')
train['gdp_ratio'] = train['gdp'] / train['total_gdp']

test = test.merge(gdp_total, on='year', how='left')
test['gdp_ratio'] = test['gdp'] / test['total_gdp']

global_product_ratio = train.groupby('product')['num_sold'].mean() / train['num_sold'].mean()
global_store_ratio = train.groupby('store')['num_sold'].mean() / train['num_sold'].mean()
global_country_ratio = train.groupby('country')['num_sold'].mean() / train['num_sold'].mean()


overall_product_ratio = global_product_ratio.mean()
overall_store_ratio = global_store_ratio.mean()
overall_country_ratio = global_country_ratio.mean()

for df in [train, test]:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['month_country'] = df['month'].astype(str) + "_" + df['country']
    df['month_store'] = df['month'].astype(str) + "_" + df['store']
    df['month_product'] = df['month'].astype(str) + "_" + df['product']
    df['year_normalized'] = df['year'] - 2010
    
    df['product_ratio'] = df['product'].map(global_product_ratio)
    df['store_ratio'] = df['store'].map(global_store_ratio)
    df['country_ratio'] = df['country'].map(global_country_ratio)
    
    df['product_ratio'].fillna(overall_product_ratio, inplace=True)
    df['store_ratio'].fillna(overall_store_ratio, inplace=True)
    df['country_ratio'].fillna(overall_country_ratio, inplace=True)


train = train.drop(columns=['date'], axis=1)
test = test.drop(columns=['date'], axis=1)

train = train.dropna(subset=['num_sold'])


train.head()


train['num_sold'] = np.log1p(train['num_sold'])


train.head()


h2o.init(max_mem_size='16G')


train_h2o = h2o.H2OFrame(train)


test_h2o = h2o.H2OFrame(test)


x = train_h2o .columns
y = 'num_sold'
x.remove(y)


gbm_model = H2OGradientBoostingEstimator(
    distribution="auto",
    ntrees=5000,
    nfolds=5,
    stopping_rounds=10,
    stopping_metric="MSE", 
    stopping_tolerance=0.001,  
    ignore_const_cols=False,
    keep_cross_validation_predictions=True,
    fold_assignment="Modulo"
)

gbm_model.train(x=x, y=y, training_frame=train_h2o, model_id="gbm_model")



xgb_model = H2OXGBoostEstimator(
    distribution="auto",           
    ntrees=5000,                   
    nfolds=5,                      
    stopping_rounds=10,           
    stopping_metric="MSE",         
    stopping_tolerance=0.001,      
    keep_cross_validation_predictions=True,  
    fold_assignment="Modulo"
         
)

xgb_model.train(x=x, y=y, training_frame=train_h2o, model_id="xgboost_model")



stack_ensemble = H2OStackedEnsembleEstimator(
    base_models=["xgboost_model", "gbm_model"]
)
stack_ensemble.train(x=x, y=y, training_frame=train_h2o, model_id="stack_ensemble")
stack_ensemble.show()



predictions = stack_ensemble.predict(train_h2o) 

predicted_values = predictions["predict"].as_data_frame().values.flatten()
actual_values = train_h2o[y].as_data_frame().values.flatten()
mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

print(f"MAPE: {mape:.4f}%")


predictions = stack_ensemble.predict(test_h2o)


predictions_df = predictions.as_data_frame()
predictions_df['predict'] = np.expm1(predictions_df['predict'])  


sub = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
sub['num_sold'] = predictions_df['predict'].values
sub.to_csv('submission.csv', index=False)
sub.head()


h2o.cluster().shutdown()

