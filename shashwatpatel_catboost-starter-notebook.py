!pip install category-encoders


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from catboost import CatBoostRegressor
import random
import category_encoders as ce
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor



train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


train.head()


# Check for Null values
print("Training Data:")
print(train.isnull().sum())
print("-------------")
print('Test Data:')
print(test.isnull().sum())


# Since num_sold is predictor variable, we drop missing values from our training data
train.dropna(inplace=True)


## Other Preprocessing
train.drop(['id'], inplace=True, axis=1)
train['date'] = pd.to_datetime(train['date'])

test_ids = test['id']
test.drop(['id'], inplace=True, axis=1)
test['date'] = pd.to_datetime(test['date'])


train.shape


# Check cardinality of categorical variables in train and test
cat_var = [var for var in train.columns if train[var].dtype=='O']

for col in cat_var:
    print('Training Data')
    print(f"{col} unique_values:{train[col].unique()}")
    print('Test Data')
    print(f"{col} unique_values:{test[col].unique()}")
    print('\n')

    


# Feature Engineering from 'date' column
def feature_engg(data):
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['day_of_week'] = data['date'].dt.dayofweek
    data['quarter'] = data['date'].dt.quarter
    data['is_month_start'] = data['date'].dt.is_month_start
    data['is_month_end'] = data['date'].dt.is_month_end
    data['is_year_start'] = data['date'].dt.is_year_start
    data['is_year_end'] = data['date'].dt.is_year_end
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x>=5 else 0)
    data['day_sin'] = np.sin(data.day*(2.*np.pi/30))
    data['day_cos'] = np.cos(data.day*(2.*np.pi/30))

    return data
    


train = feature_engg(train)
test = feature_engg(test)


test


## Add holiday data 
holiday = pd.read_csv("/kaggle/input/holiday-list/public_holidays_2010_2019.csv")

# We only consider public holidays
holiday = holiday[holiday['Type']=='Public holiday']

# Six countries: Canada, Finland, Italy, Kenya, Norway, Singapore
holiday = holiday[holiday['ADM_name'].isin(['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore'])]

holiday['Date'] = pd.to_datetime(holiday['Date'], format='%d/%m/%Y')
holiday.drop(['ISO3','Type'],inplace=True, axis = 1)

holiday.columns = ['country', 'date', 'name']
holiday['country_date'] = holiday['country'].astype(str) + '_' + holiday['date'].dt.strftime('%Y-%m-%d')
holiday = holiday.drop_duplicates(subset='country_date')
holiday.drop(['country_date'],axis=1, inplace=True)


holiday


# GDP per Capita
gdp = pd.read_csv("/kaggle/input/gdp-per-capita-2010-2019/gdp_per_capita.csv")
gdp = gdp[gdp['Country Name'].isin(['Canada', 'Finland', 'Italy', 'Norway', 'Kenya', 'Singapore'])]
gdp.drop(['Indicator Name'], axis=1, inplace =True)
gdp = gdp.melt(id_vars="Country Name", var_name="Year", value_name="GDP_per_capita")


gdp.columns = ['country', 'year', 'gdp_per_capita']
gdp['year'] = gdp['year'].astype('int32')


train=pd.merge(train,holiday,how='left')
test=pd.merge(test,holiday,how='left')

train = pd.merge(train,gdp, how='left')
test = pd.merge(test, gdp, how='left')


train['is_holiday'] = train['name'].notna()
test['is_holiday'] = test['name'].notna()




# EDA
plt.hist((np.log(train['num_sold'])))
plt.xlabel('Log of Amount Sold')
plt.ylabel('Frequency')


sns.barplot(train, x = 'country', y='num_sold',hue='store' )


sns.barplot(train, x = 'country', y='num_sold',hue='product' )


sns.barplot(train, x = 'country', y='num_sold',hue='is_holiday' )


sns.barplot(train, x = 'country', y='num_sold',hue='is_weekend' )


sns.barplot(train, x = 'country', y='num_sold',hue='quarter' )


sns.barplot(train, x = 'country', y='num_sold',hue='day_of_week' )


sns.barplot(train, x = 'country', y='num_sold',hue='is_month_start' )


sns.barplot(train, x = 'country', y='num_sold',hue='is_month_end' )


sns.barplot(train, x = 'country', y='num_sold',hue='is_year_start' )


sns.barplot(train, x = 'country', y='num_sold',hue='is_year_end' )


sns.barplot(train, x = 'country', y='num_sold',hue='month' )


train.drop(['name','date'],axis=1,inplace=True)
test.drop(['name', 'date'],axis=1, inplace=True)
y_train = train['num_sold']
x_train = train.drop(['num_sold'], axis=1)

cat_features= ['country', 'store', 'product', 'month',
       'day', 'day_of_week', 'quarter', 'is_month_start', 'is_month_end',
       'is_year_start', 'is_year_end', 'is_weekend', 'is_holiday']

encoder = ce.TargetEncoder(cols=cat_features)
X_encoder = encoder.fit_transform(x_train, y_train)
test_enc = encoder.transform(test)


def mape_objective(y_true, y_pred):
    """
    Custom objective function for LightGBM to minimize MAPE.
    Returns gradients and Hessians.
    """
    residual = (y_pred - y_true) / y_true
    grad = np.sign(residual) / np.abs(y_true)  # Gradient
    hess = 1 / (y_true**2)  # Hessian
    return grad, hess

# Custom MAPE Metric
def mape_metric(y_true, y_pred):
    """
    Custom evaluation metric for LightGBM to report MAPE.
    """
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return "MAPE", mape, False  # "False" indicates lower is better

import lightgbm as lgb

train_data = lgb.Dataset(X_encoder, label=y_train)
test_data = lgb.Dataset(test_enc)
# LightGBM Parameters
params = {
    "objective": "regression",  # Placeholder, overridden by custom loss
    "metric": "None",  # No default metric; we define MAPE
    "learning_rate": 0.116,
    "max_depth": 12,
    "seed": 42,
    "verbose": -1,
}
'''
# Train the LightGBM model
print("Training LightGBM with MAPE...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=450,
    fobj=mape_objective,  # Custom objective function
    feval=mape_metric,    # Custom evaluation metric
)
'''


# Catboost model with default parameters
random.seed(42)
cat_features= ['country', 'store', 'product', 'month',
       'day', 'day_of_week', 'quarter', 'is_month_start', 'is_month_end',
       'is_year_start', 'is_year_end', 'is_weekend', 'is_holiday']

cb_regressor=CatBoostRegressor(n_estimators  = 450, max_depth=12, learning_rate=0.115,loss_function='MAPE')

cb_regressor.fit(X_encoder,y_train,verbose=100)

#lgbm_regressor = LGBMRegressor(n_estimators=450, max_depth=10, learning_rate=0.12, loss_function='MAPE')
#lgbm_regressor.fit(X_encoder,y_train)


pred = cb_regressor.predict(test_enc)
#pred = lgbm_regressor.predict(test)


pred


df=pd.DataFrame({'id':test_ids,'num_sold':pred})
df.to_csv('submission.csv',index=False)

