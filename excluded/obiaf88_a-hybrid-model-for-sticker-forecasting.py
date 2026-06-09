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
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


train , test = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv') ,pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
# let's create a copy of test with id for submission
test_orig = test.copy()


train.drop(columns = 'id' , inplace = True)
test.drop(columns = 'id' , inplace = True)
train['date'] = pd.to_datetime(train['date'], format = '%Y-%m-%d')
test['date'] = pd.to_datetime(test['date'], format = '%Y-%m-%d')


print("Duplicated data :", train.duplicated().sum())
print("Null data : ", train.isnull().sum().sum())


#replacing missing values
train_for_imputing = train.groupby(['country','store','product'],as_index = False)['num_sold'].mean()


train_for_imputing.isnull().sum()


train_for_imputing[train_for_imputing['num_sold'].isnull()]


train_for_imputing.fillna(0,inplace = True)


train = train.merge(train_for_imputing, how = 'left', left_on = ['country','store','product'] ,  right_on = ['country','store','product'] )
train['num_sold'] = np.where(np.isnan(train['num_sold_x']), train['num_sold_y'],train['num_sold_x']) 
train.drop(['num_sold_x','num_sold_y'], axis = 1 , inplace = True)


train.isnull().sum()


class StickerPrediction(object):
    
    def __init__(self, train,test, country, store, product):
        self.train = train
        self.test = test
        self.country = country
        self.store = store
        self.product = product
        self.train_sub = self.train[(self.train['country'] == country) & (self.train['store'] == store) & (self.train['product'] == product)]
        self.test_sub = self.test[(self.test['country'] == country) & (self.test['store'] == store) & (self.test['product'] == product)]
        self.train_sub.drop(columns = ['country','store','product'], axis = 1 , inplace = True)
        self.train_sub.set_index('date', inplace = True)
        self.test_sub.set_index('date', inplace = True)
        self.train_sub = self.train_sub.asfreq('D')
        
        
    def predict(self):
        #days_to_predict
        days_to_predict = (self.test['date'].max() - self.test['date'].min()).days + 1

        # trend modelling
        dp_trend = DeterministicProcess(
            index = self.train_sub.index,
            constant=True,       
            order=3,             
            drop=True,           
            )
        X_trend = dp_trend.in_sample()
        y = self.train_sub["num_sold"]

        model_trend = LinearRegression(fit_intercept=False)
        _ = model_trend.fit(X_trend, y)
        y_trend_pred = pd.Series(model_trend.predict(X_trend), index=y.index)
        X_fore = dp_trend.out_of_sample(steps=days_to_predict)
        y_trend_fore = pd.Series(model_trend.predict(X_fore), index=X_fore.index)

        
        #seasonal_modelling
        y_detrend = y - y_trend_pred
        fourier = CalendarFourier(freq='A', order=6)  
        dp_detrend = DeterministicProcess(
            index = y_detrend.index,
            seasonal = True,               
            additional_terms=[fourier],  
            drop=True,                  
            )
        X_detrend = dp_detrend.in_sample()
        
        model_detrend = RandomForestRegressor(n_estimators = 100)
        _ = model_detrend.fit(X_detrend, y_detrend)

        y_detrend_pred = pd.Series(model_detrend.predict(X_detrend), index = y_detrend.index)
        X_detrend_fore = dp_detrend.out_of_sample(steps=days_to_predict)
        y_detrend_fore = pd.Series(model_detrend.predict(X_detrend_fore), index=X_detrend_fore.index)

    
        #residual modelling
        y_resid = y - y_trend_pred - y_detrend_pred
        model_residual = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, objective="reg:squarederror", random_state=42)
        fourier = CalendarFourier(freq='A', order=6)  # 10 sin/cos pairs for "A"nnual seasonality
        dp_resid = DeterministicProcess(
            index = y_resid.index,
            seasonal = True,               # weekly seasonality (indicators)
            additional_terms=[fourier],  # annual seasonality (fourier)
            drop=True,                   # drop terms to avoid collinearity
        )
        X_resid = dp_resid.in_sample()
        model_residual.fit(X_resid, y_resid)
        y_resid_pred = pd.Series(model_residual.predict(X_resid), index = y_resid.index)
        X_resid_fore = dp_resid.out_of_sample(steps=days_to_predict)
        y_resid_fore = pd.Series(model_residual.predict(X_resid_fore), index=X_resid_fore.index)
        y_fore_total = y_trend_fore +  y_detrend_fore + y_resid_fore
        y_fore_total_df = pd.DataFrame(y_fore_total, index = y_fore_total.index, columns = ['Predicted'])
        y_fore_total_df['country'] = self.country
        y_fore_total_df['store'] = self.store
        y_fore_total_df['product'] = self.product
        y_fore_total_df.reset_index(inplace = True)
        y_fore_total_df.rename(columns = {'index':'date'}, inplace = True)
        return y_fore_total_df


sticker_pred = StickerPrediction(train, test,country = 'Canada',store = 'Discount Stickers', product = 'Kaggle' )


df_predicted = sticker_pred.predict()


df_predicted.head()


country = 'Canada'
store = 'Discount Stickers'
product = 'Kaggle'


train_sub = train[(train['country'] == country) & (train['store'] == store) & (train['product'] == product)]


plt.plot(train_sub['date'],train_sub['num_sold'])
plt.plot(df_predicted['date'], df_predicted['Predicted'])
plt.title("Prediction for Canada - Discount Stikers - Kaggle")
;


train['check_combinations'] = train['country'] + train['store'] + train['product']
test['check_combinations'] = test['country'] + test['store'] + test['product']


set(train['check_combinations'].unique()).symmetric_difference(set(train['check_combinations'].unique()))


test.drop('check_combinations', axis = 1, inplace = True)
train.drop('check_combinations', axis = 1, inplace = True)


final_prediction = pd.DataFrame(columns = ['date','Predicted','country','store','product'] )


countries = list(test['country'].unique())
stores = list(test['store'].unique())
products = list(test['product'].unique())


for c in countries:
    for s in stores:
        for p in products:
            df_predicted = StickerPrediction(train, test,country = c,store = s, product = p).predict()
            final_prediction = pd.concat([final_prediction,df_predicted], axis = 0, ignore_index = True)


final_prediction.tail()


assert test.shape[0] == final_prediction.shape[0]


test_orig['date'] = pd.to_datetime(test_orig['date'], format = '%Y-%m-%d')
test_prediction = test_orig.merge(final_prediction, left_on = ['date','country','store','product'] , right_on = ['date','country','store','product'])


test_prediction.head()


submission = pd.DataFrame({
    'id': test_prediction['id'],
    'num_sold': test_prediction['Predicted']
})

submission.to_csv("submission.csv", index=False)



