!pip uninstall -y scikit-learn --quiet
!pip install scikit-learn==1.5.2 --quiet
!pip install ydata-profiling pyjanitor feature-engine --quiet


!pip install pmdarima --quiet


import numpy as np
import pandas as pd 
from pathlib import Path
from plotnine import *
from mizani.formatters import label_comma
import janitor
from ydata_profiling import ProfileReport
import warnings

warnings.filterwarnings('ignore')


def load_dataset(kaggle_in=Path('/kaggle/input/playground-series-s5e1')) -> dict:
    """Load datasets from playground-series"""
    return {
        n.name.strip('*.csv') : \
            pd.read_csv(n) for n in kaggle_in.iterdir() if n.suffix == '.csv'}

# load dataset and get the profiling of train and test.
dataset = load_dataset()
train_profile = ProfileReport(dataset['train'], title='Train')
test_profile = ProfileReport(dataset['test'], title='Test')


train_profile


# filter null values
nan_num_sold_df = dataset['train'][dataset['train']['num_sold'].isna()]


# number of entries that are null
nan_num_sold_df.shape[0]


# countries that are included in the null entries
nan_num_sold_df.country.unique().tolist()


# products that are included in the null features
nan_num_sold_df['product'].unique().tolist()


nan_num_sold_df


pd.crosstab(nan_num_sold_df['country'], nan_num_sold_df['store'], normalize=True) * 100


pd.crosstab(nan_num_sold_df['store'], nan_num_sold_df['product'], normalize=True) * 100


train_df = dataset['train'].dropna().reset_index(drop=True)


train_df = train_df.to_datetime('date', format='%Y-%m-%d')


ggplot(train_df)\
    + geom_line(aes('date', 'num_sold', color='product'))\
    + facet_wrap('~store', ncol=1)\
    + theme(legend_position='bottom', figure_size=(7, 7))\
    + labs(x='Date', y='Units Sold')\
    + scale_y_continuous(labels=label_comma())


ggplot(train_df)\
    + geom_line(aes('date', 'num_sold', color='product'))\
    + facet_wrap('~country', ncol=1)\
    + theme(legend_position='bottom', figure_size=(7, 14))\
    + labs(x='Date', y='Units Sold')\
    + scale_y_continuous(labels=label_comma())


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


# filter date and class variable
timeseries = train_df[['date', 'num_sold']].set_index('date')


# plot autocorrelation function (ACF)
plot_acf(timeseries, lags=1000) 
plt.title('Autocorrelation Function (ACF)')
plt.show()


# plot partial autocorrelation function (PACF)
plot_pacf(timeseries, lags=1000, method='ywm')  # 'ywm' is often used for PACF
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


from statsmodels.tsa.stattools import adfuller


def adf_test(ts: pd.Series) -> pd.DataFrame: 
    
    output = adfuller(ts)
    column_names = ['T', 'p-val', 'no. of lags', 'no. of obs']
    results = {k: v for k, v in zip(column_names, output)}
    results_df = pd.DataFrame.from_dict(results, orient='index').T
    results_df['is_stationary'] = results_df['p-val'] < 0.05
    
    return results_df


adf_results_df = adf_test(timeseries)


adf_results_df


from feature_engine.datetime import DatetimeFeatures, DatetimeSubtraction
from feature_engine.creation import CyclicalFeatures
from sklearn.preprocessing import OneHotEncoder
import pandas_flavor as pf


@pd.api.extensions.register_dataframe_accessor("ts_pipe")
class TimeSeriesPipe:

    def __init__(self, df):
        self.df = df
        self.date_features = self.make_date_features(df)

    @staticmethod
    def make_date_features(df):
        
        dtf = DatetimeFeatures(features_to_extract=['year', 'month', 'day_of_month'])
        date_features = dtf.fit_transform(df['date'].to_frame())
        return date_features
        
    def extract_datetime_features(self) -> pd.DataFrame:         
        output_df = pd.concat([self.df, self.date_features], axis=1)
        
        return output_df
    
    def transform_timeseries_cyclical(self) -> pd.DataFrame:
        
        cyclical = CyclicalFeatures(variables=None, drop_original=True)
        X = cyclical.fit_transform(self.date_features)
        output_df = pd.concat([self.df, X], axis=1)\
            .drop(['date_year', 'date_month', 'date_day_of_month', 'index'], axis=1)\
            .set_index('date')
        
        return output_df


@pf.register_dataframe_method
def encode_features(df) -> pd.DataFrame:
    
    ohc = OneHotEncoder(sparse_output=False)
    num_features = df.select_dtypes('number')
    num_col_names = num_features.columns 
    
    obj_encoded_features = ohc.fit_transform(df.reset_index().select_dtypes('object'))
    encoded_df = pd.DataFrame(obj_encoded_features, columns=ohc.get_feature_names_out())
    output_df = pd.concat([df[num_col_names].reset_index(drop=True), encoded_df], axis=1)

    return output_df


X, y = train_df.get_features_targets('num_sold')


# preprocess data
X = X\
    .drop('id', axis=1)\
    .reset_index()\
    .ts_pipe.extract_datetime_features()\
    .ts_pipe.transform_timeseries_cyclical()\
    .encode_features()


X


from statsmodels.tsa.arima.model import ARIMA


p, d, q = 1, 0, 1
mod = ARIMA(timeseries, order=(p, d, q))
mod_fit = mod.fit()
print(mod_fit.summary())


mod_fit.plot_diagnostics(figsize=(12, 8))


from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from feature_engine.timeseries.forecasting import WindowFeatures, LagFeatures
import janitor.timeseries


def get_model_score(model, input_x, input_y):

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        input_x, input_y, train_size=.8, random_state=25)

    # model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mape')

    # predict
    y_pred = model.predict(X_test)

    # score
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return mape


get_model_score(LGBMRegressor(), X, y)


get_model_score(XGBRegressor(), X, y)


train_wf = train_df\
    .sort_values(['date', 'country', 'store', 'product'])\
    .drop('id', axis=1)

unique_combination = train_wf[['country', 'store', 'product']].drop_duplicates()


def filter_timeseries(feature_model, df: pd.DataFrame, combinations: tuple) -> pd.DataFrame:
    c, s, p = combinations
    filtered_df = df[
        (df['country'] == c) & \
        (df['store'] == s) & \
        (df['product'] == p)]
    output_dataframe = feature_model.fit_transform(filtered_df)
    return output_dataframe


window_features_df = pd.concat([filter_timeseries(
    feature_model=WindowFeatures(window=2),
    df=train_wf, 
    combinations=unique_combination.iloc[i]
    ) for i in range(unique_combination.shape[0])]
) 


train_new_feat = janitor.timeseries.sort_timestamps_monotonically(window_features_df, direction='increasing')
train_new_feat_processed = window_features_df\
    .reset_index()\
    .ts_pipe.extract_datetime_features()\
    .ts_pipe.transform_timeseries_cyclical()\
    .encode_features()


X_fe, y_fe = train_new_feat_processed.get_features_targets('num_sold')


get_model_score(LGBMRegressor(), X_fe, y_fe)


get_model_score(XGBRegressor(), X_fe, y_fe)


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


train_new_feat_processed.info()


from hyperopt.pyll.base import scope


space = {'max_depth': scope.int(hp.quniform("max_depth", 1, 5, 1)),
        'gamma': hp.uniform ('gamma', 0,1),
        'reg_alpha' : hp.uniform('reg_alpha', 0,50),
        'reg_lambda' : hp.uniform('reg_lambda', 10,100),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0,1),
        'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
        'n_estimators': 10000,
        'learning_rate': hp.uniform('learning_rate', 0, .15),
        'tree_method':'gpu_hist', 
        'gpu_id': 0,
        'random_state': 5,
        'max_bin' : scope.int(hp.quniform('max_bin', 200, 550, 1))}


X_train, X_test, y_train, y_test = train_test_split(X_fe, y_fe, train_size=.8, random_state=25)


def hyperparameter_tuning(space):
    model = XGBRegressor(**space)
    evaluation = [(X_train, y_train), (X_test, y_test)]
    
    model.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=100,verbose=False)

    pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, pred)
    print ("SCORE:", mape)
    return {'loss':mape, 'status': STATUS_OK, 'model': model}


trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials)

print(best)

