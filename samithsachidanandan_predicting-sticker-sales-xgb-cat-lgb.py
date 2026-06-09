import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import requests
import warnings



from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor, XGBRFRegressor, DMatrix
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping

import sklearn
sklearn.set_config(transform_output='pandas')
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor



import category_encoders as ce


from statsmodels.graphics.tsaplots import plot_acf

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=24, titlepad=10,
       titlecolor='black'
      )

warnings.filterwarnings('ignore')


train = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")
sample_submission = pd.read_csv(r"/kaggle/input/playground-series-s5e1/sample_submission.csv")


print("train_data shape :",train.shape)
print("test_data shape :",test.shape)
print("sample_submission shape :",sample_submission.shape)


train.head()


test.head()


train.isna().sum().sort_values()


test.isna().sum().sort_values()


train.info()


train['date'] = pd.to_datetime(train['date'], format='%Y-%m-%d')
test['date'] = pd.to_datetime(test['date'], format='%Y-%m-%d')


train[train['num_sold'].isna()][['country', 'store', 'product']].value_counts()


train.describe()


train['country'].value_counts()


train = train.dropna().reset_index(drop=True)


train.duplicated().sum()


plt.figure(figsize=(28,6))
train.groupby('date')['num_sold'].sum().plot(xlabel='Date', 
                                             ylabel='Number of Products Sold', 
                                             title='Total Sales Over Time')
plt.grid()
plt.show()


plt.figure(figsize=(28, 6))
sns.barplot(x=train['date'].dt.year, y=train['num_sold'], hue=train['country'], estimator='sum', palette='deep')
plt.title('Sales Trends by Country (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Country')
plt.grid()
plt.show()


plt.figure(figsize=(28, 6))
sns.barplot(x=train['date'].dt.year, y=train['num_sold'], hue=train['store'], estimator='sum')
plt.title('Sales Trends by Store-Type (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Store Type')
plt.grid()
plt.show()


plt.figure(figsize=(28, 6))
sns.barplot(x=train['date'].dt.year, y=train['num_sold'], hue=train['product'], estimator='sum', palette='deep')
plt.title('Sales Trends by Product (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Products')
plt.grid()
plt.show()


class HolidayGDPProcessor:
    def __init__(self):
        self.country_holidays = {
            'Canada': holidays.country_holidays('CA'),
            'Finland': holidays.country_holidays('FI'),
            'Italy': holidays.country_holidays('IT'),
            'Kenya': holidays.country_holidays('KE'),
            'Norway': holidays.country_holidays('NO'),
            'Singapore': holidays.country_holidays('SG')
        }
        self.alpha3 = {
            'Canada': 'CAN', 'Finland': 'FIN', 'Italy': 'ITA',
            'Kenya': 'KEN', 'Norway': 'NOR', 'Singapore': 'SGP'
        }
        self.gdp_data = self._fetch_gdp_data()

    def _set_holiday(self, row):
        if row['country'] in self.country_holidays and row['date'] in self.country_holidays[row['country']]:
            return 0  # Holiday
        return 1  # Not a holiday

    def _fetch_gdp_data(self):      
        countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
        years = range(2010, 2020)
        gdp_data = {}

        for country in countries:
            for year in years:
                url = f"https://api.worldbank.org/v2/country/{self.alpha3[country]}/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
                response = requests.get(url).json()
                try:
                    gdp_data[(country, year)] = response[1][0]['value']
                except (IndexError, TypeError):
                    gdp_data[(country, year)] = None
        return gdp_data

    def add_gdp_feature(self, df): 
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['gdp'] = df.apply(lambda row: self.gdp_data.get((row['country'], row['year']), None), axis=1)
        return df

    def process(self, train, test):
        # Add GDP feature
        train = self.add_gdp_feature(train)
        test = self.add_gdp_feature(test)
        
        # Apply holiday feature
        train['holiday'] = train.apply(self._set_holiday, axis=1)
        test['holiday'] = test.apply(self._set_holiday, axis=1)
        
        return train, test



holiday_gdp_processor = HolidayGDPProcessor()

train, test = holiday_gdp_processor.process(train, test)


train.head()


def model_trainer(model, X, y, test, n_splits=5, random_state=42, verbose=0, model_name=None):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print("="*80)
    model_name_ = model[-1].__class__.__name__ if isinstance(model, Pipeline) else model.__class__.__name__
    print(f"Model: {model_name_}")
    print("="*80 + '\n')

    oof_mape = []
    oof_test_preds = np.zeros(len(test))
    oof_train_preds = np.zeros(len(y))
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        if model_name == 'xgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=verbose)
            booster = model.get_booster()
            
            y_pred = booster.predict(DMatrix(X_valid), iteration_range=(0, model.best_iteration+1))
            test_pred = booster.predict(DMatrix(test), iteration_range=(0, model.best_iteration+1))
            oof_train_preds[train_idx] = booster.predict(DMatrix(X_train), iteration_range=(0, model.best_iteration+1))

        elif model_name == 'cat':
            trainPool = Pool(X_train ,y_train)
            testPool = Pool(test)
            validPool = Pool(X_valid, y_valid)

            model.fit(X=trainPool, eval_set=validPool, verbose=verbose, early_stopping_rounds=200)
            y_pred = model.predict(validPool)
            test_pred = model.predict(testPool)
            oof_train_preds[train_idx] = model.predict(Pool(X_train))

        elif model_name == 'lgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[early_stopping(200, verbose=0)])
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
            test_pred = model.predict(test, num_iteration=model.best_iteration_)
            oof_train_preds[train_idx] = model.predict(X_train, num_iteration=model.best_iteration_)

        
        
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            test_pred = model.predict(test)
            oof_train_preds[train_idx] = model.predict(X_train)

        oof_test_preds += test_pred
        mape = mean_absolute_percentage_error(np.expm1(y_valid), np.expm1(y_pred))
        print(f"Fold {fold+1} --> MAPE: {mape:.4f}")
        oof_mape.append(mape)
    
    print()
    print(f"Average Fold MAPE: {np.mean(oof_mape):.4f} \xb1 {np.std(oof_mape):.4f}")
    return oof_test_preds/n_splits, oof_train_preds


class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame()
        df['day'] = X.date.dt.day
        df['month'] = X.date.dt.month
        df['year'] = X.date.dt.year
        df['quarter'] = X.date.dt.quarter
        df['sine_day'] = np.sin(2 * np.pi * df['day'] / 31)
        df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
        df['sine_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sine_year'] = np.sin(2 * np.pi * df['year']/7.0)
        df['cos_year'] = np.cos(2 * np.pi * df['year']/7.0)
        df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    
        return df


preprocessing = ColumnTransformer([
    ('categorical', 
     OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     ['country', 'store', 'product']),
    ('date', DateTransformer(),['date'])
], remainder='drop')


target = 'num_sold'


train.head()


test.head()


X.head()


X = train.copy()
y = np.log1p(X.pop(target))

X = preprocessing.fit_transform(X)
test = preprocessing.transform(test)


test_preds, train_preds = pd.DataFrame(), pd.DataFrame()


xgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.00990161328639894,
    'max_depth': 17,
    'min_child_weight': 58,
    'subsample': 0.7373527286687829,
    'colsample_bytree': 0.4544157822113165,
    'gamma': 0.0019767061497068528,
    'reg_alpha': 0.7647218923252306,
    'device': 'cuda',
    'tree_method': 'hist',
    'random_state': 0,
    'early_stopping_rounds': 200
}

xgb_reg = XGBRegressor(**xgb_params)

test_preds['xgb'], train_preds['xgb'] = model_trainer(xgb_reg, X, y, 
                                                      test, 
                                                      random_state=0, verbose=0, model_name='xgb')


cat_params = {
    'n_estimators': 10000,
    'learning_rate': 0.05, 
    'task_type': 'GPU', 
    'verbose': False, 
    'allow_writing_files': False,
}

cat_reg = CatBoostRegressor(**cat_params)

test_preds['cat'], train_preds['cat'] = model_trainer(
    cat_reg,
    X, y, test, random_state=0, model_name='cat'
)


lgb_params = {'n_estimators': 3946, 'learning_rate': 0.10203344298643195, 'max_depth': 20, 'num_leaves': 32, 'min_child_samples': 60, 'subsample': 0.7786665459484634, 'colsample_bytree': 0.7352055562065795, 'reg_alpha': 0.2840216195298897, 'reg_lambda': 6.583320975256993, "verbosity" : -1}

lgb_reg = LGBMRegressor(**lgb_params)
test_preds['lgb'], train_preds['lgb'] = model_trainer(
    lgb_reg,
    X, y, test, random_state=42, model_name='lgb'
)





test_pred = np.mean(test_preds.to_numpy(), axis=1)



sample_submission[target] = np.expm1(test_pred)
sample_submission.to_csv('submission1.csv', index=False)
sample_submission.head()




