# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

train_data_row, train_data_col = train_data.shape
print('Total rows: ', train_data_row)
print('Total columns: ', train_data_col)


train_data.tail()


train_data.info()


train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])


train_data['country'].value_counts()


print("duplicated data :", train_data.duplicated().sum())
print("null data : ", train_data.isnull().sum().sum())


train_data.describe().loc[['mean','min','max']].T


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

sold_product = train_data.groupby(['product'])['num_sold'].sum()
sold_store = train_data.groupby(['store'])['num_sold'].sum()
sold_country = train_data.groupby(['country'])['num_sold'].sum()

fig = plt.figure(figsize=(15,10))
fig.set_facecolor('white')

ax1 = fig.add_subplot(1, 2, 1)
ax1.bar(sold_country.keys(), sold_country.values)
ax1.set_title('Sales on Every Country')
ax1.ticklabel_format(style='plain', axis='y')
ax1.set_ylim(0, 3e6)

ax2 = fig.add_subplot(2, 2, 2)
ax2.pie(sold_product.values, labels=sold_product.keys(), autopct="%.1f%%")
ax2.set_title('Total Product Sales', fontsize=16)

ax3 = fig.add_subplot(2, 2, 4)
ax3.pie(sold_store.values, labels=sold_store.keys(), autopct="%.1f%%")
ax3.set_title('Total Store Sales', fontsize=16);


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

fig = plt.figure(figsize=(20,10))
fig.set_facecolor('white')

ax1 = fig.add_subplot(1, 2, 1)
sns.barplot(data=train_data, x='country', y='num_sold', hue='product')
ax1.set_title('Product Sales by Product')

ax2 = fig.add_subplot(1, 2, 2)
sns.barplot(data=train_data, x='store', y='num_sold', hue='product')
ax2.set_title("Product Sales by Store");


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

monthly_sales = train_data.groupby(['country', 'store', 'product', pd.Grouper(key='date', freq='MS')])['num_sold'].sum().reset_index()
monthly_sales


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

monthly_sales_country = monthly_sales.groupby(['country','date'])['num_sold'].sum().reset_index()
monthly_sales_product = monthly_sales.groupby(['product', 'date'])['num_sold'].sum().reset_index()
monthly_sales_store = monthly_sales.groupby(['store', 'date'])['num_sold'].sum().reset_index()

fig = plt.figure(figsize=(24, 15))

fig.set_facecolor('white')

ax1 = fig.add_subplot(3, 1, 1)
sns.lineplot(data=monthly_sales_country, x='date', y='num_sold', hue='country')
ax1.set_title("Monthly sales by Country", fontsize= 16)

ax2 = fig.add_subplot(3, 1, 2)
sns.lineplot(data=monthly_sales_product, x='date', y='num_sold', hue='product')
ax2.set_title("Monthly sales by Product", fontsize= 16)

ax3 = fig.add_subplot(3, 1, 3)
sns.lineplot(data=monthly_sales_store, x='date', y='num_sold', hue='store')
ax3.set_title("Monthly sales by Store", fontsize= 16);


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

def format_date(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayOfYear'] = df['date'].dt.dayofyear
    df['weekday'] = df['date'].dt.weekday
    return df

train_data = format_date(train_data)
test_data = format_date(test_data)


#Luca Massaron https://www.kaggle.com/code/lucamassaron/kaggle-merchandise-eda-with-xgboost
#Fixed by Kheirallah ax = fig.add_subplot(3, 6, (i*6+j+1))
#Firstly I worked with less countries (add 3 more) and Stores (add 1) 


for product in ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler','Kerneler Dark Mode']:
    fig = plt.figure(figsize=(30, 18), dpi=100)
    fig.subplots_adjust(hspace=0.25)
    for i, store in enumerate([ 'Stickers for Less', 'Premium Sticker Mart', 'Discount Stickers']):
        for j, country in enumerate(['Canada','Kenya','Singapore', 'Finland', 'Norway', 'Italy']):
            ax = fig.add_subplot(3, 6, (i*6+j+1))
            selection = (train_data['country']==country)&(train_data['store']==store)&(train_data['product']==product)
            selected = train_data[selection]
            for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016]:
                selected[selected.year==year].set_index('date').groupby('month')['num_sold'].mean().plot(ax=ax, label=year)
            ax.set_title(f"{product} | {country}:{store}")
            ax.legend()
plt.show()


#Luca Massaron https://www.kaggle.com/code/lucamassaron/kaggle-merchandise-eda-with-xgboost

for product in ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler','Kerneler Dark Mode']:
    fig = plt.figure(figsize=(20, 10), dpi=100)
    fig.subplots_adjust(hspace=0.25)
    for i, store in enumerate([ 'Stickers for Less', 'Premium Sticker Mart']):
        for j, country in enumerate(['Canada','Kenya','Singapore']):
            ax = fig.add_subplot(2, 3, (i*3+j+1))
            selection = (train_data['country']==country)&(train_data['store']==store)&(train_data['product']==product)
            selected = train_data[selection]
            for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016]:
                selected[selected.year==year].set_index('date').groupby('month')['num_sold'].mean().plot(ax=ax, label=year)
            ax.set_title(f"{product} | {country}:{store}")
            ax.legend()
plt.show()


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

import lightgbm as lgbm
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV , cross_val_score ,cross_validate


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

le = LabelEncoder()
cols = ['country', 'store', 'product']
for col in cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

train_data = train_data.drop(['date', 'id'], axis=1)
test_data = test_data.drop(['date', 'id'], axis=1)


train_data.head()


#I should have removed Missing before! 

train_data["num_sold"] = train_data["num_sold"].fillna(0)


X = train_data.drop('num_sold', axis=1)
y = train_data['num_sold']


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

lgb = LGBMRegressor(random_state=42, boosting_type='dart')
cb = CatBoostRegressor(random_state=42, verbose=0)
lr = LinearRegression()

models = [lgb, cb, lr]


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

def get_scores(cv_scores):
    scores = np.zeros(test_data.shape[0])
    for estimator in cv_scores['estimator']:
        scores += estimator.predict(test_data)
    scores /= len(cv_scores['estimator'])
    return scores


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

def smape(a, f):
    # Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

smape_score = make_scorer(smape, greater_is_better=False)

scalar = StandardScaler()

res = pd.DataFrame()
row_number = 0
results = []
names = []
prob_scores = []

for model in models:
    model_name=model.__class__.__name__
    pipeline = Pipeline([('transformer', scalar), ('estimator', model)])
    print(model_name, 'training')

    cv_results = cross_validate(pipeline, X, y, cv=42, scoring=smape_score, return_train_score=True, return_estimator=True, n_jobs=-1)

    res.loc[row_number,'Model Name'] = model_name
    res.loc[row_number, 'Train Score Mean'] = cv_results['train_score'].mean()
    res.loc[row_number, 'Test Score Mean'] = cv_results['test_score'].mean()
    res.loc[row_number, 'Fit Time Mean'] = cv_results['fit_time'].mean()
    results.append(cv_results)
    names.append(model_name)
    prob_scores.append(get_scores(cv_results))

    row_number+=1


display(res.style.background_gradient())


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

params_cb = {
    'depth'         : [8, 10],
    'learning_rate' : [0.1],
    'iterations'    : [50, 100],
    'random_state'  : [42], 
    'verbose'       : [0],
    }

grid_search_cb = GridSearchCV(
    estimator=cb,
    param_grid=params_cb,
    cv = 42,
    scoring=smape_score,
    n_jobs = -1
).fit(X, y)

cb_best = grid_search_cb.best_estimator_
print('CB Best Params',grid_search_cb.best_params_)


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

pipeline = Pipeline([('transformer', scalar), ('estimator', cb_best)])
cv_results = cross_validate(pipeline, X, y, cv=5, scoring=smape_score, return_train_score=True, return_estimator=True, n_jobs=-1)
np.mean(cv_results['test_score'])


#By Shibu Mohapatra  https://www.kaggle.com/code/shibumohapatra/tabular-playground-series-sep-2022/notebook

scores = np.zeros(test_data.shape[0])
for estimator in cv_results['estimator']:
    scores += estimator.predict(test_data)
    
scores /= len(cv_results['estimator'])


sub["num_sold"] = scores
sub.head()


sub.to_csv("sample_submission.csv", index = False)

