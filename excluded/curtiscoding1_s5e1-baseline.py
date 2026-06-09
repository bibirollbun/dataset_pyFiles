# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import *
from sklearn.model_selection import *
from matplotlib import pyplot as plt
import seaborn as sns
import optuna 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

# Finding File Paths

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

train.drop('id', axis = 1, inplace = True)
test.drop('id', axis = 1, inplace = True)

train = train[train['num_sold'].notna()]


train.head()


train.info()


train.describe()


group_df = train.copy()
country_group = group_df.groupby(by=["country"], dropna=False)['num_sold'].sum().reset_index()


sns.barplot(data=country_group, x='country', y='num_sold')
plt.title('Total Units Sold by Country')
plt.xlabel('Country')
plt.ylabel('Total Units Sold')
plt.xticks(rotation=45)
plt.show()


def date_extraction(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day.astype('float32')
    df['month'] = df['date'].dt.month.astype('category')
    df['year'] = df['date'].dt.year.astype('category')
    df.drop('date', axis = 1, inplace = True)


date_extraction(train)
date_extraction(test)
    


group_df = train.copy()
month_group = group_df.groupby(by=["month"], dropna=False)['num_sold'].sum().reset_index()


sns.barplot(data=month_group, x='month', y='num_sold')
plt.title('Total Units Sold by Month')
plt.xlabel('month')
plt.ylabel('Total Units Sold')
plt.xticks(rotation=45)
plt.show()


norway_df = train.copy()
norway_df = norway_df[norway_df['country'] == 'Norway']

norwaymonth_group = norway_df.groupby(by=["month"], dropna=False)['num_sold'].sum().reset_index()

sns.barplot(data=norwaymonth_group, x='month', y='num_sold')
plt.title('Total Units Sold by Month in Norway')
plt.xlabel('month')
plt.ylabel('Total Units Sold')
plt.ylim((0, 5000000))
plt.xticks(rotation=45)
plt.show()


kenya_df = train.copy()
kenya_df = kenya_df[kenya_df['country'] == 'Kenya']

kenyamonth_group = kenya_df.groupby(by=["month"], dropna=False)['num_sold'].sum().reset_index()

sns.barplot(data=kenyamonth_group, x='month', y='num_sold')
plt.title('Total Units Sold by Month in Kenya')
plt.xlabel('month')
plt.ylabel('Total Units Sold')
plt.ylim((0, 70000))
plt.xticks(rotation=45)
plt.show()


def country_seasonality(df):
    df['month_country'] = df['country'].astype(str) + df['month'].astype(str)
    df['year_country'] = df['country'].astype(str) + df['year'].astype(str)
    df.drop('month', axis = 1, inplace = True)
    df.drop('year', axis = 1, inplace = True)


country_seasonality(train)
country_seasonality(test)


# Cat(egorical) and Cont(inuous) columns identification

cats = test.select_dtypes(include=["object_"]).columns.tolist()
conts = [col for col in test.columns if col not in cats]


def preprocess(df):

    for col in cats:
        df[col] = df[col].astype('category')

    for col in conts:
        df[col] = df[col].astype('float32')

preprocess(train)
preprocess(test)


X = train.drop('num_sold', axis = 1).copy()

y = train['num_sold'].copy()
y = np.log(y)

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)


lgbm = lgb.LGBMRegressor(n_estimators = 100, verbose = -1)


def cross_val(model, X_df, y_df):

    kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
    scores = [] 

    for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train = X.iloc[train_idx, :]
        X_val = X.iloc[val_idx, :]

        y_train = y[train_idx]
        y_val = y[val_idx]

        model.fit(X_train, y_train)
        fold_preds = model.predict(X_val)

        score = mean_absolute_percentage_error(np.exp(y_val), np.exp(fold_preds))
        scores.append(score)

        print(f"Fold {i} Score: {score}")

    return scores
        
cross_val(lgbm, X, y)


model = lgb.LGBMRegressor(n_estimators = 300, verbose = -1)
model.fit(X, y)
preds = model.predict(test)


lgb.plot_importance(model, max_num_features=20, importance_type='split') 
plt.title("Feature Importances")
plt.show()


sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


sub['num_sold'] = np.exp(preds)


sub.to_csv('submission.csv', index = False)


sub




