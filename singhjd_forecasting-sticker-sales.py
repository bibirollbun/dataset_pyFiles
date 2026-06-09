# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import holidays
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


train_df.head()


test_df.head()


print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)


train_df.drop(columns=['id'], inplace=True)


train_df.info()


train_df['date']=pd.to_datetime(train_df['date'])


train_df.info()


test_df.info()


test_df['date']=pd.to_datetime(test_df['date'])


test_df.info()


train_df.isna().sum()


percentage=train_df['num_sold'].isna().sum()/train_df.shape[0]*100
print("Percentage of missing values in training set:", round(percentage, 2))


train_df.dropna(inplace=True)


train_df.shape


train_df.head()


print("Unique values in country attribute:", train_df['country'].unique(), "\n\n")
print("Unique values in store attribute:", train_df['store'].unique(), "\n\n")
print("Unique values in product attribute:", train_df['product'].unique())


def plotcount(attribute):
    ax=sns.countplot(x=attribute, data=train_df)
    for bars in ax.containers:
        plt.bar_label(bars)
    ax.set_title(f'Countplot- {attribute}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()


plotcount('country')
plotcount('store')
plotcount('product')


train_df['year']=train_df['date'].dt.year
train_df['month']=train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['is_weekend'] = train_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)


country_holidays={
    'Canada':holidays.CountryHoliday('CA'),
    'Finland': holidays.CountryHoliday('FI'),
    'Italy': holidays.CountryHoliday('IT'),
    'Kenya': holidays.CountryHoliday('KE'),
    'Norway': holidays.CountryHoliday('NO'),
    'Singapore': holidays.CountryHoliday('SG')
}


train_df['is_holiday'] = train_df.apply(
    lambda row: 1 if row['date'] in country_holidays.get(row['country'], []) else 0, 
    axis=1
)


train_df.head()


train_df=pd.get_dummies(train_df)


train_df.drop(columns='date', inplace=True)


train_df['country_Canada']=train_df['country_Canada'].astype(int)
train_df['country_Finland']=train_df['country_Finland'].astype(int)
train_df['country_Italy']=train_df['country_Italy'].astype(int)
train_df['country_Kenya']=train_df['country_Kenya'].astype(int)
train_df['country_Norway']=train_df['country_Norway'].astype(int)
train_df['country_Singapore']=train_df['country_Singapore'].astype(int)
train_df['store_Discount Stickers']=train_df['store_Discount Stickers'].astype(int)
train_df['store_Premium Sticker Mart']=train_df['store_Premium Sticker Mart'].astype(int)
train_df['store_Stickers for Less']=train_df['store_Stickers for Less'].astype(int)
train_df['product_Holographic Goose']=train_df['product_Holographic Goose'].astype(int)
train_df['product_Kaggle']=train_df['product_Kaggle'].astype(int)
train_df['product_Kaggle Tiers']=train_df['product_Kaggle Tiers'].astype(int)
train_df['product_Kerneler']=train_df['product_Kerneler'].astype(int)
train_df['product_Kerneler Dark Mode']=train_df['product_Kerneler Dark Mode'].astype(int)


train_df.head()


X=train_df.drop(columns=['num_sold'])
y=train_df['num_sold']


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


y_train.head()


X_train.head()


linear_model=LinearRegression()


linear_model.fit(X_train, y_train)


y_pred_linear=linear_model.predict(X_test)


print(r2_score(y_test, y_pred_linear))


decision_tree_model=DecisionTreeRegressor()


decision_tree_model.fit(X_train, y_train)


y_pred_dt=decision_tree_model.predict(X_test)


print(r2_score(y_test, y_pred_dt))


test_df.head()


test_df['year']=test_df['date'].dt.year
test_df['month']=test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.dayofweek
test_df['is_weekend'] = test_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)


test_df['is_holiday'] = test_df.apply(
    lambda row: 1 if row['date'] in country_holidays.get(row['country'], []) else 0, 
    axis=1
)


test_df.head()


test_df=pd.get_dummies(test_df)


id_test = test_df['id']

test_df.drop(columns='id', inplace=True)


test_df.drop(columns='date', inplace=True)


test_df['country_Canada']=test_df['country_Canada'].astype(int)
test_df['country_Finland']=test_df['country_Finland'].astype(int)
test_df['country_Italy']=test_df['country_Italy'].astype(int)
test_df['country_Kenya']=test_df['country_Kenya'].astype(int)
test_df['country_Norway']=test_df['country_Norway'].astype(int)
test_df['country_Singapore']=test_df['country_Singapore'].astype(int)
test_df['store_Discount Stickers']=test_df['store_Discount Stickers'].astype(int)
test_df['store_Premium Sticker Mart']=test_df['store_Premium Sticker Mart'].astype(int)
test_df['store_Stickers for Less']=test_df['store_Stickers for Less'].astype(int)
test_df['product_Holographic Goose']=test_df['product_Holographic Goose'].astype(int)
test_df['product_Kaggle']=test_df['product_Kaggle'].astype(int)
test_df['product_Kaggle Tiers']=test_df['product_Kaggle Tiers'].astype(int)
test_df['product_Kerneler']=test_df['product_Kerneler'].astype(int)
test_df['product_Kerneler Dark Mode']=test_df['product_Kerneler Dark Mode'].astype(int)


test_df.head()




