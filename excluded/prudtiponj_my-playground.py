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


import warnings
import numpy as np
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train.info()


test.info()


def check_null(df):
    print(df.isnull().sum())
    print(f'Total: {df.isnull().sum().sum()}')
    print(f'Lost Rate: {df.isnull().sum().sum() / df.size * 100:.2f}')


check_null(train)


check_null(test)


train.head()


train['date'] = pd.to_datetime(train['date'], format='%Y-%m-%d')
test['date'] = pd.to_datetime(test['date'], format='%Y-%m-%d')


import matplotlib.pyplot as plt



train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year

test['month'] = test['date'].dt.month
test['year'] = test['date'].dt.year



data = train.groupby('year')['num_sold'].sum()
plt.plot(data.index, data.values / 1000000)
plt.xlabel("Year")
plt.ylabel("1m unit")
plt.title('Annual sales')
plt.grid()
plt.show()


d1 = train.groupby(['product', 'year'])['num_sold'].sum()


plt.figure(figsize=[12, 7])
year = train['year'].sort_values().unique()
bw = 0
for product in train['product'].unique():
    plt.bar(year + bw, d1[product], width=.1, label=product)
    bw+=.1

plt.xticks(year+.2, year)
plt.tight_layout()
plt.legend(loc='best')
plt.title('Annual sales by Product')
plt.show()


train['dayOfweek'] = train['date'].dt.day_name()
test['dayOfweek'] = test['date'].dt.day_name()


train['Quarter'] = train['date'].dt.quarter
test['Quarter'] = test['date'].dt.quarter


train.set_index('id', inplace=True)
test.set_index('id', inplace=True)

d2 = train.groupby(['Quarter', 'year'])['num_sold'].sum()
bw = 0
plt.figure(figsize=[10, 7])
for q in train['Quarter'].sort_values().unique():
    plt.bar(year + bw, d2[q], width=.2, label=f'Q{q}')
    bw += .2
plt.xticks(year + .3, year)
plt.legend(loc='best')
plt.title('Annual sales by Quarter')
plt.show()


d3 = train.groupby(['product', 'country'])['num_sold'].sum()
country = train['country'].sort_values().unique()
bw = 0
plt.figure(figsize=[15, 7])
for prod in train['product'].sort_values().unique():
    plt.bar(np.arange(0, len(country)) + bw, d3[prod], width=.1, label=prod)
    bw += .1
plt.xticks(np.arange(0, len(country)) + .3, country)
plt.legend(loc='best')
plt.title('Annual sales by Country')
plt.show()


test.head()


train.head()


train.drop('date', axis='columns', inplace=True)
test.drop('date', axis='columns', inplace=True)


[f'{col}: {train[col].unique()}' for col in train.columns if train[col].dtype == 'object']


train.dtypes


numerical_col = [col for col in train.columns if (train[col].dtype == 'int') or (train[col].dtype == 'float') ]
categorical_col = [col for col in train.columns if train[col].dtype == 'object']


numerical_col


from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('ordinal_encoder', OrdinalEncoder())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_col)
    ]
)


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor())
])


pipeline


train.dropna(subset=['num_sold'], inplace=True)

target = train['num_sold']
train.drop('num_sold', axis='columns', inplace=True)


# parameters = {
#     'regressor__n_estimators': [100, 200, 500, 1000],  # จำนวนต้นไม้
#     'regressor__learning_rate': [0.1, 0.05, 0.01],  # อัตราการเรียนรู้
#     'regressor__max_depth': [3, 5, 7],  # ความลึกสูงสุดของต้นไม้
#     'regressor__subsample': [0.8, 0.9, 1.0],  # อัตราส่วนของตัวอย่าง
#     'regressor__colsample_bytree': [0.8, 0.9, 1.0],  # อัตราส่วนของคุณสมบัติ
#     'regressor__gamma': [0, 0.1, 0.5],  # ค่าต่ำสุดของการลด loss
#     'regressor__reg_alpha': [0, 0.1, 1],  # L1 regularization
#     'regressor__reg_lambda': [1, 1.1, 2]  # L2 regularization
# }


# cv = GridSearchCV(
#     estimator=pipeline,
#     param_grid=parameters,
#     scoring='neg_mean_absolute_error',
#     cv=5,
#     n_jobs=1
# )


pipeline.fit(train, target)


preds = pipeline.predict(test)


output = pd.DataFrame(
    {
        'id': test.index,
        'num_sold': preds
    }
)


output


output.to_csv('submission.csv', index=False)




