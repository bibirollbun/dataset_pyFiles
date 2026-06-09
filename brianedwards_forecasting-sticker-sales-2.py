# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings

warnings.simplefilter('ignore')


from sklearn.metrics import mean_absolute_percentage_error as mape

y_true = [10, 20, 30]
y_pred = [12, 18, 35]
loss = mape(y_true, y_pred)
loss


train_raw = pd.read_csv('../input/playground-series-s5e1/train.csv')
test_raw = pd.read_csv('../input/playground-series-s5e1/test.csv')
display(train_raw)
display(test_raw)


for i, df in enumerate([train_raw, test_raw]):
    print('test' if i else 'train')
    for country in 'Canada', 'Kenya', 'Singapore':
        df_ = df.loc[
            (df['country'] == country) &
            (df['store'] == 'Discount Stickers') &
            (df['product'] == 'Holographic Goose')
        ]
        print(country, df_.shape)
        if i == 0:
            print('  all na', df_['num_sold'].isna().all())
    print()
        


def clean(name, df):
    if name == 'train':
        df = df.drop('id', axis=1)
        df = df.dropna(subset=['num_sold'])
        assert (df['num_sold'] % 1 == 0).all()
        df['num_sold'] = df['num_sold'].astype('int')
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.to_period('D').reindex(axis=1)
    assert df[df.isna().any(axis=1)].empty
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    display(df.info())
    display(df)
    return df

train = clean('train', train_raw)
test = clean('test', test_raw)


from pprint import pprint
import matplotlib.pyplot as plt
%matplotlib inline

assert train.loc['2010-01-07'][
    (train['country'] == 'Canada') &
    (train['store'] == 'Stickers for Less') &
    (train['product'] == 'Holographic Goose')
].empty

wide = train.pivot(columns=['country', 'store', 'product'], values='num_sold')
cols_with_na = []

for col in wide.columns:
    if wide[col].isna().any():
        cols_with_na.append((wide[col].isna().sum(), col))

cols_with_na.sort(reverse=True)
pprint(cols_with_na)

for col in cols_with_na[:2]:
    print("\n", col[1])
    wide[col[1]].plot()
    plt.show()

while not wide[wide.isna().any(axis=1)].empty:
    for col in wide.columns:
        if wide[col].isna().any():
            shift = pd.concat(
                [wide[col].shift(periods=p) for p in [-365, -1, 1, 365]],
                axis=1
            )
            mean = shift.apply(
                lambda row: np.nanmean(row),
                axis=1
            )
            wide[col] = wide[col].fillna(mean)

for col in wide.columns:
    if wide[col].isna().any():
        print(col, wide[col].isna().sum(), (~wide[col].isna()).sum())

assert wide[wide.isna().any(axis=1)].empty

wide.round().astype('int')

for col in cols_with_na[:2]:
    print("\n", col[1])
    wide[col[1]].plot()
    plt.show()

print(('Kenya', 'Discount Stickers', 'Holographic Goose') in wide.columns)


from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.preprocessing import StandardScaler

order = 3
y = wide.copy(deep=True)
dp = DeterministicProcess(index=y.index, constant=True, order=order, drop=True)
X = dp.in_sample()
X['trend'] = np.log(X['trend'])
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
display(X)


from sklearn.model_selection import train_test_split

index_train, index_valid = train_test_split(
    y.index, test_size=366, shuffle=False,
)

display(index_train)
display(index_valid)
# 2010-01-01 to 2016-12-31


X_train, X_valid = X.loc[index_train, :], X.loc[index_valid, :]
y_train, y_valid = y.loc[index_train], y.loc[index_valid]
display(X_train)
display(y_train)


import random
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

y = wide.copy(deep=True)
random_col = random.choice(y.columns)
losses = []

for log_trend in ['', 'log_trend']:
    for order in range(1, 5):
        dp = DeterministicProcess(index=y.index, constant=True, order=order, drop=True)
        X = dp.in_sample()
        if log_trend:
            X['trend'] = np.log(X['trend'])
        # X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
        
        index_train, index_valid = train_test_split(
            y.index, test_size=366, shuffle=False,
        )
        
        X_train, X_valid = X.loc[index_train, :], X.loc[index_valid, :]
        y_train, y_valid = y.loc[index_train], y.loc[index_valid]
                
        for m in [LinearRegression(fit_intercept=False), Ridge(fit_intercept=False), Lasso(fit_intercept=False)]:
            m.fit(X_train, y_train)
            fig, ax = plt.subplots()
            title = f'order: {order}, {log_trend} ' + str(type(m)).split('.')[-1]
            ax.set_title(title)
            
            def predict(X, y, color='C0'):
                y_pred = pd.DataFrame(m.predict(X), index=y.index, columns=y.columns)
                y[[random_col]].plot(
                    color='0.25', subplots=True, sharex=True, ax=ax
                )
                y_pred[[random_col]].plot(
                    color=color, subplots=True, sharex=True, ax=ax
                )
                return y_pred
            
            predict(X_train, y_train)
            y_pred = predict(X_valid, y_valid, color='C3')
            ax.get_legend().remove()
            plt.plot()
    
            losses.append((mape(y_valid, y_pred), title))
        
losses.sort()
for loss in losses:
    print(f'{loss[1]:>40} {loss[0]:.4f}')


import random
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import Lasso

y = wide.copy(deep=True)
dp = DeterministicProcess(index=y.index, constant=True, order=2, drop=True)

X = dp.in_sample()
X['trend'] = np.log(X['trend'])

m = Lasso(fit_intercept=False)
m.fit(X, y)

X_test = dp.out_of_sample(steps=(test.index.max() - test.index.min()).n + 1)
X_test['trend'] = np.log(X_test['trend'])

assert (X_test.index == test.index.unique()).all()

y_fit = pd.DataFrame(m.predict(X), index=X.index, columns=y.columns)

print(('Kenya', 'Discount Stickers', 'Holographic Goose') in y.columns)

y_pred = pd.DataFrame(m.predict(X_test), index=X_test.index, columns=y.columns)

for _ in range(7):
    fig, ax = plt.subplots()
    random_col = random.choice(y.columns)
    y[[random_col]].plot(color='0.25', subplots=True, sharex=True, ax=ax)
    y_fit[[random_col]].plot(color='C0', subplots=True, sharex=True, ax=ax)
    y_pred[[random_col]].plot(color='C3', subplots=True, sharex=True, ax=ax)
    ax.set_title(random_col)
    ax.get_legend().remove()


import xgboost as xgb

X = wide.stack(('country', 'store', 'product'))
y = X.copy(deep=True)
X = X.reset_index(('country', 'store', 'product'))
X = X.drop(0, axis=1)

X_test = test[['country', 'store', 'product']]

for col in X.columns:
    codes = {k: v for k, v in zip(X[col], X[col].cat.codes)}
    X[col + '_'] = X[col].map(codes).astype('int')
    X_test[col + '_'] = X_test[col].map(codes).astype('int')

X['dayofyear'] = X.index.dayofyear
X_test['dayofyear'] = X_test.index.dayofyear

X = X.set_index(['country', 'store', 'product'], append=True)
X_test = X_test.set_index(['country', 'store', 'product'], append=True)

X = X.rename(columns={'country_': 'country', 'store_': 'store', 'product_': 'product'})
X_test = X_test.rename(columns={'country_': 'country', 'store_': 'store', 'product_': 'product'})

y_fit_long = y_fit.stack(('country', 'store', 'product')).squeeze()
y_resid = y - y_fit_long

m = xgb.XGBRegressor()
m.fit(X, y_resid)
y_fit_boosted = m.predict(X) + y_fit_long

y_pred_long = y_pred.stack(('country', 'store', 'product')).squeeze()

# (98550,) (96360,) 
y_pred_xgb = m.predict(X_test)
y_pred_xgb = pd.Series(y_pred_xgb, index=X_test.index)

y_pred_xgb_ = y_pred_xgb.loc[
    ((X_test.index.get_level_values('country').isin(['Canada', 'Kenya'])) & 
    (X_test.index.get_level_values('store') == 'Discount Stickers') &
    (X_test.index.get_level_values('product') == 'Holographic Goose'))
]

y_pred_xgb_ = y_pred_xgb_.clip(lower=0)

y_pred_xgb = y_pred_xgb.loc[
    ~((X_test.index.get_level_values('country').isin(['Canada', 'Kenya'])) & 
    (X_test.index.get_level_values('store') == 'Discount Stickers') &
    (X_test.index.get_level_values('product') == 'Holographic Goose'))
]

y_pred_long.index = y_pred_long.index.set_names('date', level=0)

# display(y_pred_xgb_)
# display(y_pred_xgb)
# display(y_pred_long)

y_pred_boosted = y_pred_xgb + y_pred_long
# display(y_pred_boosted)

y_pred_boosted = pd.concat([y_pred_boosted, y_pred_xgb_]).sort_index()
display(y_pred_boosted)

# for i in y_pred_xgb.index.difference(y_pred_long.index):
#     print(i)
# (Period('2017-01-01', 'D'), 'Canada', 'Discount Stickers', 'Holographic Goose')
# (Period('2017-01-01', 'D'), 'Kenya', 'Discount Stickers', 'Holographic Goose')


pd.read_csv('../input/playground-series-s5e1/sample_submission.csv')


y_pred_boosted = y_pred_boosted.rename('num_sold')
y_pred_boosted.index = y_pred_boosted.index.set_names('date', level=0)

submission = test.copy(deep=True)
submission = submission.set_index(['country', 'store', 'product'], append=True)

submission = submission.join(y_pred_boosted)
submission['num_sold'] = submission['num_sold'].round().astype(int)

submission.to_csv('submission.csv', index=False)

submission


for _ in range(7):
    fig, ax = plt.subplots()
    random_col = random.choice(wide.columns)

    def filter(s):
        s_ = s[
            (s.index.get_level_values('country') == random_col[0]) &
            (s.index.get_level_values('store') == random_col[1]) &
            (s.index.get_level_values('product') == random_col[2])
        ]
        s_ = s_.droplevel(3)
        s_ = s_.droplevel(2)
        s_ = s_.droplevel(1)
        return s_
    
    filter(y).plot(color='0.25', subplots=True, sharex=True, ax=ax)
    filter(y_fit_boosted).plot(color='C0', subplots=True, sharex=True, ax=ax)
    filter(y_pred_boosted).plot(color='C3', subplots=True, sharex=True, ax=ax)
    ax.set_title(random_col)
    # ax.get_legend().remove()




