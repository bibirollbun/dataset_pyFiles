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


from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from sklearn.metrics import mean_absolute_percentage_error
import optuna
import seaborn as sb
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_df.head(3), test_df.head(3)


train_df.info(), test_df.info()


train_df[train_df.duplicated()]


train_df.isna().sum()


def prepare_data(data, num_sold = False):
    df = data.copy()
    df['date'] = df['date'].apply(lambda x: dt.strptime(x, '%Y-%m-%d'))
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['year'] = df['date'].apply(lambda x: x.year)
    df['day_of_week'] = df['date'].apply(lambda x: x.day_of_week)
    df['weekdays'] = np.where(df['day_of_week'].isin([5, 6]), 0, 1)
    
    lb = LabelEncoder()
    df[['country', 'store', 'product']] = df[['country', 'store', 'product']].apply(lb.fit_transform)

    if num_sold:
        df['num_sold_fill'] = df.groupby(['month', 'country', 'product', 'weekdays'], sort=False)['num_sold'].transform(lambda x: x.fillna(int(x.mean()))).values

    return df


train = prepare_data(train_df, num_sold=True)
test = prepare_data(test_df)
test.head(), train.head()


countries = train['country'].unique()
products = train['product'].unique()
for i in countries:
    for j in products:
        train[(train['country'] == i) & (train['product'] == j)].groupby('date').aggregate(daily_sum=('num_sold', 'sum')).plot(grid=True)


train[['country', 'year', 'product', 'num_sold_fill']].groupby(['product', 'year']).aggregate(daily_sum=('num_sold_fill', 'sum'))


train


cor_mat = train.corr()
dataplot = sb.heatmap(cor_mat['num_sold_fill'].values.reshape(-1, 1), annot=True, fmt='.2f', vmax=1, vmin=-1)
dataplot.set_yticklabels(cor_mat.columns.tolist(), rotation=0);



x = train.drop(columns=["id", "date", "num_sold", "num_sold_fill", 'year', 'day'], axis=1)
y = train[["num_sold_fill"]]


x_train, x_valid, y_train, y_valid  = train_test_split(x, y, test_size=0.1, random_state=42)


model = lgbm.LGBMRegressor(random_state=42)
model.fit(x_train, y_train)
preds = model.predict(x_valid).astype('int')
mean_absolute_percentage_error(y_valid, preds)


def objective(trial):
    params = {'learning_rate': trial.suggest_loguniform("learning_rate", 0.01, 0.1),
              'max_depth': trial.suggest_int("max_depth", 5, 20),
              'min_child_samples' : trial.suggest_int("min_child_samples", 5, 50),
              'subsample': trial.suggest_float("subsample", 0.5, 1.0),
              'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
              'n_estimators': trial.suggest_int("n_estimators", 500, 5000),
              'random_state':  42,
              'device': 'gpu',
              'verbose': -1,
              'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 10.0),
              'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 10.0),
              'metric': 'mape'}


    model = lgbm.LGBMRegressor(**params)
    model.fit(x_train, y_train)


    y_pred = model.predict(x_valid).astype('int')
    accuracy = mean_absolute_percentage_error(y_valid, y_pred)
    return accuracy


%%time
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40)
print("Best trial:")
print(" Value: {}".format(study.best_trial.value))
print(" Params: {}".format(study.best_trial.params))


opt_lgbm = lgbm.LGBMRegressor(**study.best_trial.params)
opt_lgbm.fit(x_train, y_train)


x_test = test[x_train.columns]
predictions = opt_lgbm.predict(x_test).astype('int')


submission = pd.DataFrame({'id': test['id'],
                           'num_sold': predictions})

submission


submission.to_csv('submission.csv', index=False)
print("Done!")
print(submission.head())




