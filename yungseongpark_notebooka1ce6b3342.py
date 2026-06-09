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
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from bayes_opt import BayesianOptimization
import lightgbm as lgb
from lightgbm import LGBMRegressor


data_path = '/kaggle/input/playground-series-s5e1/'
train = pd.read_csv(data_path+'train.csv')
test = pd.read_csv(data_path+'test.csv')
sub = pd.read_csv(data_path+'sample_submission.csv')


cat_features = ['country','store','product']
nan_percent = train.groupby(cat_features)['num_sold'].apply(lambda x:(x.isna().sum()/len(x))*100).reset_index(name='nan_percent')
nan_percent = nan_percent[nan_percent['nan_percent']>0]
nan_percent


featrues=train.groupby(cat_features)['num_sold'].mean().reset_index(name='mean')    
featrues.sort_values('mean').head(30)


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


col = sns.color_palette('pastel')
grid = gridspec.GridSpec(3,1)
for idx, featrue in enumerate(cat_features):
    ax = plt.subplot(grid[idx])
    sns.barplot(x=featrue,y='mean',data=featrues,palette=col,ax=ax)


eda_train = train.copy()
eda_train['date'] = pd.to_datetime(eda_train['date'])

def date_feat(df):
    df['year'] = df['date'].dt.year.astype('float64')
    df['quarter'] = df['date'].dt.quarter.astype('float64')
    df['month'] =df['date'].dt.month.astype('float64')
    df['day'] = df['date'].dt.day.astype('float64')
    df['day_of_week'] =df['date'].dt.dayofweek.astype('float64')
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype('float64')
date_feat(eda_train)


date_features = ['year','quarter','month','day','day_of_week','week_of_year']

grid = gridspec.GridSpec(len(date_features),1)
for idx, featrue in enumerate(date_features):
    ax = plt.subplot(grid[idx])
    sns.barplot(x=featrue,y='num_sold',data=eda_train,palette=col,ax=ax,estimator='mean')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',va='bottom')


grid = gridspec.GridSpec(len(date_features),1)
for idx, featrue in enumerate(date_features):
    ax = plt.subplot(grid[idx])
    sns.barplot(x=featrue,y='num_sold',data=eda_train,palette=col,ax=ax,estimator='mean')


plt.figure(figsize=(40,10))
eda_train.groupby('date')['num_sold'].mean().plot()


weekly_df = eda_train.groupby([pd.Grouper(key='date',freq='W')])['num_sold'].mean().rename('num_sold').reset_index()
monthly_df = eda_train.groupby([pd.Grouper(key='date',freq='MS')])['num_sold'].mean().rename('num_sold').reset_index()
quarter_df = eda_train.groupby([pd.Grouper(key='date',freq='Q')])['num_sold'].mean().rename('num_sold').reset_index()


figure, ax = plt.subplots(figsize=(30,8))
sns.lineplot(x='date',y='num_sold',data=weekly_df)


figure, ax = plt.subplots(figsize=(30,8))
sns.lineplot(x='date',y='num_sold',data=monthly_df)


figure, ax = plt.subplots(figsize=(30,8))

sns.lineplot(x='date',y='num_sold',data=quarter_df)


from statsmodels.tsa.seasonal import seasonal_decompose

x_data_index = eda_train.set_index('date')
monthly_data = x_data_index.resample('D').sum()
monthly_data.index

monthly_data['year'] = monthly_data.index.year
monthly_data['month'] = monthly_data.index.month

for year in monthly_data['year'].unique():
    yearly_data = monthly_data[monthly_data['year']==year]
    decomposition = seasonal_decompose(yearly_data['num_sold'],model='additive',period=12)
    plt.figure(figsize=(20,5))
    decomposition.trend.plot()
    plt.show()


for year in monthly_data['year'].unique():
    yearly_data = monthly_data[monthly_data['year']==year]
    decomposition = seasonal_decompose(yearly_data['num_sold'],model='additive',period=12)
    plt.figure(figsize=(30,8))
    decomposition.seasonal.plot()


for year in monthly_data['year'].unique():
    yearly_data = monthly_data[monthly_data['year']==year]
    decomposition = seasonal_decompose(yearly_data['num_sold'],model='additive',period=52)
    plt.figure(figsize=(30,8))
    decomposition.seasonal.plot()


from statsmodels.graphics.tsaplots import plot_acf
trend_data = eda_train.groupby('date')['num_sold'].sum()
plt.figure(figsize=(20,10))
plot_acf(trend_data.dropna(),lags=50)
plt.show()


train = train.dropna(subset=['num_sold'])
all_data = pd.concat([train,test],ignore_index=True)
all_data['date'] = pd.to_datetime(all_data['date'])
date_feat(all_data)


all_data['month_sin'] = np.sin(2*np.pi*all_data['month']/12.0)
all_data['month_cos'] = np.cos(2*np.pi*all_data['month']/12.0)
all_data['year_sin'] = np.sin(2*np.pi*all_data['year']/12.0)
all_data['year_cos'] = np.cos(2*np.pi*all_data['year']/12.0)

all_data['date_num'] = (all_data['year']-2010)*48 + all_data['month']*4 + all_data['day']//7

all_data = all_data.drop(columns=['id','date','quarter','month','day','num_sold'])
all_data.head()


from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()
csr_matrix = one_hot.fit_transform(all_data[cat_features])


from scipy import sparse
all_data = all_data.drop(columns=cat_features)

all_data_sprs = sparse.hstack([sparse.csr_matrix(all_data),csr_matrix],format='csr')


train_test = len(train)
x = all_data_sprs[:train_test]
test = all_data_sprs[train_test:]
y = train['num_sold'].values


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x,y,train_size=0.8,random_state=0)

import xgboost as xgb
dtrain = xgb.DMatrix(x_train, y_train)
dvalid = xgb.DMatrix(x_valid,y_valid)


from sklearn.metrics import mean_absolute_percentage_error
params = {
    'max_depth':(4,8),
    'subsample':(0.6,0.9),
    'colsample_bytree':(0.7,0.9),
    'reg_alpha':(7,9),
    'reg_lambda':(1.1,1.5)
}
fixed_params = {
    'objective':'reg:squarederror',
    'learing_rate':0.02,
    'random_state':42
}
def mape_xgb(preds, dtrain):
    labels = dtrain.get_label()
    return 'mape', mean_absolute_percentage_error(labels,preds)



def eval_function(max_depth,subsample,colsample_bytree,reg_alpha,reg_lambda):
    params ={'max_depth':int(round(max_depth)),
             'subsample':subsample,
             'colsample_bytree':colsample_bytree,
             'reg_alpha':reg_alpha,
             'reg_lambda':reg_lambda}
    params.update(fixed_params)
    print('하이퍼 파라미터:',params)

    xgb_model = xgb.train(params=params,
                          dtrain=dtrain,
                          num_boost_round=2000,
                          evals=[(dvalid,'bayes_dvalid')],
                          maximize=True,
                          feval=mape_xgb,
                          early_stopping_rounds=200,
                          verbose_eval=False)
    best_iter = xgb_model.best_iteration
    preds = xgb_model.predict(dvalid,iteration_range=(0,best_iter))
    mape_score = mean_absolute_percentage_error(y_valid,preds)
    print(f'mape:{mape_score}\n')
    return -mape_score


from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(f=eval_function,
                                 pbounds=params,
                                 random_state=0)
optimizer.maximize(init_points=3,n_iter=6)
max_params = optimizer.max['params']
max_params['max_depth'] =int(round(max_params['max_depth']))
max_params.update(fixed_params)
max_params


from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(x.shape[0])
test_preds = np.zeros(test.shape[0])

for idx, (train_idx, valid_idx) in enumerate(kf.split(x)):
    print(f"Fold {idx + 1}")
    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    dtest = xgb.DMatrix(test)

    model = xgb.train(
        params=max_params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dvalid, 'valid')],
        feval=mape_xgb,
        maximize=False,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    best_iter = model.best_iteration
    oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, best_iter))
    test_preds += model.predict(dtest, iteration_range=(0, best_iter)) / kf.n_splits

    fold_mape = mean_absolute_percentage_error(y_valid, oof_preds[valid_idx])
    print(f"Fold {idx + 1} MAPE: {fold_mape:.4f}")


final_mape = mean_absolute_percentage_error(y, oof_preds)
print(f"Overall MAPE: {final_mape:.4f}")


sub['num_sold'] = test_preds


sub.to_csv('sample_submission.csv',index=False)


sub.head()




