%%time
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import gc

import holidays

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold, GroupKFold
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

from ydf import RandomForestLearner, GradientBoostedTreesLearner
import ydf

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, DMatrix
import xgboost as xgb
from catboost import CatBoostRegressor, Pool


%%time
train = pd.read_csv('../input/playground-series-s5e1/train.csv', index_col=0)
train['date']= pd.to_datetime(train['date'])
train = train.dropna().reset_index(drop=True)
# train['num_sold'] = train['num_sold'].fillna(0)

test = pd.read_csv('../input/playground-series-s5e1/test.csv', index_col=0)
test['date'] = pd.to_datetime(test['date'])

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


train.head()


test.head()


print('--- Train ---\n')
print(100*train.isnull().sum() / train.shape[0])
print('\n')
print('--- Test ---\n')
print(100*test.isnull().sum() / test.shape[0])


plt.figure(figsize = (12, 8))

ax = sns.barplot(data=train, x='country', y='num_sold', hue='product')
plt.xticks(rotation=30);


plt.figure(figsize = (20, 10))

ax = sns.lineplot(data = train.groupby([train.date.dt.strftime('%Y-%m'), train.country])['num_sold'].sum().reset_index(),
                  x='date',
                  y='num_sold',
                  hue='country')
plt.xticks(rotation = 45);


fig, ax = plt.subplots(2, 3, figsize = (25, 15), sharey = True)
ax = ax.flatten()
hue_order = train.country.unique()

for i, product in enumerate(train['product'].unique()):
    df = train[train['product'] == product]
    sns.lineplot(data = df.groupby([df.date.dt.strftime('%Y-%m'), df.country])['num_sold'].sum().reset_index(),
                 x = 'date',
                 y = 'num_sold',
                 hue = 'country',
                 ax = ax[i],
                 hue_order = hue_order
    )
    ax[i].set_title(product)
    ax[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    
    handles = ax[i].get_legend_handles_labels()[0]
    labels = ax[i].get_legend_handles_labels()[1]
    ax[i].legend().remove()
    
fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor=(0.5, 1.03), fontsize = 14, ncol = 6)
plt.tight_layout()


%%time
def get_holidays(df):
    years_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    holiday_FI = holidays.CountryHoliday('FI', years = years_list)
    holiday_CA = holidays.CountryHoliday('CA', years = years_list)
    holiday_IT = holidays.CountryHoliday('IT', years = years_list)
    holiday_KE = holidays.CountryHoliday('KE', years = years_list)
    holiday_NO = holidays.CountryHoliday('NO', years = years_list)
    holiday_SI = holidays.CountryHoliday('SG', years = years_list)

    holiday_dict = holiday_FI.copy()
    holiday_dict.update(holiday_CA)
    holiday_dict.update(holiday_IT)
    holiday_dict.update(holiday_KE)
    holiday_dict.update(holiday_NO)
    holiday_dict.update(holiday_SI)

    df['holiday_name'] = df['date'].map(holiday_dict)
    df['is_holiday'] = np.where(df['holiday_name'].notnull(), 1, 0)
    df['holiday_name'] = df['holiday_name'].fillna('Not Holiday')
    df.drop(columns='holiday_name', axis=1, inplace=True)
    
    return df


def feature_engineer(df):
    
    new_df = df.copy()
    new_df['year'] = new_df['date'].dt.year
    new_df['year_sin'] = np.sin(new_df['year'] * (2 * np.pi))
    new_df['year_cos'] = np.cos(new_df['year'] * (2 * np.pi))

    new_df['month'] = new_df['date'].dt.month
    new_df['month_sin'] = np.sin(new_df['month'] * (2 * np.pi / 12))
    new_df['month_cos'] = np.cos(new_df['month'] * (2 * np.pi / 12))
    
    new_df['day'] = new_df['date'].dt.day
    new_df['day_sin'] = np.sin(new_df['day'] * (2 * np.pi / 365))
    new_df['day_cos'] = np.cos(new_df['day'] * (2 * np.pi / 365))
    
    new_df['day_of_week'] = new_df['date'].dt.dayofweek
    new_df['day_of_week'] = new_df['day_of_week'].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))))
    
    # new_df = pd.get_dummies(new_df, columns=['day_of_week'], drop_first=True, dtype=int)
    
    important_dates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 124, 125, 126, 127, 140, 141, 
                       167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 
                       203, 230, 231, 232, 233, 234, 282, 289, 290, 307, 308, 309, 310, 311, 
                       312, 313, 317, 318, 319, 320, 360, 361, 362, 363, 364, 365]
    
    new_df['important_date'] = np.where(np.isin(new_df['day'], important_dates), 1, 0)
        
    return new_df.drop(columns=['date', 'month', 'day'], axis=1)

train = get_holidays(train)
train = feature_engineer(train)

test = get_holidays(test)
test = feature_engineer(test)


%%time
for cols in ['country', 'store', 'product']:
    train[cols] = train[cols].astype('category')
    test[cols] = test[cols].astype('category')
    
X = train.drop(columns=['num_sold'], axis=1)
y = np.log1p(train['num_sold'])


%%time
year_list = train['year'].unique().tolist()
lgb_params = {'objective': 'mape',
              'learning_rate': 0.08581671058380945,
              'n_estimators': 1000,
              'max_depth': 8,
              'reg_alpha': 7.4137713950946855,
              'reg_lambda': 0.03320034177331515,
              'num_leaves': 97,
              'colsample_bytree': 0.7353362488668553,
              'verbose': -1,
              'n_jobs': -1,
              'device': 'gpu'}

scores = []
for idx, yr in enumerate(year_list[:-4]):
    
    train_year = year_list[:idx+2]
    test_year = year_list[(idx+2):(idx+5)]

    X_train = X[np.isin(X.year, train_year)]
    X_train = X_train.drop(columns=['year'], axis=1)

    X_test = X[np.isin(X.year, test_year)]
    X_test = X_test.drop(columns=['year'], axis=1)

    y_train = y[np.isin(X.year, train_year)]
    y_test = y[np.isin(X.year, test_year)]

    lgb_md = LGBMRegressor(**lgb_params).fit(X_train, y_train)
    lgb_pred = lgb_md.predict(X_test)
    
    mape_oof = mean_absolute_percentage_error(y_test, lgb_pred)
    scores.append(mape_oof)
    
    print('Fold', idx, '==> LGBM oof MAPE is ==>', mape_oof)

ts_avg_score = np.mean(scores)
ts_std_score = np.std(scores)
print("\n")
print(f"The average oof MAPE of the LGBM model is {ts_avg_score}")
print(f"The std oof MAPE of the LGBM model is {ts_std_score}")


%%time
lgb_params = {'objective': 'mape',
              'learning_rate': 0.05385753736096176,
              'n_estimators': 1000,
              'max_depth': 14,
              'reg_alpha': 9.383236949308191,
              'reg_lambda': 0.30428754302668776,
              'num_leaves': 50,
              'colsample_bytree': 0.7809567901629245,
              'verbose': -1,
              'n_jobs': -1,
              'device': 'gpu'}

scores = []
skf = GroupKFold(n_splits=7)
for i, (trn_idx, test_idx) in enumerate(skf.split(X, groups=X.year)):
    
    X_train, X_test = X.iloc[trn_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[trn_idx], y.iloc[test_idx]

    X_train = X_train.drop(columns=['year'], axis=1)
    X_test = X_test.drop(columns=['year'], axis=1)

    lgb_md = LGBMRegressor(**lgb_params).fit(X_train, y_train)
    lgb_pred = lgb_md.predict(X_test)

    mape_oof = mean_absolute_percentage_error(y_test, lgb_pred)
    scores.append(mape_oof)
    
    print('Group', i, '==> LGBM oof MAPE is ==>', mape_oof)

gk_avg_score = np.mean(scores)
gk_std_score = np.std(scores)
print("\n")
print(f"The average oof MAPE of the LGBM model is {gk_avg_score}")
print(f"The std oof MAPE of the LGBM model is {gk_std_score}")


%%time
lgb_params = {'learning_rate': 0.07049928250360378,
              'n_estimators': 1000,
              'max_depth': 12,
              'reg_alpha': 0.01260164540047986,
              'reg_lambda': 5.6849501092111305,
              'num_leaves': 82,
              'colsample_bytree': 0.689643373301433,
              'verbose': -1,
              'n_jobs': -1,
              'device': 'gpu'}

test_cv = test.drop(columns='year', axis=1).copy()

scores, test_preds = [], []
skf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
for i, (trn_idx, test_idx) in enumerate(skf.split(X, groups=X.year)):
    
    X_train, X_test = X.iloc[trn_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[trn_idx], y.iloc[test_idx]

    X_train = X_train.drop(columns=['year'], axis=1)
    X_test = X_test.drop(columns=['year'], axis=1)

    lgb_md = LGBMRegressor(**lgb_params).fit(X_train, y_train)
    lgb_pred = lgb_md.predict(X_test)

    mape_oof = mean_absolute_percentage_error(y_test, lgb_pred)
    scores.append(mape_oof)
    
    print('Fold', i, '==> LGBM oof MAPE is ==>', mape_oof)

    test_preds.append(lgb_md.predict(test_cv))

kf_avg_score = np.mean(scores)
kf_std_score = np.std(scores)
print("\n")
print(f"The average oof MAPE of the LGBM model is {kf_avg_score}")
print(f"The std oof MAPE of the LGBM model is {kf_std_score}")


submission = pd.read_csv('../input/playground-series-s5e1/sample_submission.csv')
submission['num_sold'] = np.expm1(np.mean(test_preds, axis=0))
print(submission.head())

submission.to_csv('baseline_sub_1.csv', index=False)

