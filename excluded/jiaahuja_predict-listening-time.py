# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np

import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=False)
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=False)


data.shape, test.shape


df = pd.concat([data, test], axis=0, ignore_index=True)

df = df.drop(['id'], axis = 1)
df = df.drop_duplicates()

# outlier removal
df['Episode_Length_minutes'] = np.maximum(0, np.minimum(120, df['Episode_Length_minutes']))
df['Host_Popularity_percentage'] = np.maximum(20, np.minimum(100, df['Host_Popularity_percentage']))
df['Guest_Popularity_percentage'] = np.maximum(0, np.minimum(100, df['Guest_Popularity_percentage']))
df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 4


day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
df['Publication_Day'] = df['Publication_Day'].map(day_mapping)

time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
df['Publication_Time'] = df['Publication_Time'].map(time_mapping)

sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)


def epi_number(title):
    return int(title[8:])

df['Episode_num'] = df['Episode_Title'].apply(epi_number)


df.head()


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


df['Number_of_Ads'] = df['Number_of_Ads'].fillna(0)


df['Number_of_Ads_ordinal'] = df['Number_of_Ads']


df['Have_Guest'] = ~df['Guest_Popularity_percentage'].isna()
podcasts_grouped = df.groupby(['Podcast_Name'])['Guest_Popularity_percentage']
df['Guest_Popularity_percentage'] = podcasts_grouped.transform(lambda guest_popularity_: guest_popularity_.fillna(guest_popularity_.mean()))


df['Guest_pop_median_filled'] = podcasts_grouped.transform(lambda guest_popularity_: guest_popularity_.fillna(guest_popularity_.median()))


ctr = ColumnTransformer([
    ('epl_impute', SimpleImputer(strategy='median'), ['Episode_Length_minutes']),
    ('epi_title', OrdinalEncoder(), ['Episode_Title']),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first'), ['Genre']),
    ('ord_genre', OrdinalEncoder(), ['Genre']),
    ('ord_podname', OrdinalEncoder(), ['Podcast_Name']),
    ('ohe_ads', OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first'), ['Number_of_Ads'])
    ], 
    remainder='passthrough', verbose_feature_names_out=False).set_output(transform="pandas") 


df = ctr.fit_transform(df)


df['Episode_Length_minutes_sqrt'] = np.sqrt(df['Episode_Length_minutes'])
df['Episode_Length_minutes_squared'] = df['Episode_Length_minutes'] ** 2
df['Episode_Length_minutes_log'] = np.log1p(df['Episode_Length_minutes'])


df['Podcast_target_mean'] = df.groupby(['Podcast_Name'])['Listening_Time_minutes'].transform('mean')
df['Podcast_target_median'] = df.groupby(['Podcast_Name'])['Listening_Time_minutes'].transform('median')


df['Podcast_Host_Popularity_mean'] = df.groupby(['Podcast_Name'])['Host_Popularity_percentage'].transform('mean')
df['Podcast_Guest_Popularity_mean'] = df.groupby(['Podcast_Name'])['Guest_Popularity_percentage'].transform('mean')
df['Podcast_Sentiment_mean'] = df.groupby(['Podcast_Name'])['Episode_Sentiment'].transform('mean')
df['Podcast_Episode_Length_mean'] = df.groupby(['Podcast_Name'])['Episode_Length_minutes'].transform('mean')
df['Podcast_Episode_Length_median'] = df.groupby(['Podcast_Name'])['Episode_Length_minutes'].transform('median')


group_cols = ['Episode_Sentiment', 'Genre', 'Publication_Day', 'Podcast_Name', 'Episode_Title',
              'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads_ordinal']

for col in group_cols:
    df[f"{col}_EP"] = df.groupby(col)['Episode_Length_minutes'].transform('mean')


binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
df['Episode_num_binned'] = binner.fit_transform(df[['Episode_num']])


binner = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
df['Host_pop_binned'] = binner.fit_transform(df[['Host_Popularity_percentage']])
df['Guest_pop_binned'] = binner.fit_transform(df[['Guest_Popularity_percentage']])


for strategy in ['quantile', 'uniform', 'kmeans']:
    for bins in [5, 10, 20]:
        binner = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
        df[f'Ep_Length_{strategy}_{bins}bin'] = binner.fit_transform(df[['Episode_Length_minutes']])


df


df_train = df.iloc[:-len(test)]
df_test = df.iloc[-len(test):].reset_index(drop=True)

df_train = df_train[df_train['Listening_Time_minutes'].notnull()]

y = df_train.pop('Listening_Time_minutes')
df_test.pop('Listening_Time_minutes')

df_train.shape, df_test.shape


print('fitting the model')


# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.2, random_state=42)


from scipy.stats import randint, uniform

# param_dist_tree = {
#     'n_estimators': randint(100, 1000),
#     'learning_rate': uniform(0.01, 0.3),
#     'max_depth': randint(3, 15),
#     'subsample': uniform(0.5, 0.5),
#     'colsample_bytree': uniform(0.5, 0.5),
#     'reg_alpha': uniform(0, 1),
#     'reg_lambda': uniform(0, 1),
#     'gamma': uniform(0, 5)
# }

param_dist_linear = {
    'n_estimators': randint(100, 1000),         # Still can have multiple boosting rounds
    'learning_rate': uniform(0.001, 0.2),              # Step size for updates
    'reg_alpha': uniform(0.0, 10.0),                   # L1 regularization on weights
    'reg_lambda': uniform(0.0, 10.0),
    'updater': ['shotgun', 'coord_descent'],    # Different optimizers for gblinear
}


from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# # Search for gbtree
# xgb_tree = XGBRegressor(objective='reg:squarederror', booster='gbtree', random_state=42)
# search_tree = RandomizedSearchCV(
#     estimator=xgb_tree,
#     param_distributions=param_dist_tree,
#     n_iter=20,
#     scoring='neg_mean_squared_error',
#     cv=5,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# Search for gblinear
# xgb_linear = XGBRegressor(objective='reg:squarederror', booster='gblinear', random_state=42)
# search_linear = RandomizedSearchCV(
#     estimator=xgb_linear,
#     param_distributions=param_dist_linear,
#     n_iter=20,
#     scoring='neg_mean_squared_error',
#     cv=5,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )


xgb_tree_params =  {'colsample_bynode': np.float64(0.7107258159646936), 'colsample_bytree': np.float64(0.9756787150848965), 
                    'gamma': np.float64(3.0351712384334233), 'learning_rate': np.float64(0.03759991820225434), 'max_depth': 16,
                    'min_child_weight': 15, 'n_estimators': 364, 'reg_alpha': np.float64(0.1563640674119393),
                    'reg_lambda': np.float64(4.234014807063696), 'subsample': np.float64(0.6974407590877849)}

xgb_linear_params =  {'learning_rate': 0.06184844859190755, 'n_estimators': 121, 'reg_alpha': 0.07066305219717406,
                      'reg_lambda': 0.23062425041415757, 'updater': 'shotgun'}


from lightgbm import LGBMRegressor

lgbm1 = {'colsample_bytree': np.float64(0.8480148983374864), 'learning_rate': np.float64(0.119012234017873), 'max_depth': 16, 'min_child_samples': 36, 'n_estimators': 953, 'num_leaves': 136, 'reg_alpha': np.float64(0.5183296523637367), 'reg_lambda': np.float64(0.8773730719279554), 'subsample': np.float64(0.8703843088771022)}
lgbm2 = {'colsample_bytree': np.float64(0.9961057796456088), 'learning_rate': np.float64(0.128496301925543), 'max_depth': 10, 'min_child_samples': 93, 'n_estimators': 660, 'num_leaves': 68, 'reg_alpha': np.float64(0.3998609717152555), 'reg_lambda': np.float64(0.04666566321361543), 'subsample': np.float64(0.9868777594207296)}
lgbm3 = {'colsample_bytree': np.float64(0.6626651653816322), 'learning_rate': np.float64(0.082735457937896), 'max_depth': 16, 'min_child_samples': 57, 'n_estimators': 891, 'num_leaves': 133, 'reg_alpha': np.float64(0.27599918202254337), 'reg_lambda': np.float64(0.2962735057040824), 'subsample': np.float64(0.5826334695315012)}


from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor

model = StackingRegressor(estimators=[ #('lgbm1', LGBMRegressor(objective='regression', random_state=42, **lgbm1)), 
#                                       ('lgbm2', LGBMRegressor(objective='regression', random_state=42, **lgbm2)),
#                                       ('lgbm3', LGBMRegressor(objective='regression', random_state=42, **lgbm3)),
                                      ('rf', RandomForestRegressor(random_state=42)),
                                      ('xgb_tree', XGBRegressor(booster='gbtree', random_state=42, **xgb_tree_params)),
                                      ('xgb_linear', XGBRegressor(booster='gblinear', random_state=42, **xgb_linear_params)),
                                      ('catboost', CatBoostRegressor(iterations=1000, loss_function='RMSE', early_stopping_rounds=50, random_seed=42))
                                     ])



model.fit(df_train, y)


print('predicting final results')


y_pred = model.predict(df_test)


print('submitting')


df_sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
df_sub.Listening_Time_minutes = y_pred
df_sub.to_csv('submission.csv', index=False)

