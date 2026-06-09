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


#データ設定と読み込み
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from datetime import date
import glob
import re


data_path = '/kaggle/input/recruit-restaurant-visitor-forecasting-data/'

air_visit = pd.read_csv(data_path + 'air_visit_data.csv')
air_store = pd.read_csv(data_path + 'air_store_info.csv')
date_info = pd.read_csv(data_path + 'date_info.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')
weather = pd.read_csv(data_path + 'WeatherData.csv')


submission


split_results = submission['id'].apply(lambda x: x.rsplit('_', 1))
submission['air_store_id'] = split_results.str[0]
submission['visit_date'] = split_results.str[1]
submission['visit_date'] = pd.to_datetime(submission['visit_date'], errors='coerce')


submission


#air_visitデータの整形
air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])
# visitorsを対数変換（RMSLEに合わせる）
air_visit['visitors'] = np.log1p(air_visit['visitors'])



air_visit


#date_infoのcalendar_dateをvisit_dateに変更
date_info = date_info.rename(columns={'calendar_date': 'visit_date'})
date_info['visit_date'] = pd.to_datetime(date_info['visit_date'])
date_info


# air_area_nameから都道府県を抽出（例: 'Tōkyō-to Minato-ku' -> 'Tōkyō'）
air_store['prefecture'] = air_store['air_area_name'].apply(lambda x: x.split('-')[0].split(' ')[0])
air_store


# カテゴリ変数の変換
le = LabelEncoder()
air_store['prefecture'] = le.fit_transform(air_store['prefecture'])
air_store['air_genre_name'] = le.fit_transform(air_store['air_genre_name'])

air_store


air_store['prefecture'].value_counts()


#気象データの統合と都道府県別集計
#WeatherDataから都道府県を抽出
weather['prefecture'] = weather['area_name'].apply(lambda x: x.split('_')[0])

#calendar_dateをvisit_dateに変更
#date_infoのcalendar_dateをvisit_dateに変更
weather = weather.rename(columns={'calendar_date': 'visit_date'})
weather['visit_date'] = pd.to_datetime(weather['visit_date'])

weather


#prefectureをグループ化して平均気温の平均値を計算
group_col1 = 'prefecture'
group_col2 = 'visit_date'
value_col = 'avg_temperature'
agg_col_name = f'mean_{value_col}_by_{group_col1}'
agg_df = weather.groupby([group_col1,group_col2])[value_col].mean().reset_index()
agg_df = agg_df.rename(columns={value_col: agg_col_name})
agg_df


# カテゴリ変数の変換
le = LabelEncoder()
agg_df['prefecture'] = le.fit_transform(agg_df['prefecture'])
agg_df


#trainデータとtestデータの作成
#air_visitとdate_infoをマージ
train_df = pd.merge(air_visit, date_info , on='visit_date', how='left')

#店舗情報をマージ
train_df = pd.merge(train_df, air_store, on='air_store_id',how='left')

#都道府県別気象データをマージ
train_df = pd.merge(train_df, agg_df, on=['prefecture','visit_date'], how='left')
train_df


#testデータも同様の処理
test_df = pd.merge(submission, date_info, on='visit_date', how='left')
test_df = pd.merge(test_df, air_store, on='air_store_id', how='left')
test_df = pd.merge(test_df, agg_df, on=['prefecture', 'visit_date'], how='left')
test_df


#欠損値の確認
print(train_df.isnull().sum())
print(test_df.isnull().sum())


#欠損値はなさそうなので共通の特徴量エンジニアリング
def feature_engineer(df):
    df['year'] = df['visit_date'].dt.year
    df['month'] = df['visit_date'].dt.month
    df['day'] = df['visit_date'].dt.day
    df['dayofweek'] = df['visit_date'].dt.dayofweek
    df['dayofyear'] = df['visit_date'].dt.dayofyear
    df['weekofyear'] = df['visit_date'].dt.isocalendar().week.astype(int)
    return df

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)


# air_store_idをカテゴリ変数として扱うための変換
le = LabelEncoder()
train_df['air_store_id'] = le.fit_transform(train_df['air_store_id'])
train_df['air_area_name'] = le.fit_transform(train_df['air_area_name'])
test_df['air_store_id'] = le.fit_transform(test_df['air_store_id'])
test_df['air_area_name'] = le.fit_transform(test_df['air_area_name'])



#使用する特徴量
features = [
    'air_store_id', 'air_area_name', 'air_genre_name',
    'holiday_flg', 'dayofweek', 'dayofyear', 'weekofyear', 'year', 'month', 'day',
    'prefecture'
]

target = 'visitors'



#最終的なデータセット
X = train_df[features]
y = train_df[target]
X_test = test_df[features]


X


#LightGBMモデルの訓練と予測
#パラメータ設定
params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,

    #カテゴリカル特徴量の指定
    #'categorical_feature':['name:air_store_id', 'name:air_area_name', 'name:air_genre_name', 'name:prefecture']
    
}


X_test


categorical_cols = ['air_store_id', 'air_genre_name', 'prefecture']
model = lgb.LGBMRegressor(**params)
model.fit(X,y,categorical_feature=categorical_cols)


test_df['visitors'] =model.predict(X_test)


#対数変換を元に戻す
test_df['visitors'] = np.expm1(test_df['visitors'])
test_df['visitors'] = test_df['visitors'].clip(lower=0)



#提出用csv
submission_df = test_df[['id', 'visitors']].copy()
submission_df['visitors'] = submission_df['visitors'].round().astype(int)
submission_df.to_csv('submission.csv', index=False)


low_count = len(submission_df)
print(low_count)


submission_df

