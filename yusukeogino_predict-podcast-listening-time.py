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

# FutureWarning を一時的に無視
warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd

train = pd.read_csv('../input/playground-series-s5e4/train.csv')
test = pd.read_csv('../input/playground-series-s5e4/test.csv')


train.head()


test.head()


# 音楽鑑賞時間の統計情報
train['Listening_Time_minutes'].describe()


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

plt.figure(figsize=(9,8))
sns.histplot(train['Listening_Time_minutes'], kde=True, color = 'g');


plt.figure(figsize=(8, 6))
sns.regplot(
    x='Listening_Time_minutes',
    y='Episode_Length_minutes',
    data=train,
    scatter_kws={'alpha': 0.5},     # 散布図の点の透明度
    line_kws={'color': 'red'},      # 回帰線の色
)
plt.title('Listening_Time vs Episode_Length')
plt.xlabel('Listening_Time')
plt.ylabel('Episode_Length')
plt.grid(True)
plt.show()


# 上記から、Episode_Lenthが300以上のものは、全体の特性からみて大きく外れており特異点であることがわかる。
# 学習データから除外する。
train[train['Episode_Length_minutes'] > 300]


train_filtered = train[~(train['Episode_Length_minutes'] > 300)]


train_filtered


train_filtered[(train_filtered['Episode_Length_minutes'] > 100) & (train_filtered['Listening_Time_minutes'] < 20)]


train_filtered = train_filtered[~((train_filtered['Episode_Length_minutes'] > 100) & (train_filtered['Listening_Time_minutes'] < 20))]


train_filtered[(train_filtered['Episode_Length_minutes'] < 75) & (train_filtered['Listening_Time_minutes'] > 90)]


train_filtered = train_filtered[~((train_filtered['Episode_Length_minutes'] < 75) & (train_filtered['Listening_Time_minutes'] > 90))]


train_filtered


plt.figure(figsize=(8, 6))
sns.regplot(
    x='Listening_Time_minutes',
    y='Episode_Length_minutes',
    data=train_filtered,
    scatter_kws={'alpha': 0.5, 's':10},     # 散布図の点の透明度
    line_kws={'color': 'red'},      # 回帰線の色
)
plt.title('Listening_Time vs Episode_Length')
plt.xlabel('Listening_Time')
plt.ylabel('Episode_Length')
plt.grid(True)
plt.show()


sns.lmplot(
    x='Episode_Length_minutes',
    y='Listening_Time_minutes',
    hue='Number_of_Ads',
    data=train_filtered,
    scatter_kws={'alpha': 0.2, 's': 10},
    height=6,
    aspect=2
)

plt.title('Number_of_Ads', fontsize=14)
plt.xlabel('Episode_Length_minutes', fontsize=12)
plt.ylabel('Listening_Time_minutes', fontsize=12)
plt.tight_layout()
plt.show()


train['Number_of_Ads'].value_counts()


train_filtered = train_filtered[train_filtered['Number_of_Ads'].isin([0, 1, 2, 3])]


sns.lmplot(
    x='Episode_Length_minutes',
    y='Listening_Time_minutes',
    hue='Genre',
    data=train_filtered,
    scatter_kws={'alpha': 0.1, 's': 10},
    height=6,
    aspect=2
)

plt.title('Genre', fontsize=14)
plt.xlabel('Episode_Length_minutes', fontsize=12)
plt.ylabel('Listening_Time_minutes', fontsize=12)
plt.tight_layout()
plt.show()


sns.lmplot(
    x='Episode_Length_minutes',
    y='Listening_Time_minutes',
    hue='Publication_Day',
    data=train_filtered,
    scatter_kws={'alpha': 0.1, 's': 10},
    height=6,
    aspect=2
)

plt.title('Publication_Day', fontsize=14)
plt.xlabel('Episode_Length_minutes', fontsize=12)
plt.ylabel('Listening_Time_minutes', fontsize=12)
plt.tight_layout()
plt.show()


sns.lmplot(
    x='Episode_Length_minutes',
    y='Listening_Time_minutes',
    hue='Publication_Time',
    data=train_filtered,
    scatter_kws={'alpha': 0.1, 's': 10},
    height=6,
    aspect=2
)

plt.title('Publication_Time', fontsize=14)
plt.xlabel('Episode_Length_minutes', fontsize=12)
plt.ylabel('Listening_Time_minutes', fontsize=12)
plt.tight_layout()
plt.show()


sns.lmplot(
    x='Episode_Length_minutes',
    y='Listening_Time_minutes',
    hue='Episode_Sentiment',
    data=train_filtered,
    scatter_kws={'alpha': 0.1, 's': 10},
    height=6,
    aspect=2
)

plt.title('Episode_Sentiment', fontsize=14)
plt.xlabel('Episode_Length_minutes', fontsize=12)
plt.ylabel('Listening_Time_minutes', fontsize=12)
plt.tight_layout()
plt.show()


train_X = train_filtered.drop('Listening_Time_minutes', axis=1)
Y = train_filtered['Listening_Time_minutes']


train_X["is_train"] = 1
train_X["is_test"] = 0
test["is_train"] = 0
test["is_test"] = 1

data = pd.concat([train_X, test], axis=0)


data.isnull().sum()[data.isnull().any()]


data.dtypes[data.isna().any()]


missing_col_list = data.columns[data.isna().any()].tolist()
for col in missing_col_list:
    if train[col].dtype == 'float64':
        data.fillna({col:data[col].median()}, inplace=True)


data


# カテゴリ変数を取得する関数
def _get_cate_features(df):
    feats = []
    for col in df.columns:
        if df[col].dtypes == 'object':
            feats.append(col)
    return feats

# カテゴリ変数のダミー変数 (二値変数化)を作成する関数
def _get_dummies(df, feats):
    for col in feats:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df

# 型が'object'のカテゴリを整数に置換するため、カテゴリ変数をファクトライズ (整数に置換)する関数を作成する
def _factorize_categoricals(df, feats):
    for col in feats:
        df[col], uniques = pd.factorize(df[col])
    return df 


cate_feats = _get_cate_features(data)
data = _get_dummies(data, cate_feats)
data = _factorize_categoricals(data, cate_feats)


# 最終的なtrainとtestデータを用意
ignore_features = ['is_train', 'is_test']
necessary_features = [col for col in data.columns if col not in ignore_features]
trainX = data[data['is_train'] == 1][necessary_features]
testX = data[data['is_test'] == 1][necessary_features]


Y


trainX


pip install lightgbm


from lightgbm import LGBMRegressor

model = LGBMRegressor(
    colsample_bytree= 0.8, 
    learning_rate= 0.04, 
    max_depth= 30, 
    min_child_samples= 30, 
    n_estimators= 160, 
    num_leaves= 6000, 
    subsample= 0.4,
    random_state=42   # 再現性のための乱数シード
)

model.fit(trainX, Y)

# テストデータで予測
y_pred = model.predict(testX)


submission = pd.DataFrame()
submission['id'] = testX['id']
submission['Listening_Time_minutes'] = y_pred
submission.to_csv("submission.csv", index=False)
submission = pd.read_csv("submission.csv")
submission


# """
# データを分割して、手元で精度を確認する
# """

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import pandas as pd

# # --- データを訓練用と検証用に分割 ---
# X_train, X_valid, y_train, y_valid = train_test_split(trainX, Y, test_size=0.2, random_state=42)


# from lightgbm import LGBMRegressor
# from sklearn.model_selection import GridSearchCV

# # --- モデルインスタンス ---
# model = LGBMRegressor(random_state=42)

# # --- チューニングするパラメータ ---
# param_grid = {
#     'num_leaves': [5000],
#     'max_depth': [30],
#     'learning_rate': [0.04],
#     'n_estimators': [160],
#     'min_child_samples': [30],
#     'subsample': [0.4],
#     'colsample_bytree': [0.8]
# }

# # --- GridSearchCVの設定 ---
# grid_search = GridSearchCV(estimator=model,
#                            param_grid=param_grid,
#                            scoring='neg_root_mean_squared_error',
#                            cv=3,
#                            verbose=2,
#                            n_jobs=-1)

# # --- モデルの学習 ---
# grid_search.fit(X_train, y_train)

# # --- 最適なパラメータの取得 ---
# print("Best Parameters:", grid_search.best_params_)

# # --- 最適モデルで予測・評価 ---
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_valid)
# rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
# print(f"Best LightGBM RMSE: {rmse:.4f}")





