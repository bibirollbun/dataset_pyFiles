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


!pip install lifelines


import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index



train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
train_data.head(10)


for col in train_data.columns:
    print(f"Column: {col}")
    print(train_data[col].unique())
    print("-" * 30)


train_data.info()


train_data.isnull().any(axis=0)


# 数値カラムとカテゴリカラムを分ける
num_cols = train_data.select_dtypes(include=['number']).columns  # 数値カラム
cat_cols = train_data.select_dtypes(include=['object']).columns  # カテゴリカラム
for col in cat_cols:
    print(f"{col}: {train_data[col].nunique()}種類")

# 数値カラムの NaN を平均値で補完
train_data[num_cols] = train_data[num_cols].apply(lambda x: x.fillna(x.mean()))

# カテゴリカラムの NaN を 'Missing' で補完
train_data[cat_cols] = train_data[cat_cols].fillna('Missing')



train_data.isnull().any(axis=0)


# 数値カラムの処理（例：標準化）
train_data[num_cols] = (train_data[num_cols] - train_data[num_cols].mean()) / train_data[num_cols].std()

# カテゴリカラムの処理（例：One-Hot Encoding）
train_data = pd.get_dummies(train_data, columns=cat_cols, drop_first=True)


train_data.info()


# 目的変数としてefsとefs_timeを用いる
# そのほかのカラム（特徴量）を選択する
X_train = train_data.drop(columns=['efs', 'efs_time'])
Y_train = train_data[['efs', 'efs_time']]



# Cox比例ハザードモデルを使ってフィッティング
cph = CoxPHFitter()
cph.fit(train_data, duration_col='efs_time', event_col='efs')



# モデルのフィッティング後、予測を行う
# 新しいデータを予測する場合
new_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')  # 予測に使用するデータ
#risk_scores = cph.predict_partial_hazard(new_data)




# 数値カラムとカテゴリカラムを分ける
num_cols_2 = new_data.select_dtypes(include=['number']).columns  # 数値カラム
cat_cols_2 = new_data.select_dtypes(include=['object']).columns  # カテゴリカラム
#print("カテゴリカラム:", list(cat_cols_2))
for col in cat_cols_2:
    print(f"{col}: {new_data[col].nunique()}種類")

# 数値カラムの NaN を平均値で補完
new_data[num_cols_2] = new_data[num_cols_2].apply(lambda x: x.fillna(x.mean()))

# カテゴリカラムの NaN を 'Missing' で補完
new_data[cat_cols_2] = new_data[cat_cols_2].fillna('Missing')


# 学習データのカラムを取得
train_columns = X_train.columns
new_data_columns = new_data.columns
#print("カテゴリカラム:", list(cat_cols_2))

print("カラムの数:", train_data.shape[1])
print("カラムの数:", new_data.shape[1])


# 数値カラムの処理（例：標準化）
new_data[num_cols_2] = (new_data[num_cols_2] - new_data[num_cols_2].mean()) / new_data[num_cols_2].std()

# カテゴリカラムの処理（例：One-Hot Encoding）
new_data = pd.get_dummies(new_data, columns=cat_cols_2, drop_first=True)


# 訓練データにあったカラムを評価データにも追加（なければ0を入れる）
for col in train_columns:
    if col not in new_data.columns:
        new_data[col] = 0  # 訓練データにあったカラムが評価データにない場合、0を補完

# 訓練データと評価データのカラム順を揃える
#new_data = new_data[train_columns]

#print("評価データのカラム:", list(new_data.columns))



# # 説明変数（target_col 以外のすべてのカラム）
# X_2 = test_data.drop(columns=['ID'])


# 学習データのカラムを取得
train_columns = X_train.columns
new_data_columns = new_data.columns
#print("カテゴリカラム:", list(cat_cols_2))

print("カラムの数:", train_data.shape[1])
print("カラムの数:", new_data.shape[1])

# 予測用データが学習データのカラムと一致するかを確認
#new_data = new_data[train_columns]
#new_data = new_data[train_data.drop(columns=['efs_time', 'efs']).columns]


risk_scores = cph.predict_partial_hazard(new_data)


# リスクスコアを表示
print(risk_scores)



# モデルの評価（必要に応じて）
c_index = concordance_index(Y_train['efs_time'], -cph.predict_partial_hazard(X_train), Y_train['efs'])
print(f"C-index: {c_index}")





# # 目的変数を 'price' に設定
# target_col = 'efs'

# # 説明変数（target_col 以外のすべてのカラム）
# X = train_data.drop(columns=['efs', 'ID', 'efs_time'])

# # 目的変数
# y = train_data[target_col]

# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print("説明変数 X:")
# print(X)
# print("\n目的変数 y:")
# print(y)


# params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.1,
#     'num_leaves': 31,
#     'verbose': -1
# }

# #lgb_train = lgb.Dataset(X_train, Y_train)
# #lgb_test = lgb.Dataset(X_test, Y_test, reference=lgb_train)
# lgb_train = lgb.Dataset(X_train.values, label=Y_train)
# lgb_test = lgb.Dataset(X_test.values, label=Y_test, reference=lgb_train)

# model = lgb.train(params, lgb_train, valid_sets=[lgb_test], valid_names=["valid"], num_boost_round=100)


# pred_test = model.predict(X_test)
# mse = mean_squared_error(Y_test, pred_test)
# print(f'Mean Squared Error (MSE): {mse}')


# test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
# test_data.head()


# # 数値カラムとカテゴリカラムを分ける
# num_cols_2 = test_data.select_dtypes(include=['number']).columns  # 数値カラム
# cat_cols_2 = test_data.select_dtypes(include=['object']).columns  # カテゴリカラム

# # 数値カラムの NaN を平均値で補完
# test_data[num_cols_2] = test_data[num_cols_2].apply(lambda x: x.fillna(x.mean()))

# # カテゴリカラムの NaN を 'Missing' で補完
# test_data[cat_cols_2] = test_data[cat_cols_2].fillna('Missing')


# # 数値カラムの処理（例：標準化）
# test_data[num_cols_2] = (test_data[num_cols_2] - test_data[num_cols_2].mean()) / test_data[num_cols_2].std()

# # カテゴリカラムの処理（例：One-Hot Encoding）
# test_data = pd.get_dummies(test_data, columns=cat_cols_2, drop_first=True)


# # 目的変数を 'price' に設定
# #target_col = 'efs'

# # 説明変数（target_col 以外のすべてのカラム）
# X_2 = test_data.drop(columns=['ID'])



# new_predictions = model.predict(X_2)
# print(f'Prediction results for new data: {new_predictions}')

