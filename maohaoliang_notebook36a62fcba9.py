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


import sys
print(sys.version)


# 念のため train/test/submission の読み込みをもう一度
import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s4e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e7/sample_submission.csv')


train = pd.read_csv('/kaggle/input/playground-series-s4e7/train.csv')
train.head()


# 1. ターゲットの分布
print("Response value counts:")
print(train['Response'].value_counts())
print("\nResponse割合:")
print(train['Response'].value_counts(normalize=True))

# 2. 欠損値の確認
print("\n欠損値の数（カラムごと）:")
print(train.isnull().sum())


#文字のままでは使えないカテゴリ変数を学習モデルで使用できるように変形
from sklearn.preprocessing import LabelEncoder

# Gender: Male → 1, Female → 0
train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0})
test['Gender'] = test['Gender'].map({'Male': 1, 'Female': 0})

# Vehicle_Damage: Yes → 1, No → 0
train['Vehicle_Damage'] = train['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
test['Vehicle_Damage'] = test['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

# Vehicle_Age: 順序があるのでラベルで数値化
vehicle_age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
train['Vehicle_Age'] = train['Vehicle_Age'].map(vehicle_age_mapping)
test['Vehicle_Age'] = test['Vehicle_Age'].map(vehicle_age_mapping)

# 変換後の確認
train[['Gender', 'Vehicle_Damage', 'Vehicle_Age']].head()


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 目的変数と特徴量
target = 'Response'
features = [col for col in train.columns if col not in ['id', target]]

# train/valid に分割（正しいコード！）
X_train, X_valid, y_train, y_valid = train_test_split(
    train[features], train[target], test_size=0.2, random_state=42
)

# LightGBM用データセットに変換
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# ハイパーパラメータ（シンプルでOK）
params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
}

# 学習
from lightgbm import early_stopping, log_evaluation

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=100,
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=10)
    ]
)

# 検証用のAUCスコア
y_pred = model.predict(X_valid)
auc = roc_auc_score(y_valid, y_pred)
print(f"AUCスコア: {auc:.4f}")


# testデータに対する予測（確率で出力される）
test_preds = model.predict(test[features])


# 提出用データフレームを作成
submission = sample_submission.copy()
submission['Response'] = test_preds

# ファイルとして保存（Kaggle Notebook内で保存）
submission.to_csv('submission.csv', index=False)

