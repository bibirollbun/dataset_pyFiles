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

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print(train.shape)
print(train.dtypes)
print(train.head())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# データ読み込み
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# 欠損値埋め
train['Drained_after_socializing'] = train['Drained_after_socializing'].fillna('No')
for col in ['Post_frequency', 'Friends_circle_size', 'Social_event_attendance', 'Going_outside', 'Time_spent_Alone']:
    train[col] = train[col].fillna(train[col].median())

# ターゲットエンコーディング
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])  # Extrovert→1, Introvert→0

# カテゴリ変数をエンコード
for col in ['Stage_fear', 'Drained_after_socializing']:
    train[col] = le.fit_transform(train[col])

# 特徴量・ターゲット分離
X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']

# 学習データ/検証データ分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル作成
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 精度確認
preds = model.predict(X_valid)
print(f'Accuracy: {accuracy_score(y_valid, preds):.4f}')


# テストデータ読み込み
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# 欠損値埋め
test['Drained_after_socializing'] = test['Drained_after_socializing'].fillna('No')
for col in ['Post_frequency', 'Friends_circle_size', 'Social_event_attendance', 'Going_outside', 'Time_spent_Alone']:
    test[col] = test[col].fillna(train[col].median())  # trainの中央値を使う

# カテゴリ変数エンコード
for col in ['Stage_fear', 'Drained_after_socializing']:
    test[col] = le.fit_transform(test[col])

# 特徴量選択
X_test = test.drop(['id'], axis=1)

# 予測
test_preds = model.predict(X_test)

# 逆変換（0 → Introvert, 1 → Extrovert）
submission_preds = le.inverse_transform(test_preds)

# 提出ファイル作成
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = submission_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)

print('submission.csv を作成しました')

