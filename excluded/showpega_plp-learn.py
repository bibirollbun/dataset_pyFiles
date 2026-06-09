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


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train_x = train.drop(['id','loan_paid_back'],axis=1)
train_y = train['loan_paid_back']

test_x = test.drop('id',axis = 1)


train_x_onehot = pd.get_dummies(train_x, columns = ['gender','marital_status','employment_status','loan_purpose'])
train_x_onehot

test_x_onehot = pd.get_dummies(test_x, columns = ['gender','marital_status','employment_status','loan_purpose'])


from sklearn.preprocessing import OrdinalEncoder

oe_col = ['education_level', 'grade_subgrade']

encoder = OrdinalEncoder(categories=[
    ["High School", "Bachelor's", "Master's", "PhD", "Other"], [
        "A1", "A2", "A3", "A4", "A5",
        "B1", "B2", "B3", "B4", "B5",
        "C1", "C2", "C3", "C4", "C5",
        "D1", "D2", "D3", "D4", "D5",
        "E1", "E2", "E3", "E4", "E5",
        "F1", "F2", "F3", "F4", "F5"
    ]
])

# コピーを作って新しいDataFrameに格納
train_x_encoded = train_x_onehot.copy()
test_x_encoded = test_x_onehot.copy()

# 指定カラムだけOrdinal Encodingを適用
train_x_encoded[oe_col] = encoder.fit_transform(train_x_encoded[oe_col])
test_x_encoded[oe_col] = encoder.transform(test_x_encoded[oe_col])

train_x_encoded


from sklearn.model_selection import train_test_split
import lightgbm as lgb

tr_x, va_x, tr_y, va_y = train_test_split(train_x_encoded, train_y, test_size = 0.2, random_state = 71, shuffle = True)

from sklearn.metrics import accuracy_score

# データをLightGBM用のデータセットに変換
lgb_train = lgb.Dataset(tr_x, label=tr_y)
lgb_valid = lgb.Dataset(va_x, label=va_y, reference=lgb_train)

# ハイパーパラメータの設定
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 71
}

# 学習の実行
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=1000,
)

# 検証データで予測
va_pred = model.predict(va_x, num_iteration=model.best_iteration)
va_pred_label = (va_pred > 0.5).astype(int)

# 精度の算出
score = accuracy_score(va_y, va_pred_label)
print(f'Accuracy: {score:.4f}')



test_pred = model.predict(test_x_encoded, num_iteration=model.best_iteration)

submission = pd.DataFrame({"id": test["id"], "target": test_pred})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv を出力しました！")
display(submission.head())


import joblib

# 1. LightGBMネイティブ形式で保存（おすすめ）
model.save_model("lgb_model.txt")

# 2. joblibでpickle形式でも保存（オプション）
# joblib.dump(model, "lgb_model.pkl")

print("✅ モデルを保存しました！（lgb_model.txt）")

