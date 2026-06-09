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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import math


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

train.head(10)
# test.head(10)


print(train.columns)
# print(test.columns)

# つづりが違うので修正
train = train.rename(columns={'Temparature': 'Temperature'})
test = test.rename(columns={'Temparature': 'Temperature'})

# 目的変数と分離 ついでにidも
train_x = train.drop(["Fertilizer Name", "id"], axis = 1)
train_y = train["Fertilizer Name"]
test_x = test.drop(["id"], axis = 1)

train_x.head(10)


# 何が入ってるのか
soil_types = train["Soil Type"].unique()
crop_types = train["Crop Type"].unique()
fertilizer_name = train_y.unique()
print(soil_types)
print(crop_types)
print(fertilizer_name)

# 一応確認だけ
soil_types = test["Soil Type"].unique()
crop_types = test["Crop Type"].unique()
print(soil_types)
print(crop_types)




train_x.describe()


from sklearn.preprocessing   import LabelEncoder

# 一旦無視ラベルエンコーディングだけしておく
le = LabelEncoder()
train_x['Soil Type'] = le.fit_transform(train_x['Soil Type'])
train_x['Crop Type'] = le.fit_transform(train_x['Crop Type'])
test_x['Soil Type'] = le.fit_transform(test_x['Soil Type'])
test_x['Crop Type'] = le.fit_transform(test_x['Crop Type'])


train_x.head(10)


test_x.head(10)


from sklearn.model_selection import train_test_split
import xgboost as xgb


# 分割
X_tr, X_val, y_tr, y_val = train_test_split(
    train_x, train_y, random_state=71
)

# 目的変数もラベルエンコーディング
le_y = LabelEncoder()
y_tr = le_y.fit_transform(y_tr)
y_val = le_y.transform(y_val)

# 一旦XGBoost
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le_y.classes_),
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_tr, y_tr)


# 各レコードごとのaverage precisionを計算する関数
K = 3
def apk(y_i_true, y_i_pred):
    # y_predがK以下の長さで、要素がすべて異なることが必要
    assert (len(y_i_pred) <= K)
    assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision

    return sum_precision / min(len(y_i_true), K)

def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])

print(le_y.classes_)

# proba = model.predict_proba(X_val)
# print(proba[0])

# top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
# print(top3_idx[0])

# y_pred = top3_idx.tolist() 
# print(y_pred[0])

# y_true = [[lbl] for lbl in y_val]
# print(y_true[0])

# val_score = mapk(y_true, y_pred)
# print(f"Hold-out Validation MAP@3: {val_score:.5f}")

X_tr_proba = model.predict_proba(X_tr)
X_tr_top3_idx = np.argsort(X_tr_proba, axis=1)[:, -3:][:, ::-1]
y_tr_pred = X_tr_top3_idx.tolist() 
y_tr_true = [[lbl] for lbl in y_tr]
tr_score = mapk(y_tr_true, y_tr_pred)
print(f"Hold-out Train MAP@3: {tr_score:.5f}")

proba = model.predict_proba(X_val)
top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
y_pred = top3_idx.tolist() 
y_true = [[lbl] for lbl in y_val]
val_score = mapk(y_true, y_pred)
print(f"Hold-out Validation MAP@3: {val_score:.5f}")


proba  = model.predict_proba(test_x)
top3 = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
pred_labels = le_y.inverse_transform(top3.flatten()).reshape(top3.shape)

submission = pd.DataFrame({
    'id': test['id'],  # test.csv に id カラムがある想定
    'Fertilizer Name': [' '.join(row) for row in pred_labels]
})

submission.to_csv('submission.csv', index=False)
print("create csv")

