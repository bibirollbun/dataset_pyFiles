import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# -----------------------------------
# 学習データ、テストデータの読み込み
# -----------------------------------

# 学習データ、テストデータの読み込み
train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')

# 学習データを特徴量と目的変数に分ける
train_x = train.drop(['Cover_Type'], axis=1)
train_y = train['Cover_Type']

# テストデータは特徴量のみなので、そのまま
test_x = test.copy()


# -----------------------------------
# 特徴量作成
# -----------------------------------
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# IDを削除
train_x = train_x.drop(['Id'], axis=1)
test_x = test_x.drop(['Id'], axis=1)

# Wilderness_Areaをone-hotからlabelへ変換
onehot_columns_wa = []

for i in range(1, 5):
    column = 'Wilderness_Area' + str(i)
    onehot_columns_wa.append(column)

train_x['Wilderness_Area'] = train_x[onehot_columns_wa].values.argmax(axis=1)
test_x['Wilderness_Area'] = test_x[onehot_columns_wa].values.argmax(axis=1)

# 変換前の特徴量を削除
train_x = train_x.drop(onehot_columns_wa, axis=1)
test_x = test_x.drop(onehot_columns_wa, axis=1)


# Soil_Typeをone-hotからlabelへ変換
onehot_columns_st = []

for i in range(1, 41):
    column = 'Soil_Type' + str(i)
    onehot_columns_st.append(column)

train_x['Soil_Type'] = train_x[onehot_columns_st].values.argmax(axis=1)
test_x['Soil_Type'] = test_x[onehot_columns_st].values.argmax(axis=1)

# 変換前の特徴量を削除
train_x = train_x.drop(onehot_columns_st, axis=1)
test_x = test_x.drop(onehot_columns_st, axis=1)

# 地表水源までの直線距離
train_x['Distance_To_Hydrology'] = (train_x['Vertical_Distance_To_Hydrology']**2 + train_x['Horizontal_Distance_To_Hydrology']**2)**0.5
test_x['Distance_To_Hydrology'] = (test_x['Vertical_Distance_To_Hydrology']**2 + test_x['Horizontal_Distance_To_Hydrology']**2)**0.5

# 水源までの標高
train_x['Elevation_Hydrology'] = train_x['Elevation'] - train_x['Vertical_Distance_To_Hydrology']
test_x['Elevation_Hydrology'] = test_x['Elevation'] - test_x['Vertical_Distance_To_Hydrology']

# 水源と山火事
train_x['Hydrology_Fire_Points'] = train_x['Horizontal_Distance_To_Hydrology'] + train_x['Horizontal_Distance_To_Fire_Points']
test_x['Hydrology_Fire_Points'] = test_x['Horizontal_Distance_To_Hydrology'] + test_x['Horizontal_Distance_To_Fire_Points']

# 水源と道
train_x['Hydrology_Roadways'] = train_x['Horizontal_Distance_To_Hydrology'] - train_x['Horizontal_Distance_To_Roadways']
test_x['Hydrology_Roadways'] = test_x['Horizontal_Distance_To_Hydrology'] - test_x['Horizontal_Distance_To_Roadways']

# 山火事と道
train_x['Fire_Points_Roadways'] = train_x['Horizontal_Distance_To_Fire_Points'] - train_x['Horizontal_Distance_To_Roadways']
test_x['Fire_Points_Roadways'] = test_x['Horizontal_Distance_To_Fire_Points'] - test_x['Horizontal_Distance_To_Roadways']

# 各施設までの距離合計
train_x['Distance'] = train_x['Horizontal_Distance_To_Hydrology'] + train_x['Horizontal_Distance_To_Fire_Points'] + train_x['Horizontal_Distance_To_Roadways']
test_x['Distance'] = test_x['Horizontal_Distance_To_Hydrology'] + test_x['Horizontal_Distance_To_Fire_Points'] + test_x['Horizontal_Distance_To_Roadways']

# Elevationビン化
train_x['Elevation'] = pd.cut(train_x['Elevation'], 20, labels=False)
test_x['Elevation'] = pd.cut(test_x['Elevation'], 20, labels=False)

train_x.hist(bins=100, color = "blue", grid =True, label = 'pandas')
plt.show()

print(train_x)


# -----------------------------------
# モデル作成
# -----------------------------------
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# 目的変数を1~7から0~6へ
le = LabelEncoder()
train_y = le.fit_transform(train_y)

# 学習
model = XGBClassifier(random_state=71,)
model.fit(train_x, train_y)

train_pred_prob = model.predict_proba(train_x)
test_pred_prob = model.predict_proba(test_x)

# 目的変数を0~6から1~7へ
pred_label = model.predict(test_x)
pred_label = le.inverse_transform(pred_label)


# 特徴量重要度
import xgboost as xgb

# 重要度の上位を出力する
fscore = model.get_booster().get_score(importance_type='total_gain')
fscore = sorted([(k, v) for k, v in fscore.items()], key=lambda tpl: tpl[1], reverse=True)
print('xgboost importance')
for rank, (col, imp) in enumerate(fscore[:5]):
    print(f"Top {rank+1}: {imp} {col}")


# -----------------------------------
# ファイル提出
# -----------------------------------
submission = pd.DataFrame({'Id': test['Id'], 'Cover_Type': pred_label})
submission.to_csv('submission.csv', index=False)

