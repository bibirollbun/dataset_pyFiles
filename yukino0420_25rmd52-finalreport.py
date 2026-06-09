# -----------------------------------
# ライブラリのインポート
# -----------------------------------
import pandas as pd
import numpy as np
import xgboost as xgb
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------------
# 学習データ、テストデータの読み込み
# -----------------------------------
# 学習データ、テストデータの読み込み
train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')

#print(train) #訓練データの表示
#print(test)  #テストデータの表示

# 学習データを特徴量と目的変数に分ける
train_x = train.drop(['Cover_Type'], axis=1)
train_y = train['Cover_Type']

# テストデータは特徴量のみなので、そのままでよい
test_x = test.copy()

# 変数Idを除外する
train_x = train_x.drop(['Id'], axis=1)
test_x = test_x.drop(['Id'], axis=1)


# -----------------------------------
# 欠損値の確認・補完
# -----------------------------------
nan_cols = train_x.columns[train_x.isna().any()]
print("⚠️NaN を含む変数一覧：", nan_cols.tolist())


# -----------------------------------
# 変数の分析
# -----------------------------------
# 値の種類が1種類しかない変数
uninformative_cols = [col for col in train_x.columns if train_x[col].nunique() == 1]
print("⚠️Potentially uninformative variables......：", uninformative_cols)

# Soil_Type 系の出現回数
soil_cols = [col for col in train_x.columns if 'Soil_Type' in col]
for col in soil_cols:
    count = train_x[col].sum()
    if count < 10:
        print(f"⚠️{col}：{count} times → There is a risk of overfitting!!!")


# -----------------------------------
# Wilderness AreaごとのCover_Typeの分布
# -----------------------------------
# カテゴリ変数への統合（0〜3）
train_x['Wilderness_Area_cat'] = train_x[[f'Wilderness_Area{i}' for i in range(1, 5)]].idxmax(axis=1).str.extract('(\d)').astype(int)
# 分布の表示（0〜3）
sns.countplot(data=train_x.join(train_y), x='Wilderness_Area_cat', hue='Cover_Type')
plt.title("Distribution of Cover_Type against Wilderness Area")
plt.show()
# 変数を削除
train_x = train_x.drop(['Wilderness_Area_cat'], axis=1)


# -----------------------------------
# データを正規化
# -----------------------------------
cols_to_normalize = ['Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
                     'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points',
                    ]
train_x[cols_to_normalize] = normalize(train_x[cols_to_normalize])
test_x[cols_to_normalize]  = normalize(test_x[cols_to_normalize])


# -----------------------------------
# 特徴量作成
# -----------------------------------
## ここより下を自由に書き換える
## *** Elevationをbinning ***
train_x['Elevation_bin'] = [math.floor(v/50.0) for v in train_x['Elevation']]
test_x['Elevation_bin']  = [math.floor(v/50.0) for v in test_x['Elevation']]
## *** Horizontal_Distance_To_Roadwaysにlogをとる ***
train_x['Horizontal_Distance_To_Roadways_Log'] = np.log1p(train_x['Horizontal_Distance_To_Roadways'])
test_x['Horizontal_Distance_To_Roadways_Log']  = np.log1p(test_x['Horizontal_Distance_To_Roadways'])

## +++ Soil_Typeの合成変数を追加 +++
train_x['Soil_Type12_32'] = train_x['Soil_Type32'] + train_x['Soil_Type12']
test_x['Soil_Type12_32']  = test_x['Soil_Type32'] + test_x['Soil_Type12']
train_x['Soil_Type23_22_32_33'] = train_x['Soil_Type23'] + train_x['Soil_Type22'] + train_x['Soil_Type32'] + train_x['Soil_Type33']
test_x['Soil_Type23_22_32_33']  = test_x['Soil_Type23'] + test_x['Soil_Type22'] + test_x['Soil_Type32'] + test_x['Soil_Type33']

#print("infの数：", np.isinf(train_x).sum().sum())
#print(train_x.dtypes[train_x.dtypes == 'object'])


# -----------------------------------
# 変数の削除
# -----------------------------------
del_col = ['Aspect', #'Elevation', 'Slope', 
#           'Horizontal_Distance_To_Hydrology', 
#           'Vertical_Distance_To_Hydrology',
#           'Horizontal_Distance_To_Roadways',
#           'Wilderness_Area2', 
#           'Hillshade_Noon', 'Hillshade_3pm',  
#           'Soil_Type7', 'Soil_Type8',
#           'Soil_Type22', 'Soil_Type23', 'Soil_Type32',
          ]
for col in del_col:
    train_x = train_x.drop([col], axis=1)
    test_x  = test_x.drop([col], axis=1)

# Soil_Typeの変数を削除       
#soil_cols = [c for c in train_x.columns if "Soil_Type" in c]
#for col in soil_cols:
#    train_x = train_x.drop([col], axis=1)
#    test_x  = test_x.drop([col], axis=1)

print("List of the features：")
print(train_x.columns.tolist())


# -----------------------------------
# メインモデル学習と評価
# -----------------------------------
# 学習・検証用に分割
tsize = 0.2
rseed = 71
x, x_val, y, y_val = train_test_split(train_x, train_y, test_size=tsize, random_state=rseed)

# モデルの学習
model_main = RandomForestClassifier(n_estimators=100, random_state=rseed)
model_main.fit(x, y)

# 検証精度の計算
y_pred_main = model_main.predict(x_val)
accuracy_main = accuracy_score(y_val, y_pred_main)
print(f'Main Model Accuracy：{accuracy_main:.4f}') 
# Main Model Accuracy：0.8661 (Baseline_Model)
# Main Model Accuracy：0.8737


# -----------------------------------
# サブモデル（1と2のみ）学習と評価
# -----------------------------------
# 学習用
mask_train_bin = y.isin([1, 2])
x_bin = x[mask_train_bin].copy()
y_bin = y[mask_train_bin].copy()

# 検証用
mask_val_bin = y_val.isin([1, 2])
x_val_bin = x_val[mask_val_bin].copy()
y_val_bin = y_val[mask_val_bin].copy()

# モデル学習
model_bin = RandomForestClassifier(n_estimators=100, random_state=rseed)
model_bin.fit(x_bin, y_bin)
y_pred_bin = model_bin.predict(x_val_bin)
accuracy_bin = accuracy_score(y_val_bin, y_pred_bin)
print(f"Sub Model Accuracy for Class1&2：{accuracy_bin:.4f}")
# Sub Model Accuracy for Class1&2：0.8227


# -----------------------------------
# 予測上書き（補助モデルの反映）
# -----------------------------------
suspect_idx = np.where((y_pred_main == 1) | (y_pred_main == 2))[0]
x_val_suspect = x_val.iloc[suspect_idx]
y_pred_bin_suspect = model_bin.predict(x_val_suspect)

# 確信度の高いものだけ上書き
proba = model_bin.predict_proba(x_val_suspect)
conf_mask = np.max(proba, axis=1) > 0.85  # 閾値調整可
y_pred_bin_confident = model_bin.predict(x_val_suspect[conf_mask])
# インデックスを取得して置換
replace_idx = suspect_idx[conf_mask]
y_pred_main[replace_idx] = y_pred_bin_confident

# 上書き後の精度
final_accuracy = accuracy_score(y_val, y_pred_main)
print(f"Final Accuracy After Overwrite：{final_accuracy:.4f}")
# Final Accuracy After Overwrite：0.8737


# -----------------------------------
# 提出用CSVの作成
# -----------------------------------
# テストデータを予測
test_ids = test['Id']

# テストデータに対するメインモデル予測
y_pred_test = model_main.predict(test_x)

# メイン予測が 1 または 2 のインデックスを抽出
suspect_idx_test = np.where((y_pred_test == 1) | (y_pred_test == 2))[0]
# 対象データをサブモデルで再予測
x_suspect_test = test_x.iloc[suspect_idx_test]
y_pred_test_bin = model_bin.predict(x_suspect_test)

# 確信度の高いものだけ上書き
proba = model_bin.predict_proba(x_suspect_test)
conf_mask = np.max(proba, axis=1) > 0.85  # ここは調整してよい
pred_bin_confident = model_bin.predict(x_suspect_test[conf_mask])
# インデックスを取得して置換
replace_idx = suspect_idx_test[conf_mask]
y_pred_test[replace_idx] = pred_bin_confident

#print('test_x : ' + int(len(test_x)) + ', y_pred : ' + int(len(y_pred_test)))

submission = pd.DataFrame({
    "Id": test_ids, 
    "Cover_Type": y_pred_test
})
submission.to_csv("submission.csv", index=False)
print('-- Submission is completed! --') 

