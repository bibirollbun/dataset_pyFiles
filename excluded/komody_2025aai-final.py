import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

rseed = 71


# ----------------------------------------------
# データ読み込み
# ----------------------------------------------
DATA_DIR = "/kaggle/input/forest-cover-type-prediction"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

# 元テーブルからID列と目的変数をコピー
train_id = train['Id'].copy()
test_id  = test['Id'].copy()
train_y  = train['Cover_Type'].copy()

# ----------------------------------------------
# 欠損値の確認 今回は無かったのでコメントアウト
# ----------------------------------------------
# 各列の欠損値数を計算
# missing_train_counts = train.isnull().sum()
# missing_test_counts = test.isnull().sum()

# # 欠損がある列だけフィルタ
# missing_train_cols = missing_train_counts[missing_train_counts > 0]
# missing_test_cols = missing_test_counts[missing_test_counts > 0]
# print(missing_train_cols)
# print(missing_test_cols)


# ----------------------------------------------
# メインテーブルの特徴量作成
# ----------------------------------------------

# # 元テーブルから特徴量をコピー（目的変数は削除）
train_x = train.copy().drop(columns=["Cover_Type"])
test_x  = test.copy()


# ----------------------------------------------
# クラスタリング傾向の確認用
# ----------------------------------------------
# from IPython.display import Image

# def plotc(c1,c2):

#     fig = plt.figure(figsize=(16,8))
#     sel = np.array(list(train.Cover_Type.values))

#     plt.scatter(c1, c2, c=sel, s=100)
#     plt.xlabel(c1.name)
#     plt.ylabel(c2.name)
    
# plotc(train.Elevation, train.Slope)
# plotc(train.Elevation-train.Slope*0.2, train.Slope)


# -----------------------------------------------------------
# 変数 前処理
# -----------------------------------------------------------
col = 'Aspect'

# 新特徴「Aspect2」を作成
new_col = 'Aspect2'
# 0°から360°までの偏りを無くすための変換関数
def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180

train_x[new_col] = train[col].map(r)
test_x[new_col] = test[col].map(r)

# 未加工のAspectは使用しないので削除
train_x = train_x.drop([col], axis=1)
test_x = test_x.drop([col], axis=1)


# -----------------------------------------------------------
# 変数 前処理
# -----------------------------------------------------------

# 新特徴「Slope_bin」を作成 
# 平坦/緩傾斜/中傾斜/急傾斜のカテゴリで非線形な閾値効果を明示的にモデルへ

bins = [0, 15, 30, 45, train_x['Slope'].max()] #傾 斜の角度はハイパーパラメータ
labels = ['flat','gentle','moderate','steep'] # 平坦, 緩傾斜, 中傾斜, 急傾斜

# pd.cut で 'Slope' の連続値を上記ビンに区分し、'Slope_bin' 列にカテゴリラベルとして格納
# pd.get_dummies でカテゴリ列 'Slope_bin' を one-hot エンコーディング
train_x['Slope_bin'] = pd.cut(train_x['Slope'], bins=bins, labels=labels)
train_x = pd.get_dummies(train_x, columns=['Slope_bin'])

test_x['Slope_bin'] = pd.cut(test_x['Slope'], bins=bins, labels=labels)
test_x = pd.get_dummies(test_x, columns=['Slope_bin'])

# 未加工のSlopeは使用しないので削除
train_x = train_x.drop(['Slope'], axis=1)
test_x = test_x.drop(['Slope'], axis=1)


# -----------------------------------------------------------
# 変数 前処理
# -----------------------------------------------------------

# 新特徴「EVDtH」,「EHDtH」を作成
# Elevation(標高)とVertical_Distance_To_Hydorology(VDtH, 最も近い水域までの垂直距離)にクラスごとのクラスタリング傾向があるかを可視化
# →Elevation = 1(傾き) × VDtH + 定数という関係上に集まっているように見える
# →よってこの定数を特徴量として作成。
# ElevationとHorizontal_Distance_To_Hydorology(HDtH, 最も近い水域までの水平距離)にクラスごとのクラスタリング傾向があるかを可視化
# →Elevation = 0.2(傾き) × HDtH + 定数という関係上に集まっているように見える
# →よってこの定数を特徴量として作成。

train_x['EVDtH'] = train['Elevation']-train['Vertical_Distance_To_Hydrology']
test_x['EVDtH'] = test['Elevation']-test['Vertical_Distance_To_Hydrology']

train_x['EHDtH'] = train['Elevation']-train['Horizontal_Distance_To_Hydrology']*0.2
test_x['EHDtH'] = test['Elevation']-test['Horizontal_Distance_To_Hydrology']*0.2


# -----------------------------------------------------------
# 変数 前処理
# -----------------------------------------------------------

# 新特徴量を作成
# Hydro_Fire_1…最も近い水域までの水平距離と火点(過去に火災があった地点)までの距離の合計。
# →値が小さいほど「水源にも火点にも近い」→湿地帯かつ火災リスクの高いエリア
# →値が大きいほど「水源にも火点にも遠い」→乾燥地帯で火災リスクが低いエリア
# 
# Hydro_Fire_2…最も近い水域までの水平距離と火点までの距離の差の絶対値
# →値が小さいほど「水源への距離と火点への距離がほぼ同じ」→湿地帯かつ火点に近いエリア
# →値が大きいほど「どちらかに近くどちらかに遠い」→水源と火点の距離の差が大きいエリア
# 
# 同様にHydro_Road1,2やFire_Road1,2を作成している。
train_x['Hydro_Fire_1'] = train_x['Horizontal_Distance_To_Hydrology']+train_x['Horizontal_Distance_To_Fire_Points']
test_x['Hydro_Fire_1'] = test_x['Horizontal_Distance_To_Hydrology']+test_x['Horizontal_Distance_To_Fire_Points']

train_x['Hydro_Fire_2'] = abs(train_x['Horizontal_Distance_To_Hydrology']-train_x['Horizontal_Distance_To_Fire_Points'])
test_x['Hydro_Fire_2'] = abs(test_x['Horizontal_Distance_To_Hydrology']-test_x['Horizontal_Distance_To_Fire_Points'])

train_x['Hydro_Road_1'] = abs(train_x['Horizontal_Distance_To_Hydrology']+train_x['Horizontal_Distance_To_Roadways'])
test_x['Hydro_Road_1'] = abs(test_x['Horizontal_Distance_To_Hydrology']+test_x['Horizontal_Distance_To_Roadways'])

train_x['Hydro_Road_2'] = abs(train_x['Horizontal_Distance_To_Hydrology']-train_x['Horizontal_Distance_To_Roadways'])
test_x['Hydro_Road_2'] = abs(test_x['Horizontal_Distance_To_Hydrology']-test_x['Horizontal_Distance_To_Roadways'])

train_x['Fire_Road_1'] = abs(train_x['Horizontal_Distance_To_Fire_Points']+train_x['Horizontal_Distance_To_Roadways'])
test_x['Fire_Road_1'] = abs(test_x['Horizontal_Distance_To_Fire_Points']+test_x['Horizontal_Distance_To_Roadways'])

train_x['Fire_Road_2'] = abs(train_x['Horizontal_Distance_To_Fire_Points']-train_x['Horizontal_Distance_To_Roadways'])
test_x['Fire_Road_2'] = abs(test_x['Horizontal_Distance_To_Fire_Points']-test_x['Horizontal_Distance_To_Roadways'])


# -----------------------------------------------------------
# 変数 前処理
# -----------------------------------------------------------
col = 'Horizontal_Distance_To_Roadways'

# 新特徴「Horizontal_Distance_To_Roadways_Log」を作成 
# →裾が重い分布なのでログ変換することで大きな値の影響を抑制してくれるように特徴量を生成。
new_col = 'Horizontal_Distance_To_Roadways_Log'

train_x[new_col] = np.log1p(train_x[col])  # log( x + 1 )
test_x[new_col]  = np.log1p(test_x[col])

# # 未加工のHorizontal_Distance_To_Roadwaysは使用しないので削除
train_x = train_x.drop([col], axis=1)
test_x = test_x.drop([col], axis=1)


# -----------------------------------------------------------
# 変数 前処理
# -----------------------------------------------------------
col = 'Horizontal_Distance_To_Hydrology'

# 新特徴「Horizontal_Distance_To_Hydrology_Log」を作成 
# →裾が重い分布なのでログ変換することで大きな値の影響を抑制してくれるように特徴量を生成。
new_col = 'Horizontal_Distance_To_Hydrology_Log'

train_x[new_col] = np.log1p(train_x[col])  # log( x + 1 )
test_x[new_col]  = np.log1p(test_x[col])

# 未加工のHorizontal_Distance_To_Hydrologyは使用しないので削除
train_x = train_x.drop([col], axis=1)
test_x = test_x.drop([col], axis=1)


# print(train_x.head(3)) # 最初の3つを表示(テスト用)

# ID列を削除
train_x = train_x.drop(columns=["Id"])
test_x  = test_x.drop(columns=["Id"])


# ----------------------------------------------
# 使用する特徴量の可視化
# ----------------------------------------------

# 各特徴の統計量
print(pd.concat([train_x, train_y], axis=1).describe())

# 各特徴のヒストグラム
train_x.hist(bins=100, color="blue", grid=True, label='pandas')
plt.show()


# # ヒストグラムを見る用のコード
# # 数値列だけを対象にする（カテゴリ列を除外）
# num_cols = train_x.select_dtypes(include=['number']).columns

# for col in num_cols:
#     plt.figure(figsize=(10, 4))            # 図を大きく
#     train_x[col].hist(
#         bins=100,
#         grid=True
#     )
#     plt.title(col)
#     plt.xlabel(col)
#     plt.ylabel("Count")
#     plt.tight_layout()
#     plt.show()



# ------------------------------
# XGBoostの学習・推論・submit
# ------------------------------

# 1〜7 → 0〜6 に変換
train_y0 = train_y - 1

model = XGBClassifier(
    random_state=rseed,
    use_label_encoder=False, #最近のバージョンではFalse推奨
)
model.fit(train_x, train_y0)
preds = model.predict(test_x)

# 予測（0〜6 が返ってくるので +1 して元に戻す）
raw_preds = model.predict(test_x)
preds = raw_preds + 1

submission = pd.DataFrame({
    "Id": test_id,
    "Cover_Type":  preds
})

submission.to_csv("submission.csv", index=False)

