import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

rseed = 71


# ----------------------------------------------
# データ読み込み
# ----------------------------------------------
DATA_DIR = "/kaggle/input/elo-merchant-category-recommendation"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

# 元テーブルからID列と目的変数をコピー
train_card_id = train['card_id'].copy()
test_card_id  = test['card_id'].copy()
train_y       = train['target'].copy()


# ----------------------------------------------
# メインテーブルの特徴量作成
# ・年月文字列は欠損補完・Label Encoding
# ・他はそのまま
# ----------------------------------------------

# 元テーブルから特徴量をコピー（目的変数は削除）
train_x = train.copy().drop(columns=["target"])
test_x  = test.copy()

# 年月文字列に欠損があるので補完
train_x["first_active_month"] = train_x["first_active_month"].fillna("Unknown")
test_x["first_active_month"]  = test_x["first_active_month"].fillna("Unknown")

# 年月文字列をLabel Encoding
le = LabelEncoder()
all_months = pd.concat([train_x["first_active_month"], test_x["first_active_month"]])
le.fit(all_months)
train_x["fam_label"] = le.transform(train_x["first_active_month"])
test_x["fam_label"]  = le.transform(test_x["first_active_month"])

# 変換前の年月文字列を削除
train_x = train_x.drop(columns=["first_active_month"])
test_x  = test_x.drop(columns=["first_active_month"])


# ----------------------------------------------
# 追加テーブルの特徴量作成
# ----------------------------------------------
import numpy as np

# 追加テーブルの読み込み
new_trans = pd.read_csv(f"{DATA_DIR}/historical_transactions.csv")
#new_trans = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv")
#print(new_trans.describe())

# リーク確認
merged = new_trans.merge(train[['card_id', 'first_active_month']], on='card_id', how='left')
leak_mask = merged['purchase_date'] > merged['first_active_month']
print("リーク件数:", leak_mask.sum())

# データの数値化
new_trans = new_trans[new_trans['month_lag'] < 0]  # リーク防止のため、month_lagが負の行は削除
new_trans['authorized_flag'] = new_trans['authorized_flag'].map({'Y': 1, 'N': 0})
new_trans['category_1']      = new_trans['category_1'].map({'Y': 1, 'N': 0})

# 集約特徴の作成パターンを辞書化
agg_funcs = {
    # 正規化済みの購入金額（負値 = 顧客の支出）
    "purchase_amount": ["mean", "std", "min", "max"], 
    # 分割払いの分割回数
    "installments"   : ["min", "max", "first", "last"], 
    # 基準日からの経過月数
    "month_lag"      : ["min", "max", "first", "last"], 
    # 購入日時（countで空でない行を数える ⇔ 購入回数）  
    #"purchase_date"  : ["count"],    
    # 使用認証の有無  
    "authorized_flag": ["mean", "sum", "count", "nunique"],    
    # カテゴリ１  
    "category_1"     : ["mean", "sum", "count"],
    # カテゴリ２  
    "category_2"     : ["mean", "sum", "skew"],
}
# 例）"month_lag": ["min", "mean", "max"],
# 　　⇔　変数 "month_lag" について，同一IDの中で
# 　　　　最小値，平均値，最大値を取って集約特徴に追加する

# 全ての行を card_id の値でグループ化
# ⇒　上の集約パターンに従って実際に特徴を集約
agg_new = new_trans.groupby("card_id").agg(agg_funcs)

# 備考）上の集約特徴のテーブルは，列が MultiIndex の状態になっている
# 　　　⇒　列名のリストを表示して確認してみよう
print(agg_new.columns.to_list())
# 1列を指定するために「変数名と集約関数」をペアで指定する必要があることがわかる

# 新特徴の名前（列名）を機械的に生成してリスト化
# 　　命名規則：元変数名_集約方法（「purchase_amount_mean」など）
flat_cols = []
for col_name, func_name in agg_new.columns.to_flat_index():
    new_name = f"new_{col_name}_{func_name}"
    flat_cols.append(new_name)

# MultiIndex状態の列を通常の状態に変換（列方向に平坦化）
agg_new.columns = flat_cols
#print(agg_new.columns.to_list())

# groupby によるグループ化を解除（card_idを通常の列に戻す）
agg_new = agg_new.reset_index()

# メインテーブルの特徴と結合
train_x = train_x.merge(agg_new, on="card_id", how="left")
test_x  = test_x.merge(agg_new, on="card_id", how="left")

# Fill missing values created by the left join
train_x = train_x.fillna(-1)
test_x  = test_x.fillna(-1)


# ID列を削除
train_x = train_x.drop(columns=["card_id"])
test_x  = test_x.drop(columns=["card_id"])


# ----------------------------------------------
# 使用する特徴量の可視化
# ----------------------------------------------

# 各特徴の統計量
#print(pd.concat([train_x, train_y], axis=1).describe())

# 各特徴のヒストグラム
train_x.hist(bins=100, color="blue", grid=True, label='pandas')
plt.show()


# ------------------------------
# XGBoostの学習・推論・submit
# ------------------------------
model = XGBRegressor(random_state=rseed)
model.fit(train_x, train_y)
preds = model.predict(test_x)


# # 学習データ予測 & ValidationRMSE算出（過学習の心配があるためあくまで参考程度）
# import numpy as np
# from sklearn.metrics import mean_squared_error
# train_pred = model.predict(train_x)
# rmse = np.sqrt(mean_squared_error(train_y, train_pred))
# print(f"Train RMSE: {rmse:.5f}")  # Train RMSE: 3.64505

# 学習データ予測 & TrainRMSE算出
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
# データ分割
X_train, X_valid, y_train, y_valid = train_test_split(
    train_x, train_y, test_size=0.2, random_state=rseed
)
# モデル構築（n_estimatorsは700にする）
model = LGBMRegressor(
    objective='regression',
    learning_rate=0.01,
    num_leaves=31,
    n_estimators=700,  
    random_state=rseed,
    force_row_wise=True,
    verbose=-1
)
# 学習
model.fit(X_train, y_train)
# 検証予測 & RMSE算出
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Validation RMSE: {rmse:.5f}")   # RMSE: 3.80265, MinRMSE: 3.79886


submission = pd.DataFrame({
    "card_id": test_card_id,
    "target":  preds
})

submission.to_csv("submission.csv", index=False)
#Validation RMSE: 3.80026
#Validation RMSE: 3.77630

