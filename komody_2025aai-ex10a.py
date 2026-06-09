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


# # ----------------------------------------------
# # 追加テーブルの特徴量作成
# # ----------------------------------------------
# ctrl + / で複数行一気にコメントアウトの切り替えできる便利すぎる

# # 追加テーブルの読み込み
new_trans = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv")
print(new_trans.describe())

# category_1をラベルエンコーディング
new_trans['category_1'] = new_trans['category_1'].map({'Y': 1, 'N': 0})

# 集約特徴の作成パターンを辞書化
agg_funcs = {
    "purchase_amount":      ["min", "mean", "max"],# 正規化済みの購入金額（負値 = 顧客の支出）
    "installments":         ["min", "mean", "max"], # 分割払いの分割回数
    "month_lag":            ["min", "mean", "max", "var", "sem"], # 基準日からの経過月数
    "purchase_date":        ["count"],      # 購入日時（countで空でない行を数える ⇔ 購入回数）
    "subsector_id":         ["nunique"], # 加盟店カテゴリグループ（匿名化）
    "category_1":           ["mean", "sum", "nunique"],
    "category_2":           ["mean", "sum", "nunique"],
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
print(agg_new.columns.to_list())

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
print(pd.concat([train_x, train_y], axis=1).describe())

# 各特徴のヒストグラム
train_x.hist(bins=100, color="blue", grid=True, label='pandas')
plt.show()


# ------------------------------
# XGBoostの学習・推論・submit
# ------------------------------
model = XGBRegressor(random_state=rseed)
model.fit(train_x, train_y)
preds = model.predict(test_x)

submission = pd.DataFrame({
    "card_id": test_card_id,
    "target":  preds
})

submission.to_csv("submission.csv", index=False)

