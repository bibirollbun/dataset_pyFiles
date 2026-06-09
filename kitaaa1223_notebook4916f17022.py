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

# 追加テーブルの読み込み
new_trans = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv")
print(new_trans.describe())

# 追加テーブルの読み込み
new_trans = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv")
new_trans["authorized_flag"] = new_trans["authorized_flag"].map({"Y": 1, "N": 0})
new_trans["category_1"] = new_trans["category_1"].map({"Y": 1, "N": 0})
new_trans["category_3"] = new_trans["category_3"].map({'A': 0, 'B': 1, 'C': 2})
new_trans["purchase_date"] = pd.to_datetime(new_trans["purchase_date"])
new_trans["month"] = new_trans["purchase_date"].dt.month
new_trans["weekend"] = (new_trans["purchase_date"].dt.weekday >= 5).astype(int)
new_trans["hour"] = new_trans["purchase_date"].dt.hour
new_trans["day"] = new_trans["purchase_date"].dt.day
new_trans["dayofweek"] = new_trans["purchase_date"].dt.dayofweek

# より豊富な統計量で特徴量を作成
agg_funcs = {
    "purchase_amount": ["min", "max", "mean", "std", "skew", "median", "sum"],
    "installments":    ["min", "max", "mean", "std", "nunique", "median"],
    "month_lag":       ["min", "max", "mean", "std", "skew"],
    "authorized_flag": ["mean", "sum"],
    "category_1":      ["mean"],
    "category_2":      ["mean", "nunique"],
    "category_3":      ["mean", "nunique"],
    "month":           ["nunique", "median"],
    "weekend":         ["mean"],
    "hour":            ["mean", "std", "skew"],
    "day":             ["mean", "std"],
    "dayofweek":       ["mean", "nunique"],
    "purchase_date":   ["count"]
}

# 集約処理
agg_new = new_trans.groupby("card_id").agg(agg_funcs)

# 列名を平坦化
agg_new.columns = [f"new_{col}_{stat}" for col, stat in agg_new.columns.to_flat_index()]
agg_new = agg_new.reset_index()

# 結合
train_x = train_x.merge(agg_new, on="card_id", how="left")
test_x  = test_x.merge(agg_new, on="card_id", how="left")

# 欠損値処理
train_x = train_x.fillna(-1)
test_x  = test_x.fillna(-1)


# ----------------------------------------------
# historical_transactions の特徴量作成
# ----------------------------------------------

# 読み込み
hist_trans = pd.read_csv(f"{DATA_DIR}/historical_transactions.csv")

# 前処理
hist_trans["authorized_flag"] = hist_trans["authorized_flag"].map({"Y": 1, "N": 0})
hist_trans["category_1"] = hist_trans["category_1"].map({"Y": 1, "N": 0})
hist_trans["category_3"] = hist_trans["category_3"].map({'A': 0, 'B': 1, 'C': 2})
hist_trans["purchase_date"] = pd.to_datetime(hist_trans["purchase_date"])
hist_trans["month"] = hist_trans["purchase_date"].dt.month
hist_trans["weekend"] = (hist_trans["purchase_date"].dt.weekday >= 5).astype(int)
hist_trans["hour"] = hist_trans["purchase_date"].dt.hour
hist_trans["day"] = hist_trans["purchase_date"].dt.day
hist_trans["dayofweek"] = hist_trans["purchase_date"].dt.dayofweek

# ★ 重要：不正取引を除外（モデルに悪影響の場合あり）
hist_trans = hist_trans[hist_trans["authorized_flag"] == 1]

# 集約関数の設定
agg_funcs = {
    "purchase_amount": ["min", "max", "mean", "std", "skew", "median", "sum"],
    "installments":    ["min", "max", "mean", "std", "nunique", "median"],
    "month_lag":       ["min", "max", "mean", "std", "skew"],
    "authorized_flag": ["mean", "sum"],
    "category_1":      ["mean"],
    "category_2":      ["mean", "nunique"],
    "category_3":      ["mean", "nunique"],
    "month":           ["nunique", "median"],
    "weekend":         ["mean"],
    "hour":            ["mean", "std", "skew"],
    "day":             ["mean", "std"],
    "dayofweek":       ["mean", "nunique"],
    "purchase_date":   ["count"]
}

# 集約
agg_hist = hist_trans.groupby("card_id").agg(agg_funcs)

# 列名を平坦化
agg_hist.columns = [f"hist_{col}_{stat}" for col, stat in agg_hist.columns.to_flat_index()]
agg_hist = agg_hist.reset_index()

# メインテーブルへ結合
train_x = train_x.merge(agg_hist, on="card_id", how="left")
test_x  = test_x.merge(agg_hist, on="card_id", how="left")

# 欠損値補完
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

