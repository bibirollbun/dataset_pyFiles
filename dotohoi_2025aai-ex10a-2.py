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
# メインテーブルの特徴量作成
#  + 取引テーブル(historical/new)集約特徴の追加
# ----------------------------------------------
import numpy as np, gc, warnings
warnings.filterwarnings("ignore")

# ---- 1) メインテーブル: first_active_month を数値化 ----
train_x = train.copy().drop(columns=["target"])
test_x  = test.copy()

for df in (train_x, test_x):
    df["first_active_month"] = df["first_active_month"].fillna("Unknown")

le = LabelEncoder().fit(
    pd.concat([train_x["first_active_month"], test_x["first_active_month"]])
)
train_x["fam_label"] = le.transform(train_x["first_active_month"])
test_x ["fam_label"] = le.transform(test_x["first_active_month"])

train_x.drop(columns=["first_active_month"], inplace=True)
test_x .drop(columns=["first_active_month"], inplace=True)

# ---- 2) 取引テーブルを読み込み ----
usecols = [
    "card_id", "authorized_flag",
    "category_1", "category_2", "category_3",
    "city_id", "state_id", "subsector_id",
    "merchant_id", "merchant_category_id",
    "purchase_amount", "installments", "month_lag",
    "purchase_date"
]

hist = pd.read_csv(f"{DATA_DIR}/historical_transactions.csv",
                   usecols=usecols, parse_dates=["purchase_date"])
new  = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv",
                   usecols=usecols, parse_dates=["purchase_date"])

# ---- 3) 簡易前処理（型変換と欠損補完）----
for df in (hist, new):
    df["authorized_flag"] = (df["authorized_flag"] == "Y").astype(np.int8)
    df["category_1"]      = (df["category_1"]      == "Y").astype(np.int8)
    df["purchase_dt"]     = df["purchase_date"].astype(np.int64) // 10**9          # Unix 秒
    df["category_2"]      = df["category_2"].fillna(-1).astype(np.int8)
    df["category_3"]      = df["category_3"].map({"A":0, "B":1, "C":2}).fillna(-1).astype(np.int8)
    df["installments"]    = df["installments"].replace(-1, 0).astype(np.int16)

# ---- 4) 集約関数定義（min/max/count 以外も追加）----
def range_ptp(x): return x.max() - x.min()
def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)

agg_dict = {
    "purchase_amount":  ["min","max","mean","sum","std",range_ptp,q25,q75],
    "installments":     ["min","max","mean","sum","std"],
    "month_lag":        ["min","max","mean","std"],
    "purchase_dt":      ["min","max",range_ptp],
    "authorized_flag":  ["mean"],
    "category_1":       ["mean"],
    "category_2":       ["nunique"],
    "category_3":       ["nunique"],
    "merchant_id":      ["nunique"],
    "merchant_category_id":["nunique"],
    "city_id":          ["nunique"],
    "state_id":         ["nunique"],
    "subsector_id":     ["nunique"],
    "purchase_date":    ["count"],
}

def make_agg(df, prefix):
    agg = df.groupby("card_id").agg(agg_dict)
    # 列名を prefix_元変数_統計量 へ平坦化
    agg.columns = [
        f"{prefix}_{c[0]}_{(c[1] if isinstance(c[1], str) else c[1].__name__)}"
        for c in agg.columns.to_flat_index()
    ]
    return agg.reset_index()

hist_agg = make_agg(hist, "hist")
new_agg  = make_agg(new,  "new")

del hist, new
gc.collect()

# ---- 5) メインテーブルと結合 ----
train_x = train_x.merge(hist_agg, on="card_id", how="left")
train_x = train_x.merge(new_agg,  on="card_id", how="left")
test_x  = test_x .merge(hist_agg, on="card_id", how="left")
test_x  = test_x .merge(new_agg,  on="card_id", how="left")

# 欠損は -1 で埋める
for df in (train_x, test_x):
    df.fillna(-1, inplace=True)




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

