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


new_trans = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv")


new_trans["category_1"] = new_trans["category_1"].map({"Y": 1, "N": 0})

agg_funcs_new = {
    "purchase_amount": ["min", "max", "mean", "std", "sum",
                        ("range", lambda x: x.max() - x.min())],
    "installments":    ["min", "max", "mean", "std",
                        ("range", lambda x: x.max() - x.min())],
    "month_lag":       ["min", "max", "mean", "std",
                        ("range", lambda x: x.max() - x.min())],
    "purchase_date":   ["count"],
    "merchant_id":     ["nunique"],
    "category_1":      ["mean"]
}

agg_new = new_trans.groupby("card_id").agg(agg_funcs_new)
agg_new.columns = [
    f"new_{c}_{f if isinstance(f,str) else f.__name__}"
    for c, f in agg_new.columns.to_flat_index()
]
agg_new = agg_new.reset_index()

train_x = train_x.merge(agg_new, on="card_id", how="left")
test_x  = test_x.merge(agg_new, on="card_id", how="left")




hist_trans = pd.read_csv(f"{DATA_DIR}/historical_transactions.csv")
hist_trans["authorized_flag"] = hist_trans["authorized_flag"].map({"Y": 1, "N": 0})
hist_trans["category_1"] = hist_trans["category_1"].map({"Y": 1, "N": 0})

agg_funcs_hist = {
    "purchase_amount": ["min", "max", "mean", "std", "sum",
                        ("range", lambda x: x.max() - x.min())],
    "installments":    ["min", "max", "mean", "std"],
    "month_lag":       ["min", "max", "mean"],
    "authorized_flag": ["mean", "sum"],
    "category_1":      ["mean"],
    "purchase_date":   ["count"],
    "merchant_id":     ["nunique"]
}

agg_hist = hist_trans.groupby("card_id").agg(agg_funcs_hist)
agg_hist.columns = [
    f"hist_{c}_{f if isinstance(f,str) else f.__name__}"
    for c, f in agg_hist.columns.to_flat_index()
]
agg_hist = agg_hist.reset_index()

train_x = train_x.merge(agg_hist, on="card_id", how="left")
test_x  = test_x.merge(agg_hist, on="card_id", how="left")
# Fill missing values created by the left join
train_x = train_x.fillna(-1)
test_x  = test_x.fillna(-1)

illegal = r"[\[\]<>]"
train_x.columns = train_x.columns.str.replace(illegal, "", regex=True)
test_x.columns  = test_x.columns.str.replace(illegal, "", regex=True)



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


# # ------------------------------
# # XGBoostの学習・推論・submit
# # ------------------------------
# model = XGBRegressor(random_state=rseed)
# model.fit(train_x, train_y)
# preds = model.predict(test_x)

# submission = pd.DataFrame({
#     "card_id": test_card_id,
#     "target":  preds
# })

# submission.to_csv("submission.csv", index=False)


# ------------------------------
# XGBoost の学習・推論・submit
# ------------------------------
model = XGBRegressor(random_state=rseed)
model.fit(train_x, train_y)
preds = model.predict(test_x)

submission = pd.DataFrame({
    "card_id": test_card_id,
    "target": preds
})
submission.to_csv("submission.csv", index=False)

