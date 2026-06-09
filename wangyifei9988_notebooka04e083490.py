import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

rseed = 71  
DATA_DIR = "/kaggle/input/elo-merchant-category-recommendation"


#―――――――――――――――――――――――
# 1. データ読み込み
#―――――――――――――――――――――――
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

hist = pd.read_csv(
    f"{DATA_DIR}/historical_transactions.csv",
    parse_dates=["purchase_date"],
    low_memory=True
)
new_trans = pd.read_csv(
    f"{DATA_DIR}/new_merchant_transactions.csv",
    parse_dates=["purchase_date"],
    low_memory=True
)
merch = pd.read_csv(f"{DATA_DIR}/merchants.csv", low_memory=False)


#―――――――――――――――――――――――
# 2. 主テーブル前処理
#―――――――――――――――――――――――
train_y      = train["target"]
train_x      = train.drop(columns=["target", "first_active_month"])
test_x       = test .drop(columns=["first_active_month"])
test_card_id = test["card_id"]

# first_active_month → LabelEncoding
le = LabelEncoder()
all_fam = pd.concat([train["first_active_month"], test["first_active_month"]]).fillna("Unknown")
le.fit(all_fam)
train_x["fam_label"] = le.transform(train["first_active_month"].fillna("Unknown"))
test_x ["fam_label"] = le.transform(test ["first_active_month"].fillna("Unknown"))


#―――――――――――――――――――――――
# 3. 全サブテーブルのカテゴリ変数を一括変換
#―――――――――――――――――――――――
def code_columns(df, cols):
    for c in cols:
        if df[c].dtype == "object":
            df[c] = df[c].astype("category").cat.codes
    return df

# ヒストリカル＆ニューは同じ列名が多いのでまとめてコード化
hist = code_columns(hist, ["authorized_flag","category_1","category_2","category_3"])
new_trans = code_columns(new_trans, ["authorized_flag","category_1","category_2","category_3"])
merch = code_columns(merch, merch.select_dtypes("object").columns)


#―――――――――――――――――――――――
# 4. groupby＋agg の重複解消ヘルパー
#―――――――――――――――――――――――
def aggregate_features(df, group_key, agg_dict, prefix):
    """df: 入力DataFrame
       group_key: 集約キー列名 (例："card_id")
       agg_dict: {カラム名: [func1,func2,...], ...}
       prefix: 出力特徴の接頭辞 (例："hist" or "new")"""
    agg = df.groupby(group_key).agg(agg_dict)
    agg.columns = [f"{prefix}_{col}_{fn if isinstance(fn,str) else fn.__name__}"
                   for col,fn in agg.columns.to_flat_index()]
    agg.reset_index(inplace=True)
    return agg


#―――――――――――――――――――――――
# 5. 各サブテーブルの集約パターン設定
#―――――――――――――――――――――――
common_agg = {
    "purchase_amount": ["min","max","mean","var"],
    "installments":    ["min","max","mean"],
    "month_lag":       ["min","max","mean"],
    "authorized_flag": ["mean"],
    "category_1":      ["mean"],
    "category_2":      ["mean","nunique"],
    "category_3":      ["nunique"],
    "merchant_id":     ["nunique"],
    "city_id":         ["nunique"],
    "state_id":        ["nunique"],
    "subsector_id":    ["nunique"],
    "purchase_date":   ["count"]
}

new_feat  = aggregate_features(new_trans, "card_id", common_agg, "new")
hist_feat = aggregate_features(hist,      "card_id", common_agg, "hist")

# merchants だけは別パターン
merch_agg = {
    "numerical_1":       ["mean","var"],
    "numerical_2":       ["mean","var"],
    "avg_sales_lag3":    ["mean","max"],
    "avg_purchases_lag3":["mean","max"],
    "category_1":        ["nunique"],
    "city_id":           ["nunique"],
    "state_id":          ["nunique"]
}
merch_feat = aggregate_features(merch, "merchant_id", merch_agg, "merch")
# ─── merchant_id の型を文字列に揃える ───
hist["merchant_id"]        = hist["merchant_id"].astype(str)
merch_feat["merchant_id"]  = merch_feat["merchant_id"].astype(str)
# ────────────────────────────────────────

# その上でマージ
hist_merch = hist[["card_id","merchant_id"]].merge(
    merch_feat,
    on="merchant_id",
    how="left"
)
# merchant_id 列を落としてから平均を取る
mc_feat = (
    hist_merch
    .drop(columns=["merchant_id"])
    .groupby("card_id", as_index=False)
    .mean()
)

# merchant_id→card_id 再集約
hist_merch = hist[["card_id","merchant_id"]].merge(
    merch_feat,
    on="merchant_id",
    how="left"
)

mc_feat = (
    hist_merch
    .drop(columns=["merchant_id"])
    .groupby("card_id", as_index=False)
    .mean()
)



#―――――――――――――――――――――――
# 6. 主テーブルへマージ＆学習準備
#―――――――――――――――――――――――――
for feat in (new_feat, hist_feat, mc_feat):
    train_x = train_x.merge(feat, on="card_id", how="left")
    test_x  = test_x .merge(feat, on="card_id", how="left")

train_x.fillna(-1, inplace=True)
test_x .fillna(-1, inplace=True)

train_x.drop(columns=["card_id"], inplace=True)
test_x .drop(columns=["card_id"], inplace=True)


#―――――――――――――――――――――――
# 7. モデル学習・予測・出力
#―――――――――――――――――――――――
model = XGBRegressor(random_state=rseed)
model.fit(train_x, train_y)
preds = model.predict(test_x)

pd.DataFrame({"card_id": test_card_id, "target": preds})\
  .to_csv("submission.csv", index=False)

print("最終特徴量数:", train_x.shape[1])

