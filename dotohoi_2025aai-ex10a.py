# ==============================================================
# 0. 事前準備
# ==============================================================
import os, gc, warnings, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

SEED      = 71
DATA_DIR  = "/kaggle/input/elo-merchant-category-recommendation"

# --------------------------------------------------------------
# 1. メインテーブル（train / test）
# --------------------------------------------------------------
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

y      = train["target"].copy()
train_ = train.drop(columns=["target"])
test_  = test.copy()

# `first_active_month` を Label-Encode して数値化
for df in (train_, test_):
    df["first_active_month"] = df["first_active_month"].fillna("Unknown")
le = LabelEncoder().fit(pd.concat([train_["first_active_month"], test_["first_active_month"]]))
for df in (train_, test_):
    df["fam_label"] = le.transform(df["first_active_month"])
    df.drop(columns=["first_active_month"], inplace=True)

# あとで JOIN するので card_id は残す
base_cols = ["card_id", "feature_1", "feature_2", "feature_3", "fam_label"]

train_ = train_[base_cols].copy()
test_  = test_ [base_cols].copy()

# 使いやすい関数をいくつか定義
def range_ptp(x):        # (最大−最小)
    return x.max() - x.min()

def quartile25(x):
    return x.quantile(0.25)

def quartile75(x):
    return x.quantile(0.75)

# --------------------------------------------------------------
# 2. 取引テーブル（historical / new）を読む
# --------------------------------------------------------------
usecols = [                # 最小限に絞ってメモリ節約
    "card_id", "authorized_flag",
    "category_1", "category_2", "category_3",
    "city_id", "state_id", "subsector_id",
    "merchant_id", "merchant_category_id",
    "purchase_amount", "installments", "month_lag",
    "purchase_date"
]

hist_trans = pd.read_csv(f"{DATA_DIR}/historical_transactions.csv",
                         usecols=usecols, parse_dates=["purchase_date"])
new_trans  = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv",
                         usecols=usecols, parse_dates=["purchase_date"])

# 3. 前処理：使いやすい数値型に変換
# ================================================
for df in (hist_trans, new_trans):

    # フラグを 0/1 に
    df["authorized_flag"] = (df["authorized_flag"] == "Y").astype(np.int8)
    df["category_1"]      = (df["category_1"]      == "Y").astype(np.int8)

    # purchase_date → Unix time（秒）
    df["purchase_dt"] = df["purchase_date"].astype(np.int64) // 10**9

    # category_2 : 数値 (1–5) + NaN
    df["category_2"] = df["category_2"].fillna(-1).astype(np.int8)

    # ----- ★ 修正ポイント ★ -----
    # category_3 : 'A','B','C' → 0,1,2 へマッピング
    cat3_map = {"A": 0, "B": 1, "C": 2}
    df["category_3"] = df["category_3"].map(cat3_map).fillna(-1).astype(np.int8)

    # installments : -1 を 0（⟷一括払い）とみなす
    df["installments"] = df["installments"].replace(-1, 0).astype(np.int16)

# --------------------------------------------------------------
# 4. 集約関数を定義
#    ※ min / max / count 以外も大量投入！
# --------------------------------------------------------------
agg_dict = {
    # 数値系
    "purchase_amount":  ["min", "max", "mean", "sum", "std", "var", range_ptp,
                         quartile25, quartile75],
    "installments":     ["min", "max", "mean", "sum", "std", "var"],
    "month_lag":        ["min", "max", "mean", "std", skew := "skew"],
    "purchase_dt":      ["min", "max", range_ptp],
    # カテゴリ系 → 集計すると「多様性」を表せる
    "authorized_flag":  ["mean"],                  # 承認率
    "category_1":       ["mean"],                  # C1=1 の比率
    "category_2":       ["nunique", quartile75],   # 分布の粗さ + 値
    "category_3":       ["nunique"],
    "merchant_id":      ["nunique"],
    "merchant_category_id": ["nunique"],
    "city_id":          ["nunique"],
    "state_id":         ["nunique"],
    "subsector_id":     ["nunique"],
    "purchase_date":    ["count"],                 # 取引件数
}

# --------------------------------------------------------------
# 5. 「historical」「new」別に集約し、列名に接頭辞を付ける
# --------------------------------------------------------------
def aggregate_transactions(df, prefix):
    """card_id ごとに集約し、列名を prefix_... にする"""
    agg = (df
           .groupby("card_id")
           .agg(agg_dict))
    # MultiIndex → 1階層化
    agg.columns = [f"{prefix}_{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()]
    return agg.reset_index()

hist_agg = aggregate_transactions(hist_trans, "hist")
new_agg  = aggregate_transactions(new_trans,  "new")

del hist_trans, new_trans
gc.collect()

# --------------------------------------------------------------
# 6. メインテーブルと結合   ★修正版★
# --------------------------------------------------------------
# train データ
train_ = train_.merge(hist_agg, on="card_id", how="left")
train_ = train_.merge(new_agg,  on="card_id", how="left")

# test データ
test_  = test_.merge(hist_agg,  on="card_id", how="left")
test_  = test_.merge(new_agg,   on="card_id", how="left")

# 欠損値を -1 で埋める
for df in (train_, test_):
    df.fillna(-1, inplace=True)

print("最終的な特徴量数:", train_.shape[1] - 1)  # card_id を除いた数

# --------------------------------------------------------------
# 7. 学習・推論（シンプルなワンショット）
#    ※ CV／パラメータ調整は必ず行ってください
# --------------------------------------------------------------
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=3,
    random_state=SEED,
    tree_method="hist",
    eval_metric="rmse"
)
model.fit(train_.drop(columns=["card_id"]), y)

pred = model.predict(test_.drop(columns=["card_id"]))

pd.DataFrame({
    "card_id": test_["card_id"],
    "target":  pred
}).to_csv("submission.csv", index=False)


