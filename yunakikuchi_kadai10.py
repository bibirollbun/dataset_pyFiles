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


import pandas as pd
import numpy as np

# -----------------------------
# 新サブテーブルの読み込み
# -----------------------------
new_trans = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv")
hist_trans = pd.read_csv(f"{DATA_DIR}/historical_transactions.csv")

# -----------------------------
# サブテーブル共通前処理
# -----------------------------
def preprocess(trans):
    # 日付データを datetime に変換（集約で min/max が正しく動作するように）
    trans['purchase_date'] = pd.to_datetime(trans['purchase_date'])

    # フラグ/カテゴリ変数の変換
    trans['authorized_flag'] = trans['authorized_flag'].map({'Y': 1, 'N': 0})
    trans['category_1'] = trans['category_1'].map({'Y': 1, 'N': 0})

    # category_2, category_3 を数値に変換（欠損あり）
    trans['category_2'] = trans['category_2'].fillna(-1)
    trans['category_3'] = trans['category_3'].map({'A': 0, 'B': 1, 'C': 2}).fillna(-1)

    return trans

new_trans = preprocess(new_trans)
hist_trans = preprocess(hist_trans)

# -----------------------------
# 集約用関数定義
# -----------------------------
agg_funcs = {
    'purchase_amount': ['min', 'max', 'mean', 'sum', 'std', 'skew'],
    'installments': ['min', 'max', 'mean', 'std'],
    'month_lag': ['min', 'max', 'mean'],
    'purchase_date': ['count', 'min', 'max'],
    'authorized_flag': ['mean', 'sum'],
    'category_1': ['mean', 'sum'],
    'category_2': ['mean', 'nunique'],
    'category_3': ['mean', 'nunique']
}

# -----------------------------
# 集約特徴量作成関数
# -----------------------------
def create_agg_features(trans, prefix):
    agg = trans.groupby("card_id").agg(agg_funcs)
    flat_cols = [f"{prefix}_{col}_{func}" for col, func in agg.columns.to_flat_index()]
    agg.columns = flat_cols
    agg = agg.reset_index()
    return agg

# -----------------------------
# 集約テーブル作成
# -----------------------------
agg_new = create_agg_features(new_trans, "new")
agg_hist = create_agg_features(hist_trans, "hist")

# -----------------------------
# メインテーブルに結合
# -----------------------------

# card_id を結合キーとして追加しておく（必須）
train_x["card_id"] = train["card_id"]
test_x["card_id"]  = test["card_id"]

# 特徴量結合
train_x = train_x.merge(agg_new, on="card_id", how="left")
train_x = train_x.merge(agg_hist, on="card_id", how="left")

test_x = test_x.merge(agg_new, on="card_id", how="left")
test_x = test_x.merge(agg_hist, on="card_id", how="left")


# 特徴量の結合まで完了した後で
datetime_format = "%Y-%m-%d %H:%M:%S"
# 日付列を timestamp に変換
for col in train_x.columns:
    if train_x[col].dtype == 'datetime64[ns]' or train_x[col].dtype == 'object':
        try:
            train_x[col] = pd.to_datetime(train_x[col], errors='coerce')
            test_x[col] = pd.to_datetime(test_x[col], errors='coerce')

            # view の代わりに astype を使う
            train_x[col] = train_x[col].astype('int64') // 10**9
            test_x[col] = test_x[col].astype('int64') // 10**9
        except Exception:
            pass


# 欠損値補完
train_x = train_x.fillna(-1)
test_x  = test_x.fillna(-1)

# モデル学習へ
model = XGBRegressor(random_state=rseed)
model.fit(train_x, train_y)



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

