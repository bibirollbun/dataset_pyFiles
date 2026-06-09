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


historical = pd.read_csv("/kaggle/input/elo-merchant-category-recommendation/historical_transactions.csv")
new_merchant = pd.read_csv("/kaggle/input/elo-merchant-category-recommendation/new_merchant_transactions.csv")


# 集約特徴量を強化（historical + new_merchant、関数拡張）
def aggregate_transactions(df, prefix):
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['purchase_month'] = df['purchase_date'].dt.month

    aggs = {
        'purchase_amount': ['min', 'max', 'mean', 'std', 'sum', 'count'],
        'installments': ['min', 'max', 'mean', 'std'],
        'month_lag': ['min', 'max', 'mean', 'std'],
        'purchase_month': ['nunique'],
    }

    agg_df = df.groupby('card_id').agg(aggs)
    agg_df.columns = [f"{prefix}_{k}_{stat}" for k, stats in aggs.items() for stat in stats]
    agg_df.reset_index(inplace=True)
    return agg_df

agg_hist = aggregate_transactions(historical, "hist")
agg_new  = aggregate_transactions(new_merchant, "new")

train_x = train.merge(agg_hist, on="card_id", how="left")
train_x = train_x.merge(agg_new, on="card_id", how="left")
test_x  = test.merge(agg_hist, on="card_id", how="left")
test_x  = test_x.merge(agg_new, on="card_id", how="left")

# モデルに与える列を整理（object列などを除外）
drop_cols = ["card_id", "first_active_month", "target"]
train_y = train_x["target"]
train_x = train_x.drop(columns=drop_cols, errors='ignore')
test_x = test_x.drop(columns=drop_cols, errors='ignore')



# ID列を削除
# 'card_id' が存在する場合のみ削除する
if 'card_id' in train_x.columns:
    train_x = train_x.drop(columns=["card_id"])
if 'card_id' in test_x.columns:
    test_x = test_x.drop(columns=["card_id"])



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
print(submission)

submission.to_csv("submission.csv", index=False)

