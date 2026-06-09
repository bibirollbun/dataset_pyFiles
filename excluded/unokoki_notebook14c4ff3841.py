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


DATA_DIR = '/kaggle/input/elo-merchant-category-recommendation'
new = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv")
# ── new_merchant_transactions の集約（min,max,count 以外を追加）
agg_dict_new = {
    'purchase_amount': ['min', 'max', 'mean', 'var', 'skew',
                        lambda x: x.quantile(0.25),  # 25%
                        lambda x: x.quantile(0.75)], # 75%
    'installments':    ['min', 'max', 'mean', 'nunique', 'size'],
    'month_lag':       ['min', 'max', 'mean', 'nunique'],
    'purchase_date':   ['min', 'max', 'nunique']
}

new_agg = new.groupby('card_id').agg(agg_dict_new)

# MultiIndex→フラット化
new_agg.columns = [
    'new_' + '_'.join(
        str(c) if not callable(c) else
        ('q25' if '0.25' in repr(c) else 'q75')
        for c in col
    )
    for col in new_agg.columns
]
new_agg = new_agg.reset_index()



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

