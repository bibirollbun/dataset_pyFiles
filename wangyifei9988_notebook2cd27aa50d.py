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
# new_merchant_transactions.csv 高级聚合特征生成（修正版）
# ----------------------------------------------

import pandas as pd
from scipy.stats import sem

# 自定义函数
def range_func(x):
    return x.max() - x.min()

def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

# 读取数据
new_trans = pd.read_csv(f"{DATA_DIR}/new_merchant_transactions.csv")

# 定义聚合方式（修正后的）
agg_funcs = {
    "purchase_amount": ["min", "max", "mean", "var", pd.Series.skew, pd.Series.kurt, sem, q25, q75, range_func],
    "installments":    ["min", "max", "mean", "nunique", "var", sem, q25, q75, range_func],
    "month_lag":       ["min", "max", "mean", pd.Series.skew, pd.Series.kurt, sem, range_func],
    "purchase_date":   ["count", "first", "last"]
}

# 分组聚合
agg_new = new_trans.groupby("card_id").agg(agg_funcs)

# 列名扁平化 + 清除非法字符
flat_cols = []
for col_name, func in agg_new.columns.to_flat_index():
    if isinstance(func, str):
        func_name = func
    elif hasattr(func, '__name__'):
        func_name = func.__name__
    else:
        func_name = str(func)
    new_col = f"new_{col_name}_{func_name}"
    new_col = new_col.replace("[", "").replace("]", "") \
                     .replace("<", "").replace(">", "") \
                     .replace("(", "").replace(")", "") \
                     .replace(",", "_").replace(" ", "")
    flat_cols.append(new_col)

agg_new.columns = flat_cols
agg_new = agg_new.reset_index()

# 合并进主特征表
train_x = train_x.merge(agg_new, on="card_id", how="left")
test_x  = test_x.merge(agg_new, on="card_id", how="left")

# 填补缺失值
train_x = train_x.fillna(-1)
test_x  = test_x.fillna(-1)



# 删除 ID 列后继续训练
train_x = train_x.drop(columns=["card_id"])
test_x  = test_x.drop(columns=["card_id"])

model = XGBRegressor(random_state=rseed)
model.fit(train_x, train_y)
preds = model.predict(test_x)

submission = pd.DataFrame({
    "card_id": test_card_id,
    "target":  preds
})
submission.to_csv("submission.csv", index=False)



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

