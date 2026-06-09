


# ================================================================
# 1. ライブラリ & データ読み込み
# ================================================================
import pandas as pd, numpy as np, gc, warnings
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 71
DATA_DIR = "/kaggle/input/rossmann-store-sales"

train = pd.read_csv(f"{DATA_DIR}/train.csv", low_memory=False)
test  = pd.read_csv(f"{DATA_DIR}/test.csv",  low_memory=False)
store = pd.read_csv(f"{DATA_DIR}/store.csv")

# 結合（Competition系が欲しいため）
train = train.merge(store, on="Store", how="left")
test  = test.merge(store, on="Store", how="left")


# ================================================================
# 2. 基本カレンダー特徴
# ================================================================
def add_date_parts(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(np.int16)
    df["DayOfWeek"] = df["Date"].dt.dayofweek        # 0=Mon
    df["Day"] = df["Date"].dt.day
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(np.int8)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(np.int8)
    df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(np.int8)
    return df

train = add_date_parts(train)
test  = add_date_parts(test)

# ================================================================
# 3. ラグ & 移動統計（店舗ごと、リーク防止に shift(1)）
# ================================================================
train = train.sort_values(["Store", "Date"])
test  = test.sort_values(["Store", "Date"])

WINDOWS = [7, 30]
for w in WINDOWS:
    train[f"Sales_ma_{w}"] = (
        train.groupby("Store")["Sales"]
             .transform(lambda x: x.shift(1).rolling(w).mean())
    )
    train[f"Sales_std_{w}"] = (
        train.groupby("Store")["Sales"]
             .transform(lambda x: x.shift(1).rolling(w).std())
    )
# 前年同日ラグ
train["Sales_lag_365"] = train.groupby("Store")["Sales"].shift(365)


# ================================================================
# 4. イベント経過日数特徴
# ================================================================
# Competition open days
for df in (train, test):
    df["CompetitionOpenSinceYear"].fillna(1900, inplace=True)
    df["CompetitionOpenSinceMonth"].fillna(1, inplace=True)
    comp_open = pd.to_datetime(dict(year=df["CompetitionOpenSinceYear"],
                                    month=df["CompetitionOpenSinceMonth"],
                                    day=15))
    df["CompDays"] = (df["Date"] - comp_open).dt.days
    df.loc[df["CompDays"] < 0, "CompDays"] = 0    # 未来→0

    # Promo2 open days
    df["Promo2SinceYear"].fillna(1900, inplace=True)
    df["Promo2SinceWeek"].fillna(1, inplace=True)
    promo2_open = pd.to_datetime(df["Promo2SinceYear"].astype(int)
                                 .astype(str) + "-1-1") + \
                  pd.to_timedelta((df["Promo2SinceWeek"] - 1) * 7, unit="d")
    df["Promo2Days"] = (df["Date"] - promo2_open).dt.days
    df.loc[df["Promo2Days"] < 0, "Promo2Days"] = 0


# ================================================================
# 5. 交互作用特徴
# ================================================================
train["Promo_DOW"] = train["Promo"] * train["DayOfWeek"]
test ["Promo_DOW"] = test ["Promo"] * test ["DayOfWeek"]

train["Holiday_Month"] = (
    train["StateHoliday"].astype(str).replace({"0": "None"}) +
    "_M" + train["Month"].astype(str)
)
test["Holiday_Month"] = (
    test["StateHoliday"].astype(str).replace({"0": "None"}) +
    "_M" + test["Month"].astype(str)
)



# ================================================================
# 6. 店舗固定統計（平均 Sales・Promo 率）
# ================================================================
store_stats = (
    train.groupby("Store")
         .agg(
             store_sales_mean=("Sales", "mean"),
             store_sales_median=("Sales", "median"),
             store_promo_rate=("Promo", "mean")
         )
         .reset_index()
)
train = train.merge(store_stats, on="Store", how="left")
test  = test .merge(store_stats, on="Store", how="left")


# ================================================================
# 7. カテゴリ列エンコード & 不要列削除
# ================================================================
#cat_cols = ["StoreType", "Assortment", "StateHoliday", "Holiday_Month"]

#cat_cols = ["StoreType", "Assortment", "StateHoliday",
#            "Holiday_Month",          # 既存
#            "Assort_Qtr", "StoreType_DOW"]   # ← 追加
# 既に object の列をすべて抽出する 1 行に置き換える
cat_cols = train.select_dtypes('object').columns


for col in cat_cols:
    le = LabelEncoder().fit(pd.concat([train[col].fillna("NA"),
                                       test [col].fillna("NA")]))
    train[col] = le.transform(train[col].fillna("NA"))
    test [col] = le.transform(test [col].fillna("NA"))

drop_cols = ["Date", "Sales", "Customers",
             "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth",
             "Promo2SinceYear", "Promo2SinceWeek", "PromoInterval"]
train_x = train.drop(columns=drop_cols)
test_x  = test .drop(columns=[c for c in drop_cols if c in test.columns])

train_y = train["Sales"]
test_id = test["Id"]

# 欠損を -1 で埋める
train_x.fillna(-1, inplace=True)
test_x .fillna(-1, inplace=True)

# ===== 追加ここから =========================
common_cols = train_x.columns
test_x = test_x.reindex(columns=common_cols, fill_value=-1)
# ===== 追加ここまで =========================



# ================================================================
# 8. XGBoost 学習 & 提出ファイル  ★ハイパラ設定はここだけ★
# ================================================================
TREE_NUM = 100   # ← 木の本数を増やす（デフォルト 100 → 800）
DEPTH    = 6    # ← 木の最大深さ（デフォルト 6 → 10）

model = XGBRegressor(
    n_estimators=TREE_NUM,
    max_depth=DEPTH,
    random_state=SEED           # ← これ以外のパラメータはデフォルト
)
model.fit(train_x, train_y)

pred = model.predict(test_x)
pd.DataFrame({"Id": test_id, "Sales": pred}).to_csv("submission.csv", index=False)


'''

# --- 追加インポート ---
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor

# --- ベースモデルを定義 ---
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=4
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=4
)

# --- スタッキングモデルを構築 ---
stack = StackingRegressor(
    estimators=[("xgb", xgb_model), ("lgb", lgb_model)],
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    cv=5,
    n_jobs=1,            # メモリやCPUに余裕があれば -1
    passthrough=False    # True にすると一段目の予測値も二段目に渡す
)

# --- 学習 ---
stack.fit(train_x, train_y)

# --- 予測＆提出ファイル作成 ---
pred = stack.predict(test_x)
pd.DataFrame({"Id": test_id, "Sales": pred}) \
  .to_csv("submission_stack.csv", index=False)


# ================================================================
# 8. XGBoost（デフォルト）学習 & 提出ファイル
# ================================================================
model = XGBRegressor(random_state=SEED)
model.fit(train_x, train_y)

pred = model.predict(test_x)
pd.DataFrame({"Id": test_id, "Sales": pred}) \
  .to_csv("submission.csv", index=False)






# ================================================================
# 3. ラグ & 移動統計（店舗ごと、リーク防止に shift(1)）
# ================================================================



train = train.sort_values(["Store", "Date"])
test  = test.sort_values(["Store", "Date"])

WINDOWS = [7, 30]
for w in WINDOWS:
    train[f"Sales_ma_{w}"] = (
        train.groupby("Store")["Sales"]
             .transform(lambda x: x.shift(1).rolling(w).mean())
    )
    train[f"Sales_std_{w}"] = (
        train.groupby("Store")["Sales"]
             .transform(lambda x: x.shift(1).rolling(w).std())
    )
# 前年同日ラグ
train["Sales_lag_365"] = train.groupby("Store")["Sales"].shift(365)







# ================================================================
# 7. カテゴリ列エンコード & 不要列削除
# ================================================================
#cat_cols = ["StoreType", "Assortment", "StateHoliday", "Holiday_Month"]

#cat_cols = ["StoreType", "Assortment", "StateHoliday",
#            "Holiday_Month",          # 既存
#            "Assort_Qtr", "StoreType_DOW"]   # ← 追加
# 既に object の列をすべて抽出する 1 行に置き換える
cat_cols = train.select_dtypes('object').columns


for col in cat_cols:
    le = LabelEncoder().fit(pd.concat([train[col].fillna("NA"),
                                       test [col].fillna("NA")]))
    train[col] = le.transform(train[col].fillna("NA"))
    test [col] = le.transform(test [col].fillna("NA"))

drop_cols = ["Date", "Sales", "Customers",
             "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth",
             "Promo2SinceYear", "Promo2SinceWeek", "PromoInterval"]
train_x = train.drop(columns=drop_cols)
test_x  = test .drop(columns=[c for c in drop_cols if c in test.columns])

train_y = train["Sales"]
test_id = test["Id"]

# 欠損を -1 で埋める
train_x.fillna(-1, inplace=True)
test_x .fillna(-1, inplace=True)

# ===== 追加ここから =========================
common_cols = train_x.columns
test_x = test_x.reindex(columns=common_cols, fill_value=-1)
'''

