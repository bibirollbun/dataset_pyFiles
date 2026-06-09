# ============================================================
# PGS501 (Playground Series S5E1)
# 「前処理はそのまま」＋「全体（日次合計）だけ予測」＋「国・店・製品の比率で配分」
# 完全版コード（省略なし） 2025-07-27  fix: leap-day collision -> dedup & renorm
# ============================================================

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error

sns.set_style('darkgrid')

# -----------------------------
# 0. パス
# -----------------------------
TRAIN_PATH = "/kaggle/input/playground-series-s5e1/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e1/test.csv"
GDP_PATH   = "/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv"
SAMPLE_SUB = "/kaggle/input/playground-series-s5e1/sample_submission.csv"

# ============================================================
# 1. Utility 関数群（前処理は既存を維持）
# ============================================================

def fit_mape_linear_model(X, y):
    X = np.asarray(X)
    y = np.asarray(y).squeeze()
    X_aug = np.column_stack((np.ones(X.shape[0]), X))
    def mape_loss(beta, X, y):
        y_pred = X @ beta
        return np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-9))) * 100
    init_params = np.zeros(X_aug.shape[1])
    result = minimize(mape_loss, init_params, args=(X_aug, y), method='L-BFGS-B')
    beta_opt = result.x
    model = LinearRegression()
    model.coef_ = beta_opt[1:]
    model.intercept_ = beta_opt[0]
    return model

def add_leap_adjusted_n_day(df, date_col="date"):
    df = df.copy()
    df["n_day"] = (df[date_col] - df[date_col].min()).dt.days
    for leap in [dt.datetime(2012, 2, 29), dt.datetime(2016, 2, 29)]:
        df.loc[df[date_col] > leap, "n_day"] -= 1
    return df

def make_fourier_cols(df, base_col, max_harm=9, period=365, prefix="wave"):
    df = df.copy()
    cols = []
    for i in range(1, max_harm + 1):
        df[f"{prefix}_sin{i}"] = np.sin(np.pi * i * df[base_col] / period)
        df[f"{prefix}_cos{i}"] = np.cos(np.pi * i * df[base_col] / period)
        cols += [f"{prefix}_sin{i}", f"{prefix}_cos{i}"]
    return df, cols

def create_near_holiday_flag(df, window=10, date_col="date", country_col="country"):
    df = df.copy()
    df["near_holiday"] = 0
    years = df[date_col].dt.year.unique()
    for c in df[country_col].unique():
        try:
            hdays = [d for d in holidays.CountryHoliday(c, years=years)]
        except Exception:
            hdays = []
        for day in hdays:
            mask = (df[country_col] == c) & (df[date_col].dt.date < (day + dt.timedelta(days=window))) & \
                   (df[date_col].dt.date > (day - dt.timedelta(days=window)))
            df.loc[mask, "near_holiday"] = 1
    return df

def build_holiday_shift_cols(df, date_col="date", country_col="country", max_shift=9):
    years = range(2010, 2020)
    countries = df[country_col].unique()
    dfs = []
    for y in years:
        for c in countries:
            try:
                for d, _ in sorted(holidays.CountryHoliday(c, years=y).items()):
                    dfs.append(pd.DataFrame({"date": [pd.to_datetime(d)], "country": [c], "tmp": [1]}))
            except Exception:
                pass
    if len(dfs) == 0:
        df_h = pd.DataFrame(columns=["date", "country", "tmp"])
    else:
        df_h = pd.concat(dfs, ignore_index=True)
    df_h["date"] = pd.to_datetime(df_h["date"])

    out = df.copy()
    holiday_cols = []
    for i in range(0, max_shift + 1):
        col = f"holiday_{i}"
        shifted = df_h.copy()
        shifted["date"] = shifted["date"] + dt.timedelta(days=i)
        shifted = shifted.rename(columns={"tmp": col})
        out = out.merge(shifted, on=[country_col, date_col], how="left")
        out[col] = out[col].fillna(0).astype(float)
        holiday_cols.append(col)
    return out, holiday_cols

def build_newyear_cols(df, date_col="date"):
    out = df.copy()
    cols = []
    for day in range(25, 32):
        col = f"day_12_{day}"
        out[col] = ((out[date_col].dt.month == 12) & (out[date_col].dt.day == day)).astype(float)
        cols.append(col)
    for day in range(1, 11):
        col = f"day_1_{day}"
        out[col] = ((out[date_col].dt.month == 1) & (out[date_col].dt.day == day)).astype(float)
        cols.append(col)
    return out, cols

def feature_engineer_day(df_dates):
    new_df = df_dates.copy()
    new_df["month"] = new_df["date"].dt.month
    new_df["month_sin"] = np.sin(new_df["month"] * (2 * np.pi / 12))
    new_df["month_cos"] = np.cos(new_df["month"] * (2 * np.pi / 12))
    new_df["day_of_week"] = new_df["date"].dt.dayofweek
    new_df["day_of_week_group"] = new_df["day_of_week"].apply(
        lambda x: 0 if x <= 3 else (1 if x == 4 else (2 if x == 5 else 3))
    )
    doy = new_df["date"].apply(
        lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month > 2) else x.timetuple().tm_yday - 1
    )
    new_df["day_of_year"] = doy
    new_df["day_sin4"] = np.sin(new_df["day_of_year"] * (8 * np.pi / 365.0))
    new_df["day_cos4"] = np.cos(new_df["day_of_year"] * (8 * np.pi / 365.0))
    new_df["day_sin3"] = np.sin(new_df["day_of_year"] * (6 * np.pi / 365.0))
    new_df["day_cos3"] = np.cos(new_df["day_of_year"] * (6 * np.pi / 365.0))
    new_df["day_sin2"] = np.sin(new_df["day_of_year"] * (4 * np.pi / 365.0))
    new_df["day_cos2"] = np.cos(new_df["day_of_year"] * (4 * np.pi / 365.0))
    new_df["day_sin"]  = np.sin(new_df["day_of_year"] * (2 * np.pi / 365.0))
    new_df["day_cos"]  = np.cos(new_df["day_of_year"] * (2 * np.pi / 365.0))
    new_df["day_sin_0.5"] = np.sin(new_df["day_of_year"] * (1 * np.pi / 365.0))
    new_df["day_cos_0.5"] = np.cos(new_df["day_of_year"] * (1 * np.pi / 365.0))
    important = [1,2,3,4,5,6,7,8,9,10,99,100,101,125,126,355,256,357,358,359,360,361,362,363,364,365]
    new_df["important_dates"] = new_df["day_of_year"].apply(lambda x: x if x in important else 0)
    new_df = pd.get_dummies(new_df,
                            columns=["important_dates", "day_of_week_group"],
                            drop_first=True)
    new_df = new_df.drop(columns=["month", "day_of_year"])
    return new_df.drop(columns=['date'])

def fill_with_ratio_by_date(df, dst_mask, src_mask, ratio):
    src_ser = (df.loc[src_mask, ["date", "num_sold"]]
                 .dropna()
                 .set_index("date")["num_sold"] * ratio)
    dst_idx = df.index[dst_mask]
    dst_dates = df.loc[dst_idx, "date"]
    df.loc[dst_idx, "num_sold"] = dst_dates.map(src_ser).values

# ============================================================
# 2. 読み込み & 欠損補完（前処理そのまま）
# ============================================================

train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
original_train_df = train_df.copy()
test_df  = pd.read_csv(TEST_PATH,  parse_dates=["date"])

# GDP CSV（Cabaxiom式 ratio 用）
gdp_pc = pd.read_csv(GDP_PATH)
years = [str(y) for y in range(2010, 2021)]
gdp_filtered = gdp_pc.loc[gdp_pc["Country Name"].isin(train_df["country"].unique()),
                          ["Country Name"] + years].set_index("Country Name")
for y in years:
    gdp_filtered[f"{y}_ratio"] = gdp_filtered[y] / gdp_filtered[y].sum()
gdp_ratios = gdp_filtered[[f"{y}_ratio" for y in years]].copy()
gdp_ratios.columns = [int(y) for y in years]
gdp_ratios = gdp_ratios.unstack().reset_index().rename(
    columns={"level_0":"year", 0:"ratio", "Country Name":"country"}
)
gdp_ratios["year"] = pd.to_datetime(gdp_ratios["year"], format="%Y")

gdp_ratios_2 = gdp_ratios.copy()
gdp_ratios_2["year"] = pd.to_datetime(gdp_ratios_2["year"].astype(str)) + pd.offsets.YearEnd(1)
_ = pd.concat([gdp_ratios, gdp_ratios_2]).reset_index(drop=True)  # 互換用
gdp_ratios_2["year"] = gdp_ratios_2["year"].dt.year

# ===== 欠損補完 =====
train_df_imputed = train_df.copy()
print(f"[Before Impute] Missing num_sold = {train_df_imputed['num_sold'].isna().sum()}")

train_df_imputed["year"] = train_df_imputed["date"].dt.year
for year in train_df_imputed["year"].unique():
    target_ratio = gdp_ratios_2.loc[(gdp_ratios_2["year"] == year) & (gdp_ratios_2["country"] == "Norway"), "ratio"].values[0]
    ratio_can = gdp_ratios_2.loc[(gdp_ratios_2["year"] == year) & (gdp_ratios_2["country"] == "Canada"), "ratio"].values[0] / target_ratio
    ratio_ken = gdp_ratios_2.loc[(gdp_ratios_2["year"] == year) & (gdp_ratios_2["country"] == "Kenya"),  "ratio"].values[0] / target_ratio

    mask_nor_ds_hg = (train_df_imputed["country"]=="Norway") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_nor_pm_hg = (train_df_imputed["country"]=="Norway") & (train_df_imputed["store"]=="Premium Sticker Mart") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_nor_sl_hg = (train_df_imputed["country"]=="Norway") & (train_df_imputed["store"]=="Stickers for Less") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_nor_ds_kern= (train_df_imputed["country"]=="Norway") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Kerneler") & (train_df_imputed["year"]==year)

    mask_can_ds_hg = (train_df_imputed["country"]=="Canada") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_can_pm_hg = (train_df_imputed["country"]=="Canada") & (train_df_imputed["store"]=="Premium Sticker Mart") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())
    mask_can_sl_hg = (train_df_imputed["country"]=="Canada") & (train_df_imputed["store"]=="Stickers for Less") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())

    mask_ken_ds_hg = (train_df_imputed["country"]=="Kenya") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_ken_pm_hg = (train_df_imputed["country"]=="Kenya") & (train_df_imputed["store"]=="Premium Sticker Mart") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())
    mask_ken_sl_hg = (train_df_imputed["country"]=="Kenya") & (train_df_imputed["store"]=="Stickers for Less") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())
    mask_ken_ds_kern= (train_df_imputed["country"]=="Kenya") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Kerneler") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())

    fill_with_ratio_by_date(train_df_imputed, mask_can_ds_hg,  mask_nor_ds_hg, ratio_can)
    fill_with_ratio_by_date(train_df_imputed, mask_can_pm_hg,  mask_nor_pm_hg, ratio_can)
    fill_with_ratio_by_date(train_df_imputed, mask_can_sl_hg,  mask_nor_sl_hg, ratio_can)
    fill_with_ratio_by_date(train_df_imputed, mask_ken_ds_hg,  mask_nor_ds_hg, ratio_ken)
    fill_with_ratio_by_date(train_df_imputed, mask_ken_pm_hg,  mask_nor_pm_hg, ratio_ken)
    fill_with_ratio_by_date(train_df_imputed, mask_ken_sl_hg,  mask_nor_sl_hg, ratio_ken)
    fill_with_ratio_by_date(train_df_imputed, mask_ken_ds_kern,mask_nor_ds_kern, ratio_ken)

print(f"[After Impute] Missing num_sold = {train_df_imputed['num_sold'].isna().sum()}")
train_df_imputed.loc[train_df_imputed["id"] == 23719,  "num_sold"] = 4
train_df_imputed.loc[train_df_imputed["id"] == 207003, "num_sold"] = 195
print(f"[After Manual Fix] Missing num_sold = {train_df_imputed['num_sold'].isna().sum()}")

# ============================================================
# 3. 全体（日次合計）だけを予測（Ridge）
# ============================================================

day_train = train_df_imputed.groupby("date", as_index=False)["num_sold"].sum()
day_train = day_train.rename(columns={"num_sold": "total_num_sold"})

alpha2 = dict(zip(np.sort(train_df.country.unique()), ['CA','FI','IT','KE','NO','SG']))
h = {c: holidays.country_holidays(a, years=range(2010,2020)) for c,a in alpha2.items()}
train_df_tmp = train_df.copy()
test_df_tmp  = test_df.copy()
train_df_tmp['is_holiday'] = 0
test_df_tmp['is_holiday']  = 0
for c in alpha2:
    train_df_tmp.loc[train_df_tmp.country==c, 'is_holiday'] = train_df_tmp.date.isin(h[c]).astype(int)
    test_df_tmp.loc[test_df_tmp.country==c,  'is_holiday']  = test_df_tmp.date.isin(h[c]).astype(int)

holiday_count_train = train_df_tmp.groupby("date")["is_holiday"].sum().reset_index()
holiday_count_test  = test_df_tmp.groupby("date")["is_holiday"].sum().reset_index()

day_train = day_train.merge(holiday_count_train, on="date", how="left")
X_day = feature_engineer_day(day_train[["date","is_holiday"]].copy())
y_day = day_train["total_num_sold"].values

ridge = Ridge(tol=1e-2, max_iter=1_000_000, random_state=0)
ridge.fit(X_day, y_day)
pred_train_day = ridge.predict(X_day)
print("[Train-Day MAPE]", mean_absolute_percentage_error(y_day, pred_train_day))

test_day_df = test_df.groupby("date", as_index=False)["id"].first().drop(columns="id")
test_day_df = test_day_df.merge(holiday_count_test, on="date", how="left")
X_test_day = feature_engineer_day(test_day_df[["date","is_holiday"]].copy())

missing_cols = set(X_day.columns) - set(X_test_day.columns)
for mc in missing_cols:
    X_test_day[mc] = 0
extra_cols = set(X_test_day.columns) - set(X_day.columns)
if extra_cols:
    X_test_day = X_test_day.drop(columns=list(extra_cols))
X_test_day = X_test_day[X_day.columns]

pred_test_day = ridge.predict(X_test_day)
test_total_sales_dates = test_day_df[["date"]].copy()
test_total_sales_dates["day_num_sold"] = pred_test_day

train_total_sales_dates = day_train[["date"]].copy()
train_total_sales_dates["day_num_sold"] = pred_train_day

# ============================================================
# 4. 比率での割り振り（Store×Country×Product）
# ============================================================

# Store 比率（一定）
store_weights = (train_df_imputed.groupby("store")["num_sold"].sum()
                 / train_df_imputed["num_sold"].sum())
store_weights_df = store_weights.rename("store_ratio").reset_index()

# Country 年次比率（GDP）
# gdp_ratios_2: ["year"(int), "country", "ratio"]

# Product：2年周期の比率
product_df = train_df_imputed.groupby(["date","product"])["num_sold"].sum().reset_index()
product_ratio_df = product_df.pivot(index="date", columns="product", values="num_sold")
product_ratio_df = product_ratio_df.apply(lambda x: x/x.sum(), axis=1)
product_ratio_df = product_ratio_df.stack().rename("product_ratio").reset_index()

product_ratio_2017_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()
product_ratio_2018_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2016].copy()
product_ratio_2019_df = product_ratio_df.loc[product_ratio_df["date"].dt.year == 2015].copy()

product_ratio_2017_df["date"] = product_ratio_2017_df["date"] + pd.DateOffset(years=2)  # 2015→2017
product_ratio_2018_df["date"] = product_ratio_2018_df["date"] + pd.DateOffset(years=2)  # 2016→2018 (leap 注意)
product_ratio_2019_df["date"] = product_ratio_2019_df["date"] + pd.DateOffset(years=4)  # 2015→2019

forecasted_ratios_df = pd.concat([product_ratio_2017_df, product_ratio_2018_df, product_ratio_2019_df],
                                 ignore_index=True)

# NEW: うるう日シフトで 2018-02-28 に (date, product) の重複が出るため、平均で束ねて日別に再正規化
forecasted_ratios_df = (forecasted_ratios_df
                        .groupby(["date","product"], as_index=False)["product_ratio"].mean())

sum_per_day = forecasted_ratios_df.groupby("date")["product_ratio"].transform("sum")
forecasted_ratios_df["product_ratio"] = forecasted_ratios_df["product_ratio"] / sum_per_day
# 安全確認
dup_ct = forecasted_ratios_df.duplicated(["date","product"]).sum()
if dup_ct > 0:
    print(f"[WARN] Still duplicated (date,product): {dup_ct}")

# ============================================================
# 5. 予測合計を配分して最終予測作成
# ============================================================

test_sub_df = test_df[["id","date","country","store","product"]].copy()

# 日次合計
test_sub_df = test_sub_df.merge(test_total_sales_dates, on="date", how="left")
# Store 比率
test_sub_df = test_sub_df.merge(store_weights_df, on="store", how="left")
# Country 年次比率（GDP）
test_sub_df["year"] = test_sub_df["date"].dt.year
test_sub_df = test_sub_df.merge(
    gdp_ratios_2.rename(columns={"ratio":"country_ratio"}),
    on=["year","country"], how="left"
)
# Product 日次比率
test_sub_df = test_sub_df.merge(forecasted_ratios_df, on=["date","product"], how="left")

# 欠損が出た場合の安全策
for c in ["day_num_sold", "store_ratio", "country_ratio", "product_ratio"]:
    if c in test_sub_df:
        test_sub_df[c] = test_sub_df[c].fillna(0.0)

# 最終予測
test_sub_df["num_sold"] = test_sub_df["day_num_sold"] * test_sub_df["store_ratio"] \
                           * test_sub_df["country_ratio"] * test_sub_df["product_ratio"]
test_sub_df["num_sold"] = np.round(test_sub_df["num_sold"]).astype(int)

# 長さチェック（98,550 行のはず）
print("[Rows] test_sub_df:", len(test_sub_df))

# ============================================================
# 6. 提出
# ============================================================

submission = pd.read_csv(SAMPLE_SUB)

# NEW: id で安全に merge（順序ズレや万一の重複に強い）
submission = submission[["id"]].merge(
    test_sub_df[["id","num_sold"]],
    on="id", how="left"
)
assert len(submission) == len(pd.read_csv(SAMPLE_SUB)), "submission length mismatch"
submission.to_csv("submission.csv", index=False)
print("submission.csv written")

# （任意）学習期間で「合計の当てはまり」を可視化
try:
    plt.figure(figsize=(18,4))
    plt.plot(day_train["date"], day_train["total_num_sold"], label="train actual (daily total)", alpha=0.7)
    plt.plot(train_total_sales_dates["date"], train_total_sales_dates["day_num_sold"], label="train pred (daily total)", alpha=0.7)
    plt.title("Daily total: actual vs pred (train period)")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Plot skipped:", e)

# 確認
try:
    from IPython.display import display
    display(submission.head(2))
except:
    print(submission.head(2))


