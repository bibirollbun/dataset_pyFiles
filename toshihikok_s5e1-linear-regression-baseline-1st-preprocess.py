# ============================================================
# PGS501 (Playground Series S5E1)  分解型前処理 × Cabaxiom改造版
# 完全版コード（省略なし） 2025-07-24 fix: impute length mismatch
# ============================================================

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge  # 参考用

sns.set_style('darkgrid')

# -----------------------------
# 0. パス
# -----------------------------
TRAIN_PATH = "/kaggle/input/playground-series-s5e1/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e1/test.csv"
GDP_PATH   = "/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv"
SAMPLE_SUB = "/kaggle/input/playground-series-s5e1/sample_submission.csv"

# ============================================================
# 1. Utility 関数群
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
        hdays = [d for d in holidays.CountryHoliday(c, years=years)]
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
            for d, _ in sorted(holidays.CountryHoliday(c, years=y).items()):
                dfs.append(pd.DataFrame({"date": [pd.to_datetime(d)], "country": [c], "tmp": [1]}))
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
    new_df["day_of_week_group"] = new_df["day_of_week"].apply(lambda x: 0 if x <= 3 else (1 if x == 4 else (2 if x == 5 else 3)))

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

# ====== ★ 追加：安全な欠損補完ヘルパー関数 ★ ======
def fill_with_ratio_by_date(df, dst_mask, src_mask, ratio):
    """
    df: DataFrame (train_df_imputed)
    dst_mask: 埋めたい側のbool mask
    src_mask: 参照する側のbool mask (Norway等)
    ratio: 倍率
    - 日付をキーにmapして長さ不一致を防ぐ
    """
    # 参照元 series
    src_ser = (df.loc[src_mask, ["date", "num_sold"]]
                 .dropna()
                 .set_index("date")["num_sold"] * ratio)
    # 埋め先 index & date
    dst_idx = df.index[dst_mask]
    dst_dates = df.loc[dst_idx, "date"]
    df.loc[dst_idx, "num_sold"] = dst_dates.map(src_ser).values

# ============================================================
# 2. 読み込み & 欠損補完
# ============================================================

train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
original_train_df = train_df.copy()
test_df  = pd.read_csv(TEST_PATH,  parse_dates=["date"])

# GDP CSV（Cabaxiom式 ratio 用）
gdp_pc = pd.read_csv(GDP_PATH)
years = [str(y) for y in range(2010, 2021)]
gdp_filtered = gdp_pc.loc[gdp_pc["Country Name"].isin(train_df["country"].unique()), ["Country Name"] + years].set_index("Country Name")
for y in years:
    gdp_filtered[f"{y}_ratio"] = gdp_filtered[y] / gdp_filtered[y].sum()
gdp_ratios = gdp_filtered[[f"{y}_ratio" for y in years]].copy()
gdp_ratios.columns = [int(y) for y in years]
gdp_ratios = gdp_ratios.unstack().reset_index().rename(columns={"level_0":"year", 0:"ratio", "Country Name":"country"})
gdp_ratios["year"] = pd.to_datetime(gdp_ratios["year"], format="%Y")

# 可視化用複製（省略なし）
gdp_ratios_2 = gdp_ratios.copy()
gdp_ratios_2["year"] = pd.to_datetime(gdp_ratios_2["year"].astype(str)) + pd.offsets.YearEnd(1)
gdp_ratios_plot = pd.concat([gdp_ratios, gdp_ratios_2]).reset_index(drop=True)
gdp_ratios_2["year"] = gdp_ratios_2["year"].dt.year

train_df_imputed = train_df.copy()
print(f"[Before Impute] Missing num_sold = {train_df_imputed['num_sold'].isna().sum()}")

train_df_imputed["year"] = train_df_imputed["date"].dt.year

for year in train_df_imputed["year"].unique():
    target_ratio = gdp_ratios_2.loc[(gdp_ratios_2["year"] == year) & (gdp_ratios_2["country"] == "Norway"), "ratio"].values[0]
    ratio_can = gdp_ratios_2.loc[(gdp_ratios_2["year"] == year) & (gdp_ratios_2["country"] == "Canada"), "ratio"].values[0] / target_ratio
    ratio_ken = gdp_ratios_2.loc[(gdp_ratios_2["year"] == year) & (gdp_ratios_2["country"] == "Kenya"),  "ratio"].values[0] / target_ratio

    # Norway masks
    mask_nor_ds_hg = (train_df_imputed["country"]=="Norway") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_nor_pm_hg = (train_df_imputed["country"]=="Norway") & (train_df_imputed["store"]=="Premium Sticker Mart") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_nor_sl_hg = (train_df_imputed["country"]=="Norway") & (train_df_imputed["store"]=="Stickers for Less") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_nor_ds_kern= (train_df_imputed["country"]=="Norway") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Kerneler") & (train_df_imputed["year"]==year)

    # Canada targets
    mask_can_ds_hg = (train_df_imputed["country"]=="Canada") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_can_pm_hg = (train_df_imputed["country"]=="Canada") & (train_df_imputed["store"]=="Premium Sticker Mart") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())
    mask_can_sl_hg = (train_df_imputed["country"]=="Canada") & (train_df_imputed["store"]=="Stickers for Less") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())

    # Kenya targets
    mask_ken_ds_hg = (train_df_imputed["country"]=="Kenya") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year)
    mask_ken_pm_hg = (train_df_imputed["country"]=="Kenya") & (train_df_imputed["store"]=="Premium Sticker Mart") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())
    mask_ken_sl_hg = (train_df_imputed["country"]=="Kenya") & (train_df_imputed["store"]=="Stickers for Less") & (train_df_imputed["product"]=="Holographic Goose") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())
    mask_ken_ds_kern= (train_df_imputed["country"]=="Kenya") & (train_df_imputed["store"]=="Discount Stickers") & (train_df_imputed["product"]=="Kerneler") & (train_df_imputed["year"]==year) & (train_df_imputed["num_sold"].isna())

    # --- Canada DS HG（全置換：NaNも非NaNも）---
    fill_with_ratio_by_date(train_df_imputed, mask_can_ds_hg, mask_nor_ds_hg, ratio_can)

    # --- Canada PM HG (Missing only) ---
    fill_with_ratio_by_date(train_df_imputed, mask_can_pm_hg, mask_nor_pm_hg, ratio_can)

    # --- Canada SL HG (Missing only) ---
    fill_with_ratio_by_date(train_df_imputed, mask_can_sl_hg, mask_nor_sl_hg, ratio_can)

    # --- Kenya DS HG (全置換) ---
    fill_with_ratio_by_date(train_df_imputed, mask_ken_ds_hg, mask_nor_ds_hg, ratio_ken)

    # --- Kenya PM HG ---
    fill_with_ratio_by_date(train_df_imputed, mask_ken_pm_hg, mask_nor_pm_hg, ratio_ken)

    # --- Kenya SL HG ---
    fill_with_ratio_by_date(train_df_imputed, mask_ken_sl_hg, mask_nor_sl_hg, ratio_ken)

    # --- Kenya DS Kerneler ---
    fill_with_ratio_by_date(train_df_imputed, mask_ken_ds_kern, mask_nor_ds_kern, ratio_ken)

print(f"[After Impute] Missing num_sold = {train_df_imputed['num_sold'].isna().sum()}")
# 手動埋め
train_df_imputed.loc[train_df_imputed["id"] == 23719,  "num_sold"] = 4
train_df_imputed.loc[train_df_imputed["id"] == 207003, "num_sold"] = 195
print(f"[After Manual Fix] Missing num_sold = {train_df_imputed['num_sold'].isna().sum()}")

# ============================================================
# 3. 分解処理（前回と同じ。変更なし）
# ============================================================

train = train_df_imputed.drop(columns=["year"])
test  = test_df.copy()
df = pd.concat([train, test], sort=False).reset_index(drop=True)

df["year"]    = df["date"].dt.year
df["weekday"] = df["date"].dt.weekday
df["day_of_year"] = df["date"].dt.dayofyear

df = add_leap_adjusted_n_day(df, "date")
df, wave_cols = make_fourier_cols(df, "n_day", 9, 365, "wave")
df = create_near_holiday_flag(df, 10, "date", "country")

# GDP factor
gdp_pc_long = gdp_pc.loc[gdp_pc["Country Name"].isin(df["country"].unique()), ["Country Name"] + years]\
                 .rename(columns={"Country Name":"country"})
gdp_pc_long = gdp_pc_long.melt(id_vars="country", var_name="year", value_name="gdp")
gdp_pc_long["year"] = gdp_pc_long["year"].astype(int)
df = df.merge(gdp_pc_long, on=["country","year"], how="left")
df["gdp"] = df["gdp"].fillna(method="ffill")
df["gdp_factor"] = (-17643.346899 + 85.42355636 * df["gdp"]) / 365.0

# Store factor
exclude_mask = ~df["country"].isin(["Canada", "Kenya"])
df_no_can_ken = df[exclude_mask & df["num_sold"].notna()]
store_tmp = df_no_can_ken.groupby(["date","store"])["num_sold"].sum().reset_index()
total_per_day_tmp = df_no_can_ken.groupby("date")["num_sold"].sum().reset_index().rename(columns={"num_sold":"num_sold_total"})
store_tmp = store_tmp.merge(total_per_day_tmp, on="date")
store_tmp["store_factor"] = store_tmp["num_sold"] / store_tmp["num_sold_total"]
store_df = store_tmp.groupby("store")["store_factor"].mean().reset_index()
df = df.merge(store_df, on="store", how="left")

# Product factor
df["ratio"] = df["store_factor"] * df["gdp_factor"]
df["total"] = df["num_sold"] / df["ratio"]
mask_product_fit = (~df["country"].isin(["Canada","Kenya"])) & (df["date"] < dt.datetime(2017,1,1)) & df["num_sold"].notna()
df_pfit = df[mask_product_fit].copy()

df["product_factor"] = np.nan
for prod in df["product"].unique():
    tmp = df_pfit[df_pfit["product"]==prod]
    if len(tmp)==0:
        continue
    grp = tmp.groupby("date")
    Xp = grp[wave_cols].mean()
    yp = grp["total"].sum()
    model = fit_mape_linear_model(Xp, yp)
    mask_prod = (df["product"]==prod)
    df.loc[mask_prod, "product_factor"] = model.predict(df.loc[mask_prod, wave_cols])

# Day of week factor
df["ratio"] = df["store_factor"] * df["gdp_factor"] * df["product_factor"]
df["total"] = df["num_sold"] / df["ratio"]
mask_dow = (~df["country"].isin(["Canada","Kenya"])) & (df["near_holiday"]==0) & df["num_sold"].notna()
tmp = df[mask_dow]
mean_per_weekday = tmp.groupby("weekday")["total"].mean()
mean_mon_thu = mean_per_weekday[mean_per_weekday.index < 4].mean()
dow_factor = (mean_per_weekday / mean_mon_thu).rename("day_of_week_factor").reset_index()
df = df.merge(dow_factor, on="weekday", how="left")

# Sincos factor
df["ratio"] = df["store_factor"] * df["gdp_factor"] * df["product_factor"] * df["day_of_week_factor"]
df["total"] = df["num_sold"] / df["ratio"]
mask_sincos = (~df["country"].isin(["Canada","Kenya"])) & (df["near_holiday"]==0) & (df["date"] < dt.datetime(2017,1,1)) & df["num_sold"].notna()
grp = df[mask_sincos].groupby("date")
Xs = grp[wave_cols].mean()
ys = grp["total"].mean()
model_sincos = fit_mape_linear_model(Xs, ys)
df["sincos_factor"] = model_sincos.predict(df[wave_cols])

# Trend factor (optional)
use_trend = False
if use_trend:
    df["ratio"] = df["store_factor"] * df["gdp_factor"] * df["product_factor"] * df["day_of_week_factor"] * df["sincos_factor"]
    df["total"] = df["num_sold"] / df["ratio"]
    grp_tr = df.groupby(["date","n_day"])["total"].mean().reset_index()
    tr_train = grp_tr[(grp_tr["date"] >= dt.datetime(2013,1,1)) & (grp_tr["date"] < dt.datetime(2017,1,1))]
    Xtr = tr_train["n_day"].values.reshape(-1,1)
    ytr = tr_train["total"].values
    ridge = Ridge(alpha=0.1)
    ridge.fit(Xtr, ytr)
    df["trend_factor"] = ridge.predict(df["n_day"].values.reshape(-1,1))
    df.loc[df["date"] < dt.datetime(2013,1,1), "trend_factor"] = 1.0
else:
    df["trend_factor"] = 1.0

# Country factor
df["ratio"] = df["store_factor"] * df["gdp_factor"] * df["product_factor"] * df["day_of_week_factor"] * df["sincos_factor"] * df["trend_factor"]
df["total"] = df["num_sold"] / df["ratio"]
mask_country = df["product"]=="Kaggle"
country_factor_ser = df[mask_country & df["num_sold"].notna()].groupby("country")["total"].sum()
country_factor_ser = (country_factor_ser / country_factor_ser.median()).rename("country_factor")
df = df.merge(country_factor_ser, on="country", how="left")

# Holiday factor
df["ratio"] = df["store_factor"] * df["gdp_factor"] * df["product_factor"] * df["day_of_week_factor"] * df["sincos_factor"] * df["country_factor"] * df["trend_factor"]
df["total"] = df["num_sold"] / df["ratio"]
df, holiday_cols = build_holiday_shift_cols(df, "date", "country", 9)
mask_holy = (df["date"] > dt.datetime(2012,12,31)) & (df["date"] < dt.datetime(2017,1,1)) & df["num_sold"].notna()
Xh = df.loc[mask_holy, holiday_cols]
yh = df.loc[mask_holy, "total"]
model_hol = fit_mape_linear_model(Xh, yh)
df["holiday_factor"] = model_hol.predict(df[holiday_cols])

# New Year factor
df["ratio"] = df["store_factor"] * df["gdp_factor"] * df["product_factor"] * df["day_of_week_factor"] * df["sincos_factor"] * df["country_factor"] * df["holiday_factor"] * df["trend_factor"]
df["total"] = df["num_sold"] / df["ratio"]
df, ny_cols = build_newyear_cols(df, "date")
mask_ny = (df["date"] > dt.datetime(2012,12,31)) & (df["date"] < dt.datetime(2017,1,1)) & df["num_sold"].notna()
Xny = df.loc[mask_ny, ny_cols]
yny = df.loc[mask_ny, "total"]
model_ny = fit_mape_linear_model(Xny, yny)
df["new_years_factor"] = model_ny.predict(df[ny_cols])

# ============================================================
# 4. 基準需要（日次）をLasso
# ============================================================

factor_cols = [
    "country_factor", "store_factor", "gdp_factor", "product_factor",
    "day_of_week_factor", "sincos_factor", "holiday_factor", "new_years_factor",
    "trend_factor"
]
df["ratio_all"] = 1.0
for c in factor_cols:
    df["ratio_all"] *= df[c]

df["total_core"] = df["num_sold"] / df["ratio_all"]

train_mask = df["num_sold"].notna() & (df["date"] < dt.datetime(2017,1,1))
day_train = df.loc[train_mask, ["date","total_core"]].groupby("date")["total_core"].mean().reset_index()

# holiday count per day
alpha2 = dict(zip(np.sort(train_df.country.unique()), ['CA','FI','IT','KE','NO','SG']))
h = {c: holidays.country_holidays(a, years=range(2010,2020)) for c,a in alpha2.items()}
train_df_tmp = train_df.copy()
test_df_tmp  = test_df.copy()
train_df_tmp['is_holiday'] = 0
test_df_tmp['is_holiday']  = 0
for c in alpha2:
    train_df_tmp.loc[train_df_tmp.country==c, 'is_holiday'] = train_df_tmp.date.isin(h[c]).astype(int)
    test_df_tmp.loc[test_df_tmp.country==c,  'is_holiday']  = test_df_tmp.date.isin(h[c]).astype(int)
day_train = day_train.merge(train_df_tmp.groupby("date")["is_holiday"].sum().reset_index(), on="date", how="left")

train_feat = feature_engineer_day(day_train)
y_day = train_feat["total_core"]
X_day = train_feat.drop(columns=["total_core"])

lasso = Ridge(tol=1e-2, max_iter=1000000, random_state=0)
lasso.fit(X_day, y_day)
pred_train_day = lasso.predict(X_day)
print("[Train Day MAPE]", mean_absolute_percentage_error(y_day, pred_train_day))

test_day_df = test_df.groupby("date")["id"].first().reset_index().drop(columns="id")
test_day_df = test_day_df.merge(test_df_tmp.groupby("date")["is_holiday"].sum().reset_index(), on="date", how="left")
test_feat = feature_engineer_day(test_day_df)

missing_cols = set(X_day.columns) - set(test_feat.columns)
for mc in missing_cols:
    test_feat[mc] = 0
extra_cols = set(test_feat.columns) - set(X_day.columns)
test_feat = test_feat.drop(columns=list(extra_cols))
test_feat = test_feat[X_day.columns]

pred_test_day_base = lasso.predict(test_feat)
pred_day_base_df = pd.DataFrame({"date": test_day_df["date"], "base_total_pred": pred_test_day_base})
pred_day_base_train_df = pd.DataFrame({"date": day_train["date"], "base_total_pred": pred_train_day})
df = df.merge(pd.concat([pred_day_base_df, pred_day_base_train_df]), on="date", how="left")

df["pred_num_sold"] = df["ratio_all"] * df["base_total_pred"]
df.loc[df["country"]=="Kenya", "pred_num_sold"] *= (1 - 0.0007/2)  # 微調整

df["pred_num_sold_round"] = np.round(df["pred_num_sold"]).astype(int)

train_eval_mask = df["num_sold"].notna() & (df["date"] < dt.datetime(2017,1,1))
mape_train_final = mean_absolute_percentage_error(df.loc[train_eval_mask, "num_sold"],
                                                 df.loc[train_eval_mask, "pred_num_sold"])
print(f"[Final Train MAPE] {mape_train_final:.5f}")

submission = pd.read_csv(SAMPLE_SUB)
test_mask = df["num_sold"].isna()
submission["num_sold"] = df.loc[test_mask, "pred_num_sold_round"].values
submission.to_csv("submission.csv", index=False)
print("submission.csv written")

fig, ax = plt.subplots(1,1, figsize=(18,4))
ax.plot(df.loc[train_eval_mask, "date"], df.loc[train_eval_mask, "num_sold"], label="train actual", alpha=0.6)
ax.plot(df.loc[train_eval_mask, "date"], df.loc[train_eval_mask, "pred_num_sold"], label="train pred", alpha=0.6)
ax.set_title("Train period actual vs pred (sum of all rows)")
ax.legend()
plt.tight_layout()
plt.show()

display(submission.head(2))


