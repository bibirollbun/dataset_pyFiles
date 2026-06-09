# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# --- Imports ---
import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import ExtraTreesRegressor

# --- Paths ---
pth = "/kaggle/input/china-real-estate-demand-prediction"

# --- Load data ---
ci = (pl.read_csv(f"{pth}/train/city_indexes.csv")
        .head(6).fill_null(-1)
        .drop("total_fixed_asset_investment_10k")
        .rename(lambda c: ("" if c in ["sector","month"] else "ci_") + c))

csi = pl.read_csv(f"{pth}/train/city_search_index.csv")  # optional

sp = (pl.read_csv(f"{pth}/train/sector_POI.csv")
        .fill_null(-1)
        .rename(lambda c: ("" if c in ["sector","month"] else "sp_") + c))

train_lt   = pl.read_csv(f"{pth}/train/land_transactions.csv", infer_schema_length=10000
             ).rename(lambda c: ("" if c in ["sector","month"] else "lt_") + c)

train_ltns = pl.read_csv(f"{pth}/train/land_transactions_nearby_sectors.csv"
             ).rename(lambda c: ("" if c in ["sector","month"] else "ltns_") + c)

train_pht  = pl.read_csv(f"{pth}/train/pre_owned_house_transactions.csv"
             ).rename(lambda c: ("" if c in ["sector","month"] else "pht_") + c)

train_phtns = pl.read_csv(f"{pth}/train/pre_owned_house_transactions_nearby_sectors.csv"
              ).rename(lambda c: ("" if c in ["sector","month"] else "phtns_") + c)

train_nht  = pl.read_csv(f"{pth}/train/new_house_transactions.csv"
             ).rename(lambda c: ("" if c in ["sector","month"] else "nht_") + c)

train_nhtns = pl.read_csv(f"{pth}/train/new_house_transactions_nearby_sectors.csv"
              ).rename(lambda c: ("" if c in ["sector","month"] else "nhtns_") + c)

test = pl.read_csv(f"{pth}/test.csv").with_columns(
    id_=pl.col("id").str.split("_")
).with_columns(
    month=pl.col("id_").list.get(0),
    sector=pl.col("id_").list.get(1)
).drop("id_")

month_codes = {
    'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
    'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12
}

# --- Join & feature prep ---
data = (pl.DataFrame(train_nht["month"].unique())
    .join(pl.DataFrame(train_nht["sector"].unique().to_list()+["sector 95"]).rename({"column_0":"sector"}), how="cross")
    .with_columns(
        sector_id=pl.col("sector").str.split(" ").list.get(1).cast(pl.Int8),
        year=pl.col("month").str.split("-").list.get(0).cast(pl.Int16),
        month_num=pl.col("month").str.split("-").list.get(1).replace(month_codes).cast(pl.Int8),
    )
    .with_columns(time=((pl.col("year")-2019)*12 + pl.col("month_num") - 1).cast(pl.Int8))
    .sort("sector_id","time")
    .join(train_nht,   on=["sector","month"], how="left").fill_null(0)
    .join(train_nhtns, on=["sector","month"], how="left").fill_null(-1)
    .join(train_pht,   on=["sector","month"], how="left").fill_null(-1)
    .join(train_phtns, on=["sector","month"], how="left").fill_null(-1)
    .join(ci.rename({"ci_city_indicator_data_year":"year"}), on=["year"], how="left").fill_null(-1)
    .join(sp, on=["sector"], how="left").fill_null(-1)
    .join(train_lt,   on=["sector","month"], how="left").fill_null(-1)
    .join(train_ltns, on=["sector","month"], how="left").fill_null(-1)
    .with_columns(cs.float().cast(pl.Float32))
)

# downcast ints, drop all-zero int columns
for col in data.columns:
    if data[col].dtype == pl.Int64:
        cmin, cmax = data[col].min(), data[col].max()
        if cmin == 0 and cmax == 0:
            data = data.drop(col)
        elif cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
            data = data.with_columns(pl.col(col).cast(pl.Int8))
        elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
            data = data.with_columns(pl.col(col).cast(pl.Int16))
        elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
            data = data.with_columns(pl.col(col).cast(pl.Int32))

data = data.drop("month","sector","year")

# simple temporal joins (as in your script)
data2 = data.sort("time","sector_id")
for m in [1,2,12]:
    data2 = data2.join(
        data.drop("month_num").with_columns(pl.col("time")+m),
        on=["sector_id","time"], how="left", suffix=f"_{m}"
    )
data2 = data2.sort("time","sector_id")

# label & seasonality
lag = -1
data3 = (data2.with_columns(
            pl.col("nht_amount_new_house_transactions").shift(lag).over("sector_id").alias("label"),
            cs=((pl.col("month_num")-1)/6*np.pi).cos(),
            sn=((pl.col("month_num")-1)/6*np.pi).sin(),
            cs6=((pl.col("month_num")-1)/3*np.pi).cos(),
            sn6=((pl.col("month_num")-1)/3*np.pi).sin(),
            cs3=((pl.col("month_num")-1)/1.5*np.pi).cos(),
            sn3=((pl.col("month_num")-1)/1.5*np.pi).sin(),
         )
        # .drop_nulls(subset=["label"])  # keep NA for CatBoost handling via fill
)
data3 = data3.drop("sector_id")  # matches your original

# --- Splits / pools ---
cat_features = ["month_num"]
border  = 66+lag-1
border1 = 6*3

train_mask = (pl.col("time")<=border) & (pl.col("time")>border1)
val_mask   = (pl.col("time")>border) & (pl.col("time")<=66+lag)
t66_mask   = (pl.col("time")==66)

# CatBoost pools
trainPool = Pool(
    data = data3.filter(train_mask).drop(["label"]).to_pandas().fillna(-2),
    label= data3.filter(train_mask)["label"].to_pandas(),
    cat_features=cat_features
)
testPool = Pool(
    data = data3.filter(val_mask).drop(["label"]).to_pandas().fillna(-2),
    label= data3.filter(val_mask)["label"].to_pandas(),
    cat_features=cat_features
)
testPool2 = Pool(  # time==66
    data = data3.filter(t66_mask).drop(["label"]).to_pandas().fillna(-2),
    cat_features=cat_features
)

# Pandas frames for ExtraTrees + stacking
X_train_df = data3.filter(train_mask).drop(["label"]).to_pandas().fillna(-2)
y_train    = data3.filter(train_mask)["label"].to_pandas().values.ravel()

X_val_df   = data3.filter(val_mask).drop(["label"]).to_pandas().fillna(-2)
y_val      = data3.filter(val_mask)["label"].to_pandas().values.ravel()

X_t66_df   = data3.filter(t66_mask).drop(["label"]).to_pandas().fillna(-2)

# --- Custom metric for blending selection ---
def custom_score(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0: return 0.0
    ape = np.abs((y_true - np.maximum(y_pred,0)) / np.maximum(y_true, eps))
    bad_rate = np.mean(ape > 1.0)
    if bad_rate > 0.30: return 0.0
    good_ape = ape[ape <= 1.0]
    if good_ape.size == 0: return 0.0
    mape = np.mean(good_ape)
    fraction = good_ape.size / y_true.size
    scaled_mape = mape / (fraction + eps)
    return max(0.0, 1.0 - scaled_mape)

class CustomMetric:
    def is_max_optimal(self): return True
    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        score = custom_score(target, approxes[0])
        return score, 1
    def get_final_error(self, error, weight): return error

class CustomObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        result = []
        for i in range(len(targets)):
            # non-convex custom: amplify when underpredicting heavily
            der1 = np.sign(targets[i] - approxes[i]) if (2*targets[i] - approxes[i]) < 0 else np.sign(targets[i] - approxes[i]) * 5
            result.append((der1, 0.0))
        return result

# --- Train CatBoost ---
cb = CatBoostRegressor(
    iterations=21000,
    learning_rate=0.0125,
    one_hot_max_size=256,
    custom_metric=["RMSE","MAPE","SMAPE","MAE"],
    loss_function=CustomObjective(),
    eval_metric=CustomMetric(),
    l2_leaf_reg=0.3,
    random_seed=4,
    verbose=1000
)
cb.fit(trainPool, eval_set=testPool, verbose=1000)

# --- Train ExtraTrees (stack component) ---
et = ExtraTreesRegressor(
    n_estimators=800,
    n_jobs=-1,
    random_state=42
)
et.fit(X_train_df, y_train)

# --- Blend weight tuned on validation by custom_score ---
cb_val = cb.predict(testPool)
et_val = et.predict(X_val_df)

best_alpha, best_score = 0.5, -1.0
for a in np.linspace(0.0, 1.0, 21):
    blend_val = a*cb_val + (1.0-a)*et_val
    sc = custom_score(y_val, blend_val)
    if sc > best_score:
        best_score, best_alpha = sc, a

print(f"[Blend] best_alpha={best_alpha:.2f} | val_score={best_score:.6f} | "
      f"cat_val={custom_score(y_val, cb_val):.6f} | et_val={custom_score(y_val, et_val):.6f}")

# --- Final predictions for time==66 (BLENDED) ---
cb_t66 = cb.predict(testPool2)
et_t66 = et.predict(X_t66_df)
month = np.maximum(best_alpha*cb_t66 + (1.0-best_alpha)*et_t66, 0.0)

# Safety: ensure not identical to CatBoost-only (prevents accidental overwrite)
assert not np.allclose(month, cb_t66), "Stacked blend was not used; 'month' equals CatBoost-only."

# --- Index pruning (as in your script) ---
for i in [11,38,40,43,48,51,52,57,71,72,73,74,81,86,88,94,95]:
    month[i] = 0

# --- Write submission ---
sub = pd.read_csv(f"{pth}/sample_submission.csv")
for m in range(12):
    sub.loc[[i + m*96 for i in range(96)], "new_house_transaction_amount"] = month
sub.to_csv("submission.csv", index=False)

print("submission.csv written with CatBoost + ExtraTrees blended predictions.")


