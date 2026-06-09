"""
 XGBoost Baseline for China_Real_Estate_Demand_Prediction_V2
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ========= CONFIG =========
DATA_DIR = "/kaggle/input/china-real-estate-demand-prediction/train"
USE_XGB = True

# --------------------------
# Helpers
# --------------------------
_month_map = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
              'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}

def month_to_ym(s):
    if pd.isna(s): return np.nan
    s = str(s).replace("_"," ").replace("/", " ").strip()
    parts = s.split()
    if len(parts) == 1 and "-" in parts[0]:
        parts = parts[0].split("-")
    if len(parts) >= 2:
        year = parts[0]
        mon = parts[1][:3]
        mm = _month_map.get(mon, None)
        if mm:
            return f"{year}-{mm}"
    return s

def parse_id(id_str):
    s = str(id_str).strip()
    if "_sector" in s:
        left, right = s.split("_sector",1)
        monthpart, sector = left, "sector " + right
    elif " sector " in s:
        left, right = s.split(" sector ",1)
        monthpart, sector = left, "sector " + right
    else:
        toks = s.split()
        monthpart = " ".join(toks[:-1])
        sector = toks[-1]
    ym = month_to_ym(monthpart.replace(" ", "-"))
    return ym, sector

# --------------------------
# Load data
# --------------------------
new_house = pd.read_csv(f"{DATA_DIR}/new_house_transactions.csv")
pre = pd.read_csv(f"{DATA_DIR}/pre_owned_house_transactions.csv")
land = pd.read_csv(f"{DATA_DIR}/land_transactions.csv")
poi = pd.read_csv(f"{DATA_DIR}/sector_POI.csv")

sample_sub = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/sample_submission.csv")
test = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/test.csv")

print("✅ Files loaded.")

# --------------------------
# Target setup
# --------------------------
target_col = [c for c in new_house.columns if "amount" in c.lower()][0]
base = new_house.rename(columns={target_col:"amount_new_house_transactions"}).copy()
base["ym"] = base["month"].apply(month_to_ym)
base["period"] = pd.to_datetime(base["ym"]+"-01")
base = base.rename(columns={"sector":"sector_norm"}).sort_values(["sector_norm","period"])
base["y"] = np.log1p(base["amount_new_house_transactions"])

# --------------------------
# Merge additional features
# --------------------------
def merge_on(base, tbl, n_keep=5, prefix=None):
    t = tbl.rename(columns={"sector": "sector_norm"})
    num_cols = [c for c in t.columns if pd.api.types.is_numeric_dtype(t[c]) and c not in ["month"]]
    num_cols = num_cols[:n_keep]
    t["ym"] = t["month"].apply(month_to_ym)
    keep = ["ym","sector_norm"]+num_cols
    t = t[keep]
    if prefix:
        t = t.rename(columns={c:f"{prefix}_{c}" for c in num_cols})
    return base.merge(t, on=["ym","sector_norm"], how="left")

base = merge_on(base, pre, 5, "pre")
base = merge_on(base, land, 5, "land")
base = base.merge(poi.rename(columns={"sector":"sector_norm"}), on="sector_norm", how="left")
base = base.fillna(0)

# --------------------------
# Feature engineering
# --------------------------
base = base.sort_values(["sector_norm","period"])
for L in [1,3,6,12]:
    base[f"lag_amt_{L}"] = base.groupby("sector_norm")["amount_new_house_transactions"].shift(L).fillna(0)

base["rmean_3"] = base.groupby("sector_norm")["amount_new_house_transactions"].shift(1).rolling(3,min_periods=1).mean().reset_index(level=0,drop=True).fillna(0)
base["rmean_6"] = base.groupby("sector_norm")["amount_new_house_transactions"].shift(1).rolling(6,min_periods=1).mean().reset_index(level=0,drop=True).fillna(0)

base["month_num"] = base["period"].dt.month
base["year"] = base["period"].dt.year
base["trend"] = (base["period"].dt.year - base["period"].dt.year.min()) * 12 + \
                (base["period"].dt.month - base["period"].dt.month.min())

# --------------------------
# Train / validation split
# --------------------------
last_period = base["period"].max()
train_df = base[base["period"]<last_period]
val_df = base[base["period"]==last_period]

features = [c for c in base.columns if c not in ["month","ym","period","sector_norm","amount_new_house_transactions","y"]]
X_train,y_train = train_df[features],train_df["y"]
X_val,y_val = val_df[features],val_df["y"]

# --------------------------
# XGBoost model
# --------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective":"reg:squarederror",
    "eval_metric":"rmse",
    "eta":0.05,
    "max_depth":6,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "seed":42
}

watchlist = [(dtrain,"train"),(dval,"valid")]
model = xgb.train(params, dtrain, num_boost_round=2000, evals=watchlist, 
                  early_stopping_rounds=100, verbose_eval=100)

val_pred = model.predict(dval)
print("Val RMSE log:", mean_squared_error(y_val,val_pred,squared=False))
print("Val RMSE orig:", mean_squared_error(np.expm1(y_val),np.expm1(val_pred),squared=False))

# --------------------------
# Predict test
# --------------------------
test = test.copy()
test[["ym","sector_norm"]] = test["id"].apply(lambda x: pd.Series(parse_id(x)))
test["period"] = pd.to_datetime(test["ym"]+"-01")
test["month_num"] = test["period"].dt.month
test["year"] = test["period"].dt.year
test["trend"] = (test["period"].dt.year - base["period"].dt.year.min()) * 12 + \
                (test["period"].dt.month - base["period"].dt.month.min())

latest = base.sort_values("period").groupby("sector_norm").tail(1).set_index("sector_norm")
for f in features:
    test[f] = test["sector_norm"].map(latest[f]).fillna(0)

dtest = xgb.DMatrix(test[features])
test_pred = model.predict(dtest)
test["new_house_transaction_amount"] = np.expm1(test_pred)

# enforce rule: unseen (ym,sector) → 0
train_pairs = set(zip(base["ym"], base["sector_norm"]))
test["new_house_transaction_amount"] = test.apply(
    lambda r: 0 if (r["ym"],r["sector_norm"]) not in train_pairs else r["new_house_transaction_amount"],
    axis=1
)

submission = test[["id","new_house_transaction_amount"]]
submission.to_csv("submission_xgb.csv",index=False)
print("✅ Submission saved: submission_xgb.csv")
print(submission.head())


