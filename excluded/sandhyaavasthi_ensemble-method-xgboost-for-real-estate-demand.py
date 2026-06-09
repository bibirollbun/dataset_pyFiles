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


# import required libraries

import os, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import calendar



# load required data into dataframe

df_new = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv")
df_new_nb = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv")
df_pre = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv")
df_pre_nb = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv")
df_land = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv")
df_land_nb = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv")
poi = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv")
city_idx = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv")
city_search_idx = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv")
comp_test_df = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/test.csv")


df_new.shape


df_new.head()


def custom_competition_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    ape = np.empty_like(y_true, dtype=float)
    zero_mask = (y_true == 0)
    nonzero_mask = ~zero_mask
    ape[nonzero_mask] = np.abs(y_pred[nonzero_mask] - y_true[nonzero_mask]) / np.abs(y_true[nonzero_mask])
    ape[zero_mask] = np.where(np.abs(y_pred[zero_mask]) == 0, 0.0, np.inf)
    frac_over_1 = np.mean(ape > 1.0)
    if frac_over_1 > 0.30:
        return 0.0
    ok_mask = (ape <= 1.0)
    frac_le_1 = np.mean(ok_mask)
    mape_ok = np.mean(ape[ok_mask])
    scaled_mape = mape_ok / frac_le_1
    score = 1.0 - scaled_mape
    return float(score)


# ==== 2 Extract Year Month and Sector Num ====
df_new[["Year", "Month"]] = df_new["month"].str.split("-", expand=True)
df_new["Year"] = df_new["Year"].astype(int)
df_new["sector_num"] = df_new["sector"].str.extract(r'(\d+)').astype(int)


# ==== 3 Convert month to number ====
month_codes = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
df_new["Month_num"] = df_new["Month"].map(month_codes)
df_new.drop(columns=["month","sector","Month"],inplace=True) # I droped previous columns


# ==== Prepare Missing Months ====

years   = sorted(df_new["Year"].unique())
months  = sorted(df_new["Month_num"].unique())
sectors = sorted(df_new["sector_num"].unique())

# I filtered months greater than 07.2024
CUT_Y, CUT_M = 2024, 7

full_index = pd.DataFrame(
    [(y, m, s)
     for y in years
     for m in months
     for s in sectors
     if (y < CUT_Y) or (y == CUT_Y and m <= CUT_M)],
    columns=["Year", "Month_num", "sector_num"]
)


# ==== Add Missing Months to Data ====

df_full = pd.merge(full_index, df_new, on=["Year", "Month_num", "sector_num"], how="left")
df_full = df_full.sort_values(["sector_num", "Year", "Month_num"])# ==== Add Missing Months to Data ====

df_full = pd.merge(full_index, df_new, on=["Year", "Month_num", "sector_num"], how="left")
df_full = df_full.sort_values(["sector_num", "Year", "Month_num"])


# Our target is the amount_new_house_transactions value after 12 months
df_full["target_12m_ahead"] = (
    df_full.groupby("sector_num")["amount_new_house_transactions"]
      .shift(-12)
)


#create a train data set for XGBoost which is between 01.01.2019 to 01.07.2023 (DD.MM.YEAR)
df_train = df_full[
    (df_full["Year"] < 2023) | ((df_full["Year"] == 2023) & (df_full["Month_num"] <= 7))
]
df_train_sorted = df_train.sort_values(["Year", "Month_num", "sector_num"]).reset_index(drop=True)

#Rest of the df_new will be used as a Test. 


TARGET_COL = "target_12m_ahead"
drop_cols = [c for c in [TARGET_COL] if c in df_train.columns]
X = df_train.drop(columns=drop_cols) #Features
y = df_train[TARGET_COL] #Target


# I split the train data as train and validation set.
# Each sector's first %80 month is used as train

X_tr_list, X_val_list, y_tr_list, y_val_list = [], [], [], []

for sector, group in df_train_sorted.groupby("sector_num"):
    group = group.sort_values(["Year", "Month_num"]).reset_index(drop=True)
    drop_cols = [c for c in [TARGET_COL] if c in group.columns]
    X_grp = group.drop(columns=drop_cols)
    y_grp = group[TARGET_COL]
# %80 / %20 split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_grp, y_grp, test_size=0.20, shuffle=False
    )
    X_tr_list.append(X_tr)
    X_val_list.append(X_val)
    y_tr_list.append(y_tr)
    y_val_list.append(y_val)

X_tr = pd.concat(X_tr_list).reset_index(drop=True)
X_val = pd.concat(X_val_list).reset_index(drop=True)
y_tr = pd.concat(y_tr_list).reset_index(drop=True)
y_val = pd.concat(y_val_list).reset_index(drop=True)

print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)


def clean_xy(X, y):
    y = pd.to_numeric(y, errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)
    ok = y.notna()
    Xc, yc = X.loc[ok].copy(), y.loc[ok].copy()
    return Xc, yc

#Some target values has NaN, so I dropped them for train.
#You can fill them as, 0 if you want
X_tr, y_tr = clean_xy(X_tr, y_tr)
X_val, y_val = clean_xy(X_val, y_val)


model = XGBRegressor(
    objective="reg:tweedie", #I assumed that target value is non-negative number          
    tweedie_variance_power=1.4,   
    n_estimators=20000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",
    eval_metric="rmse"
)


model.fit(X_tr, y_tr, verbose=False)


y_val_pred = model.predict(X_val)


y_true = np.asarray(y_val)
y_pred = np.asarray(y_val_pred)

mse  = mean_squared_error(y_true, y_pred) 
rmse = np.sqrt(mse)                        # RMSE
r2   = r2_score(y_true, y_pred)
custom = custom_competition_score(y_true,y_pred)

print(f"[VAL] RMSE: {rmse:,.3f} | R^2: {r2:,.4f}| Custom: {custom:,.4f}")


X_train_all = pd.concat([X_tr, X_val], ignore_index=False)
y_train_all = pd.concat([y_tr, y_val ], ignore_index=False)


model.fit(X_train_all, y_train_all, verbose=False)


df_test = df_full[
    (df_full["Year"] > 2023) | ((df_full["Year"] == 2023) & (df_full["Month_num"] > 7))
]
df_test.drop(columns=["target_12m_ahead"],inplace=True)


test_result = model.predict(df_test)


df_test["target_predicted"] = pd.Series(test_result, index=df_test.index, dtype="float32")


mabbr = {i: calendar.month_abbr[i] for i in range(1, 13)}

map_df = df_test[["Year", "Month_num", "sector_num", "target_predicted"]].copy()
map_df["id"] = (
    (map_df["Year"].astype(int) + 1).astype(str) + " " +
    map_df["Month_num"].astype(int).map(mabbr) + "_sector " +
    map_df["sector_num"].astype(int).astype(str)
)

map_df = map_df.groupby("id", as_index=False, sort=False)["target_predicted"].last()

comp_test_df = comp_test_df.merge(map_df, on="id", how="left")
comp_test_df["new_house_transaction_amount"] = (
    comp_test_df["new_house_transaction_amount"]
    .fillna(comp_test_df["target_predicted"])
)

comp_test_df = comp_test_df.drop(columns=["target_predicted"])


comp_test_df=comp_test_df.fillna(0) #Sector 95 is not exist in train data, so I filled them as 0


comp_test_df


comp_test_df.to_csv("submission.csv",index=False)


submission=pd.read_csv("/kaggle/working/submission.csv")


submission

