# Import basic libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: set some display or plot styles
pd.set_option("display.max_columns", 50)
sns.set_style("whitegrid")

print("Libraries imported successfully.")


import os

# Path to the competition data folder
input_path = "/kaggle/input/m5-forecasting-accuracy"

print("Files in the input directory:")
for filename in os.listdir(input_path):
    print("  -", filename)


calendar_path = f"{input_path}/calendar.csv"

calendar = pd.read_csv(calendar_path)
print("Calendar shape:", calendar.shape)

# Display the first 5 rows
calendar.head()


print("-- Missing values in calendar --")
print(calendar.isnull().sum())

print("\n-- Data types (info) --")
calendar.info()

print("\n-- Descriptive statistics --")
print(calendar.describe(include='all'))


sell_prices_path = f"{input_path}/sell_prices.csv"

sell_prices = pd.read_csv(sell_prices_path)
print("Sell Prices shape:", sell_prices.shape)

# Display first 5 rows
sell_prices.head()


print("-- Missing values in sell_prices --")
print(sell_prices.isnull().sum())

print("\n-- Data types (info) --")
sell_prices.info()

print("\n-- Descriptive stats for sell_price --")
print(sell_prices["sell_price"].describe())


sales_eval_path = f"{input_path}/sales_train_evaluation.csv"

sales_eval = pd.read_csv(sales_eval_path)
print("Sales Evaluation shape:", sales_eval.shape)

# Display first 5 rows
sales_eval.head()


print("-- Missing values in sales_eval --")
print(sales_eval.isnull().sum().sum())  # sum of all missing across columns

print("\n-- Data types (info) --")
sales_eval.info(verbose=False)  # 'verbose=False' to avoid printing all 1947 columns


print("Unique item IDs:", sales_eval["item_id"].nunique())
print("Unique store IDs:", sales_eval["store_id"].nunique())
print("Unique dept IDs:", sales_eval["dept_id"].nunique())
print("Unique cat IDs:", sales_eval["cat_id"].nunique())
print("Unique state IDs:", sales_eval["state_id"].nunique())


import gc
import pandas as pd
import numpy as np

# Optional: If you want to reduce memory usage further, define a helper function
def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"  Memory usage: {start_mem:.2f} MB ->", end=" ")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type not in [object, 'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
        else:
            # optionally convert object columns to category if appropriate
            if col_type == object:
                if df[col].nunique() / len(df[col]) < 0.5:
                    df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"{end_mem:.2f} MB (reduced by {(start_mem - end_mem)/start_mem*100:.1f}%)")
    return df

# Make sure you already have "sales_eval", "calendar", and "sell_prices" dataframes loaded
# from your previous steps.

all_stores = sales_eval["store_id"].unique()
print("Found store_ids:", all_stores)

for store in all_stores:
    print(f"\nProcessing store: {store}")

    # (1) Filter sales_eval for this store
    df_store = sales_eval[sales_eval["store_id"] == store].copy()

    # (Optional) reduce memory usage for df_store
    df_store = reduce_memory_usage(df_store)

    # (2) Melt the subset
    fixed_cols = ["id","item_id","dept_id","cat_id","store_id","state_id"]
    date_cols = [c for c in df_store.columns if c.startswith("d_")]
    df_melted_sub = pd.melt(
        df_store,
        id_vars=fixed_cols,
        value_vars=date_cols,
        var_name="d",
        value_name="sales"
    )
    del df_store
    gc.collect()

    # (3) Merge with calendar on 'd'
    df_cal_sub = pd.merge(df_melted_sub, calendar, how="left", on="d")
    del df_melted_sub
    gc.collect()

    # (4) Filter sell_prices for this store, then merge
    sp_sub = sell_prices[sell_prices["store_id"] == store].copy()
    sp_sub = reduce_memory_usage(sp_sub)

    df_merged_sub = pd.merge(
        df_cal_sub,
        sp_sub,
        how="left",
        on=["store_id","item_id","wm_yr_wk"]
    )
    del df_cal_sub, sp_sub
    gc.collect()

    # (Optional) reduce memory usage again
    df_merged_sub = reduce_memory_usage(df_merged_sub)

    # (5) Save to disk
    out_path = f"/kaggle/working/merged_{store}.pkl"
    df_merged_sub.to_pickle(out_path)
    print(f"  Saved merged data for store={store}, shape={df_merged_sub.shape} -> {out_path}")

    # (6) Clear memory
    del df_merged_sub
    gc.collect()


df_ca1 = pd.read_pickle("/kaggle/working/merged_CA_1.pkl")
print("CA_1 data shape:", df_ca1.shape)
print(df_ca1.head())

# For instance, check distribution of 'sales'
import matplotlib.pyplot as plt
import seaborn as sns

sample_ca1 = df_ca1.sample(100000, random_state=42)  # sample to avoid heavy plotting
plt.figure(figsize=(8,4))
sns.histplot(data=sample_ca1, x="sales", bins=50, kde=True)
plt.title("Distribution of sales (sample of 100k) - CA_1")
plt.show()


############################
# NAIVE SUBMISSION (One Code Block)
############################

# 1) Read the wide "sales_train_evaluation.csv"
#    (Make sure you've already imported pandas, etc. in earlier cells.)
eval_path = "/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv"
df_eval = pd.read_csv(eval_path)

# 2) Compute the average sales for the last 28 days (d_1914 .. d_1941)
last_28_cols = [f"d_{i}" for i in range(1914, 1942)]
df_eval["naive_mean"] = df_eval[last_28_cols].mean(axis=1)

# 3) Duplicate these rows to form "_validation" counterparts
df_val = df_eval.copy()
df_val["id"] = df_val["id"].str.replace("_evaluation", "_validation")

# 4) Fill F1..F28 columns with the naive_mean for each part
def fill_forecasts(df, pred_col="naive_mean"):
    """
    Create columns F1..F28 from a single prediction column (pred_col).
    df must have columns ["id", pred_col].
    """
    result = df[["id", pred_col]].copy()
    for i in range(1, 29):
        result[f"F{i}"] = result[pred_col]
    result.drop(columns=[pred_col], inplace=True)
    return result

val_part = fill_forecasts(df_val, pred_col="naive_mean")   # "_validation"
eval_part = fill_forecasts(df_eval, pred_col="naive_mean") # "_evaluation"

# 5) Combine them into one final DataFrame of 60,980 rows
final_sub = pd.concat([val_part, eval_part], axis=0)
cols = ["id"] + [f"F{i}" for i in range(1,29)]
final_sub = final_sub[cols]

# 6) Write out "submission.csv" for Kaggle
final_sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", final_sub.shape)

############################
# End of naive submission code
############################

# NOTE:
#  - Now you have a file "submission.csv" with 30490 "_validation" rows + 30490 "_evaluation" rows.
#  - You can go to the right side panel or "Output" section, verify the file, and click "Submit to Competition".
#  - This baseline won't be very accurate, but ensures you have a valid ID structure for Kaggle's scoring.


# Load merged data for store CA_1 
df_ca1 = pd.read_pickle("/kaggle/working/merged_CA_1.pkl")

print("Shape of df_ca1:", df_ca1.shape)
print(df_ca1.head(3))

# Check the columns to see what information is available
print("\nColumns in df_ca1:", df_ca1.columns.tolist())


# Convert "d" (like "d_1") into an integer day index, and sample a portion
df_ca1["d_num"] = df_ca1["d"].str[2:].astype(int)

print("Added 'd_num' column. First few values:")
print(df_ca1[["d", "d_num"]].head(5))

# Optional: Take a small sample for further exploration
df_ca1_sample = df_ca1.sample(100_000, random_state=42)

print("\nSampled 100,000 rows. Shape:", df_ca1_sample.shape)
print(df_ca1_sample.head(3))


# Group by the "date" column to see average sales per day in the sample
avg_sales_by_date = (
    df_ca1_sample
    .groupby("date", as_index=False)["sales"]
    .mean()
    .rename(columns={"sales": "avg_sales_sample"})
)

print("Shape of avg_sales_by_date:", avg_sales_by_date.shape)
print(avg_sales_by_date.head(5))

# (Optional) Sort by date if needed for a clearer chronological view
avg_sales_by_date = avg_sales_by_date.sort_values("date")
print("\nFirst 5 rows after sorting by date:")
print(avg_sales_by_date.head(5))


# Plot avg_sales_sample vs. date
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,4))
sns.lineplot(data=avg_sales_by_date, x="date", y="avg_sales_sample")
plt.title("Daily Average Sales (Sample of 100k rows)")
plt.xlabel("Date")
plt.ylabel("Avg Sales in Sample")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Sort by item (id) and d_num, then group by 'id' to apply a shift(7) on the 'sales' column

df_ca1_sample = df_ca1_sample.sort_values(["id", "d_num"]).copy()

df_ca1_sample["sales_lag7"] = (
    df_ca1_sample
    .groupby("id")["sales"]
    .shift(7)
)

print("Created 'sales_lag7' feature (lag=7 days). First 10 rows after sorting:")
print(df_ca1_sample[["id","d_num","sales","sales_lag7"]].head(10))


# 1) Fill missing values in 'sales_lag7' with 0 (a quick choice)
df_ca1_sample["sales_lag7"] = df_ca1_sample["sales_lag7"].fillna(0)

# 2) Define a cutoff, for example d_num < 1500 as "train", d_num >= 1500 as "val"
train_mask = df_ca1_sample["d_num"] < 1500
val_mask   = df_ca1_sample["d_num"] >= 1500

df_train = df_ca1_sample[train_mask].copy()
df_val   = df_ca1_sample[val_mask].copy()

print("Training set shape:", df_train.shape)
print("Validation set shape:", df_val.shape)

# Show a sample of each
print("\nSample rows from training set:")
print(df_train.sample(5))

print("\nSample rows from validation set:")
print(df_val.sample(5))


# We assume df_ca1_sample is already available
# and you have already split df_train, df_val based on d_num < 1500, etc.
# So let's combine them back if needed, or create features in both sets.

import pandas as pd

# Combine train + val for feature engineering (optional):
df_all = pd.concat([df_train, df_val], axis=0, sort=False).copy()
df_all = df_all.sort_values(["id", "d_num"]).reset_index(drop=True)

# Define which lags and rolling windows we want
lags = [7, 14, 28]
rolling_windows = [7, 28]

# Group by item id
grouped = df_all.groupby("id", observed=False)

# Create lag features
for lag in lags:
    col_name = f"sales_lag{lag}"
    df_all[col_name] = grouped["sales"].shift(lag)

# Create rolling mean features
for w in rolling_windows:
    col_name = f"sales_rollmean{w}"
    df_all[col_name] = (
        grouped["sales"]
        .shift(1)                      # shift by 1 to avoid including current day
        .rolling(w, min_periods=1)
        .mean()
    )

# Fill missing values for all new features
feature_cols = [f"sales_lag{x}" for x in lags] + [f"sales_rollmean{x}" for x in rolling_windows]
df_all[feature_cols] = df_all[feature_cols].fillna(0)

print("Created multiple lag/rolling features. Sample of columns:")
print(df_all[["id", "d_num", "sales"] + feature_cols].head(10))

# Now re-split into train and val
df_train = df_all[df_all["d_num"] < 1500].copy()
df_val   = df_all[df_all["d_num"] >= 1500].copy()

print(f"\nTrain shape: {df_train.shape}, Val shape: {df_val.shape}")


from lightgbm import LGBMRegressor
import numpy as np

# Define our feature columns and target
feature_cols = [f"sales_lag{x}" for x in [7, 14, 28]] + [f"sales_rollmean{x}" for x in [7, 28]]
target_col = "sales"

X_train = df_train[feature_cols]
y_train = df_train[target_col]

X_val   = df_val[feature_cols]
y_val   = df_val[target_col]

print("Using features:", feature_cols)

# Initialize and fit the model
model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse"
)

# Predict on validation
pred_val = model.predict(X_val)

# Calculate RMSE
rmse = np.sqrt(np.mean((pred_val - y_val)**2))
print("Validation RMSE:", rmse)

# best_iteration_ may always be the final iteration if early stopping isn't used
print("Best iteration:", model.best_iteration_)


import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import os

# Example set of stores (adjust to your actual list)
all_stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]

# A place to store predictions or intermediate results
all_preds = []

# Loop over store IDs
for store in all_stores:
    pickle_path = f"/kaggle/working/merged_{store}.pkl"
    
    if not os.path.exists(pickle_path):
        print(f"File not found for {store}: {pickle_path}, skipping.")
        continue
    
    print(f"Processing {store}...")

    # 1) Load store's merged pickle
    df_merged_sub = pd.read_pickle(pickle_path).copy()

    # 2) Sort by [id, d], convert "d" -> numeric if needed
    df_merged_sub["d_num"] = df_merged_sub["d"].str[2:].astype(int)
    df_merged_sub = df_merged_sub.sort_values(["id", "d_num"])

    # 3) Create lag/rolling features (example: lags 7,14,28; rolling mean 7,28)
    lags = [7, 14, 28]
    rolling_windows = [7, 28]

    grouped = df_merged_sub.groupby("id", observed=False)
    
    for lag in lags:
        df_merged_sub[f"sales_lag{lag}"] = grouped["sales"].shift(lag)

    for w in rolling_windows:
        df_merged_sub[f"sales_rollmean{w}"] = (
            grouped["sales"].shift(1).rolling(w, min_periods=1).mean()
        )
    
    # Fill missing in newly created features
    feature_cols = ([f"sales_lag{x}" for x in lags] 
                    + [f"sales_rollmean{x}" for x in rolling_windows])
    df_merged_sub[feature_cols] = df_merged_sub[feature_cols].fillna(0)
    
    # 4) Example time-based split: d_num < 1500 => train, >= 1500 => val
    df_train = df_merged_sub[df_merged_sub["d_num"] < 1500].copy()
    df_val   = df_merged_sub[df_merged_sub["d_num"] >= 1500].copy()

    # 5) Build a simple model (optional); in real usage, you might train on full data
    #    or do a more careful time split, or skip training if you're purely generating future predictions
    if len(df_train) == 0 or len(df_val) == 0:
        print(f"No valid train/val split for {store}, skipping training.")
        continue

    X_train = df_train[feature_cols]
    y_train = df_train["sales"]
    X_val   = df_val[feature_cols]
    y_val   = df_val["sales"]

    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predict on validation just to check RMSE
    pred_val = model.predict(X_val)
    rmse = np.sqrt(np.mean((pred_val - y_val)**2))
    print(f"{store} validation RMSE: {rmse:.4f}")

    # 6) In a real scenario, predict the final 28 days (d_1942~d_1969) or wherever needed
    #    Then store those predictions (for eventual submission)
    # example placeholder:
    # final_pred = model.predict(X_future)
    # all_preds.append( (store, final_pred) )

print("\nFinished multi-store loop. Collected predictions:", all_preds)


import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor
from sklearn.model_selection import ParameterGrid

all_stores = ["CA_1", "CA_2", "CA_3", "CA_4",
              "TX_1", "TX_2", "TX_3",
              "WI_1", "WI_2", "WI_3"]

param_grid = {
    "num_leaves": [31, 63],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200]
}

weekday_map = {
    "Monday":1, "Tuesday":2, "Wednesday":3,
    "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7
}

store_metrics = {}

for store in all_stores:
    pickle_path = f"/kaggle/working/merged_{store}.pkl"
    if not os.path.exists(pickle_path):
        print(f"File not found for {store}, skipping.")
        continue

    print(f"\nProcessing {store} ...")
    df_merged = pd.read_pickle(pickle_path).copy()
    
    # 1) Convert "d" -> numeric
    df_merged["d_num"] = df_merged["d"].str[2:].astype(int)
    df_merged = df_merged.sort_values(["id","d_num"])

    # 2) Convert weekday to a numeric column
    if df_merged["weekday"].dtype in [object, "category"]:
        # Map strings to integers
        df_merged["weekday_num"] = (
            df_merged["weekday"].astype(str)
            .map(weekday_map)
            .fillna(0)
            .astype(int)
        )
    else:
        # If it's already numeric
        df_merged["weekday_num"] = df_merged["weekday"].fillna(0).astype(int)

    # Convert month, year to int if needed
    df_merged["month"] = df_merged["month"].fillna(0).astype(int)
    df_merged["year"]  = df_merged["year"].fillna(0).astype(int)

    # 3) Price feature fill
    df_merged["sell_price"] = df_merged["sell_price"].fillna(0)

    # 4) Create sales lag/rolling features
    lags = [7, 14, 28]
    rolling_windows = [7, 28]
    grouped = df_merged.groupby("id", observed=False)
    
    for lag in lags:
        col_name = f"sales_lag{lag}"
        df_merged[col_name] = grouped["sales"].shift(lag)
    
    for w in rolling_windows:
        col_name = f"sales_rollmean{w}"
        df_merged[col_name] = (
            grouped["sales"].shift(1).rolling(w, min_periods=1).mean()
        )
    
    df_merged[["snap_CA","snap_TX","snap_WI"]] = df_merged[["snap_CA","snap_TX","snap_WI"]].fillna(0)
    
    # Additional price-based rolling as an example
    df_merged["price_roll7"] = grouped["sell_price"].shift(1).rolling(7, min_periods=1).mean()
    
    # Fill any missing new features
    feature_cols = []
    for lag in lags:
        feature_cols.append(f"sales_lag{lag}")
    for w in rolling_windows:
        feature_cols.append(f"sales_rollmean{w}")
    feature_cols.append("price_roll7")

    extra_cols = ["snap_CA","snap_TX","snap_WI","sell_price","weekday_num","month","year"]
    feature_cols += extra_cols

    df_merged[feature_cols] = df_merged[feature_cols].fillna(0)

    # 5) Simple time-based split
    df_train = df_merged[df_merged["d_num"] < 1800].copy()
    df_val   = df_merged[df_merged["d_num"] >= 1800].copy()

    if len(df_train)==0 or len(df_val)==0:
        print(f"No data for {store}, skip training.")
        continue

    X_train = df_train[feature_cols]
    y_train = df_train["sales"]
    X_val   = df_val[feature_cols]
    y_val   = df_val["sales"]

    best_rmse = float("inf")
    best_params = None

    for params in ParameterGrid(param_grid):
        model = LGBMRegressor(random_state=42, **params)
        model.fit(X_train, y_train)

        pred_val = model.predict(X_val)
        rmse = np.sqrt(np.mean((pred_val - y_val)**2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    print(f"{store} best params: {best_params}, RMSE: {best_rmse:.4f}")
    store_metrics[store] = (best_params, best_rmse)

print("\nAll store metrics:")
for st, (pars, sc) in store_metrics.items():
    print(f"{st} -> RMSE: {sc:.4f}, best params: {pars}")


import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor

# Example store list
all_stores = ["CA_1","CA_2","CA_3","CA_4",
              "TX_1","TX_2","TX_3",
              "WI_1","WI_2","WI_3"]

# We'll collect final predictions in a list of dataframes, then merge them for submission
submission_rows = []

for store in all_stores:
    pickle_path = f"/kaggle/working/merged_{store}.pkl"
    if not os.path.exists(pickle_path):
        print(f"No file for {store}, skipping.")
        continue

    print(f"\nPreparing final predictions for {store}...")

    df_merged = pd.read_pickle(pickle_path).copy()
    df_merged["d_num"] = df_merged["d"].str[2:].astype(int)
    df_merged = df_merged.sort_values(["id","d_num"])

    # (A) Suppose we have discovered best hyperparams from the previous step
    # For illustration, we just fix some params or store-specific best
    # In real usage, you might store each store's best_params from "store_metrics"
    # or do final training on all data to get a single best model.
    best_params = {"learning_rate":0.05,"n_estimators":200,"num_leaves":63}
    model = LGBMRegressor(random_state=42, **best_params)

    # (B) Create features for historical days (e.g. d_num < 1942) to train on
    #  plus we must build the same lag/rolling for d_1942..d_1969 for prediction
    # We'll do a quick approach: if you have d up to 1941, let's add rows for 1942..1969 with sales=NaN
    # Then build features, fill the future's lag from known history.
    
    max_dnum = df_merged["d_num"].max()  # likely 1941
    future_days = range(1942, 1970)      # 28 days

    # For each id in df_merged, create empty rows for future days
    all_ids = df_merged["id"].unique()
    future_df = []
    for _id in all_ids:
        for d_future in future_days:
            future_df.append({"id": _id, "d_num": d_future, "sales": np.nan})
    future_df = pd.DataFrame(future_df)
    
    # Merge these future rows into df_merged
    df_merged = pd.concat([df_merged, future_df], ignore_index=True, sort=False)
    df_merged = df_merged.sort_values(["id","d_num"]).reset_index(drop=True)

    # (C) Re-create features (lags, rolling, etc.)
    # As we did in training, so the future days can also get "sales_lag7" etc. from prior
    grouped = df_merged.groupby("id", observed=False)

    # We'll define same lags/rolling
    lags = [7,14,28]
    rolling_windows = [7,28]

    for lag in lags:
        df_merged[f"sales_lag{lag}"] = grouped["sales"].shift(lag)
    for w in rolling_windows:
        df_merged[f"sales_rollmean{w}"] = (
            grouped["sales"].shift(1).rolling(w, min_periods=1).mean()
        )

    # Snap / price / weekday transformations as well (like the step in training)
    df_merged["sell_price"] = df_merged["sell_price"].fillna(0)
    # If "weekday" is object, map it; else keep numeric
    # (Skipping code to re-check weekday type for brevity; you might do the same map approach)

    # Example: fill or create "weekday_num", "month", "year" if not present
    # Or carry them forward if you have them in future rows
    # We'll do a simplistic fill:
    df_merged["weekday_num"] = df_merged.get("weekday_num",0)
    df_merged["month"] = df_merged.get("month",0)
    df_merged["year"]  = df_merged.get("year",0)

    # Additional price-based rolling, etc. 
    df_merged["price_roll7"] = grouped["sell_price"].shift(1).rolling(7, min_periods=1).mean()

    # fill any new columns
    feature_cols = [f"sales_lag{x}" for x in lags] + [f"sales_rollmean{x}" for x in rolling_windows]
    feature_cols += ["snap_CA","snap_TX","snap_WI","sell_price","weekday_num","month","year","price_roll7"]
    df_merged[feature_cols] = df_merged[feature_cols].fillna(0)

    # (D) Train the model on all known historical data: d_num <= 1941
    train_mask = (df_merged["d_num"] <= 1941) & (~df_merged["sales"].isna())
    X_train = df_merged.loc[train_mask, feature_cols]
    y_train = df_merged.loc[train_mask, "sales"]
    model.fit(X_train, y_train)

    # (E) Predict future days: d_num in [1942..1969], which is 28 days
    pred_mask = (df_merged["d_num"] >= 1942) & (df_merged["d_num"] <= 1969)
    X_pred = df_merged.loc[pred_mask, feature_cols]
    preds = model.predict(X_pred)

    # Store predictions in df_merged
    df_merged.loc[pred_mask, "pred_sales"] = preds

    # (F) Build submission rows for each id. We have `_evaluation` IDs for the future
    # plus we replicate them as `_validation` if needed. For each day in 1942..1969,
    # map to F1..F28. 
    # We'll illustrate just `_evaluation` to keep it short:
    #  - Real submission also needs `_validation` lines.
    
    # Build a pivot of (id, d_num) -> predicted sales
    pivot_df = df_merged.loc[pred_mask, ["id","d_num","pred_sales"]].copy()
    pivot_df["day_index"] = pivot_df["d_num"] - 1941  # so 1942 becomes day_index=1 (F1), 1969 -> 28
    # pivot to wide F1..F28
    sub_wide = pivot_df.pivot(index="id", columns="day_index", values="pred_sales").reset_index()
    sub_wide.columns = ["id"] + [f"F{i}" for i in range(1,29)]

    # filter `_evaluation` only
    eval_mask = sub_wide["id"].str.endswith("_evaluation")
    store_eval = sub_wide[eval_mask].copy()

    # optionally replicate for _validation
    # an example for naive approach:
    val_df = store_eval.copy()
    val_df["id"] = val_df["id"].str.replace("_evaluation","_validation")
    
    # combine them
    final_store_sub = pd.concat([store_eval, val_df], axis=0)
    submission_rows.append(final_store_sub)

# (G) Concatenate all stores' submissions
submission_df = pd.concat(submission_rows, axis=0)
submission_df = submission_df.sort_values("id").reset_index(drop=True)

print("Submission shape:", submission_df.shape)
print(submission_df.head(10))

# (H) Save to CSV
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv. Ready to upload or refine further.")

