import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")


train = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv', parse_dates=['date'])
sample_submission = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')


# ==============================
# ğŸ”� First Data Exploration
# ==============================

# Check dataset dimensions
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)

# Show first rows of the training set
print("\nFirst rows of the training data:")
display(train.head())

# Check data types and null values
print("\nInfo about train dataset:")
print(train.info())

print("\nMissing values in train dataset:")
print(train.isnull().sum())



train.describe()


# ==============================
# ğŸ“Š Global Sales Trend
# ==============================

# Aggregate sales per day (all stores/items combined)
daily_sales = train.groupby("date")["sales"].sum().reset_index()

plt.figure(figsize=(15,5))
plt.plot(daily_sales["date"], daily_sales["sales"], color="steelblue", linewidth=1)
plt.title("Total Daily Sales over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()



# ==============================
# ğŸ“† Monthly Sales Aggregation
# ==============================

# Resample sales per month (sum across all stores/items)
monthly_sales = train.set_index("date").resample("M")["sales"].sum()

plt.figure(figsize=(15,5))
plt.plot(monthly_sales.index, monthly_sales.values, color="darkorange", linewidth=2)
plt.title("Total Monthly Sales", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()



# ==============================
# ğŸ“… Weekly Sales Pattern
# ==============================

# Add weekday column (0 = Monday, 6 = Sunday)
train["weekday"] = train["date"].dt.dayofweek

# Average sales per weekday
weekday_sales = train.groupby("weekday")["sales"].mean()

plt.figure(figsize=(10,5))
sns.barplot(x=weekday_sales.index, y=weekday_sales.values, palette="viridis")
plt.title("Average Sales by Weekday", fontsize=14)
plt.xlabel("Weekday (0=Mon, 6=Sun)")
plt.ylabel("Average Sales")
plt.show()

print("Average sales per weekday:\n", weekday_sales)



# ==============================
# ğŸ�¬ Store-Level Sales Comparison
# ==============================

# Average daily sales per store
store_sales = train.groupby("store")["sales"].mean().reset_index()

plt.figure(figsize=(10,5))
sns.barplot(x="store", y="sales", data=store_sales, palette="coolwarm")
plt.title("Average Daily Sales per Store", fontsize=14)
plt.xlabel("Store ID")
plt.ylabel("Average Sales")
plt.show()

print("Average daily sales per store:\n", store_sales)



# ==============================
# ğŸ›’ Top-10 Items Overall
# ==============================

# Total sales per item (all stores combined)
item_sales = train.groupby("item")["sales"].sum().reset_index()

# Top-10 items
top_items = item_sales.sort_values(by="sales", ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x="item", y="sales", data=top_items, palette="magma")
plt.title("Top-10 Items (Total Sales across all Stores)", fontsize=14)
plt.xlabel("Item ID")
plt.ylabel("Total Sales")
plt.show()

print("Top-10 items overall:\n", top_items)



# ==============================
# ğŸ�¬ + ğŸ›’ Top Item per Store
# ==============================

# Total sales per store-item combination
store_item_sales = train.groupby(["store", "item"])["sales"].sum().reset_index()

# For each store, find the item with maximum sales
top_item_per_store = store_item_sales.loc[store_item_sales.groupby("store")["sales"].idxmax()]

print("Top-Selling Item per Store:")
display(top_item_per_store)



# ==============================
# ğŸ”¥ Heatmap Store Ã— Item
# ==============================

# Pivot table: stores (rows), items (columns), total sales as values
pivot_sales = train.pivot_table(index="store", columns="item", values="sales", aggfunc="sum")

plt.figure(figsize=(15,6))
sns.heatmap(pivot_sales, cmap="YlOrRd", cbar=True)
plt.title("Store Ã— Item â€” Total Sales Heatmap", fontsize=14)
plt.xlabel("Item ID")
plt.ylabel("Store ID")
plt.show()



# ==============================
# âš¡ Baseline: Mean Predictor
# ==============================

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
import numpy as np

# Train/validation split (time-based: last 3 months = validation)
cutoff_date = "2017-10-01"
train_baseline = train[train["date"] < cutoff_date]
val_baseline   = train[train["date"] >= cutoff_date]

# Compute mean sales per (store, item) from training part
mean_sales = train_baseline.groupby(["store", "item"])["sales"].mean()

# Predict validation: use group mean (fallback to global mean if missing)
val_preds = val_baseline.set_index(["store","item"]).index.map(
    lambda idx: mean_sales.get(idx, train_baseline["sales"].mean())
)

# True values
y_true = val_baseline["sales"].values
y_pred = np.array(val_preds)

# RMSLE (Kaggle metric)
score = np.sqrt(mean_squared_log_error(y_true, y_pred))
print(f"Baseline RMSLE: {score:.5f}")



# =========================
# âš™ï¸� Compact FE + Split
# =========================

# 0) concat train+test so features exist on both (safe for lags)
tr = train.copy(); te = test.copy()
tr["split"] = "train"; te["split"] = "test"
df = pd.concat([tr, te], ignore_index=True).sort_values(["store","item","date"])
df.reset_index(drop=True, inplace=True)

# 1) calendar features
dt = df["date"]
df["dow"]       = dt.dt.dayofweek.astype("int8")
df["is_weekend"]= df["dow"].isin([5,6]).astype("int8")
df["month"]     = dt.dt.month.astype("int8")
df["year"]      = dt.dt.year.astype("int16")
df["day"]       = dt.dt.day.astype("int8")
df["woy"]       = dt.dt.isocalendar().week.astype("int16")
df["quarter"]   = dt.dt.quarter.astype("int8")
df["m_start"]   = dt.dt.is_month_start.astype("int8")
df["m_end"]     = dt.dt.is_month_end.astype("int8")

# 2) lags & rolling (shift first to avoid leakage)
g = df.groupby(["store","item"], group_keys=False)
for L in (1,7,14,28):
    df[f"lag_{L}"] = g["sales"].shift(L)
df["roll7"]  = g["sales"].shift(1).rolling(7,  min_periods=1).mean()
df["roll28"] = g["sales"].shift(1).rolling(28, min_periods=1).mean()

# 3) dtypes (smaller & categorical for LGBM)
for c in ["store","item"]:
    df[c] = df[c].astype("int16").astype("category")
num_cols = ["lag_1","lag_7","lag_14","lag_28","roll7","roll28"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast="float")

# 4) split back + time-based validation
feat_cols = ["store","item","dow","is_weekend","month","year","day","woy","quarter","m_start","m_end",
             "lag_1","lag_7","lag_14","lag_28","roll7","roll28"]

df_tr = df[df["split"]=="train"].copy()
df_te = df[df["split"]=="test"].copy()

cutoff = pd.Timestamp("2017-10-01")  # last ~3 months as validation
tr_mask = df_tr["date"] < cutoff
va_mask = df_tr["date"] >= cutoff

X_train = df_tr.loc[tr_mask, feat_cols]
y_train = df_tr.loc[tr_mask, "sales"].astype("float32")
X_val   = df_tr.loc[va_mask, feat_cols]
y_val   = df_tr.loc[va_mask, "sales"].astype("float32")
X_test  = df_te[feat_cols]

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)



# =========================
# ğŸŒ¿ Tiny LightGBM train
# =========================
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np

cat_feats = ["store","item","dow","month","quarter"]  # small, effective set

model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    categorical_feature=cat_feats,
    callbacks=[]
    # early_stopping removed in some LightGBM builds; if available:
    # callbacks=[lightgbm.early_stopping(100), lightgbm.log_evaluation(100)]
)

# Evaluate (RMSLE)
val_pred = np.clip(model.predict(X_val), 0, None)
rmsle = np.sqrt(mean_squared_log_error(y_val, val_pred))
print(f"Validation RMSLE: {rmsle:.5f}")



# =========================
# ğŸ“� Predict & Submission
# =========================
test_pred = np.clip(model.predict(X_test), 0, None)
sub = sample_submission.copy()
sub["sales"] = test_pred
print(sub.head())
sub.to_csv("submission.csv", index=False)



# =========================
# ğŸ“Š Plot: True vs Pred (Validation)
# =========================
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(y_val.values[:200], label="True", alpha=0.8)
plt.plot(val_pred[:200], label="Predicted", alpha=0.8)
plt.title("Validation: True vs Predicted Sales (first 200 samples)")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.show()



# =========================
# ğŸ’¾ Save trained model
# =========================
import joblib

joblib.dump(model, "lgbm_demand_model.pkl")
print("âœ… Model saved as lgbm_demand_model.pkl")


