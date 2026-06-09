import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# reload data
train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

# monthly aggregation
monthly = (
    train
    .groupby(["date_block_num", "shop_id", "item_id"], as_index=False)["item_cnt_day"]
    .sum()
    .rename(columns={"item_cnt_day": "item_cnt_month"})
)

# lag feature
monthly_sorted = monthly.sort_values(["shop_id", "item_id", "date_block_num"])
monthly_sorted["item_cnt_month_lag1"] = (
    monthly_sorted
    .groupby(["shop_id", "item_id"])["item_cnt_month"]
    .shift(1)
)

# drop missing lag
data = monthly_sorted.dropna(subset=["item_cnt_month_lag1"]).copy()

# train / validation split
train_data = data[data["date_block_num"] < 33]
val_data   = data[data["date_block_num"] == 33]

X_train = train_data[["item_cnt_month_lag1"]]
y_train = train_data["item_cnt_month"]
X_val   = val_data[["item_cnt_month_lag1"]]
y_val   = val_data["item_cnt_month"]

# fit model and get validation predictions
model = LinearRegression()
model.fit(X_train, y_train)
val_pred = model.predict(X_val)

rmse = mean_squared_error(y_val, val_pred, squared=False)
print("Validation RMSE:", rmse)
print("Train rows:", train_data.shape[0], "Val rows:", val_data.shape[0])



import pandas as pd

train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



monthly = (
    train
    .groupby(["date_block_num", "shop_id", "item_id"], as_index=False)["item_cnt_day"]
    .sum()
    .rename(columns={"item_cnt_day": "item_cnt_month"})
)

print(monthly.shape)
monthly.head()



train_data = data[data["date_block_num"] < 33]
val_data = data[data["date_block_num"] == 33]

print(train_data["date_block_num"].unique()[:5], "...")
print("Train rows:", train_data.shape[0])
print("Val rows:", val_data.shape[0])
val_data.head()



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1) Define features (X) and target (y)
X_train = train_data[["item_cnt_month_lag1"]]
y_train = train_data["item_cnt_month"]

X_val = val_data[["item_cnt_month_lag1"]]
y_val = val_data["item_cnt_month"]

# 2) Fit a simple linear regression model
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# 3) Predict on validation and evaluate
val_pred = baseline_model.predict(X_val)

rmse = mean_squared_error(y_val, val_pred, squared=False)
print("Validation RMSE:", rmse)

# Show first few predictions vs actuals
comparison = val_data[["shop_id", "item_id", "item_cnt_month"]].copy()
comparison["predicted_cnt_month"] = val_pred
comparison.head()



data_enhanced = data.copy()

data_enhanced["prev_month_nonzero"] = (data_enhanced["item_cnt_month_lag1"] > 0).astype(int)

feature_cols = ["item_cnt_month_lag1", "date_block_num", "prev_month_nonzero"]

train_enh = data_enhanced[data_enhanced["date_block_num"] < 33]
val_enh = data_enhanced[data_enhanced["date_block_num"] == 33]

X_train_enh = train_enh[feature_cols]
y_train_enh = train_enh["item_cnt_month"]

X_val_enh = val_enh[feature_cols]
y_val_enh = val_enh["item_cnt_month"]

X_train_enh.head()



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_enh, y_train_enh)

val_pred_rf = rf_model.predict(X_val_enh)

rmse_rf = mean_squared_error(y_val_enh, val_pred_rf, squared=False)
print("Baseline Linear Regression RMSE (from before): 14.1330 (approx)")
print("RandomForest RMSE:", rmse_rf)

comparison_rf = val_enh[["shop_id", "item_id", "item_cnt_month"]].copy()
comparison_rf["predicted_cnt_month_rf"] = val_pred_rf
comparison_rf.head()



from sklearn.linear_model import LinearRegression

# Use all rows that have a valid lag (that's already what `data` is)
X_all = data[["item_cnt_month_lag1"]]
y_all = data["item_cnt_month"]

final_model = LinearRegression()
final_model.fit(X_all, y_all)

print("Trained final linear model on", X_all.shape[0], "rows.")



from sklearn.linear_model import LinearRegression

# Use all rows that have a valid lag (that's already what `data` is)
X_all = data[["item_cnt_month_lag1"]]
y_all = data["item_cnt_month"]

final_model = LinearRegression()
final_model.fit(X_all, y_all)

print("Trained final linear model on", X_all.shape[0], "rows.")



# assumes you already have `train` loaded from train.csv
monthly = (
    train
    .groupby(["date_block_num", "shop_id", "item_id"], as_index=False)["item_cnt_day"]
    .sum()
    .rename(columns={"item_cnt_day": "item_cnt_month"})
)

print(monthly.shape)
monthly.head()



import pandas as pd

train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



# 1) Aggregate to monthly level
monthly = (
    train
    .groupby(["date_block_num", "shop_id", "item_id"], as_index=False)["item_cnt_day"]
    .sum()
    .rename(columns={"item_cnt_day": "item_cnt_month"})
)

print("Monthly shape:", monthly.shape)
monthly.head()



import warnings
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in greater",
    category=RuntimeWarning
)

# 2) Sort and add lag-1 feature
monthly_sorted = monthly.sort_values(["shop_id", "item_id", "date_block_num"])

monthly_sorted["item_cnt_month_lag1"] = (
    monthly_sorted
    .groupby(["shop_id", "item_id"])["item_cnt_month"]
    .shift(1)
)

print(monthly_sorted.head().to_string())



# 3) Drop rows without lag
data = monthly_sorted.dropna(subset=["item_cnt_month_lag1"]).copy()
print("Data shape:", data.shape)
data.head()



from sklearn.linear_model import LinearRegression

X_all = data[["item_cnt_month_lag1"]]
y_all = data["item_cnt_month"]

final_model = LinearRegression()
final_model.fit(X_all, y_all)

print("Trained final linear model on", X_all.shape[0], "rows.")



# Get monthly data for October 2015 (date_block_num == 33)
october = monthly[monthly["date_block_num"] == 33][
    ["shop_id", "item_id", "item_cnt_month"]
].copy()

october = october.rename(columns={"item_cnt_month": "item_cnt_month_lag1"})
print("October rows:", october.shape[0])
october.head()



# Start from test.csv pairs
test_with_lag = test.merge(
    october,
    on=["shop_id", "item_id"],
    how="left"
)

# Items that never sold in October get NaN; replace with 0
test_with_lag["item_cnt_month_lag1"] = test_with_lag["item_cnt_month_lag1"].fillna(0)

print(test_with_lag.shape)
test_with_lag.head()



X_test = test_with_lag[["item_cnt_month_lag1"]]

test_with_lag["item_cnt_month_pred"] = final_model.predict(X_test)

# Kaggle expects predictions between 0 and 20
test_with_lag["item_cnt_month_pred"] = test_with_lag["item_cnt_month_pred"].clip(0, 20)

test_with_lag[["ID", "item_cnt_month_pred"]].head()



submission = test_with_lag[["ID", "item_cnt_month_pred"]].rename(
    columns={"item_cnt_month_pred": "item_cnt_month"}
)

# THIS is what actually creates the file Kaggle looks for:
submission.to_csv("submission.csv", index=False)

# Optional preview (keeps what you already see)
submission.head()



train_data = monthly_sorted[monthly_sorted["date_block_num"] < 33]
val_data = monthly_sorted[monthly_sorted["date_block_num"] == 33]



import numpy as np
import matplotlib.pyplot as plt

# Turn to arrays/Series if needed
y_val_array = y_val.values
val_pred_array = val_pred

# Choose sample size (e.g., 10,000 points)
sample_size = 10000
n = len(y_val_array)
idx = np.random.choice(n, size=min(sample_size, n), replace=False)

y_sample = y_val_array[idx]
pred_sample = val_pred_array[idx]

plt.figure(figsize=(6, 6))
plt.scatter(y_sample, pred_sample, alpha=0.3, s=5)

max_val = max(y_sample.max(), pred_sample.max())
plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Perfect prediction")

plt.xlabel("Actual item_cnt_month (validation month 33)")
plt.ylabel("Predicted item_cnt_month")
plt.title("Predicted vs Actual Monthly Sales (Sampled Validation Points)")
plt.legend()
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.hist(data["item_cnt_month"], bins=50, range=(0, 50))
plt.xlabel("item_cnt_month (monthly units sold)")
plt.ylabel("Number of (shop, item, month) records")
plt.title("Distribution of Monthly Item Sales")
plt.grid(True)
plt.show()



from math import sqrt
from sklearn.metrics import mean_squared_error

pred_train = model.predict(X_train)
rmse_train = sqrt(mean_squared_error(y_train, pred_train))
rmse_train


