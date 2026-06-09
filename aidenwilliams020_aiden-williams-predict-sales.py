import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

train.head(), test.head()



monthly = (
    train
    .groupby(["date_block_num", "shop_id", "item_id"], as_index=False)["item_cnt_day"]
    .sum()
    .rename(columns={"item_cnt_day": "item_cnt_month"})
)

print("Monthly shape:", monthly.shape)
monthly.head()



monthly_sorted = monthly.sort_values(["shop_id", "item_id", "date_block_num"])

monthly_sorted["item_cnt_month_lag1"] = (
    monthly_sorted
    .groupby(["shop_id", "item_id"])["item_cnt_month"]
    .shift(1)
)

data = monthly_sorted.dropna(subset=["item_cnt_month_lag1"]).copy()
print("Modeling data shape (non-missing lag):", data.shape)
data.head()



train_data = data[data["date_block_num"] < 33]
val_data   = data[data["date_block_num"] == 33]

print("Train rows:", train_data.shape[0])
print("Validation rows:", val_data.shape[0])

X_train = train_data[["item_cnt_month_lag1"]]
y_train = train_data["item_cnt_month"]

X_val = val_data[["item_cnt_month_lag1"]]
y_val = val_data["item_cnt_month"]



model = LinearRegression()
model.fit(X_train, y_train)

val_pred = model.predict(X_val)

rmse_val = mean_squared_error(y_val, val_pred, squared=False)
rmse_train = mean_squared_error(y_train, model.predict(X_train), squared=False)

print("Training RMSE:", rmse_train)
print("Validation RMSE (month 33):", rmse_val)



plt.figure(figsize=(6, 4))
plt.hist(data["item_cnt_month"], bins=50, range=(0, 50))
plt.xlabel("item_cnt_month (monthly units sold)")
plt.ylabel("Number of records")
plt.title("Distribution of Monthly Item Sales")
plt.grid(True)
plt.show()



# Convert to arrays
y_val_array = y_val.values
val_pred_array = val_pred

# Sample up to 10,000 points
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



# Get October 2015 (month 33) monthly sales
oct_2015 = monthly[monthly["date_block_num"] == 33][["shop_id", "item_id", "item_cnt_month"]]

oct_2015 = oct_2015.rename(columns={"item_cnt_month": "item_cnt_month_lag1"})

test_with_lag = test.merge(oct_2015, on=["shop_id", "item_id"], how="left")
test_with_lag["item_cnt_month_lag1"] = test_with_lag["item_cnt_month_lag1"].fillna(0)

print("Test with lag shape:", test_with_lag.shape)
test_with_lag.head()



X_full = data[["item_cnt_month_lag1"]]
y_full = data["item_cnt_month"]

final_model = LinearRegression()
final_model.fit(X_full, y_full)

X_test_final = test_with_lag[["item_cnt_month_lag1"]]
test_pred = final_model.predict(X_test_final)

# Optional clipping like many kernels do
test_pred_clipped = np.clip(test_pred, 0, 20)

submission = pd.DataFrame({
    "ID": test_with_lag["ID"],
    "item_cnt_month": test_pred_clipped
})

submission.to_csv("submission.csv", index=False)
submission.head()


