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


# load data 
import os
import pandas as pd
import numpy as np

DATA_DIR = '/kaggle/input/demand-forecasting-kernels-only'
OUT_DIR = '/kaggle/working'

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), parse_dates=['date'])
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), parse_dates=['date'])
sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

print('Shapes -> train:', train.shape, ' test:', test.shape, ' sample:', sample.shape)
print(' train head')
display(train.head())
print('test head')
display(test.head())
print('sample head')
display(sample.head())



#SMAPE metric 
import numpy as np

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom == 0
    denom[mask] = 1.0
    num = np.abs(y_true - y_pred)
    num[mask] = 0.0
    return 100.0 * np.mean(num / denom)



#EDA (Exploratory Data Analysis — statistics only)

print("Date Range:", train['date'].min(), "to", train['date'].max())
print("Total Unique Stores:", train['store'].nunique())
print("Total Unique Items:", train['item'].nunique())

print("\nSales Summary Statistics:")
display(train['sales'].describe())

print("\nMissing Values:")
display(train.isnull().sum())

print("\nStore-wise Average Sales:")
display(train.groupby("store")["sales"].mean().round(2))

print("\nItem-wise Average Sales (Top 10):")
display(train.groupby("item")["sales"].mean().sort_values(ascending=False).head(10))



# Daily sales trend visualization
import matplotlib.pyplot as plt
import pandas as pd

daily = train.groupby("date")["sales"].sum()

plt.figure(figsize=(14,4))
plt.plot(daily.index, daily.values, color="blue", alpha=0.7)
plt.title("Daily Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()



#Rolling mean smoothing
import matplotlib.pyplot as plt
import pandas as pd

daily = train.groupby("date")["sales"].sum().reset_index()
daily["rolling30"] = daily["sales"].rolling(30, min_periods=1).mean()

plt.figure(figsize=(14,4))
plt.plot(daily["date"], daily["rolling30"], color="red")
plt.title("30-Day Rolling Average of Sales")
plt.xlabel("Date")
plt.ylabel("Rolling Sales")
plt.show()



# Cell 5 — Monthly seasonality
import matplotlib.pyplot as plt
import pandas as pd

train["month"] = train["date"].dt.month
monthly = train.groupby("month")["sales"].mean()

plt.figure(figsize=(8,4))
plt.bar(monthly.index, monthly.values, color="orange")
plt.title("Average Sales by Month")
plt.xlabel("Month")
plt.ylabel("Avg Sales")
plt.xticks(range(1,13))
plt.show()



# Cell 6 — Average sales by day of week
import matplotlib.pyplot as plt

dow = train.groupby(train["date"].dt.dayofweek)["sales"].mean()

plt.figure(figsize=(8,4))
plt.plot(dow.index, dow.values, marker='o', color="green")
plt.xticks(range(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
plt.title("Average Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Avg Sales")
plt.grid(alpha=0.3)
plt.show()



#Sales distribution
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.hist(train["sales"], bins=60, color="pink")
plt.title("Sales Distribution")
plt.xlabel("Sales")

plt.subplot(1,2,2)
plt.hist(np.log1p(train["sales"]), bins=60, color="hotpink")
plt.title("Log-transformed Sales Distribution")
plt.xlabel("log1p(Sales)")

plt.tight_layout()
plt.show()



# Boxplot of monthly sales
import matplotlib.pyplot as plt

data_by_month = [train.loc[train["month"] == m, "sales"] for m in range(1,13)]

plt.figure(figsize=(14,5))
plt.boxplot(data_by_month, labels=range(1,13), showfliers=False)
plt.title("Sales Distribution per Month (Outliers Hidden)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()



# Store x Item mean sales heatmap
import matplotlib.pyplot as plt
import numpy as np

pivot = train.groupby(["store","item"])["sales"].mean().unstack()

rows = min(10, pivot.shape[0])
cols = min(20, pivot.shape[1])

plt.figure(figsize=(12,5))
plt.imshow(pivot.iloc[:rows, :cols], cmap="viridis", aspect="auto")
plt.colorbar(label="Mean Sales")
plt.title(f"Store × Item Mean Sales (subset {rows} stores × {cols} items)")
plt.xlabel("Item ID (subset)")
plt.ylabel("Store ID (subset)")
plt.show()



# Cell 2 — Load data
import pandas as pd
import os

DATA_DIR = "/kaggle/input/demand-forecasting-kernels-only"

train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["date"])
test  = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["date"])
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("Train:", train.shape, " Test:", test.shape)
display(train.head())



# Cell 3 — Date-based features
import pandas as pd

def add_date_features(df):
    df = df.copy()
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    return df

train_fe = add_date_features(train)
test_fe  = add_date_features(test)

display(train_fe.head())



# Cell 4 — Aggregate statistics for each (store, item)
import pandas as pd

aggs = train.groupby(["store", "item"])["sales"].agg(["mean", "median", "std"]).reset_index()
aggs.columns = ["store", "item", "sales_mean", "sales_median", "sales_std"]

train_fe = train_fe.merge(aggs, on=["store", "item"], how="left")
test_fe  = test_fe.merge(aggs, on=["store", "item"], how="left")

display(train_fe.head())



# Cell 5 — Sort by store, item, date for proper lag calculation
train_fe = train_fe.sort_values(["store", "item", "date"]).reset_index(drop=True)



import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



# Cell 6 — Lag features
import pandas as pd

lags = [7, 14, 28, 365]  # weekly, bi-weekly, monthly, yearly

for lag in lags:
    train_fe[f"lag_{lag}"] = (
        train_fe
        .groupby(["store", "item"])["sales"]
        .shift(lag)
    )

display(train_fe.head(10))



# Cell 7 — Drop rows without lag_7 (minimum required)
train_fe = train_fe.dropna(subset=["lag_7"]).reset_index(drop=True)
print("Train FE shape after dropping lag-NaNs:", train_fe.shape)



# Cell 8 — Extend timeline to compute test lags
import pandas as pd
import numpy as np

combined = pd.concat([
    train_fe[["date","store","item","sales"]],
    test_fe[["date","store","item"]].assign(sales=np.nan)
], ignore_index=True)

combined = combined.sort_values(["store","item","date"])

# compute lags on combined timeline
for lag in lags:
    combined[f"lag_{lag}"] = (
        combined.groupby(["store","item"])["sales"]
        .shift(lag)
    )

# extract test rows with the new lag values
test_lagged = combined[combined["date"].isin(test_fe["date"])][
    ["date","store","item"] + [f"lag_{l}" for l in lags]
]

# merge into test_fe
test_fe = test_fe.merge(test_lagged, on=["date","store","item"], how="left")

display(test_fe.head())



# Cell 9 — Final list of modeling features
FEATURES = [
    "day","month","year","dayofweek","weekofyear",
    "sales_mean","sales_median","sales_std",
    "lag_7","lag_14","lag_28","lag_365"
]

print("Final feature set:", FEATURES)



# Cell 10 — Save processed datasets into Kaggle working directory
train_fe.to_csv("/kaggle/working/train_fe.csv", index=False)
test_fe.to_csv("/kaggle/working/test_fe.csv", index=False)

print("Saved train_fe.csv and test_fe.csv successfully.")



train_fe.head()


test_fe.head()


#  Load engineered datasets
import pandas as pd

train_fe = pd.read_csv("/kaggle/working/train_fe.csv", parse_dates=["date"])
test_fe  = pd.read_csv("/kaggle/working/test_fe.csv",  parse_dates=["date"])

print("Train:", train_fe.shape, " Test:", test_fe.shape)
display(train_fe.head())



FEATURES = [
    "day","month","year","dayofweek","weekofyear",
    "sales_mean","sales_median","sales_std",
    "lag_7","lag_14","lag_28","lag_365"
]

TARGET = "sales"

print("Using features:", FEATURES)


# Cell 5 — Time-based split (last 90 days for validation)
import pandas as pd

val_start = train_fe["date"].max() - pd.Timedelta(days=90)

train_data = train_fe[train_fe["date"] < val_start]
val_data   = train_fe[train_fe["date"] >= val_start]

X_train = train_data[FEATURES].fillna(0)
y_train = train_data[TARGET]

X_val = val_data[FEATURES].fillna(0)
y_val = val_data[TARGET]

print("Train rows:", X_train.shape[0], 
      " Validation rows:", X_val.shape[0])



#Seasonal average baseline
import pandas as pd

seasonal_mean = train_fe.groupby(train_fe["date"].dt.dayofyear)["sales"].mean()

val_seasonal = val_data["date"].dt.dayofyear.map(seasonal_mean).fillna(y_train.mean())

smape_seasonal = smape(y_val, val_seasonal)
print("Seasonal Baseline SMAPE:", smape_seasonal)



import numpy as np

global_mean = y_train.mean()
val_global = np.full(len(y_val), global_mean)

smape_global = smape(y_val, val_global)
print("Global Mean SMAPE:", smape_global)


#  Linear-regression-based baselines
from sklearn.linear_model import LinearRegression, Ridge, Lasso

results = {}

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_val)
results["LinearRegression"] = smape(y_val, lr_pred)

# Ridge
ridge = Ridge().fit(X_train, y_train)
ridge_pred = ridge.predict(X_val)
results["Ridge"] = smape(y_val, ridge_pred)

# Lasso
lasso = Lasso(max_iter=5000).fit(X_train, y_train)
lasso_pred = lasso.predict(X_val)
results["Lasso"] = smape(y_val, lasso_pred)

results



# Cell 10 — XGBoost baseline model
import xgboost as xgb
import numpy as np

dtrain = xgb.DMatrix(X_train, label=np.log1p(y_train))
dval   = xgb.DMatrix(X_val)

params = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    verbose_eval=False
)

xgb_pred = np.expm1(xgb_model.predict(dval))
results["XGBoost"] = smape(y_val, xgb_pred)

print("XGBoost SMAPE:", results["XGBoost"])



# Cell 11 — Compare all baseline models in a table
import pandas as pd

baseline_results = pd.DataFrame([
    ["GlobalMean", smape_global],
    ["SeasonalMean", smape_seasonal],
] + [
    [model, score] for model, score in results.items()
], columns=["Model", "SMAPE"]).sort_values("SMAPE")

display(baseline_results)



# Cell 12 — Select best model
best_model_name = baseline_results.iloc[0]["Model"]
best_smape = baseline_results.iloc[0]["SMAPE"]

print("BEST BASELINE MODEL:", best_model_name)
print("BEST BASELINE SMAPE:", best_smape)



# Define hyperparameter search space
param_grid = {
    "eta": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [4, 5, 6, 7, 8],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5, 7],
    "lambda": [0, 0.1, 1, 2],
    "alpha": [0, 0.1, 1]
}



# Cell 7 — Randomized search tuning
import random

best_params = None
best_smape = 999

for i in range(20):   # 20 fast iterations
    params = {
        "objective": "reg:squarederror",
        "eta": random.choice(param_grid["eta"]),
        "max_depth": random.choice(param_grid["max_depth"]),
        "subsample": random.choice(param_grid["subsample"]),
        "colsample_bytree": random.choice(param_grid["colsample_bytree"]),
        "min_child_weight": random.choice(param_grid["min_child_weight"]),
        "lambda": random.choice(param_grid["lambda"]),
        "alpha": random.choice(param_grid["alpha"]),
        "seed": 42
    }

    model = xgb.train(params, dtrain, num_boost_round=400, verbose_eval=False)
    pred = np.expm1(model.predict(dval))
    score = smape(y_val, pred)

    if score < best_smape:
        best_smape = score
        best_params = params

    print(f"Trial {i+1}/20 → SMAPE: {score:.4f}")

print("\nBest SMAPE:", best_smape)
print("Best Parameters:", best_params)



#Final tuned model on training split
best_model = xgb.train(best_params, dtrain, num_boost_round=600)
val_pred = np.expm1(best_model.predict(dval))

tuned_smape = smape(y_val, val_pred)

print("Tuned XGBoost SMAPE:", tuned_smape)



baseline_results = baseline_results.drop_duplicates(subset=["Model"]).reset_index(drop=True)



# Create a new row for Tuned XGBoost
tuned_row = pd.DataFrame([{
    "Model": "TunedXGBoost",
    "SMAPE": tuned_smape
}])

# Combine with the baseline results
baseline_results = pd.concat([baseline_results, tuned_row], ignore_index=True)

# Make sure baseline XGBoost SMAPE is correct
baseline_results.loc[baseline_results["Model"] == "XGBoost", "SMAPE"] = baseline_smape

# Remove duplicates (if the cell runs multiple times)
baseline_results = baseline_results.drop_duplicates(subset=["Model"]).reset_index(drop=True)

# Sort by SMAPE (lower = better)
baseline_results = baseline_results.sort_values("SMAPE").reset_index(drop=True)

display(baseline_results)



# Prepare full training matrix
X_full = train_fe[FEATURES].fillna(0)
y_full = train_fe["sales"]

import xgboost as xgb
dtrain_full = xgb.DMatrix(X_full, label=np.log1p(y_full))

# Prepare test matrix
X_test = test_fe[FEATURES].fillna(0)
dtest = xgb.DMatrix(X_test)



# Train final XGBoost model using tuned parameters
final_model = xgb.train(
    best_params,
    dtrain_full,
    num_boost_round=600
)

print("Final model trained successfully.")



# Predict on test set
test_pred = final_model.predict(dtest)
test_pred = np.expm1(test_pred)  # reverse log1p



test_pred


# Create submission.csv
submission = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv")
submission["sales"] = test_pred

submission.to_csv("/kaggle/working/submission.csv", index=False)
submission.head()



# Save tuned model
final_model.save_model("/kaggle/working/tuned_xgboost_model.json")
print("Model saved: tuned_xgboost_model.json")



# Inference function for manual predictions
def predict_sales(input_features):
    df = pd.DataFrame([input_features])
    df = df[FEATURES].fillna(0)
    dmat = xgb.DMatrix(df)
    pred = np.expm1(final_model.predict(dmat))
    return float(pred[0])

# Example test
example = {
    "day": 12, "month": 5, "year": 2018,
    "dayofweek": 6, "weekofyear": 20,
    "sales_mean": 21, "sales_median": 20, "sales_std": 6,
    "lag_7": 23, "lag_14": 22, "lag_28": 21, "lag_365": 19
}

predict_sales(example)


