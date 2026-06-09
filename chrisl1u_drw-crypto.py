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


import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
import optuna

# Load Data
train_df = pd.read_parquet("train.parquet")
test_df  = pd.read_parquet("test.parquet")
sample_submission = pd.read_csv("sample_submission.csv")

def add_timestamp(df):
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    df = df.reset_index() 
    for cand in ("timestamp", "index", "level_0"):
        if cand in df.columns:
            df = df.rename(columns={cand: "timestamp"})
            break
    else:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

train_df = add_timestamp(train_df)
test_df  = add_timestamp(test_df)

# Prophet-based anomaly detection
def prophet_filter_outliers(df, column="volume", timestamp_col="timestamp", threshold=3.0):
    df_prophet = df[[timestamp_col, column]].dropna().copy()
    df_prophet.columns = ["ds", "y"]
    df_prophet = df_prophet.sort_values("ds")

    m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False)
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=0, freq="min")
    forecast = m.predict(future)

    df_forecast = df_prophet.copy()
    df_forecast["yhat"] = forecast["yhat"].values
    df_forecast["residual"] = df_forecast["y"] - df_forecast["yhat"]
    std = df_forecast["residual"].std()

    mask = df_forecast["residual"].abs() < threshold * std
    valid_timestamps = df_forecast.loc[mask, "ds"]
    return df[df[timestamp_col].isin(valid_timestamps)].copy()

train_df = prophet_filter_outliers(train_df, column="volume")

#IQR-based clipping
numeric_cols = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]

def remove_outliers(df, columns, lower_quantile=0.01, upper_quantile=0.99):
    for col in columns:
        lower = df[col].quantile(lower_quantile)
        upper = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(lower, upper)
    return df

train_df = remove_outliers(train_df, numeric_cols)
test_df = remove_outliers(test_df, numeric_cols)

#Isolation Forest
def isolate_outliers(df, features, contamination=0.01):
    iso = IsolationForest(contamination=contamination, random_state=42)
    mask = iso.fit_predict(df[features]) == 1
    return df[mask]

train_df = isolate_outliers(train_df, numeric_cols)

#features prep
features = [c for c in train_df.columns if c not in ["timestamp", "label"]]
target = "label"

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

#train data split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Optuna
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "tree_method": "hist",
        "random_state": 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=25,
        verbose=False
    )
    preds = model.predict(X_valid)
    return mean_squared_error(y_valid, preds)

#optimizing
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
best_params["objective"] = "reg:squarederror"
best_params["tree_method"] = "hist"
best_params["random_state"] = 42

print("Best parameters found:", best_params)

# find the best model
xgb_model = xgb.XGBRegressor(**best_params)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=25,
    verbose=50
)

# predicting
preds = xgb_model.predict(X_test)
min_val = y.quantile(0.01)
max_val = y.quantile(0.99)
preds = np.clip(preds, min_val, max_val)

# generate the file
submission = sample_submission.copy()
submission["label"] = preds
submission.to_csv("xgb_prophet_optuna_submission.csv", index=False)

# save data and model
import os
os.makedirs("artifacts", exist_ok=True)

xgb_model.save_model("artifacts/xgb_model.json")
np.savez_compressed(
    "artifacts/valid_data_for_plot.npz",
    idx=X_valid.index.values,
    y=y_valid.values.astype(np.float32),
    ts=train_df.loc[X_valid.index, "timestamp"].astype(str).values
)

print("module and data have been saved as 'xgb_prophet_optuna_submission.csv'")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

BASE_DIR = os.path.dirname(__file__)
ART_DIR  = os.path.join(BASE_DIR, "artifacts")

model_path = os.path.join(ART_DIR, "xgb_model.json")
valid_path = os.path.join(ART_DIR, "valid_data_for_plot.npz")

assert os.path.exists(model_path), f"model file does not exist: {model_path}"
assert os.path.exists(valid_path), f"valid data does not exist: {valid_path}"

# 1. load model
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(model_path)

# 2. load valid data
data = np.load(valid_path, allow_pickle=True)
idx = data["idx"]
y_valid = data["y"]
ts = pd.to_datetime(data["ts"])

# 3. rebuild X_valid
train_df = pd.read_parquet("train.parquet")
train_df = train_df.reset_index().rename(columns={"index": "timestamp"})
features = [c for c in train_df.columns if c not in ["timestamp", "label"]]
X = train_df[features]
X_valid = X.loc[idx]

# 4. predict
best_iter = getattr(xgb_model, "best_iteration", None)
if best_iter is not None:
    preds_valid = xgb_model.predict(X_valid, iteration_range=(0, best_iter + 1))
else:
    best_ntree = getattr(xgb_model, "best_ntree_limit", xgb_model.n_estimators)
    preds_valid = xgb_model.predict(X_valid, ntree_limit=best_ntree)

# 5. graph
plot_df = pd.DataFrame({"timestamp": ts, "actual": y_valid, "pred": preds_valid}).sort_values("timestamp")

pearson = np.corrcoef(plot_df["actual"], plot_df["pred"])[0, 1]
print(f"Pearson corr (valid): {pearson:.4f}")

plt.figure(figsize=(12, 4))
plt.plot(plot_df["timestamp"], plot_df["actual"], label="Actual", linewidth=1)
plt.plot(plot_df["timestamp"], plot_df["pred"],   label="Prediction", linewidth=1, alpha=0.8)
plt.title(f"Validation Actual vs Prediction (Pearson={pearson:.3f})")
plt.xlabel("Timestamp"); plt.ylabel("Label"); plt.legend(); plt.tight_layout()
plt.savefig("valid_pred_vs_actual_time.png", dpi=150); plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(plot_df["actual"], plot_df["pred"], s=5, alpha=0.5)
lims = [min(plot_df["actual"].min(), plot_df["pred"].min()),
        max(plot_df["actual"].max(), plot_df["pred"].max())]
plt.plot(lims, lims, "--", linewidth=1)
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("Actual"); plt.ylabel("Prediction")
plt.title(f"Pred vs Actual Scatter (Pearson={pearson:.3f})")
plt.tight_layout()
plt.savefig("valid_pred_vs_actul_scatter.png", dpi=150); plt.show()

