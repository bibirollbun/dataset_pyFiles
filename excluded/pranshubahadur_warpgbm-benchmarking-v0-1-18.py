# ğŸ“¦ Install required packages
!pip install warpgbm lightgbm catboost xgboost


# ğŸ“š Import libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder



playground_series_s5e4_path = '/kaggle/input/playground-series-s5e4'


# ğŸ“‚ Load data
train_df = pd.read_csv(f'{playground_series_s5e4_path}/train.csv')
test_df  = pd.read_csv(f'{playground_series_s5e4_path}/test.csv')

print("Train shape:", train_df.shape)
train_df.head()



# ğŸ§¼ Encode categorical columns and drop rows with missing target
categorical_cols = train_df.select_dtypes('object').columns.tolist()
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_df[categorical_cols] = encoder.fit_transform(train_df[categorical_cols].astype(str))
test_df[categorical_cols] = encoder.transform(test_df[categorical_cols].astype(str))

train_df = train_df.dropna(subset=['Listening_Time_minutes'])

# ğŸ�¯ Define features and target
X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']

# âœ‚ï¸� Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



import time

def benchmark(model_class, model_name, model_kwargs=None):
    print(f"â�± Benchmarking {model_name}...")

    if model_kwargs is None:
        model_kwargs = {}

    # ğŸ’¥ Time from instantiation â†’ fit â†’ predict
    start = time.time()

    # â�³ Instantiate
    model = model_class(**model_kwargs)

    # ğŸ�‹ï¸� Fit
    if model_name == "WarpGBM":
        model.fit(X_train.values.astype('int8'), y_train.values.astype('float32'))
    else:
        model.fit(X_train, y_train)

    total_time = time.time() - start


    # ğŸ”® Predict
    if model_name == "WarpGBM":
        preds = model.predict(X_val.values.astype('int8'), chunk_size=1000000)
    else:
        preds = model.predict(X_val)

    rmse = mean_squared_error(y_val, preds, squared=False)

    print(f"{model_name:10s} | ğŸ•’ Time: {total_time:.2f}s | ğŸ�¯ RMSE: {rmse:.4f}")
    return model_name, total_time, rmse



from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from warpgbm import WarpGBM

n_estimators = 100
max_depth = 5
learning_rate = 0.1
max_bin = 128

results = []

# LightGBM
results.append(benchmark(
    LGBMRegressor,
    "LightGBM",
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "max_bin": max_bin,
        "device": "gpu",
        "colsample_bytree": 1.0,
        'num_leaves': 2**max_depth - 1,


    }
))

# CatBoost
results.append(benchmark(
    CatBoostRegressor,
    "CatBoost",
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "task_type": "GPU",
        "devices": "0",
        "verbose": 0,
        'num_leaves': 2**max_depth - 1,
    }
))

# XGBoost
results.append(benchmark(
    XGBRegressor,
    "XGBoost",
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "max_bin": max_bin,
        "device": "cuda",
        "predictor": "gpu_predictor",
        "colsample_bytree": 1.0,
        'num_leaves': 2**max_depth - 1,

    }
))

# WarpGBM
results.append(benchmark(
    WarpGBM,
    "WarpGBM",
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_bins": max_bin,
    }
))



# ğŸ“ˆ Compile results into DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Train Time (s)", "RMSE"])
results_df = results_df.sort_values("Train Time (s)").reset_index(drop=True)
results_df

