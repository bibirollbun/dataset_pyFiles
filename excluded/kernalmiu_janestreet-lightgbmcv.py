!pip install /kaggle/input/janestreet2025-code/janestreet-0.1-py3-none-any.whl --force-reinstall --no-deps


import lightgbm as lgb
import polars as pl
import numpy as np
import os
import joblib
import gc
import time 

from janestreet.data_processor import DataProcessor
from janestreet.config import PATH_MODELS
from janestreet.utils import create_folder


MODEL_NAME = "lgbm_cv_v1"
TARGET_COL = "responder_6"
N_FOLDS = 3             
SKIP_DAYS = 677           

LGB_PARAMS = {
    "objective": "regression_l2",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 62,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
    'device' : 'gpu',
    'gpu_use_dp': True,
}


def r2_weighted(y_true, y_pred, weights):
    if len(y_true) == 0:
        return np.nan
        
    weights = np.array(weights)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_mean = np.average(y_true, weights=weights)
    ss_tot = np.sum(weights * (y_true - weighted_mean) ** 2)

    if ss_tot == 0:
        return np.nan
        
    return 1 - ss_res / ss_tot




processor = DataProcessor(
    name=MODEL_NAME,
    skip_days=SKIP_DAYS,
    use_aux_targets=False
)

df_all = processor.get_train_valid_data()
features = processor.features

print(f"总行数: {df_all.height}")
print(f"使用特征数: {len(features)}")


all_dates = sorted(df_all["date_id"].unique().to_list())
fold_size = len(all_dates) // (N_FOLDS + 1)

scores_r2 = []
scores_rmse = []

print("\n开始时间序列 CV...")
print("=" * 60)

fold_Lgbpredictions = []

cv_start_time = time.time()
for fold in range(N_FOLDS):
    fold_start_time = time.time()
    print(f"\nFOLD {fold+1}/{N_FOLDS}")
    print("-" * 60)

    valid_start = fold_size * (fold + 1)
    valid_end = fold_size * (fold + 2)

    valid_dates = set(all_dates[valid_start : valid_end])
    train_dates = set(all_dates[:valid_start])

    df_train = df_all.filter(pl.col("date_id").is_in(train_dates))
    df_valid = df_all.filter(pl.col("date_id").is_in(valid_dates))

    df_train = df_train.with_columns([pl.col(c).cast(pl.Float32) for c in features])
    df_valid = df_valid.with_columns([pl.col(c).cast(pl.Float32) for c in features])

    print(f"Train rows = {df_train.height}, Valid rows = {df_valid.height}")
    data_prep_start = time.time()
    # 提取矩阵
    X_train = df_train.select(features).to_numpy()
    y_train = df_train[TARGET_COL].to_numpy()
    w_train = df_train["weight"].to_numpy()

    X_valid = df_valid.select(features).to_numpy()
    y_valid = df_valid[TARGET_COL].to_numpy()
    w_valid = df_valid["weight"].to_numpy()
    print(f"Data prep time: {time.time() - data_prep_start:.2f} sec")
    train_set = lgb.Dataset(X_train, y_train, weight=w_train, feature_name=features)
    valid_set = lgb.Dataset(X_valid, y_valid, weight=w_valid, feature_name=features, reference=train_set)
    train_start = time.time()
    # 训练
    model = lgb.train(
        params=LGB_PARAMS,
        train_set=train_set,
        num_boost_round=2000,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    print(f"Training time: {time.time() - train_start:.2f} sec")
    # RMSE
    rmse = model.best_score["valid"]["rmse"]
    scores_rmse.append(rmse)

    # Weighted R²
    y_pred = model.predict(X_valid)
    r2 = r2_weighted(y_valid, y_pred, w_valid)
    scores_r2.append(r2)

    print(f"Fold {fold+1} RMSE = {rmse:.6f}")
    print(f"Fold {fold+1} R²   = {r2:.6f}")

    fold_Lgbpredictions.append({
        "fold": fold,
        "y_valid": y_valid.copy(),
        "w_valid": w_valid.copy(),
        "lgb_pred": y_pred.copy(),
        "valid_dates": list(valid_dates),  # optional
    })

    del df_train, df_valid
    del X_train, y_train, w_train
    del X_valid, y_valid, w_valid
    del train_set, valid_set, model
    gc.collect()
    print(f"Fold {fold+1} total time: {time.time() - fold_start_time:.2f} sec")

print("\n" + "=" * 60)
print("CV 完成！")
print(f"Total CV time: {time.time() - cv_start_time:.2f} sec") 
print("RMSE per fold:", scores_rmse)
print("   Mean RMSE:", np.mean(scores_rmse))
print("    Std RMSE:", np.std(scores_rmse))

print("\nR² per fold:", scores_r2)
print("   Mean R² :", np.mean(scores_r2))
print("    Std R² :", np.std(scores_r2))



joblib.dump(fold_Lgbpredictions, "/kaggle/working/lgb_cv_preds.joblib")




