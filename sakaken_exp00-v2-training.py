# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import joblib
from functools import reduce

# Import evaluation API
import kaggle_evaluation.cmi_inference_server
pd.set_option('display.max_rows', 500)
random_seed = 42


# def of file paths
data_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/"

# load datasets
train = pl.read_csv(os.path.join(data_path, "train.csv"))
test = pl.read_csv(os.path.join(data_path, "test.csv"))


train = train.drop(["sequence_type","orientation","behavior","phase"])

# check concat data
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


train.head(1)



test.head(1)



def extract_acc_features(data: pl.DataFrame) -> pl.DataFrame:
    """
    閾値ベースの加速度特徴量を抽出（Polars使用、agg_exprsスタイル）。
    
    Parameters
    ----------
    data : pl.DataFrame
        入力データ。以下のカラムが必要：
        - sequence_id
        - acc_x, acc_y, acc_z

    Returns
    -------
    pl.DataFrame
        sequence_idごとの特徴量を格納したDataFrame
    """
    # ---- 前処理（特徴量計算） ----
    data = data.with_columns([
        # delta_acc = √(Δx² + Δy² + Δz²)
        ((pl.col("acc_x").diff().pow(2) +
          pl.col("acc_y").diff().pow(2) +
          pl.col("acc_z").diff().pow(2)).sqrt()).alias("delta_acc"),
    ])
    data = data.with_columns([
        # AVM = √(x² + y² + z²)
        ((pl.col("acc_x").pow(2) +
          pl.col("acc_y").pow(2) +
          pl.col("acc_z").pow(2)).sqrt()).alias("avm")
    ])
    data = data.with_columns([
        # 閾値フラグ
        (pl.col("delta_acc") > 0.2).cast(pl.Int8).alias("motion_gt_0.2g"),
        (pl.col("avm") > 1.5).cast(pl.Int8).alias("high_intensity_flag")
    ])

    # ---- 姿勢変化 GM ----
    gm_df = (
        data.group_by("sequence_id")
        .agg([
            pl.mean("acc_x").alias("mean_x"),
            pl.mean("acc_y").alias("mean_y"),
            pl.mean("acc_z").alias("mean_z"),
        ])
        .with_columns([
            ((pl.col("mean_x").pow(2) + pl.col("mean_y").pow(2) + pl.col("mean_z").pow(2)).sqrt()).alias("gm")
        ])
        .select(["sequence_id", "gm"])
    )

    # ---- 閾値ベース特徴量集約 ----
    agg_exprs = [
        pl.sum("motion_gt_0.2g").alias("motion_count_gt_0.2g"),
        pl.sum("high_intensity_flag").alias("high_intensity_count"),
        pl.mean("high_intensity_flag").alias("high_intensity_ratio"),
    ]

    thresh_features = data.group_by("sequence_id").agg(agg_exprs)

    # ---- 結合 ----
    result = thresh_features.join(gm_df, on="sequence_id", how="left")
    result = data.join(result, on= "sequence_id", how="left")

    return result


def feature_engineering_train(data: pl.DataFrame) -> pl.DataFrame:
    target_col = "gesture"

    # 特徴量の増加
    data = extract_acc_features(data=data)

    # 数値センサーカラムの抽出
    stat_cols = [
        c for c in data.columns
        if c not in [target_col, "sequence_id", "row_id", "sequence_counter", "subject"]
    ]

    # ---- 通常の sequence_id 単位の集約 ----
    agg_exprs = []
    for c in stat_cols:
        agg_exprs.extend([
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
        ])

    final = data.group_by("sequence_id").agg(agg_exprs)
    final = final.join(data[["sequence_id", target_col]].unique(), on="sequence_id", how="left")

    return final


cleaned_data = feature_engineering_train(train)
cleaned_data.head(1)


target_col = "gesture"
pdf = cleaned_data.to_pandas()  # keeps nullable dtypes

le = LabelEncoder()
y = le.fit_transform(pdf[target_col])
X = pdf.drop(columns=[target_col, "sequence_id"])         # drop id + label

joblib.dump(le, 'le.joblib')


def competition_metric(y_true, y_pred, le_instance, all_original_gestures):
    """
    Competition metric calculation
    """
    bfrb_gestures = [g for g in all_original_gestures if g in le_instance.classes_]
    
    # Binary F1: All are Target in this filtered dataset
    y_true_binary = np.ones_like(y_true, dtype=int)
    y_pred_binary = np.ones_like(y_pred, dtype=int)
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label=1, zero_division=0)
    
    # Macro F1: specific gesture classification
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    final_score = (binary_f1 + macro_f1) / 2
    return final_score, binary_f1, macro_f1


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

all_original_gestures_in_train = pdf['gesture'].unique()

callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=100)]

# LightGBM model with cross-validation
print("\nTraining LightGBM models with cross-validation...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold + 1}/5")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # LightGBM model with GPU acceleration
    model = lgb.LGBMClassifier(
        objective='multiclass',
        n_estimators= 1000,
        learning_rate= 0.08,
        max_depth= 15,
        reg_alpha= 0.8,
        lambda_l2= 4.0,  
        num_leaves=31, 
        min_child_samples= 32,
        colsample_bytree= 0.85,
        subsample= 0.5,
        subsample_freq=0,
        cat_smooth=20.0,
        is_unbalance=True,
        max_bin=127,
        verbose=-1,  
        metric='multi_logloss',   
        device='gpu',  
    )

    # Train model with verbose output
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],  
        eval_metric='multi_logloss',  
        callbacks=callbacks
    )
    
    # Predict
    y_pred_fold = model.predict(X_val_fold)

    # Calculate score
    score, binary_f1, macro_f1 = competition_metric(
        y_true=y_val_fold,
        y_pred=y_pred_fold,
        all_original_gestures=all_original_gestures_in_train,
        le_instance=le
    )
    
    cv_scores.append(score)
    models.append(model)
    joblib.dump(model, f'model_lgb{fold}.joblib')
    
    print(f"Fold {fold + 1} - Competition Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")

    # feature importances
    feature_importances = model.feature_importances_    
    feature_names = X.columns    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)    
    print(feature_importance_df)



print(f"\nCross-validation results:")
print(f"Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
print(f"Individual fold scores: {cv_scores}")

# Train final model on all data with GPU acceleration
print("\nTraining final model on all training data...")


def feature_engineering_inference(data: pl.DataFrame) -> pl.DataFrame:
    # 特徴量の増加
    data = extract_acc_features(data=data)

    # 数値センサーカラムの抽出
    stat_cols = [
        c for c in data.columns
        if c not in ["sequence_id", "row_id", "sequence_counter", "subject"]
    ]

    # ---- 通常の sequence_id 単位の集約 ----
    agg_exprs = []
    for c in stat_cols:
        agg_exprs.extend([
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
        ])

    final = data.group_by("sequence_id").agg(agg_exprs)

    return final


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    cleaned_data = feature_engineering_inference(sequence)
    pdf = cleaned_data.to_pandas().drop(columns=["sequence_id"])
    # Predict using ensemble of CV models
    predictions = []
    for model in models:
        pred = model.predict(pdf)
        # Ensure we get a scalar value
        if isinstance(pred, np.ndarray):
            pred = pred[0]
        predictions.append(int(pred))
    
    # Use majority vote or most confident prediction
    predicted_label_id = max(set(predictions), key=predictions.count)
    
    # Convert back to gesture string
    predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]
    
    return predicted_gesture_str


import os
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

