import numpy as np
import polars as pl
import joblib

models=[]
for fold in range(5):
    model = joblib.load(f'/kaggle/input/exp01-training/model_lgb{fold}.joblib')
    models.append(model)
le = joblib.load('/kaggle/input/exp01-training/le.joblib')


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


def feature_engineering_inference(data: pl.DataFrame) -> pl.DataFrame:
    demographic_cols = [
        "adult_child", "age", "sex", "handedness",
        "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"
    ]

    # 特徴量の増加
    data = extract_acc_features(data=data)

    # 数値センサーカラムの抽出
    stat_cols = [
        c for c in data.columns
        if c not in demographic_cols + ["sequence_id", "row_id", "sequence_counter", "subject"]
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

    base_features = data.group_by("sequence_id").agg(agg_exprs)

    # ---- カテゴリカル変数 × 数値カラムのクロス統計量 ----
    cat_cols = ["sex", "handedness", "adult_child"]
    cat_stats = []

    for cat_col in cat_cols:
        if cat_col in data.columns:
            for c in stat_cols:
                group = (
                    data.group_by(["sequence_id", cat_col])
                    .agg([
                        pl.col(c).mean().alias(f"{c}_mean"),
                        pl.col(c).std().alias(f"{c}_std"),
                    ])
                )

                # 各カテゴリごとにフィルタしてリネームしてマージ
                for category in group.select(cat_col).unique().to_series(0):
                    filtered = group.filter(pl.col(cat_col) == category).select([
                        pl.col("sequence_id"),
                        pl.col(f"{c}_mean").alias(f"{c}_{cat_col}_{category}_mean"),
                        pl.col(f"{c}_std").alias(f"{c}_{cat_col}_{category}_std"),
                    ])
                    cat_stats.append(filtered)
    
    # ✅ sequence_id を残して安全に順次 join
    cat_features = cat_stats[0]
    for df in cat_stats[1:]:
        cat_features = cat_features.join(df, on="sequence_id", how="left")
    cat_features

    # ---- 最終マージ ----
    if cat_features is not None:
        final = base_features.join(cat_features, on="sequence_id", how="left")
        # final = base_features.join(data[["sequence_id", target_col]].unique(), on="sequence_id", how="left")
    else:
        final = base_features

    return final


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> list[str]:
    predictions = []

    for seq_id in sequence["sequence_id"].unique():
        # 個別にシーケンスを抽出
        seq_df = sequence.filter(pl.col("sequence_id") == seq_id)
        demo_df = demographics.filter(pl.col("sequence_id") == seq_id)

        # 特徴量生成
        cleaned_data = feature_engineering_inference(seq_df)
        pdf = cleaned_data.to_pandas().drop(columns=["sequence_id"])

        # アンサンブル推論
        pred_list = []
        for model in models:
            pred = model.predict(pdf)
            if isinstance(pred, np.ndarray):
                pred = pred[0]
            pred_list.append(int(pred))

        # 投票で最終予測
        predicted_label_id = max(set(pred_list), key=pred_list.count)
        predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]

        predictions.append(predicted_gesture_str)

    return predictions


# def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
#     cleaned_data = feature_engineering_inference(sequence)
#     pdf = cleaned_data.to_pandas().drop(columns=["sequence_id"])
#     # Predict using ensemble of CV models
#     predictions = []
#     for model in models:
#         pred = model.predict(pdf)
#         # Ensure we get a scalar value
#         if isinstance(pred, np.ndarray):
#             pred = pred[0]
#         predictions.append(int(pred))
    
#     # Use majority vote or most confident prediction
#     predicted_label_id = max(set(predictions), key=predictions.count)
    
#     # Convert back to gesture string
#     predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]
    
#     return predicted_gesture_str


import kaggle_evaluation.cmi_inference_server
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




