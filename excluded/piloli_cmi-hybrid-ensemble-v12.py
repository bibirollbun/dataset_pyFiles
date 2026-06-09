import os
import sys
import numpy as np
import pandas as pd
import polars as pl
import pickle
import warnings

warnings.filterwarnings("ignore")

# 環境変数を確認 - Kaggle環境の複数の判定方法
IS_KAGGLE = (
    os.getenv("KAGGLE_IS_COMPETITION_RERUN") is not None
    or os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None
    or os.path.exists("/kaggle/input")
)
print(f"Running on Kaggle: {IS_KAGGLE}")

if IS_KAGGLE:
    DATA_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test"
    # V11モデル（合成データモデル）- 正しいDataset名を使用
    V11_MODEL_PATH = "/kaggle/input/cmi-v11-models"
    # V12モデル（実データモデル）
    V12_MODEL_PATH = "/kaggle/input/cmi-v12-real-data-models"
else:
    DATA_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test"
    V11_MODEL_PATH = "cmi-v11-dataset"
    V12_MODEL_PATH = "models_v12"


# モデルをロード
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading V11 models (synthetic data)...")
# V11モデル - 最終モデルのみ使用
lgb_model_v11 = lgb.Booster(model_file=os.path.join(V11_MODEL_PATH, "lgb_model_v11.txt"))
cat_model_v11 = CatBoostClassifier()
cat_model_v11.load_model(os.path.join(V11_MODEL_PATH, "cat_model_v11.cbm"))

print("Loading V12 models (real data)...")
# V12モデル - 最終モデルのみ使用
lgb_model_v12 = lgb.Booster(
    model_file=os.path.join(V12_MODEL_PATH, "lgb_model_v12_final.txt")
)
xgb_model_v12 = xgb.Booster()
xgb_model_v12.load_model(os.path.join(V12_MODEL_PATH, "xgb_model_v12_final.json"))
cat_model_v12 = CatBoostClassifier()
cat_model_v12.load_model(os.path.join(V12_MODEL_PATH, "cat_model_v12_final.cbm"))

# V12用のscalerとその他のメタデータ
with open(os.path.join(V12_MODEL_PATH, "scaler_v12.pkl"), "rb") as f:
    scaler_v12 = pickle.load(f)
with open(os.path.join(V12_MODEL_PATH, "feature_names_v12.pkl"), "rb") as f:
    feature_names_v12 = pickle.load(f)
with open(os.path.join(V12_MODEL_PATH, "gesture_mapping_v12.pkl"), "rb") as f:
    gesture_mapping_v12 = pickle.load(f)

print(f"Loaded models successfully")
print(f"V12 features: {len(feature_names_v12)}")
print(f"Gesture mapping: {gesture_mapping_v12}")


def extract_advanced_features(df, seq_id, metadata_row):
    """V12で使用する高度な特徴量抽出（V12と同じ実装）"""
    features = {}

    # 基本統計量
    for col in df.columns:
        if col != "timestamp":
            features[f"{col}_mean"] = np.mean(df[col])
            features[f"{col}_std"] = np.std(df[col])
            features[f"{col}_min"] = np.min(df[col])
            features[f"{col}_max"] = np.max(df[col])
            features[f"{col}_median"] = np.median(df[col])
            features[f"{col}_q25"] = np.percentile(df[col], 25)
            features[f"{col}_q75"] = np.percentile(df[col], 75)
            features[f"{col}_iqr"] = features[f"{col}_q75"] - features[f"{col}_q25"]
            features[f"{col}_skew"] = df[col].skew() if len(df[col]) > 2 else 0
            features[f"{col}_kurtosis"] = df[col].kurtosis() if len(df[col]) > 3 else 0

    # 重力除去線形加速度
    gravity_x = np.mean(df["acc_x"])
    gravity_y = np.mean(df["acc_y"])
    gravity_z = np.mean(df["acc_z"])

    linear_acc_x = df["acc_x"] - gravity_x
    linear_acc_y = df["acc_y"] - gravity_y
    linear_acc_z = df["acc_z"] - gravity_z

    features["linear_acc_x_mean"] = np.mean(linear_acc_x)
    features["linear_acc_y_mean"] = np.mean(linear_acc_y)
    features["linear_acc_z_mean"] = np.mean(linear_acc_z)
    features["linear_acc_magnitude"] = np.mean(
        np.sqrt(linear_acc_x**2 + linear_acc_y**2 + linear_acc_z**2)
    )

    # クォータニオンからオイラー角
    qw = df["quat_w"].values
    qx = df["quat_x"].values
    qy = df["quat_y"].values
    qz = df["quat_z"].values

    # ロール、ピッチ、ヨー
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    features["euler_roll_mean"] = np.mean(roll)
    features["euler_pitch_mean"] = np.mean(pitch)
    features["euler_yaw_mean"] = np.mean(yaw)
    features["euler_roll_std"] = np.std(roll)
    features["euler_pitch_std"] = np.std(pitch)
    features["euler_yaw_std"] = np.std(yaw)

    # 角速度（クォータニオンの時間微分）
    if len(df) > 1:
        dt = np.diff(df["timestamp"].values) / 1000.0
        dqw = np.diff(qw) / dt
        dqx = np.diff(qx) / dt
        dqy = np.diff(qy) / dt
        dqz = np.diff(qz) / dt

        features["angular_velocity_mean"] = np.mean(np.sqrt(dqx**2 + dqy**2 + dqz**2))
        features["angular_velocity_std"] = np.std(np.sqrt(dqx**2 + dqy**2 + dqz**2))
    else:
        features["angular_velocity_mean"] = 0
        features["angular_velocity_std"] = 0

    # ジャーク（加速度の変化率）
    if len(df) > 1:
        jerk_x = np.diff(df["acc_x"]) / dt
        jerk_y = np.diff(df["acc_y"]) / dt
        jerk_z = np.diff(df["acc_z"]) / dt

        features["jerk_x_mean"] = np.mean(jerk_x)
        features["jerk_y_mean"] = np.mean(jerk_y)
        features["jerk_z_mean"] = np.mean(jerk_z)
        features["jerk_magnitude"] = np.mean(np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2))
    else:
        features["jerk_x_mean"] = 0
        features["jerk_y_mean"] = 0
        features["jerk_z_mean"] = 0
        features["jerk_magnitude"] = 0

    # ToFセンサー統計
    tof_cols = [col for col in df.columns if col.startswith("tof_")]
    if tof_cols:
        tof_values = df[tof_cols].values.flatten()
        features["tof_global_mean"] = np.mean(tof_values)
        features["tof_global_std"] = np.std(tof_values)
        features["tof_global_min"] = np.min(tof_values)
        features["tof_global_max"] = np.max(tof_values)

    # シーケンス長
    features["sequence_length"] = len(df)
    features["duration_ms"] = df["timestamp"].max() - df["timestamp"].min()

    return features


def extract_simple_features(df, seq_id, metadata_row):
    """V11で使用するシンプルな特徴量抽出"""
    features = {}

    # IMUセンサーデータの基本統計量
    imu_cols = ["acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"]
    for col in imu_cols:
        if col in df.columns:
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_min"] = df[col].min()
            features[f"{col}_max"] = df[col].max()

    # ToFセンサーの統計量
    tof_cols = [col for col in df.columns if col.startswith("tof_")]
    if tof_cols:
        tof_values = df[tof_cols].values.flatten()
        features["tof_mean"] = np.mean(tof_values)
        features["tof_std"] = np.std(tof_values)
        features["tof_min"] = np.min(tof_values)
        features["tof_max"] = np.max(tof_values)

    # 加速度の大きさ
    if all(col in df.columns for col in ["acc_x", "acc_y", "acc_z"]):
        acc_magnitude = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
        features["acc_magnitude_mean"] = acc_magnitude.mean()
        features["acc_magnitude_std"] = acc_magnitude.std()

    # シーケンス長
    features["sequence_length"] = len(df)

    return features


# CMI推論サーバーのセットアップ（競技提出用）
if IS_KAGGLE:
    from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

    def predict(test_data: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
        """ハイブリッドアンサンブル予測"""
        print(f"Processing {len(metadata)} sequences for prediction")

        # pandas DataFrameに変換
        test_df = test_data.to_pandas()
        metadata_df = metadata.to_pandas()

        predictions = []

        for idx, row in metadata_df.iterrows():
            seq_id = row["sequence_id"]

            # シーケンスデータを取得
            seq_data = test_df[test_df["sequence_id"] == seq_id].copy()

            if len(seq_data) == 0:
                print(f"Warning: No data for sequence {seq_id}")
                # デフォルトはジェスチャー名で返す
                predictions.append("Wave hello")
                continue

            # V11用の特徴量抽出（シンプル）
            features_v11 = extract_simple_features(seq_data, seq_id, row)
            X_v11 = pd.DataFrame([features_v11])

            # V11予測（LightGBM + CatBoost）
            pred_lgb_v11 = lgb_model_v11.predict(
                X_v11, num_iteration=lgb_model_v11.best_iteration
            )[0]
            pred_cat_v11 = cat_model_v11.predict_proba(X_v11)[0]

            # V11アンサンブル（LightGBM 50% + CatBoost 50%）
            pred_v11 = 0.5 * pred_lgb_v11 + 0.5 * pred_cat_v11

            # V12用の特徴量抽出（高度）
            features_v12 = extract_advanced_features(seq_data, seq_id, row)
            X_v12 = pd.DataFrame([features_v12])

            # V12用の特徴量を整形
            X_v12_aligned = X_v12[feature_names_v12]
            X_v12_scaled = scaler_v12.transform(X_v12_aligned)

            # V12予測（LightGBM + XGBoost + CatBoost）
            pred_lgb_v12 = lgb_model_v12.predict(
                X_v12_scaled, num_iteration=lgb_model_v12.best_iteration
            )[0]

            dmatrix_v12 = xgb.DMatrix(X_v12_scaled, feature_names=feature_names_v12)
            pred_xgb_v12 = xgb_model_v12.predict(dmatrix_v12)[0]

            pred_cat_v12 = cat_model_v12.predict_proba(X_v12_scaled)[0]

            # V12アンサンブル（LightGBM 40% + XGBoost 30% + CatBoost 30%）
            pred_v12 = 0.4 * pred_lgb_v12 + 0.3 * pred_xgb_v12 + 0.3 * pred_cat_v12

            # ハイブリッドアンサンブル
            # V11（合成データ）20% + V12（実データ）80%
            hybrid_probs = 0.2 * pred_v11 + 0.8 * pred_v12

            # 最も確率の高いクラスを選択
            pred_class = np.argmax(hybrid_probs)

            # gesture_mappingで元のジェスチャー名に戻す
            # gesture_mapping_v12は{'gesture_to_id': {...}, 'id_to_gesture': {...}}の構造
            if (
                isinstance(gesture_mapping_v12, dict)
                and "id_to_gesture" in gesture_mapping_v12
            ):
                gesture_name = gesture_mapping_v12["id_to_gesture"].get(
                    pred_class, "Wave hello"
                )
            else:
                # フォールバック：直接マッピング
                gesture_names = [
                    "Above ear - pull hair",
                    "Cheek - pinch skin",
                    "Drink from bottle/cup",
                    "Eyebrow - pull hair",
                    "Eyelash - pull hair",
                    "Feel around in tray and pull out an object",
                    "Forehead - pull hairline",
                    "Forehead - scratch",
                    "Glasses on/off",
                    "Neck - pinch skin",
                    "Neck - scratch",
                    "Pinch knee/leg skin",
                    "Pull air toward your face",
                    "Scratch knee/leg skin",
                    "Text on phone",
                    "Wave hello",
                    "Write name in air",
                    "Write name on leg",
                ]
                gesture_name = (
                    gesture_names[pred_class] if pred_class < 18 else "Wave hello"
                )

            predictions.append(gesture_name)

        # Polars DataFrameとして返す
        result = pl.DataFrame(
            {"sequence_id": metadata["sequence_id"], "gesture": predictions}
        )

        print(f"Predictions completed. Shape: {result.shape}")
        print(f"Sample predictions: {result.head()}")

        return result

    # 推論サーバーを開始
    print("Starting CMI Inference Server...")
    inference_server = CMIInferenceServer(predict)
    print("CMI Inference Server initialized")

    # serve()を呼ぶ - 競技データを処理
    print("Calling serve() - processing competition data...")
    inference_server.serve()
    print("serve() completed")

else:
    print("Local testing mode - CMI inference server not available")
    print("Creating dummy submission for local testing...")

    # ローカルテスト用のダミー出力
    gesture_names = [
        "Above ear - pull hair",
        "Cheek - pinch skin",
        "Drink from bottle/cup",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Feel around in tray and pull out an object",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Glasses on/off",
        "Neck - pinch skin",
        "Neck - scratch",
        "Pinch knee/leg skin",
        "Pull air toward your face",
        "Scratch knee/leg skin",
        "Text on phone",
        "Wave hello",
        "Write name in air",
        "Write name on leg",
    ]

    dummy_submission = pl.DataFrame(
        {
            "sequence_id": list(range(1, 101)),
            "gesture": [gesture_names[i % 18] for i in range(100)],
        }
    )
    dummy_submission.write_parquet("submission.parquet")
    print(f"Dummy submission.parquet created with shape {dummy_submission.shape}")

