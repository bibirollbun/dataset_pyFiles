import os
import numpy as np
import pandas as pd
import polars as pl
import pickle
import warnings
import kaggle_evaluation.cmi_inference_server

warnings.filterwarnings("ignore")

# 環境判定
IS_KAGGLE = (
    os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None or
    os.getenv('KAGGLE_KERNEL_RUN_TYPE') is not None or
    os.path.exists('/kaggle/input')
)

print(f"Running on Kaggle: {IS_KAGGLE}")

# モデルパス
V12_MODEL_PATH = "/kaggle/input/cmi-v12-real-data-models"
print("Loading V14 corrected models...")


import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib

# ジェスチャー名定義
GESTURE_NAMES = [
    "Wave hello",
    "Text on phone",
    "Raise hand",
    "Put hand to mouth",
    "Point",
    "Clap",
    "Shake head",
    "Nod head",
    "Check watch or wrist",
    "Cross arms",
    "Adjust glasses",
    "Touch face",
    "Scratch head",
    "Brush hair",
    "Stretch",
    "Yawn",
    "Take a photo",
    "Other gesture",
]

# モデルとメタデータのロード
print("Loading V12 models...")
lgb_model = lgb.Booster(model_file=f"{V12_MODEL_PATH}/lgb_model_v12_final.txt")
xgb_model = xgb.Booster()
xgb_model.load_model(f"{V12_MODEL_PATH}/xgb_model_v12_final.json")
cat_model = CatBoostClassifier()
cat_model.load_model(f"{V12_MODEL_PATH}/cat_model_v12_final.cbm")

# スケーラーと特徴量名のロード
scaler = joblib.load(f"{V12_MODEL_PATH}/scaler_v12.pkl")
feature_names = joblib.load(f"{V12_MODEL_PATH}/feature_names_v12.pkl")

print(f"Models loaded. Expected features: {len(feature_names)}")
print(f"First 5 features: {feature_names[:5]}")
print(f"Last 5 features: {feature_names[-5:]}")


def extract_v12_compatible_features(seq_df):
    """V12モデルと完全に互換性のある特徴量抽出（70個の特徴量）"""
    features = {}

    # Polarsの場合はPandasに変換
    if isinstance(seq_df, pl.DataFrame):
        df = seq_df.to_pandas()
    else:
        df = seq_df.copy()

    # Basic info
    features["subject_id"] = df["subject"].iloc[0] if "subject" in df.columns else 0
    features["seq_length"] = len(df)
    features["seq_length_log"] = np.log1p(len(df))

    # Sensor features for acc_x, acc_y, acc_z
    for sensor in ["acc_x", "acc_y", "acc_z"]:
        if sensor in df.columns:
            values = df[sensor].values
            
            # Basic statistics
            features[f"{sensor}_mean"] = np.mean(values)
            features[f"{sensor}_std"] = np.std(values)
            features[f"{sensor}_max"] = np.max(values)
            features[f"{sensor}_min"] = np.min(values)
            features[f"{sensor}_range"] = np.max(values) - np.min(values)
            features[f"{sensor}_skew"] = pd.Series(values).skew() if len(values) > 1 else 0
            features[f"{sensor}_kurt"] = pd.Series(values).kurtosis() if len(values) > 1 else 0
            
            # Quantiles
            features[f"{sensor}_q25"] = np.percentile(values, 25)
            features[f"{sensor}_q50"] = np.percentile(values, 50)
            features[f"{sensor}_q75"] = np.percentile(values, 75)
            features[f"{sensor}_iqr"] = features[f"{sensor}_q75"] - features[f"{sensor}_q25"]
            
            # FFT features
            if len(values) > 1:
                fft = np.abs(np.fft.fft(values))
                features[f"{sensor}_fft_max"] = np.max(fft)
                features[f"{sensor}_fft_mean"] = np.mean(fft)
                features[f"{sensor}_fft_std"] = np.std(fft)
                features[f"{sensor}_fft_energy"] = np.sum(fft**2)
                features[f"{sensor}_dominant_freq"] = np.argmax(fft)
            else:
                features[f"{sensor}_fft_max"] = 0
                features[f"{sensor}_fft_mean"] = 0
                features[f"{sensor}_fft_std"] = 0
                features[f"{sensor}_fft_energy"] = 0
                features[f"{sensor}_dominant_freq"] = 0
            
            # Jerk features
            if len(values) > 1:
                jerk = np.diff(values)
                features[f"{sensor}_jerk_mean"] = np.mean(jerk)
                features[f"{sensor}_jerk_std"] = np.std(jerk)
                features[f"{sensor}_jerk_max"] = np.max(np.abs(jerk))
            else:
                features[f"{sensor}_jerk_mean"] = 0
                features[f"{sensor}_jerk_std"] = 0
                features[f"{sensor}_jerk_max"] = 0
            
            # Zero crossing
            features[f"{sensor}_zero_cross"] = np.sum(np.diff(np.sign(values)) != 0) if len(values) > 1 else 0
        else:
            # センサーが存在しない場合はデフォルト値
            for suffix in ["_mean", "_std", "_max", "_min", "_range", "_skew", "_kurt", 
                          "_q25", "_q50", "_q75", "_iqr", "_fft_max", "_fft_mean", "_fft_std", 
                          "_fft_energy", "_dominant_freq", "_jerk_mean", "_jerk_std", "_jerk_max", "_zero_cross"]:
                features[f"{sensor}{suffix}"] = 0

    # Magnitude features
    if all(col in df.columns for col in ["acc_x", "acc_y", "acc_z"]):
        acc_mag = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
        features["acc_magnitude_mean"] = np.mean(acc_mag)
        features["acc_magnitude_std"] = np.std(acc_mag)
        features["acc_magnitude_max"] = np.max(acc_mag)
        features["acc_magnitude_energy"] = np.sum(acc_mag**2)
        
        # Linear acceleration (gravity removed)
        gravity = np.array([np.mean(df["acc_x"]), np.mean(df["acc_y"]), np.mean(df["acc_z"])])
        linear_acc = acc_mag - np.linalg.norm(gravity)
        features["linear_acc_mean"] = np.mean(linear_acc)
        features["linear_acc_std"] = np.std(linear_acc)
        features["linear_acc_max"] = np.max(np.abs(linear_acc))
    else:
        features["acc_magnitude_mean"] = 0
        features["acc_magnitude_std"] = 0
        features["acc_magnitude_max"] = 0
        features["acc_magnitude_energy"] = 0
        features["linear_acc_mean"] = 0
        features["linear_acc_std"] = 0
        features["linear_acc_max"] = 0

    return features

print("V12-compatible feature extraction function defined.")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """V14修正版: V12モデルと完全互換の予測関数"""
    
    try:
        # 特徴量抽出
        features = extract_v12_compatible_features(sequence)
        
        # DataFrameに変換
        X = pd.DataFrame([features])
        
        # V12期待特徴量の順序で整理
        X_aligned = pd.DataFrame(index=X.index)
        for feature_name in feature_names:
            if feature_name in X.columns:
                X_aligned[feature_name] = X[feature_name]
            else:
                X_aligned[feature_name] = 0  # 欠損特徴量はゼロで埋める
        
        print(f"Feature alignment: Expected {len(feature_names)}, Got {len(X_aligned.columns)}")
        
        # スケーリング
        X_scaled = scaler.transform(X_aligned)
        
        # 予測
        pred_lgb = lgb_model.predict(X_scaled, num_iteration=lgb_model.best_iteration)[0]
        
        dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
        pred_xgb = xgb_model.predict(dmatrix)[0]
        
        pred_cat = cat_model.predict_proba(X_scaled)[0]
        
        # アンサンブル (V12と同じ重み)
        ensemble_probs = 0.4 * pred_lgb + 0.3 * pred_xgb + 0.3 * pred_cat
        
        # 最も確率の高いクラス
        pred_class = np.argmax(ensemble_probs)
        
        # ジェスチャー名を返す
        gesture_name = GESTURE_NAMES[pred_class] if pred_class < len(GESTURE_NAMES) else "Wave hello"
        
        print(f"Predicted: {gesture_name} (class {pred_class}, prob {ensemble_probs[pred_class]:.3f})")
        
        return gesture_name
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Wave hello"  # フォールバック

print("V14 prediction function defined.")


# 推論サーバー設定
print("Setting up CMI Inference Server...")
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if IS_KAGGLE:
    print("Competition environment detected - serving predictions...")
    inference_server.serve()
else:
    print("Local environment - running gateway test...")
    # ローカルテスト用
    test_data_paths = (
        "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
        "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv",
    )
    
    try:
        inference_server.run_local_gateway(data_paths=test_data_paths)
    except Exception as e:
        print(f"Local test failed: {e}")
        print("This is expected in a local environment without test data.")

print("V14 Corrected inference complete!")

