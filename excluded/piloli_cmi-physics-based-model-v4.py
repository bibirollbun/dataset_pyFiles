import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import gc
import sys
import warnings

warnings.filterwarnings("ignore")

# Add path for inference server
sys.path.append("/kaggle/input/cmi-detect-behavior-with-sensor-data")

print("Libraries imported successfully")


# Import inference server
from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

print("CMI Inference Server imported")


def remove_gravity_from_acc(df_seq, alpha=0.8):
    """重力除去加速度の計算（トップ解法の必須要素）"""
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    acc_data = df_seq[acc_cols].values

    # Low-pass filter for gravity estimation
    gravity = np.zeros_like(acc_data)
    gravity[0] = acc_data[0]

    for i in range(1, len(acc_data)):
        gravity[i] = alpha * gravity[i - 1] + (1 - alpha) * acc_data[i]

    # Linear acceleration = total - gravity
    linear_acc = acc_data - gravity
    return linear_acc


def calculate_angular_velocity_from_quat(df_seq):
    """クォータニオンから角速度を計算（0.852スコア達成手法）"""
    quat_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
    quat = df_seq[quat_cols].values

    # Quaternion difference
    quat_diff = np.diff(quat, axis=0)

    # Angular velocity approximation
    angular_vel = np.zeros((len(df_seq), 3))
    angular_vel[1:, 0] = 2 * quat_diff[:, 0]
    angular_vel[1:, 1] = 2 * quat_diff[:, 1]
    angular_vel[1:, 2] = 2 * quat_diff[:, 2]

    return angular_vel


def calculate_jerk(df_seq):
    """ジャーク（加速度変化率）の計算"""
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    acc_data = df_seq[acc_cols].values

    # Jerk = derivative of acceleration
    jerk = np.zeros_like(acc_data)
    jerk[1:] = np.diff(acc_data, axis=0)

    return jerk


def extract_tof_statistics(df_seq):
    """ToFセンサーの統計量抽出（320次元→20次元）"""
    tof_features = {}

    for sensor_id in range(1, 6):
        pixel_cols = [f"tof_{sensor_id}_v{p}" for p in range(64)]
        if all(col in df_seq.columns for col in pixel_cols):
            tof_data = df_seq[pixel_cols].replace(-1, np.nan)

            # 統計量のみ保持（メモリ効率化）
            tof_features[f"tof_{sensor_id}_mean"] = tof_data.mean(axis=1).mean()
            tof_features[f"tof_{sensor_id}_std"] = tof_data.std(axis=1).mean()
            tof_features[f"tof_{sensor_id}_min"] = tof_data.min(axis=1).mean()
            tof_features[f"tof_{sensor_id}_max"] = tof_data.max(axis=1).mean()

    return tof_features


def create_physics_features(df_seq):
    """物理ベース特徴量の生成（トップ解法の要素統合）"""
    features = {}

    # 1. 重力除去加速度
    linear_acc = remove_gravity_from_acc(df_seq)
    features["linear_acc_x_mean"] = linear_acc[:, 0].mean()
    features["linear_acc_y_mean"] = linear_acc[:, 1].mean()
    features["linear_acc_z_mean"] = linear_acc[:, 2].mean()
    features["linear_acc_x_std"] = linear_acc[:, 0].std()
    features["linear_acc_y_std"] = linear_acc[:, 1].std()
    features["linear_acc_z_std"] = linear_acc[:, 2].std()

    # Linear acceleration magnitude
    linear_acc_mag = np.sqrt(np.sum(linear_acc**2, axis=1))
    features["linear_acc_mag_mean"] = linear_acc_mag.mean()
    features["linear_acc_mag_std"] = linear_acc_mag.std()
    features["linear_acc_mag_max"] = linear_acc_mag.max()

    # 2. 角速度
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    features["angular_vel_x_mean"] = angular_vel[:, 0].mean()
    features["angular_vel_y_mean"] = angular_vel[:, 1].mean()
    features["angular_vel_z_mean"] = angular_vel[:, 2].mean()
    features["angular_vel_x_std"] = angular_vel[:, 0].std()
    features["angular_vel_y_std"] = angular_vel[:, 1].std()
    features["angular_vel_z_std"] = angular_vel[:, 2].std()

    # Angular velocity magnitude
    angular_vel_mag = np.sqrt(np.sum(angular_vel**2, axis=1))
    features["angular_vel_mag_mean"] = angular_vel_mag.mean()
    features["angular_vel_mag_std"] = angular_vel_mag.std()

    # 3. ジャーク
    jerk = calculate_jerk(df_seq)
    jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
    features["jerk_mag_mean"] = jerk_mag.mean()
    features["jerk_mag_std"] = jerk_mag.std()
    features["jerk_mag_max"] = jerk_mag.max()

    # 4. 基本的な加速度統計
    acc_mag = np.sqrt(df_seq["acc_x"] ** 2 + df_seq["acc_y"] ** 2 + df_seq["acc_z"] ** 2)
    features["acc_mag_mean"] = acc_mag.mean()
    features["acc_mag_std"] = acc_mag.std()
    features["acc_mag_max"] = acc_mag.max()
    features["acc_mag_min"] = acc_mag.min()

    # 5. 回転統計
    features["rot_x_mean"] = df_seq["rot_x"].mean()
    features["rot_y_mean"] = df_seq["rot_y"].mean()
    features["rot_z_mean"] = df_seq["rot_z"].mean()
    features["rot_w_mean"] = df_seq["rot_w"].mean()

    # 6. 温度センサー（実際は5個のみ）
    temp_cols = [f"thm_{i}" for i in range(1, 6)]
    if all(col in df_seq.columns for col in temp_cols):
        temp_data = df_seq[temp_cols]
        features["temp_mean"] = temp_data.mean().mean()
        features["temp_std"] = temp_data.std().mean()
        features["temp_max"] = temp_data.max().max()
        features["temp_min"] = temp_data.min().min()
    else:
        # 温度センサーがない場合のデフォルト値
        features["temp_mean"] = 0
        features["temp_std"] = 0
        features["temp_max"] = 0
        features["temp_min"] = 0

    # 7. ToF統計量（メモリ効率化）
    tof_stats = extract_tof_statistics(df_seq)
    features.update(tof_stats)

    # 8. 時系列長
    features["sequence_length"] = len(df_seq)

    return features


# Load training data
print("Loading training data...")
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demo = pd.read_csv(
    "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
)

# Get actual gesture labels from data
actual_gestures = sorted(train_df["gesture"].unique())
print(f"Actual gesture classes: {actual_gestures}")

print(f"Data loaded: {len(train_df)} samples")
print(f"Unique sequences: {train_df['sequence_id'].nunique()}")
print(f"Number of gesture classes: {len(actual_gestures)}")


# Feature extraction
print("Extracting features with physics-based approach...")

X = []
y = []
sequence_ids = train_df["sequence_id"].unique()

for i, seq_id in enumerate(sequence_ids):
    if i % 100 == 0:
        print(f"Processing sequence {i}/{len(sequence_ids)}")

    # Get sequence data
    seq_data = train_df[train_df["sequence_id"] == seq_id].copy()

    # Extract physics features
    features = create_physics_features(seq_data)

    # Add demographics using subject column
    subject_id = seq_data["subject"].iloc[0]
    demo_data = train_demo[train_demo["subject"] == subject_id].iloc[0]
    features["age"] = demo_data["age"]
    features["is_male"] = 1 if demo_data["sex"] == "M" else 0
    features["height"] = demo_data.get("height_cm", 170)  # Default height

    X.append(features)
    y.append(seq_data["gesture"].iloc[0])

    # Memory cleanup
    if i % 500 == 0:
        gc.collect()

# Convert to DataFrame
X_df = pd.DataFrame(X)
y_series = pd.Series(y)

print(f"Feature extraction complete: {X_df.shape}")
print(f"Features: {list(X_df.columns)[:10]}...")


# Train LightGBM model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode labels with actual gesture classes
le = LabelEncoder()
le.fit(actual_gestures)  # Use actual gestures from data
y_encoded = le.transform(y_series)

print(f"Number of classes: {len(le.classes_)}")
print(f"Classes: {le.classes_}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# LightGBM parameters (optimized for memory)
num_classes = len(le.classes_)
lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,  # Reduced for memory
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "seed": 42,
    "num_threads": 2,  # Limited threads for stability
}

# Train model
print("Training LightGBM model...")
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val)

model_lgb = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

print("LightGBM training complete")


# Train XGBoost model
import xgboost as xgb

print("Training XGBoost model...")

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,  # Use same number of classes
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "eval_metric": "mlogloss",
    "seed": 42,
    "nthread": 2,  # Limited threads
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

model_xgb = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, "validation")],
    early_stopping_rounds=50,
    verbose_eval=100,
)

print("XGBoost training complete")


# Validation performance
from sklearn.metrics import accuracy_score

# LightGBM predictions
pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
pred_lgb_class = np.argmax(pred_lgb, axis=1)

# XGBoost predictions
dval = xgb.DMatrix(X_val)
pred_xgb = model_xgb.predict(dval)
pred_xgb_class = np.argmax(pred_xgb, axis=1)

# Ensemble predictions (weighted average)
ensemble_weights = [0.6, 0.4]  # LightGBM, XGBoost
pred_ensemble = ensemble_weights[0] * pred_lgb + ensemble_weights[1] * pred_xgb
pred_ensemble_class = np.argmax(pred_ensemble, axis=1)

print(f"LightGBM Accuracy: {accuracy_score(y_val, pred_lgb_class):.4f}")
print(f"XGBoost Accuracy: {accuracy_score(y_val, pred_xgb_class):.4f}")
print(f"Ensemble Accuracy: {accuracy_score(y_val, pred_ensemble_class):.4f}")


# Save models
print("Saving models...")

with open("model_lgb.pkl", "wb") as f:
    pickle.dump(model_lgb, f)

with open("model_xgb.pkl", "wb") as f:
    pickle.dump(model_xgb, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Models saved successfully")


def predict(sequence, demographics):
    """
    Physics-based prediction with ensemble model.
    Memory-optimized for stable inference server operation.
    """
    try:
        # Convert to DataFrame
        df_seq = pd.DataFrame(sequence)

        # Extract physics-based features
        features = create_physics_features(df_seq)

        # Add demographics (using lowercase keys from CMI server)
        features["age"] = demographics.get("age", 30)
        features["is_male"] = 1 if demographics.get("sex", "M") == "M" else 0
        features["height"] = demographics.get("height_cm", 170)

        # Convert to DataFrame for prediction
        X = pd.DataFrame([features])

        # LightGBM prediction
        pred_lgb = model_lgb.predict(X, num_iteration=model_lgb.best_iteration)

        # XGBoost prediction
        dtest = xgb.DMatrix(X)
        pred_xgb = model_xgb.predict(dtest)

        # Ensemble
        pred_ensemble = 0.6 * pred_lgb + 0.4 * pred_xgb

        # Get predicted class
        pred_class_idx = np.argmax(pred_ensemble[0])
        pred_gesture = le.inverse_transform([pred_class_idx])[0]

        # Memory cleanup
        del df_seq, features, X, pred_lgb, pred_xgb, pred_ensemble
        gc.collect()

        return pred_gesture

    except Exception as e:
        print(f"Prediction error: {e}")
        # Fallback to most common gesture from training data
        return "Text on phone"  # Most common in training data


print("Prediction function defined")


# Test the prediction function
print("Testing prediction function...")

# Get a sample sequence
test_seq_id = train_df["sequence_id"].iloc[0]
test_seq = train_df[train_df["sequence_id"] == test_seq_id].drop(
    [
        "sequence_id",
        "behavior",
        "gesture",
        "subject",
        "orientation",
        "phase",
        "row_id",
        "sequence_type",
        "sequence_counter",
    ],
    axis=1,
)

# Get demographics for test
test_subject = train_df[train_df["sequence_id"] == test_seq_id]["subject"].iloc[0]
test_demo_row = train_demo[train_demo["subject"] == test_subject].iloc[0]
test_demo = {
    "age": test_demo_row["age"],
    "sex": test_demo_row["sex"],
    "height_cm": test_demo_row.get("height_cm", 170),
}

# Test prediction
test_pred = predict(test_seq.to_dict("records"), test_demo)
actual_gesture = train_df[train_df["sequence_id"] == test_seq_id]["gesture"].iloc[0]

print(f"Test prediction: {test_pred}")
print(f"Actual gesture: {actual_gesture}")
print(f"Match: {test_pred == actual_gesture}")


# Start inference server
print("Starting CMI Inference Server...")
print("Server will handle predictions for test data")

server = CMIInferenceServer(predict)
server.serve()

print("Inference complete")

