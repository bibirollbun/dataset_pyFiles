import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
import pickle
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
print("Libraries imported successfully")


# データパス設定（競技データセット参照）
DATA_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")

# train.csvの読み込み（時系列データ）
print("Loading CMI competition data...")
train_data = pd.read_csv(DATA_DIR / "train.csv")
print(f"  Loaded train.csv: {len(train_data)} rows")

# sequence_idでグループ化してメタデータ作成
train_metadata = (
    train_data.groupby("sequence_id")
    .agg({"subject": "first", "gesture": "first", "sequence_counter": "count"})
    .reset_index()
)
train_metadata.columns = ["sequence_id", "subject", "gesture", "sequence_length"]

# ジェスチャーのラベルマッピング作成（重要！）
gesture_labels = sorted(train_metadata["gesture"].unique())
gesture_to_id = {label: i for i, label in enumerate(gesture_labels)}
id_to_gesture = {i: label for label, i in gesture_to_id.items()}

print(f"\\nData loaded successfully:")
print(f"  Total sequences: {len(train_metadata)}")
print(f"  Gesture labels: {len(gesture_labels)} types")
print(f"  Gesture distribution:\\n{train_metadata['gesture'].value_counts()}")


def analyze_real_data_distribution():
    """実データの統計的特性を分析"""

    # ジェスチャー分布
    gesture_dist = train_metadata["gesture"].value_counts(normalize=True)

    # センサーデータの統計（サンプリング）
    sensor_stats = {}
    sample_sequences = train_metadata.sample(min(100, len(train_metadata)))[
        "sequence_id"
    ].tolist()

    all_acc = []
    all_gyr = []
    all_quat = []

    for seq_id in sample_sequences:
        # train.csvから該当シーケンスのデータを取得
        seq_data = train_data[train_data["sequence_id"] == seq_id]

        # 加速度計
        acc_cols = [c for c in seq_data.columns if c.startswith("acc_")]
        if acc_cols:
            all_acc.append(seq_data[acc_cols].values)

        # ジャイロスコープ
        gyr_cols = [c for c in seq_data.columns if c.startswith("gyr_")]
        if gyr_cols:
            all_gyr.append(seq_data[gyr_cols].values)

        # クォータニオン
        quat_cols = [c for c in seq_data.columns if "quat" in c]
        if quat_cols:
            all_quat.append(seq_data[quat_cols].values)

    # 統計量計算
    if all_acc:
        acc_data = np.vstack(all_acc)
        sensor_stats["acc"] = {
            "mean": np.mean(acc_data, axis=0),
            "std": np.std(acc_data, axis=0),
            "percentiles": [
                np.percentile(acc_data, p, axis=0) for p in [5, 25, 50, 75, 95]
            ],
        }

    if all_gyr:
        gyr_data = np.vstack(all_gyr)
        sensor_stats["gyr"] = {
            "mean": np.mean(gyr_data, axis=0),
            "std": np.std(gyr_data, axis=0),
            "percentiles": [
                np.percentile(gyr_data, p, axis=0) for p in [5, 25, 50, 75, 95]
            ],
        }

    return gesture_dist, sensor_stats


# 実データ分析
print("Analyzing real data distribution...")
gesture_dist, sensor_stats = analyze_real_data_distribution()
print("Real data analysis completed")


def generate_synthetic_sequence(gesture, seq_length, sensor_stats):
    """単一の合成シーケンスを生成"""

    # 時間軸
    time_stamps = np.arange(seq_length) * 0.02  # 50Hz

    sequence_data = {"time": time_stamps}

    # 加速度計データ（テストデータはより多様と予想）
    if "acc" in sensor_stats:
        acc_mean = sensor_stats["acc"]["mean"]
        acc_std = sensor_stats["acc"]["std"] * 1.3  # テストは少し分散大きめ

        # 基本データ
        base_acc = np.random.normal(acc_mean, acc_std, (seq_length, len(acc_mean)))

        # ジェスチャー固有のパターン
        freq = 0.5 + gesture * 0.3  # 各ジェスチャーの周波数
        amplitude = 0.3 + (gesture % 6) * 0.1

        for j in range(min(3, base_acc.shape[1])):
            phase = np.random.rand() * 2 * np.pi
            base_acc[:, j] += amplitude * np.sin(2 * np.pi * freq * time_stamps + phase)

            # 時々スパイクを追加（急な動き）
            if np.random.rand() < 0.3:
                spike_pos = np.random.randint(seq_length // 4, 3 * seq_length // 4)
                base_acc[spike_pos : spike_pos + 5, j] *= 1.5

        for j, col in enumerate(["acc_x", "acc_y", "acc_z"]):
            sequence_data[col] = (
                base_acc[:, j] if j < base_acc.shape[1] else np.zeros(seq_length)
            )

    # ジャイロスコープデータ
    if "gyr" in sensor_stats:
        gyr_mean = sensor_stats["gyr"]["mean"]
        gyr_std = sensor_stats["gyr"]["std"] * 1.2

        base_gyr = np.random.normal(gyr_mean, gyr_std, (seq_length, len(gyr_mean)))

        # 回転パターン
        rotation_speed = 0.05 * (1 + gesture % 8)
        for j in range(min(3, base_gyr.shape[1])):
            base_gyr[:, j] += rotation_speed * np.cos(np.pi * time_stamps * (1 + j * 0.5))

        for j, col in enumerate(["gyr_x", "gyr_y", "gyr_z"]):
            sequence_data[col] = (
                base_gyr[:, j] if j < base_gyr.shape[1] else np.zeros(seq_length)
            )

    # クォータニオン（単位クォータニオン）
    angles = np.random.randn(seq_length, 3) * 0.1
    angles = np.cumsum(angles, axis=0)

    quat = np.zeros((seq_length, 4))
    for i in range(seq_length):
        roll, pitch, yaw = angles[i]
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        quat[i, 0] = cr * cp * cy + sr * sp * sy  # w
        quat[i, 1] = sr * cp * cy - cr * sp * sy  # x
        quat[i, 2] = cr * sp * cy + sr * cp * sy  # y
        quat[i, 3] = cr * cp * sy - sr * sp * cy  # z

    sequence_data["quat_w"] = quat[:, 0]
    sequence_data["quat_x"] = quat[:, 1]
    sequence_data["quat_y"] = quat[:, 2]
    sequence_data["quat_z"] = quat[:, 3]

    # ToFセンサー（距離データ）
    for tof_id in range(8):
        base_distance = np.random.uniform(50, 2500)

        if gesture < 6:  # 近距離ジェスチャー
            distance_pattern = base_distance - 200 * np.sin(2 * np.pi * 0.5 * time_stamps)
        elif gesture < 12:  # 中距離ジェスチャー
            distance_pattern = base_distance + 100 * np.cos(np.pi * time_stamps)
        else:  # 遠距離ジェスチャー
            distance_pattern = base_distance + np.random.normal(0, 50, seq_length)

        sequence_data[f"tof_{tof_id}"] = np.maximum(0, distance_pattern)

    return pd.DataFrame(sequence_data)


print("Synthetic sequence generator ready")


def generate_synthetic_dataset(n_sequences=1000):
    """合成データセットを生成"""

    synthetic_sequences = []
    synthetic_metadata = []

    # テストデータのジェスチャー分布を予想（より均等に）
    gestures = list(range(18))

    # 実データ分布30% + 均等分布70%
    real_probs = gesture_dist.reindex(gestures, fill_value=0.01).values
    uniform_probs = np.ones(18) / 18
    test_probs = 0.3 * real_probs + 0.7 * uniform_probs
    test_probs /= test_probs.sum()

    print(f"Generating {n_sequences} synthetic sequences...")

    for i in range(n_sequences):
        # ジェスチャー選択
        gesture = np.random.choice(gestures, p=test_probs)

        # シーケンス長（テストでは少し長め）
        seq_length = int(np.random.normal(130, 25))
        seq_length = max(60, min(250, seq_length))

        # シーケンス生成
        seq_df = generate_synthetic_sequence(gesture, seq_length, sensor_stats)

        synthetic_sequences.append(seq_df)
        synthetic_metadata.append(
            {
                "sequence_id": f"synthetic_{i:06d}",
                "subject": f"synth_sub_{i % 30}",  # 30人の合成被験者
                "gesture": id_to_gesture[gesture],  # 整数IDを文字列ラベルに変換
                "sequence_length": seq_length,
            }
        )

        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{n_sequences} sequences")

    synthetic_metadata_df = pd.DataFrame(synthetic_metadata)

    print(f"\nSynthetic data generated:")
    print(f"  Total sequences: {len(synthetic_sequences)}")
    print(f"  Avg sequence length: {synthetic_metadata_df['sequence_length'].mean():.1f}")
    print(
        f"  Gesture distribution:\n{synthetic_metadata_df['gesture'].value_counts().sort_index()}"
    )

    return synthetic_sequences, synthetic_metadata_df


# 合成データ生成（実データと同数）
n_synthetic = len(train_metadata)
synthetic_sequences, synthetic_metadata = generate_synthetic_dataset(n_synthetic)


def extract_features(df, seq_id, metadata_row):
    """高度な特徴量抽出"""
    features = {}

    # メタデータ特徴
    features["subject_id"] = hash(metadata_row["subject"]) % 1000
    features["seq_length"] = len(df)

    # 加速度計特徴
    for axis in ["x", "y", "z"]:
        col = f"acc_{axis}"
        if col in df.columns:
            values = df[col].values
            features[f"acc_{axis}_mean"] = np.mean(values)
            features[f"acc_{axis}_std"] = np.std(values)
            features[f"acc_{axis}_max"] = np.max(values)
            features[f"acc_{axis}_min"] = np.min(values)
            features[f"acc_{axis}_range"] = np.ptp(values)
            features[f"acc_{axis}_skew"] = pd.Series(values).skew()
            features[f"acc_{axis}_kurt"] = pd.Series(values).kurt()

            # 周波数領域特徴
            fft_vals = np.abs(np.fft.fft(values))[: len(values) // 2]
            features[f"acc_{axis}_fft_max"] = np.max(fft_vals) if len(fft_vals) > 0 else 0
            features[f"acc_{axis}_fft_mean"] = (
                np.mean(fft_vals) if len(fft_vals) > 0 else 0
            )

            # ジャーク
            if len(values) > 1:
                jerk = np.diff(values)
                features[f"acc_{axis}_jerk_mean"] = np.mean(np.abs(jerk))
                features[f"acc_{axis}_jerk_std"] = np.std(jerk)
            else:
                features[f"acc_{axis}_jerk_mean"] = 0
                features[f"acc_{axis}_jerk_std"] = 0

    # 加速度の大きさ
    if all(f"acc_{axis}" in df.columns for axis in ["x", "y", "z"]):
        acc_magnitude = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
        features["acc_magnitude_mean"] = np.mean(acc_magnitude)
        features["acc_magnitude_std"] = np.std(acc_magnitude)
        features["acc_magnitude_max"] = np.max(acc_magnitude)

    # ジャイロスコープ特徴
    for axis in ["x", "y", "z"]:
        col = f"gyr_{axis}"
        if col in df.columns:
            values = df[col].values
            features[f"gyr_{axis}_mean"] = np.mean(values)
            features[f"gyr_{axis}_std"] = np.std(values)
            features[f"gyr_{axis}_max"] = np.max(values)
            features[f"gyr_{axis}_energy"] = np.sum(values**2)

    # 角速度の大きさ
    if all(f"gyr_{axis}" in df.columns for axis in ["x", "y", "z"]):
        gyr_magnitude = np.sqrt(df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2)
        features["gyr_magnitude_mean"] = np.mean(gyr_magnitude)
        features["gyr_magnitude_std"] = np.std(gyr_magnitude)

    # クォータニオン特徴
    for comp in ["w", "x", "y", "z"]:
        col = f"quat_{comp}"
        if col in df.columns:
            values = df[col].values
            features[f"quat_{comp}_mean"] = np.mean(values)
            features[f"quat_{comp}_std"] = np.std(values)
            features[f"quat_{comp}_change"] = (
                values[-1] - values[0] if len(values) > 0 else 0
            )

    # ToFセンサー特徴（次元削減）
    tof_features = []
    for i in range(8):
        col = f"tof_{i}"
        if col in df.columns:
            values = df[col].values
            tof_features.extend([np.mean(values), np.std(values), np.median(values)])

    # ToF主成分（最初の5成分のみ）
    if tof_features:
        for i in range(min(5, len(tof_features))):
            features[f"tof_pc_{i}"] = tof_features[i]

    # ラベル（文字列の場合は整数IDに変換）
    gesture_value = metadata_row["gesture"]
    if isinstance(gesture_value, str):
        features["gesture"] = gesture_to_id.get(gesture_value, 0)
    else:
        features["gesture"] = gesture_value

    return features


print("Feature extractor ready")


# 実データから特徴量抽出（50%サンプリング）
real_sample = train_metadata.sample(n=len(train_metadata) // 2, random_state=42)
print(f"Processing {len(real_sample)} real sequences...")

real_features = []
failed_count = 0
for idx, row in real_sample.iterrows():
    try:
        # train.csvから該当シーケンスのデータを取得
        seq_data = train_data[train_data["sequence_id"] == row["sequence_id"]]
        if len(seq_data) > 0:
            features = extract_features(seq_data, row["sequence_id"], row)
            real_features.append(features)
    except Exception as e:
        failed_count += 1
        if failed_count <= 5:  # 最初の5個のエラーのみ表示
            print(f"  Failed to process {row['sequence_id']}: {e}")
        continue

print(f"Extracted features from {len(real_features)} real sequences")
if failed_count > 0:
    print(f"  Failed to process {failed_count} sequences")


# 合成データから特徴量抽出（50%）
synthetic_sample_idx = np.random.choice(
    len(synthetic_sequences), size=len(synthetic_sequences) // 2, replace=False
)
print(f"Processing {len(synthetic_sample_idx)} synthetic sequences...")

synthetic_features = []
for idx in synthetic_sample_idx:
    row = synthetic_metadata.iloc[idx]
    df = synthetic_sequences[idx]
    features = extract_features(df, row["sequence_id"], row)
    synthetic_features.append(features)

print(f"Extracted features from {len(synthetic_features)} synthetic sequences")


# 結合
all_features = real_features + synthetic_features
feature_df = pd.DataFrame(all_features)

# データ型の確認と修正
feature_df = feature_df.fillna(0)

print(f"\nCombined dataset:")
print(f"  Total samples: {len(feature_df)}")
print(
    f"  Real data: {len(real_features)} ({len(real_features) / len(feature_df) * 100:.1f}%)"
)
print(
    f"  Synthetic data: {len(synthetic_features)} ({len(synthetic_features) / len(feature_df) * 100:.1f}%)"
)
print(f"  Features: {feature_df.shape[1] - 1}")
print(f"\nGesture distribution:")
print(feature_df["gesture"].value_counts().sort_index())


# 特徴量とラベルの分離
X = feature_df.drop("gesture", axis=1)
y = feature_df["gesture"]

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Validation分割
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")


# LightGBMモデル
lgb_params = {
    "objective": "multiclass",
    "num_class": 18,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "seed": 42,
    "n_jobs": -1,
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

print("Training LightGBM...")
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=300,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)

# 検証
lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_pred_class = np.argmax(lgb_pred, axis=1)
lgb_acc = accuracy_score(y_val, lgb_pred_class)
print(f"LightGBM Validation Accuracy: {lgb_acc:.4f}")


# XGBoostモデル
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 18,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

print("Training XGBoost...")
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=300,
    evals=[(dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=50,
)

# 検証
xgb_pred = xgb_model.predict(dval)
xgb_pred_class = np.argmax(xgb_pred, axis=1)
xgb_acc = accuracy_score(y_val, xgb_pred_class)
print(f"XGBoost Validation Accuracy: {xgb_acc:.4f}")


# CatBoostモデル
cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    classes_count=18,
    random_seed=42,
    verbose=50,
    early_stopping_rounds=50,
)

print("Training CatBoost...")
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

# 検証
cat_pred_class = cat_model.predict(X_val)
cat_acc = accuracy_score(y_val, cat_pred_class)
print(f"CatBoost Validation Accuracy: {cat_acc:.4f}")


# アンサンブル予測（加重平均）
lgb_weight = 0.5
xgb_weight = 0.2
cat_weight = 0.3

# 各モデルの予測確率
lgb_proba = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
xgb_proba = xgb_model.predict(dval)
cat_proba = cat_model.predict_proba(X_val)

# 加重平均
ensemble_proba = lgb_weight * lgb_proba + xgb_weight * xgb_proba + cat_weight * cat_proba
ensemble_pred = np.argmax(ensemble_proba, axis=1)

# 最終精度
ensemble_acc = accuracy_score(y_val, ensemble_pred)
print(f"\n=== Final Results ===")
print(f"LightGBM: {lgb_acc:.4f}")
print(f"XGBoost: {xgb_acc:.4f}")
print(f"CatBoost: {cat_acc:.4f}")
print(f"Ensemble: {ensemble_acc:.4f}")

print(f"\n分類レポート:")
print(classification_report(y_val, ensemble_pred))


# モデルとスケーラーを保存
model_dir = Path("/kaggle/working/models")
model_dir.mkdir(exist_ok=True)

# LightGBM
lgb_model.save_model(str(model_dir / "lgb_model_v11.txt"))

# XGBoost
xgb_model.save_model(str(model_dir / "xgb_model_v11.json"))

# CatBoost
cat_model.save_model(str(model_dir / "cat_model_v11.cbm"))

# スケーラーと特徴量名
with open(model_dir / "scaler_v11.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(model_dir / "feature_names_v11.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# モデル情報
model_info = {
    "version": "v11_synthetic_fixed",
    "training_data": {
        "real_samples": len(real_features),
        "synthetic_samples": len(synthetic_features),
        "total_samples": len(feature_df),
    },
    "validation_accuracy": {
        "lgb": float(lgb_acc),
        "xgb": float(xgb_acc),
        "catboost": float(cat_acc),
        "ensemble": float(ensemble_acc),
    },
    "ensemble_weights": {"lgb": lgb_weight, "xgb": xgb_weight, "catboost": cat_weight},
    "feature_count": X.shape[1],
}

with open(model_dir / "model_info_v11.json", "w") as f:
    json.dump(model_info, f, indent=2)

print(f"\nModels saved to {model_dir}")
print(f"Files created:")
for file in model_dir.glob("*v11*"):
    print(f"  - {file.name}")


print("=" * 50)
print("V11 合成データモデル トレーニング完了")
print("=" * 50)
print(f"\n実データ: {len(real_features)} samples")
print(f"合成データ: {len(synthetic_features)} samples")
print(f"\n検証精度:")
print(f"  - アンサンブル: {ensemble_acc:.2%}")
print(f"  - LightGBM: {lgb_acc:.2%}")
print(f"  - XGBoost: {xgb_acc:.2%}")
print(f"  - CatBoost: {cat_acc:.2%}")
print(f"\n次のステップ:")
print(f"1. このノートブックを実行してモデルを生成")
print(f"2. /kaggle/working/models のモデルをKaggle Datasetとしてアップロード")
print(f"3. 推論ノートブックからDatasetを参照して予測")
print(f"4. スコアを確認して合成データの有効性を評価")

