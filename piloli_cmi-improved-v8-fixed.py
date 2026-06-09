import numpy as np
import pandas as pd
import pickle
import gc
import warnings
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from scipy import signal, stats
from scipy.fft import fft, fftfreq

warnings.filterwarnings("ignore")

print("Loading data...")
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demo = pd.read_csv(
    "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
)

# Get labels
gesture_labels = sorted(train_df["gesture"].unique())
num_classes = len(gesture_labels)

print(f"Classes: {num_classes}")
print(f"Total samples: {len(train_df)}")
print(f"Unique sequences: {train_df['sequence_id'].nunique()}")


def remove_gravity_from_acc(df_seq, alpha=0.8):
    """é‡�åŠ›é™¤å�»ã�«ã‚ˆã‚‹ç·šå½¢åŠ é€Ÿåº¦ã�®è¨ˆç®—"""
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    if not all(col in df_seq.columns for col in acc_cols):
        return None

    acc_data = df_seq[acc_cols].values
    gravity = np.zeros_like(acc_data)
    gravity[0] = acc_data[0]

    for i in range(1, len(acc_data)):
        gravity[i] = alpha * gravity[i - 1] + (1 - alpha) * acc_data[i]

    linear_acc = acc_data - gravity
    return linear_acc


def quaternion_to_angular_velocity(df_seq):
    """ã‚¯ã‚©ãƒ¼ã‚¿ãƒ‹ã‚ªãƒ³ã�‹ã‚‰è§’é€Ÿåº¦ã‚’è¨ˆç®—"""
    quat_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
    if not all(col in df_seq.columns for col in quat_cols):
        return None

    q = df_seq[quat_cols].values

    # æ­£è¦�åŒ–
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / (q_norm + 1e-10)

    # æ™‚é–“å¾®åˆ†ã�§è§’é€Ÿåº¦ã‚’æ�¨å®š
    dq = np.gradient(q, axis=0)

    # è§’é€Ÿåº¦ãƒ™ã‚¯ãƒˆãƒ«è¨ˆç®—
    w = np.zeros((len(q), 3))
    for i in range(len(q)):
        w[i, 0] = 2 * (
            q[i, 3] * dq[i, 0]
            - q[i, 0] * dq[i, 3]
            - q[i, 1] * dq[i, 2]
            + q[i, 2] * dq[i, 1]
        )
        w[i, 1] = 2 * (
            q[i, 3] * dq[i, 1]
            - q[i, 1] * dq[i, 3]
            - q[i, 2] * dq[i, 0]
            + q[i, 0] * dq[i, 2]
        )
        w[i, 2] = 2 * (
            q[i, 3] * dq[i, 2]
            - q[i, 2] * dq[i, 3]
            - q[i, 0] * dq[i, 1]
            + q[i, 1] * dq[i, 0]
        )

    return w


def extract_frequency_features(signal_data, sampling_rate=50):
    """å‘¨æ³¢æ•°é ˜åŸŸç‰¹å¾´é‡�ã�®æŠ½å‡º"""
    if len(signal_data) < 10:
        return {}

    fft_values = np.abs(fft(signal_data))
    freqs = fftfreq(len(signal_data), 1 / sampling_rate)

    # æ­£ã�®å‘¨æ³¢æ•°ã�®ã�¿
    pos_idx = freqs > 0
    fft_values = fft_values[pos_idx]
    freqs = freqs[pos_idx]

    features = {}
    if len(fft_values) > 0:
        features["fft_max"] = np.max(fft_values)
        features["fft_mean"] = np.mean(fft_values)
        features["fft_std"] = np.std(fft_values)

        # ãƒ”ãƒ¼ã‚¯å‘¨æ³¢æ•°
        peak_idx = np.argmax(fft_values)
        features["peak_freq"] = freqs[peak_idx] if len(freqs) > 0 else 0

    return features


def extract_comprehensive_features(df_seq):
    """åŒ…æ‹¬çš„ã�ªç‰¹å¾´é‡�æŠ½å‡º"""
    features = {}

    # åŸºæœ¬çµ±è¨ˆé‡�
    for col in ["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "rot_w"]:
        if col in df_seq.columns:
            values = df_seq[col].values
            features[f"{col}_mean"] = np.mean(values)
            features[f"{col}_std"] = np.std(values)
            features[f"{col}_max"] = np.max(values)
            features[f"{col}_min"] = np.min(values)
            features[f"{col}_range"] = features[f"{col}_max"] - features[f"{col}_min"]
            features[f"{col}_skew"] = stats.skew(values)
            features[f"{col}_kurtosis"] = stats.kurtosis(values)

            # å¤‰åŒ–é‡�çµ±è¨ˆ
            diff = np.diff(values)
            if len(diff) > 0:
                features[f"{col}_diff_mean"] = np.mean(diff)
                features[f"{col}_diff_std"] = np.std(diff)

    # é‡�åŠ›é™¤å�»åŠ é€Ÿåº¦
    linear_acc = remove_gravity_from_acc(df_seq)
    if linear_acc is not None:
        for i, axis in enumerate(["x", "y", "z"]):
            features[f"linear_acc_{axis}_mean"] = np.mean(linear_acc[:, i])
            features[f"linear_acc_{axis}_std"] = np.std(linear_acc[:, i])
            features[f"linear_acc_{axis}_max"] = np.max(np.abs(linear_acc[:, i]))

    # è§’é€Ÿåº¦
    angular_vel = quaternion_to_angular_velocity(df_seq)
    if angular_vel is not None:
        for i, axis in enumerate(["x", "y", "z"]):
            features[f"angular_vel_{axis}_mean"] = np.mean(angular_vel[:, i])
            features[f"angular_vel_{axis}_std"] = np.std(angular_vel[:, i])
            features[f"angular_vel_{axis}_max"] = np.max(np.abs(angular_vel[:, i]))

    # åŠ é€Ÿåº¦ãƒ�ã‚°ãƒ‹ãƒ�ãƒ¥ãƒ¼ãƒ‰
    if all(col in df_seq.columns for col in ["acc_x", "acc_y", "acc_z"]):
        acc_mag = np.sqrt(
            df_seq["acc_x"] ** 2 + df_seq["acc_y"] ** 2 + df_seq["acc_z"] ** 2
        )
        features["acc_mag_mean"] = np.mean(acc_mag)
        features["acc_mag_std"] = np.std(acc_mag)
        features["acc_mag_max"] = np.max(acc_mag)

        # ã‚¸ãƒ£ãƒ¼ã‚¯ï¼ˆåŠ é€Ÿåº¦ã�®å¤‰åŒ–ç�‡ï¼‰
        jerk = np.diff(acc_mag)
        if len(jerk) > 0:
            features["jerk_mean"] = np.mean(np.abs(jerk))
            features["jerk_std"] = np.std(jerk)
            features["jerk_max"] = np.max(np.abs(jerk))

        # å‘¨æ³¢æ•°ç‰¹å¾´
        freq_features = extract_frequency_features(acc_mag.values)
        for key, value in freq_features.items():
            features[f"acc_mag_{key}"] = value

    # æ¸©åº¦ã‚»ãƒ³ã‚µãƒ¼
    temp_cols = [f"thm_{i}" for i in range(1, 6)]
    temp_cols = [col for col in temp_cols if col in df_seq.columns]
    if temp_cols:
        temp_data = df_seq[temp_cols].values
        features["temp_mean"] = np.mean(temp_data)
        features["temp_std"] = np.std(temp_data)
        features["temp_range"] = np.max(temp_data) - np.min(temp_data)

    # ToFã‚»ãƒ³ã‚µãƒ¼ï¼ˆæ¬¡å…ƒå‰Šæ¸›ï¼‰
    tof_cols = [col for col in df_seq.columns if col.startswith("tof_")]
    if tof_cols:
        tof_data = df_seq[tof_cols].values
        # çµ±è¨ˆé‡�ã�§è¦�ç´„
        features["tof_mean"] = np.mean(tof_data)
        features["tof_std"] = np.std(tof_data)
        features["tof_max"] = np.max(tof_data)
        features["tof_min"] = np.min(tof_data)

        # å�„ãƒ•ãƒ¬ãƒ¼ãƒ ã�®çµ±è¨ˆé‡�
        features["tof_frame_mean_std"] = np.std(np.mean(tof_data, axis=1))
        features["tof_frame_max_mean"] = np.mean(np.max(tof_data, axis=1))

    # ã‚·ãƒ¼ã‚±ãƒ³ã‚¹é•·
    features["sequence_length"] = len(df_seq)

    return features


# å…¨ãƒ‡ãƒ¼ã‚¿ã�§ç‰¹å¾´é‡�æŠ½å‡º
print("Extracting features from all sequences...")
X_features = []
y_labels = []
groups = []

# Label encoder
le = LabelEncoder()
le.fit(gesture_labels)

sequence_ids = train_df["sequence_id"].unique()
total_sequences = len(sequence_ids)

# å…¨ãƒ‡ãƒ¼ã‚¿ã‚’å‡¦ç�†ï¼ˆãƒ�ãƒƒãƒ�å‡¦ç�†ã�§åŠ¹ç�‡åŒ–ï¼‰
batch_size = 1000
for batch_start in range(0, total_sequences, batch_size):
    batch_end = min(batch_start + batch_size, total_sequences)
    batch_seq_ids = sequence_ids[batch_start:batch_end]

    if batch_start % 2000 == 0:
        print(f"Processing sequences {batch_start}/{total_sequences}...")

    for seq_id in batch_seq_ids:
        seq_data = train_df[train_df["sequence_id"] == seq_id]

        # ç‰¹å¾´é‡�æŠ½å‡º
        features = extract_comprehensive_features(seq_data)
        X_features.append(features)

        # ãƒ©ãƒ™ãƒ«
        gesture = seq_data["gesture"].iloc[0]
        label = le.transform([gesture])[0]
        y_labels.append(label)

        # Subject IDï¼ˆGroupKFoldç”¨ï¼‰
        subject = seq_data["subject"].iloc[0]
        groups.append(subject)

    # ãƒ¡ãƒ¢ãƒªè§£æ”¾
    if batch_start % 2000 == 0:
        gc.collect()

# DataFrameã�«å¤‰æ�›
X_df = pd.DataFrame(X_features)
y_array = np.array(y_labels)
groups_array = np.array(groups)

print(f"Feature shape: {X_df.shape}")
print(f"Number of features: {len(X_df.columns)}")

# æ¬ æ��å€¤ã‚’0ã�§åŸ‹ã‚�ã‚‹
X_df = X_df.fillna(0)


# GroupKFoldã�§äº¤å·®æ¤œè¨¼
print("Training models with GroupKFold...")
gkf = GroupKFold(n_splits=5)

# æ¤œè¨¼ã‚¹ã‚³ã‚¢ä¿�å­˜ç”¨
val_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_df, y_array, groups_array)):
    print(f"\nFold {fold + 1}/5")

    X_train = X_df.iloc[train_idx]
    X_val = X_df.iloc[val_idx]
    y_train = y_array[train_idx]
    y_val = y_array[val_idx]

    # LightGBM
    lgb_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42 + fold,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    model_lgb = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    # äºˆæ¸¬
    pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    pred_class = np.argmax(pred_lgb, axis=1)
    accuracy = np.mean(pred_class == y_val)

    print(f"LightGBM Validation Accuracy: {accuracy:.4f}")
    val_scores.append(accuracy)
    models.append(model_lgb)

# å¹³å�‡ç²¾åº¦
mean_accuracy = np.mean(val_scores)
std_accuracy = np.std(val_scores)
print(f"\n=== Cross-Validation Results ===")
print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
print(f"Target 87%: {'âœ… ACHIEVED' if mean_accuracy >= 0.87 else 'â�Œ NOT YET'}")


# XGBoostã‚‚è¿½åŠ ã�§ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«
print("\nTraining XGBoost for ensemble...")

# æœ€è‰¯ã�®foldã�®ãƒ‡ãƒ¼ã‚¿ã�§XGBoostã‚’ãƒˆãƒ¬ãƒ¼ãƒ‹ãƒ³ã‚°
best_fold = np.argmax(val_scores)
train_idx, val_idx = list(gkf.split(X_df, y_array, groups_array))[best_fold]

X_train = X_df.iloc[train_idx]
X_val = X_df.iloc[val_idx]
y_train = y_array[train_idx]
y_val = y_array[val_idx]

# XGBoost
xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "mlogloss",
    "seed": 42,
    "tree_method": "hist",
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

model_xgb = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, "val")],
    early_stopping_rounds=100,
    verbose_eval=False,
)

# XGBoostäºˆæ¸¬
pred_xgb = model_xgb.predict(dval)
pred_xgb_class = np.argmax(pred_xgb, axis=1)
xgb_accuracy = np.mean(pred_xgb_class == y_val)
print(f"XGBoost Validation Accuracy: {xgb_accuracy:.4f}")

# ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«
pred_lgb = models[best_fold].predict(
    X_val, num_iteration=models[best_fold].best_iteration
)
pred_ensemble = 0.6 * pred_lgb + 0.4 * pred_xgb
pred_ensemble_class = np.argmax(pred_ensemble, axis=1)
ensemble_accuracy = np.mean(pred_ensemble_class == y_val)

print(f"\nğŸ�¯ Ensemble Accuracy: {ensemble_accuracy:.4f}")
print(f"Target 87%: {'âœ… ACHIEVED' if ensemble_accuracy >= 0.87 else 'â�Œ NOT YET'}")


# CatBoostã‚‚è¿½åŠ 
print("\nTraining CatBoost for triple ensemble...")

cat_params = {
    "loss_function": "MultiClass",
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 100,
}

train_pool = cb.Pool(X_train, y_train)
val_pool = cb.Pool(X_val, y_val)

model_cat = cb.CatBoostClassifier(**cat_params)
model_cat.fit(train_pool, eval_set=val_pool, verbose=False)

# CatBoostäºˆæ¸¬
pred_cat = model_cat.predict_proba(X_val)
pred_cat_class = np.argmax(pred_cat, axis=1)
cat_accuracy = np.mean(pred_cat_class == y_val)
print(f"CatBoost Validation Accuracy: {cat_accuracy:.4f}")

# ãƒˆãƒªãƒ—ãƒ«ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«
pred_triple = 0.4 * pred_lgb + 0.3 * pred_xgb + 0.3 * pred_cat
pred_triple_class = np.argmax(pred_triple, axis=1)
triple_accuracy = np.mean(pred_triple_class == y_val)

print(f"\nğŸ�¯ Triple Ensemble Accuracy: {triple_accuracy:.4f}")
print(f"Target 87%: {'âœ… ACHIEVED!!!' if triple_accuracy >= 0.87 else 'â�Œ NOT YET'}")

# æœ€çµ‚ç²¾åº¦
final_accuracy = max(mean_accuracy, ensemble_accuracy, triple_accuracy)
print(f"\nğŸ�† Best Validation Accuracy: {final_accuracy:.4f}")


# ãƒ¢ãƒ‡ãƒ«ä¿�å­˜
print("\nSaving models...")

# æœ€è‰¯ã�®LightGBMãƒ¢ãƒ‡ãƒ«
with open("lgb_model_v8.pkl", "wb") as f:
    pickle.dump(models[best_fold], f)

# XGBoostãƒ¢ãƒ‡ãƒ«
with open("xgb_model_v8.pkl", "wb") as f:
    pickle.dump(model_xgb, f)

# CatBoostãƒ¢ãƒ‡ãƒ«
model_cat.save_model("cat_model_v8.cbm")

# Label encoder
with open("label_encoder_v8.pkl", "wb") as f:
    pickle.dump(le, f)

print("Models saved successfully")


# CMIæ�¨è«–ã‚µãƒ¼ãƒ�ãƒ¼
import sys

sys.path.append("/kaggle/input/cmi-detect-behavior-with-sensor-data")
from kaggle_evaluation.cmi_inference_server import CMIInferenceServer


def predict(sequence, demographics):
    """CMIæ�¨è«–ã‚µãƒ¼ãƒ�ãƒ¼ç”¨äºˆæ¸¬é–¢æ•°"""
    try:
        # DataFrameã�«å¤‰æ�›
        df_seq = pd.DataFrame(sequence)

        # ç‰¹å¾´é‡�æŠ½å‡º
        features = extract_comprehensive_features(df_seq)
        X_feat = pd.DataFrame([features])
        X_feat = X_feat.fillna(0)

        # å�„ãƒ¢ãƒ‡ãƒ«ã�§äºˆæ¸¬
        pred_lgb = models[best_fold].predict(
            X_feat, num_iteration=models[best_fold].best_iteration
        )

        dtest = xgb.DMatrix(X_feat)
        pred_xgb = model_xgb.predict(dtest)

        pred_cat = model_cat.predict_proba(X_feat)

        # ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«
        pred_ensemble = 0.4 * pred_lgb + 0.3 * pred_xgb + 0.3 * pred_cat

        # äºˆæ¸¬ã‚¯ãƒ©ã‚¹
        pred_class = np.argmax(pred_ensemble[0])
        pred_gesture = le.inverse_transform([pred_class])[0]

        return pred_gesture

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Text on phone"


print("Starting CMI Inference Server...")
server = CMIInferenceServer(predict)
server.serve()
print("Inference complete")

