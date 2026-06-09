# ============================================================
# SECTION 1 â€” SETUP & IMPORTS (KAGGLE VERSION)
# Clean unified pipeline for the MABe Challenge
# ============================================================

import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

# ML Models (âœ”ï¸� NOW INCLUDING LightGBM + XGBoost)
import lightgbm as lgb
import xgboost as xgb

import matplotlib.pyplot as plt

# Clean output
import warnings
warnings.filterwarnings("ignore")

# Global reproducibility
SEED = 42
np.random.seed(SEED)

print("âœ”ï¸� Kaggle imports loaded successfully.")



# ============================================================
# SECTION 2 â€” CONFIGURATION (KAGGLE VERSION)
# Dataset root and model settings
# ============================================================

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# ------------------------------------------------
# âœ” Correct Kaggle dataset root
# ------------------------------------------------
DATA_ROOT = Path("/kaggle/input/mabe-mouse-behavior-detection")

TRAIN_ANN_DIR = DATA_ROOT / "train_annotation"
TRAIN_TRACK_DIR = DATA_ROOT / "train_tracking"

# ------------------------------------------------
# IMPORTANT:
# Build a mapping: video_id (int) â†’ folder_name (e.g. "AdaptableSnail")
# ------------------------------------------------
VIDEO_TO_FOLDER = {}

for folder in sorted(os.listdir(TRAIN_ANN_DIR)):
    folder_path = TRAIN_ANN_DIR / folder
    for fname in os.listdir(folder_path):
        vid = int(Path(fname).stem)
        VIDEO_TO_FOLDER[vid] = folder

# Ensure mapping is not empty
assert len(VIDEO_TO_FOLDER) > 0, "â�Œ Could not build VIDEO_TO_FOLDER mapping!"
print(f"âœ”ï¸� VIDEO_TO_FOLDER mapping built: {len(VIDEO_TO_FOLDER)} videos found.")

# ------------------------------------------------
# Feature extraction settings
# ------------------------------------------------
WINDOW_SIZE = 15        # annotation window
SLIDE_WINDOW = 15       # test sliding window
SLIDE_STEP = 5

# ------------------------------------------------
# Output Directory
# ------------------------------------------------
OUTPUT_DIR = Path("./outputs_clean_pipeline")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------
USE_RANDOM_FOREST = True
USE_EXTRATREES = True
USE_LIGHTGBM = True
USE_XGBOOST = True

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "n_jobs": -1,
    "random_state": 42,
}

EXTRA_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "bootstrap": False,
    "n_jobs": -1,
    "random_state": 42,
}

# ------------------------------------------------
# LightGBM Parameters
# ------------------------------------------------
LGBM_PARAMS = {
    "objective": "multiclass",   # will fill num_class later
    "learning_rate": 0.05,
    "n_estimators": 300,
    "num_leaves": 64,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}

# ------------------------------------------------
# XGBoost Parameters
# ------------------------------------------------
XGB_PARAMS = {
    "objective": "multi:softprob",
    "learning_rate": 0.05,
    "max_depth": 8,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "n_jobs": -1,
}

print("âœ”ï¸� Kaggle configuration loaded.")



# ============================================================
# SECTION 3 â€” DATA LOADING (KAGGLE, MEMORY-SAFE & CORRECT)
# ============================================================

def load_annotations_train(data_root):
    """
    Load ALL training annotation parquet files.
    Each annotation file corresponds to one video.
    Folder structure:
        train_annotation/<folder>/<video_id>.parquet
    """
    ann_root = Path(data_root) / "train_annotation"
    dfs = []

    print("ğŸ”� Scanning annotation folders...")

    for folder in sorted(ann_root.iterdir()):
        if not folder.is_dir():
            continue

        for f in folder.glob("*.parquet"):
            df = pd.read_parquet(f)

            # Numeric video ID from filename
            video_id = int(f.stem)
            df["video_id"] = video_id

            dfs.append(df)

    if not dfs:
        raise ValueError("â�Œ No annotation parquet files found!")

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df



def load_tracking_file(video_id, split="train", data_root=DATA_ROOT):
    """
    Load a tracking parquet file and pivot keypoints
    so each frame has ONE row per mouse:
      mouse_id, video_frame, x_bodypart, y_bodypart, ...
    """
    track_root = Path(data_root) / f"{split}_tracking"
    matches = list(track_root.rglob(f"{video_id}.parquet"))

    if not matches:
        raise FileNotFoundError(f"â�Œ Tracking parquet for video_id={video_id} not found.")

    df = pd.read_parquet(matches[0])

    # -------------------------------------
    # Normalize column names
    # -------------------------------------
    if "frame_idx" in df.columns:
        df.rename(columns={"frame_idx": "video_frame"}, inplace=True)

    if "frame" in df.columns and "video_frame" not in df.columns:
        df.rename(columns={"frame": "video_frame"}, inplace=True)

    if "mouse" in df.columns:
        df.rename(columns={"mouse": "mouse_id"}, inplace=True)

    df["video_frame"] = df["video_frame"].astype(int)

    # -------------------------------------
    # ğŸ”¥ Pivot: bodypart rows â†’ wide keypoints per frame
    # -------------------------------------
    df_pivot = df.pivot_table(
        index=["mouse_id", "video_frame"],
        columns="bodypart",
        values=["x", "y"],
        aggfunc="first",
    )

    # Flatten x/y column names â†’ x_head, y_head, x_nose, y_nose, etc.
    df_pivot.columns = [f"{coord}_{bp}" for coord, bp in df_pivot.columns]

    # Reset index â†’ normal dataframe
    df_pivot = df_pivot.reset_index()

    # Add video ID
    df_pivot["video_id"] = video_id

    # Fill missing keypoints
    df_pivot = df_pivot.fillna(0)

    return df_pivot



def list_train_video_ids(data_root=DATA_ROOT):
    """
    Return all train video IDs by searching nested train_tracking folders.
    """
    track_dir = Path(data_root) / "train_tracking"
    files = track_dir.rglob("*.parquet")
    video_ids = sorted({int(f.stem) for f in files})

    if len(video_ids) == 0:
        print("âš ï¸� No train video IDs found! Check path.")

    return video_ids



def list_test_video_ids(data_root=DATA_ROOT):
    """
    Return all test video IDs by searching nested test_tracking folders.
    """
    test_dir = Path(data_root) / "test_tracking"
    files = test_dir.rglob("*.parquet")
    video_ids = sorted({int(f.stem) for f in files})

    if len(video_ids) == 0:
        print("âš ï¸� No test video IDs found! Check path.")

    return video_ids


print("âœ”ï¸� Data loader (Kaggle + memory-safe + folder-aware) ready.")



# ============================================================
# SECTION 4A â€” FEATURE UTILITIES (FIXED FOR MABe)
# ============================================================

def detect_xy_columns(df):
    """
    Detect columns that represent x/y coordinates in MABe tracking.

    Expected valid patterns:
        x, y
        x_head, y_head
        x_tail, y_tail
        nose_x, nose_y

    Falls back to the first matching x_*/y_* pattern.
    If nothing is found, tries plain 'x' and 'y'.
    """
    
    # Explicit known patterns
    xy_cols = [
        (c.lower(), c) for c in df.columns
        if c.lower() in [
            "x", "y",
            "x_head", "y_head",
            "x_tail", "y_tail",
            "nose_x", "nose_y"
        ]
    ]

    if xy_cols:
        # return first match in original column name format
        return xy_cols[0][1], xy_cols[1][1] if len(xy_cols) > 1 else xy_cols[0][1]

    # Generic fallback: first x_*, y_*
    generic_x = [c for c in df.columns if c.lower().startswith("x_")]
    generic_y = [c for c in df.columns if c.lower().startswith("y_")]

    if generic_x and generic_y:
        return generic_x[0], generic_y[0]

    # Basic fallback: raw x and y
    if "x" in df.columns and "y" in df.columns:
        return "x", "y"

    raise ValueError(
        f"â�Œ Could not detect XY columns in tracking df.\n"
        f"Available columns: {list(df.columns)}"
    )



def compute_stats(arr):
    """
    Compute basic statistical summary features from a 1D numeric array.
    
    Returns a 7-element vector:
        [mean, std, min, max, p25, p50, p75]

    Handles empty inputs gracefully by returning zeros.
    """
    if arr is None or len(arr) == 0:
        return [0, 0, 0, 0, 0, 0, 0]

    arr = np.array(arr, dtype=np.float32)

    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
    ]



# ============================================================
# SECTION 4B â€” TRAINING FEATURES (video-by-video loading)
# FINAL FIXED VERSION â€” SAFE FOR SPARSE / KEYPOINT TRACKING
# ============================================================

def generate_training_features(annotations_df, window_size=WINDOW_SIZE, data_root=DATA_ROOT):
    """
    Safe feature generation for sparse / keypoint tracking.
    
    Works even if:
      - annotations reference frames not present in tracking
      - tracking has missing or irregular frame numbering
      - tracking is keypoint-based (multiple bodyparts per frame)
    """

    print("ğŸ”§ Generating TRAINING features (video-by-video)...")

    ann_by_video = annotations_df.groupby("video_id")
    X_rows = []
    y_rows = []

    for video_id, ann_group in tqdm(ann_by_video, desc="Videos"):

        # ----------------------------------------------------
        # 1. Load tracking (pivoted â†’ wide format)
        # ----------------------------------------------------
        try:
            track_df = load_tracking_file(video_id, split="train", data_root=data_root)
        except Exception as e:
            print(f"âš ï¸� Cannot load tracking for video {video_id}: {e}")
            continue

        # Detect XY columns
        try:
            x_col, y_col = detect_xy_columns(track_df)
        except Exception:
            print(f"âš ï¸� Cannot detect XY columns for video {video_id}")
            continue

        # Sort and index by (mouse_id, frame)
        track_df = track_df.sort_values(["mouse_id", "video_frame"])
        idx = track_df.set_index(["mouse_id", "video_frame"])

        # ----------------------------------------------------
        # 2. Process annotation events for this video
        # ----------------------------------------------------
        for _, row in ann_group.iterrows():

            agent = row["agent_id"]
            target = row["target_id"]

            start = int(row["start_frame"])
            stop  = int(row["stop_frame"])

            # Expanded window around event
            s = max(0, start - window_size)
            e = stop + window_size

            # ----------------------------
            # Agent frames (safe extraction)
            # ----------------------------
            try:
                agent_sub = idx.loc[agent]
                agent_frames = agent_sub.index.values
            except KeyError:
                continue

            valid_agent_frames = agent_frames[(agent_frames >= s) & (agent_frames <= e)]
            if len(valid_agent_frames) == 0:
                continue

            valid_agent_frames = np.intersect1d(valid_agent_frames, agent_frames)
            if len(valid_agent_frames) == 0:
                continue

            at = agent_sub.loc[valid_agent_frames]

            # ----------------------------
            # Target frames (safe extraction)
            # ----------------------------
            try:
                target_sub = idx.loc[target]
                target_frames = target_sub.index.values
            except KeyError:
                continue

            valid_target_frames = target_frames[(target_frames >= s) & (target_frames <= e)]
            if len(valid_target_frames) == 0:
                continue

            valid_target_frames = np.intersect1d(valid_target_frames, target_frames)
            if len(valid_target_frames) == 0:
                continue

            tt = target_sub.loc[valid_target_frames]

            # ----------------------------
            # Align by overlapping frames
            # ----------------------------
            merged = (
                at.reset_index()
                  .merge(tt.reset_index(), on="video_frame", suffixes=("_a", "_t"))
                  .fillna(0)
            )

            if merged.empty:
                continue

            # ----------------------------------------------------
            # Compute geometric & motion features
            # ----------------------------------------------------
            ax = merged[f"{x_col}_a"].astype(np.float32)
            ay = merged[f"{y_col}_a"].astype(np.float32)
            tx = merged[f"{x_col}_t"].astype(np.float32)
            ty = merged[f"{y_col}_t"].astype(np.float32)

            dx = ax - tx
            dy = ay - ty
            dist = np.sqrt(dx * dx + dy * dy)

            vx = np.diff(ax, prepend=ax.iloc[0])
            vy = np.diff(ay, prepend=ay.iloc[0])
            speed = np.sqrt(vx * vx + vy * vy)

            # Final feature vector (28 dims)
            feats = (
                compute_stats(dx) +
                compute_stats(dy) +
                compute_stats(dist) +
                compute_stats(speed)
            )

            X_rows.append(feats)
            y_rows.append(row["action"])

        # Memory cleanup per video
        del track_df
        gc.collect()

    # Convert to DataFrame / Series
    X = pd.DataFrame(X_rows)
    y = pd.Series(y_rows, name="action")

    print(f"âœ”ï¸� Training features generated: X={X.shape}, y={len(y)}")
    return X, y



# ============================================================
# SECTION 5 â€” MODEL DEFINITION (RF + ExtraTrees + LightGBM + XGBoost)
# With correct LabelEncoder handling
# ============================================================

from sklearn.preprocessing import LabelEncoder

# -------------------------------------------
# Global LabelEncoder (shared by all models)
# -------------------------------------------
label_encoder = LabelEncoder()


# ============================================================
# Base Wrapper (common utilities)
# ============================================================
class BaseModel:
    def predict(self, X):
        """Return encoded predictions."""
        return self.model.predict(X)

    def predict_labels(self, X):
        """Return string (original) action labels."""
        preds = self.model.predict(X)
        return label_encoder.inverse_transform(preds)

    def predict_proba(self, X):
        """Every model must expose probability predictions."""
        return self.model.predict_proba(X)


# ============================================================
# RandomForest Wrapper
# ============================================================
class RFModel(BaseModel):
    def __init__(self, params):
        params = params.copy()
        params.setdefault("class_weight", "balanced")
        self.model = RandomForestClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)


# ============================================================
# ExtraTrees Wrapper
# ============================================================
class ExtraTreesModel(BaseModel):
    def __init__(self, params):
        params = params.copy()
        params.setdefault("class_weight", "balanced")
        self.model = ExtraTreesClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)


# ============================================================
# LightGBM Wrapper
# ============================================================
class LGBMModel(BaseModel):
    def __init__(self, params):
        params = params.copy()
        params.setdefault("objective", "multiclass")
        params.setdefault("num_class", NUM_CLASSES)
        params.setdefault("verbosity", -1)
        params.setdefault("random_state", 42)
        self.model = lgb.LGBMClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)


# ============================================================
# XGBoost Wrapper
# ============================================================
class XGBModel(BaseModel):
    def __init__(self, params):
        params = params.copy()
        params.setdefault("objective", "multi:softprob")
        params.setdefault("num_class", NUM_CLASSES)
        params.setdefault("eval_metric", "mlogloss")
        params.setdefault("verbosity", 0)
        params.setdefault("n_jobs", -1)
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)


# ============================================================
# Factory: initialize selected models (from config)
# ============================================================
def get_models():
    models = []

    if USE_RANDOM_FOREST:
        print("ğŸŒ² Initializing RandomForest...")
        models.append(("rf", RFModel(RF_PARAMS)))

    if USE_EXTRATREES:
        print("ğŸŒ³ Initializing ExtraTrees...")
        models.append(("extra", ExtraTreesModel(EXTRA_PARAMS)))

    if USE_LIGHTGBM:
        print("ğŸ’¡ Initializing LightGBM...")
        models.append(("lgbm", LGBMModel(LGBM_PARAMS)))

    if USE_XGBOOST:
        print("ğŸ”¥ Initializing XGBoost...")
        models.append(("xgb", XGBModel(XGB_PARAMS)))

    if not models:
        raise ValueError("â�Œ No models selected in Section 2 config.")

    return models


# ============================================================
# Ensemble utility (probability averaging)
# ============================================================
def ensemble_probabilities(list_of_probas):
    """
    Average probability matrices from multiple models.
    Ensures consistent class order (same NUM_CLASSES) across models.
    """
    arr = np.stack(list_of_probas, axis=0)  # shape: (num_models, N, num_classes)
    return np.mean(arr, axis=0)


print("âœ”ï¸� Models ready (RF + ExtraTrees + LightGBM + XGBoost).")



# ============================================================
# SECTION 6 â€” TRAINING PIPELINE (Stable Kaggle Version)
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------------
# 1. Load training annotations
# ---------------------------
print("ğŸ“¥ Loading training annotations...")
annotations_train = load_annotations_train(DATA_ROOT)
print("âœ”ï¸� Annotations:", annotations_train.shape)


# ---------------------------
# 2. Generate training features
# ---------------------------
print("\nğŸ”§ Generating TRAINING features...")
X_train_raw, y_train_raw = generate_training_features(
    annotations_train,
    window_size=WINDOW_SIZE,
    data_root=DATA_ROOT
)

# fix NaNs
X_train_raw = X_train_raw.fillna(0)

if X_train_raw.empty:
    raise ValueError("â�Œ No training features produced. Likely tracking lookup mismatch.")

print("âœ”ï¸� X shape:", X_train_raw.shape)
print("âœ”ï¸� y count:", len(y_train_raw))


# ---------------------------
# 3. Encode labels
# ---------------------------
print("\nğŸ”§ Encoding labels...")
y_encoded = label_encoder.fit_transform(y_train_raw)

classes = list(label_encoder.classes_)
NUM_CLASSES = len(classes)

print("âœ”ï¸� Classes:", classes)
print("âœ”ï¸� NUM_CLASSES =", NUM_CLASSES)

# Inject NUM_CLASSES into LGBM & XGB params (required for multiclass)
LGBM_PARAMS["num_class"] = NUM_CLASSES
XGB_PARAMS["num_class"] = NUM_CLASSES


# ---------------------------
# 4. Train/Validation Split
# ---------------------------
print("\nğŸ”§ Splitting train/val...")

try:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )
except Exception as e:
    print("âš ï¸� Stratified split failed:", e)
    print("â�¡ï¸� Using NON-STRATIFIED split instead.")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw,
        y_encoded,
        test_size=0.25,
        random_state=42
    )

print("âœ”ï¸� Train:", X_train.shape)
print("âœ”ï¸� Val:", X_val.shape)


# ---------------------------
# 5. Initialize selected models
# ---------------------------
print("\nâš™ï¸� Initializing models...")
models = get_models()
trained_models = {}


# ---------------------------
# 6. Train + Validate each model
# ---------------------------
for name, model_obj in models:
    print(f"\nğŸš€ Training: {name.upper()}")

    # Train
    model_obj.fit(X_train, y_train)
    trained_models[name] = model_obj

    # Validate
    val_proba = model_obj.predict_proba(X_val)
    val_pred = np.argmax(val_proba, axis=1)

    # Metrics
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)

    print(f"âœ”ï¸� {name} Accuracy: {acc:.4f}")
    print(f"âœ”ï¸� {name} Macro F1 : {f1:.4f}")

    # Classification report (safe for missing classes)
    print("\nğŸ”� Classification Report:")
    print(classification_report(
        y_val,
        val_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=classes,
        zero_division=0
    ))

    gc.collect()


# ---------------------------
# 7. Save all trained models + encoder
# ---------------------------
for name, model_obj in trained_models.items():
    path = OUTPUT_DIR / f"{name}_model.pkl"
    joblib.dump(model_obj, path)
    print(f"ğŸ’¾ Saved model â†’ {path}")

# Save LabelEncoder
joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.pkl")
print("ğŸ’¾ Saved LabelEncoder â†’ label_encoder.pkl")

print("\nğŸ�¯ Training pipeline completed successfully!")



def load_test_tracking_recursive(video_id, data_root=DATA_ROOT):
    """
    Custom loader ONLY for test inference.
    Recursively searches for <video_id>.parquet inside:
        test_tracking/**/<video_id>.parquet
    
    Returns a pandas DataFrame or None if missing.
    """
    root = Path(data_root) / "test_tracking"

    # recursive search
    files = list(root.glob(f"**/{video_id}.parquet"))

    if not files:
        print(f"âš ï¸� No test tracking file found for video_id={video_id}")
        return None

    try:
        df = pd.read_parquet(files[0])
    except Exception as e:
        print(f"â�Œ Failed to read test tracking for {video_id}: {e}")
        return None

    # ------------------------------------------------------
    # Normalize column names (same as training loader)
    # ------------------------------------------------------
    if "frame_idx" in df.columns:
        df.rename(columns={"frame_idx": "video_frame"}, inplace=True)
    if "frame" in df.columns and "video_frame" not in df.columns:
        df.rename(columns={"frame": "video_frame"}, inplace=True)
    if "mouse" in df.columns:
        df.rename(columns={"mouse": "mouse_id"}, inplace=True)

    df["video_frame"] = df["video_frame"].astype(int)
    df = df.fillna(0)

    return df



# ============================================================
# SECTION 7 â€” TEST INFERENCE (Sliding-Window)
# ============================================================

def generate_test_features_for_video(video_id, slide_window=SLIDE_WINDOW, slide_step=SLIDE_STEP):
    """
    Safe sliding-window feature extraction for TEST videos.
    Handles sparse frames, missing keypoints, and irregular tracking.
    """

    # Load test tracking
    try:
        track_df = load_test_tracking_recursive(video_id, data_root=DATA_ROOT)
        if track_df is None:
            raise FileNotFoundError("Returned None from loader.")
    except Exception as e:
        print(f"âš ï¸� Cannot load test tracking for video {video_id}: {e}")
        return None

    # Detect XY columns
    try:
        x_col, y_col = detect_xy_columns(track_df)
    except:
        print(f"âš ï¸� Cannot detect xy for test video {video_id}")
        return None

    # Sort & index
    track_df = track_df.sort_values(["mouse_id", "video_frame"])
    idx = track_df.set_index(["mouse_id", "video_frame"])

    all_frames = track_df["video_frame"].unique()
    if len(all_frames) == 0:
        return None

    max_frame = all_frames.max()
    rows = []

    # Test: typically mouse_id = [1,2]
    mice = sorted(track_df["mouse_id"].unique())
    if len(mice) < 2:
        return None

    agent, target = mice[0], mice[1]

    # Sliding window
    for start in range(0, max_frame - slide_window, slide_step):

        end = start + slide_window

        # -------- AGENT --------
        try:
            agent_sub = idx.loc[agent]
        except KeyError:
            continue

        agent_frames = agent_sub.index.values
        valid_af = agent_frames[(agent_frames >= start) & (agent_frames <= end)]
        if len(valid_af) == 0:
            continue

        valid_agent_df = agent_sub.loc[valid_af]

        # -------- TARGET --------
        try:
            target_sub = idx.loc[target]
        except KeyError:
            continue

        target_frames = target_sub.index.values
        valid_tf = target_frames[(target_frames >= start) & (target_frames <= end)]
        if len(valid_tf) == 0:
            continue

        valid_target_df = target_sub.loc[valid_tf]

        # -------- MERGE --------
        merged = (
            valid_agent_df.reset_index()
            .merge(valid_target_df.reset_index(), on="video_frame",
                   suffixes=("_a", "_t"))
            .fillna(0)
        )

        if merged.empty:
            continue

        ax = merged[f"{x_col}_a"].astype(np.float32)
        ay = merged[f"{y_col}_a"].astype(np.float32)
        tx = merged[f"{x_col}_t"].astype(np.float32)
        ty = merged[f"{y_col}_t"].astype(np.float32)

        dx = ax - tx
        dy = ay - ty
        dist = np.sqrt(dx * dx + dy * dy)

        vx = np.diff(ax, prepend=ax.iloc[0])
        vy = np.diff(ay, prepend=ay.iloc[0])
        speed = np.sqrt(vx * vx + vy * vy)

        feats = (
            compute_stats(dx)
            + compute_stats(dy)
            + compute_stats(dist)
            + compute_stats(speed)
        )

        rows.append((start, feats))

    if len(rows) == 0:
        return None

    frame_starts, feats = zip(*rows)
    return agent, target, frame_starts, pd.DataFrame(list(feats))



# ============================================================
# SEGMENTATION LOGIC (Convert frame predictions â†’ segments)
# ============================================================

def convert_to_segments(df):
    """
    Convert frame-level predictions into continuous action segments.
    Kaggle format required:
    row_id, video_id, agent_id, target_id, action, start_frame, stop_frame
    """

    segments = []
    row_id = 0

    for vid, group in df.groupby("video_id"):

        group = group.sort_values("start_frame")

        prev_action = None
        seg_start = None
        prev_frame = None
        agent_id = None
        target_id = None

        for _, row in group.iterrows():
            action = row["action"]
            frame = row["start_frame"]
            agent_id = row["agent_id"]
            target_id = row["target_id"]

            if action != prev_action:
                # close previous segment
                if prev_action is not None:
                    segments.append([
                        row_id, vid, agent_id, target_id,
                        prev_action, seg_start, prev_frame
                    ])
                    row_id += 1

                seg_start = frame
                prev_action = action

            prev_frame = frame

        # close last segment
        if prev_action is not None:
            segments.append([
                row_id, vid, agent_id, target_id,
                prev_action, seg_start, prev_frame
            ])
            row_id += 1

    return pd.DataFrame(
        segments,
        columns=[
            "row_id",
            "video_id",
            "agent_id",
            "target_id",
            "action",
            "start_frame",
            "stop_frame"
        ]
    )



# ============================================================
# PREDICT TEST SET & EXPORT FINAL KAGGLE SUBMISSION
# ============================================================

def run_test_inference():
    print("\nğŸš€ Running test inference...")

    test_video_files = list((DATA_ROOT / "test_tracking").glob("**/*.parquet"))
    print("ğŸ“‚ Found test test files:", len(test_video_files))

    test_video_ids = sorted({int(p.stem) for p in test_video_files})

    outputs = []

    for vid in tqdm(test_video_ids, desc="Test Videos"):
        result = generate_test_features_for_video(vid)

        if result is None:
            continue

        agent, target, frame_starts, X_test = result
        X_test = X_test.fillna(0)

        # Ensemble prediction
        model_preds = []
        for _, model_obj in trained_models.items():
            proba = model_obj.predict_proba(X_test)
            model_preds.append(proba)

        final_proba = ensemble_probabilities(model_preds)
        final_labels = np.argmax(final_proba, axis=1)
        final_actions = label_encoder.inverse_transform(final_labels)

        for frame, action in zip(frame_starts, final_actions):
            outputs.append([vid, agent, target, frame, action])

    df_frames = pd.DataFrame(
        outputs,
        columns=["video_id", "agent_id", "target_id", "start_frame", "action"]
    ).sort_values(["video_id", "start_frame"])

    submission = convert_to_segments(df_frames)

    # Save required Kaggle file
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("ğŸ“� Saved Kaggle submission.csv to /kaggle/working/submission.csv")

    return submission


print("âœ”ï¸� Test inference module ready.")



sub = run_test_inference()

sub.head()


