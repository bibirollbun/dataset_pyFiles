# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

directory = '/kaggle/input/MABe-mouse-behavior-detection/'
pd.set_option('display.max_columns',100)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# ============================================================
# ğŸ“¦ Imports
# ============================================================
import pandas as pd
import numpy as np
import re, ast, os, gc, warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.seterr(invalid="ignore")

# ============================================================
# 1ï¸�âƒ£ Parse light/time info per mouse
# ============================================================
def parse_light_info(df, col, num):
    pattern = r"day\s*(\d+)\s*-\s*(\d{1,2}):(\d{2})\s*\(([^)]+)\)"
    base_col = col.removeprefix(f"mouse{num}_")
    time_col = f"mouse{num}_{base_col}_time_in_hours"
    light_col = f"mouse{num}_{base_col}_light_status"

    def extract(entry):
        if isinstance(entry, str):
            m = re.match(pattern, entry.lower().strip())
            if m:
                day, hour, minute, status = m.groups()
                return [int(day)*24 + int(hour) + int(minute)/60, status]
        return [-1, "unknown"]

    vals = df[col].map(extract)
    df[time_col] = [v[0] for v in vals]
    df[light_col] = [v[1] for v in vals]
    return df


# ============================================================
# 2ï¸�âƒ£ Behavior parsing
# ============================================================
def safe_parse_behavior_set(x):
    if pd.isna(x) or not isinstance(x, str) or not x.strip():
        return set()
    try:
        parsed = ast.literal_eval(x)
        if not isinstance(parsed, (list, set)):
            return set()
        result = set()
        for i in parsed:
            if isinstance(i, str):
                parts = i.split(',')
                if len(parts) == 3:
                    if parts[1].strip() == "self":
                        result.add(f"self, {parts[2].strip()}")
                    else:
                        result.add(parts[2].strip())
        return result
    except Exception:
        return set()


def prepare_behavior_sets(train):
    print("â†’ Parsing behavior sets...")
    tqdm.pandas()
    train["behavior_set"] = train["behaviors_labeled"].progress_apply(safe_parse_behavior_set)
    final_set = set().union(*train["behavior_set"])
    print(f"   âœ… Unique behaviors: {len(final_set)}")
    train["behavior_set"] = train["behavior_set"].apply(lambda s: s if s else final_set)
    return train, final_set


# ============================================================
# 3ï¸�âƒ£ Feature extraction (groupby + unstack = lean pivot)
# ============================================================
def extract_mouse_features(df, body_parts, video_width_pix=None, video_height_pix=None, arena_width_cm=None, arena_height_cm=None):
    # Group and unstack instead of pivot_table (much leaner)
    df = (
        df.groupby(["video_frame", "mouse_id", "bodypart"], observed=True)[["x", "y"]]
        .mean()
        .unstack("bodypart")
    )

    df.columns = [f"{c1}_{c2}" for c1, c2 in df.columns]
    df = df.astype("float32", copy=False).reset_index()

    present_parts = [bp for bp in body_parts if f"x_{bp}" in df.columns]
    if not present_parts:
        return pd.DataFrame()

    arr_x = df[[f"x_{bp}" for bp in present_parts]].to_numpy(dtype=np.float32, copy=False)
    arr_y = df[[f"y_{bp}" for bp in present_parts]].to_numpy(dtype=np.float32, copy=False)

    df["COM_x"] = np.nanmean(arr_x, axis=1)
    df["COM_y"] = np.nanmean(arr_y, axis=1)

    # ====================================================
    # ğŸ§® Convert COM coordinates from pixels â†’ centimeters
    # ====================================================
    if all(v is not None for v in [video_width_pix, arena_width_cm, video_height_pix, arena_height_cm]):
        scale_x = video_width_pix / arena_width_cm
        scale_y = video_height_pix / arena_height_cm
        df["COM_x"] = df["COM_x"] / scale_x
        df["COM_y"] = df["COM_y"] / scale_y
    else:
        print("âš ï¸� Missing scaling info; COM remains in pixels.")


    for bp in ["ear_left", "ear_right", "tail_base"]:
        if f"x_{bp}" not in df.columns:
            df[f"x_{bp}"] = np.nan
            df[f"y_{bp}"] = np.nan

    df["ear_mid_x"] = df[["x_ear_left", "x_ear_right"]].mean(axis=1)
    df["ear_mid_y"] = df[["y_ear_left", "y_ear_right"]].mean(axis=1)

    dx = df["x_tail_base"] - df["ear_mid_x"]
    dy = df["y_tail_base"] - df["ear_mid_y"]
    df["orientation_angle"] = np.arctan2(dy, dx).astype("float32")

    df.sort_values(["mouse_id", "video_frame"], inplace=True, ignore_index=True)

    df[["vel_x", "vel_y"]] = (
        df.groupby("mouse_id", sort=False)[["COM_x", "COM_y"]].diff().astype("float32")
    )
    df["velocity"] = np.hypot(df["vel_x"], df["vel_y"]).astype("float32")
    df[["accel_x", "accel_y"]] = (
        df.groupby("mouse_id", sort=False)[["vel_x", "vel_y"]].diff().astype("float32")
    )
    df["acceleration"] = np.hypot(df["accel_x"], df["accel_y"]).astype("float32")

    del arr_x, arr_y
    gc.collect()

    return df[
        [
            "video_frame",
            "mouse_id",
            "COM_x",
            "COM_y",
            "orientation_angle",
            "velocity",
            "acceleration",
        ]
    ].astype("float32", copy=False)
    

# ============================================================
# 4ï¸�âƒ£ Pairwise features
# ============================================================
def compute_pairwise_features(df):
    mice = df["mouse_id"].unique()
    out = []

    for a in mice:
        s = df[df.mouse_id == a].copy()
        for b in mice:
            if a == b:
                continue

            # Target mouse b
            t = df[df.mouse_id == b][
                ["video_frame", "COM_x", "COM_y", "orientation_angle", "velocity", "acceleration"]
            ].rename(columns=lambda c: f"{c}_tgt" if c != "video_frame" else c)

            merged = pd.merge(s, t, on="video_frame", how="inner", sort=False)

            # Vector from mouse a â†’ mouse b
            dx = merged["COM_x_tgt"] - merged["COM_x"]
            dy = merged["COM_y_tgt"] - merged["COM_y"]
            dist = np.hypot(dx, dy).astype("float32")
            merged["distance"] = dist

            # Unit vector toward target
            ux = dx / (dist + 1e-6)
            uy = dy / (dist + 1e-6)

            # Mouse a's velocity direction components
            vx = merged.groupby("mouse_id", sort=False)["COM_x"].diff().fillna(0).astype("float32")
            vy = merged.groupby("mouse_id", sort=False)["COM_y"].diff().fillna(0).astype("float32")

            # Project velocity along the line toward target (dot product)
            v_toward = vx * ux + vy * uy
            merged["relative_velocity"] = v_toward.astype("float32")

            # Same for acceleration
            ax = vx.diff().fillna(0)
            ay = vy.diff().fillna(0)
            a_toward = ax * ux + ay * uy
            merged["relative_acceleration"] = a_toward.astype("float32")

            # Relative orientation: angle between mouse's facing direction and line to target
            orientation_to_target = np.arctan2(dy, dx)
            delta_angle = merged["orientation_angle"] - orientation_to_target
            merged["relative_orientation"] = np.arctan2(np.sin(delta_angle), np.cos(delta_angle)).astype("float32")

            # Attach identifiers
            merged["mouse_id_target"] = b

            # Keep both coordinates + kinematics
            out.append(
                merged[
                    [
                        "video_frame",
                        "mouse_id",
                        "mouse_id_target",
                        "COM_x",
                        "COM_y",
                        "COM_x_tgt",
                        "COM_y_tgt",
                        "distance",
                        "relative_orientation",
                        "relative_velocity",
                        "relative_acceleration",
                    ]
                ]
            )

    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)



# ============================================================
# 5ï¸�âƒ£ Metadata enrichment
# ============================================================
def enrich_with_metadata(df_pairs, train_row, num_mice=4):
    if df_pairs.empty:
        return df_pairs

    meta_cols = [
        "lab_id",
        "video_id",
        "pix_per_cm_approx",
        "video_width_pix",
        "video_height_pix",
        "arena_width_cm",
        "arena_height_cm",
        "arena_shape",
        "arena_type",
        "body_parts_tracked",
        "behaviors_labeled",
        "tracking_method",
        "no_of_mice",
        "behavior_set",
    ]
    for c in meta_cols:
        val = train_row.get(c, np.nan)
        if isinstance(val, (list, set, tuple)):
            val = ",".join(map(str, val))
        df_pairs[c] = val

    mouse_fields = [
        "strain",
        "sex",
        "age",
        "color",
        "condition",
        "condition_time_in_hours",
        "condition_light_status",
    ]
    for mid in range(1, num_mice + 1):
        for f in mouse_fields:
            v = train_row.get(f"mouse{mid}_{f}", np.nan)
            mask_self = df_pairs["mouse_id"] == mid
            mask_tgt = df_pairs["mouse_id_target"] == mid
            if mask_self.any():
                df_pairs.loc[mask_self, f"mouse_{f}"] = v
            if mask_tgt.any():
                df_pairs.loc[mask_tgt, f"target_mouse_{f}"] = v

    return df_pairs
    

# ============================================================
# 7ï¸�âƒ£ Annotation merger
# ============================================================
def attach_annotations(df_pairs, lab_id, video_id, base_annot_path):
    """
    Attach behavior annotations for matching agent-target pairs.
    If annotation file not found, leaves 'Target' column blank.
    """
    annot_path = f"{base_annot_path}/{lab_id}/{video_id}.parquet"

    # If annotation missing, just create blank column
    if not os.path.exists(annot_path):
        tqdm.write(f"âš ï¸� No annotation file for {lab_id}/{video_id}")
        df_pairs["Target"] = np.nan
        return df_pairs
        
    try:
        ann = pd.read_parquet(annot_path)

        # Expected columns: ['agent_id','target_id','action','start_frame','stop_frame']
        if not set(["agent_id", "target_id", "action", "start_frame", "stop_frame"]).issubset(ann.columns):
            tqdm.write(f"âš ï¸� Annotation format invalid for {lab_id}/{video_id}")
            df_pairs["Target"] = np.nan
            return df_pairs

        # Prepare annotation mapping per (agent,target)
        df_pairs["Target"] = np.nan  # initialize column

        # Iterate over unique combinations for efficient assignment
        for _, row in ann.iterrows():
            agent = row["agent_id"]
            target = row["target_id"]
            start_f = int(row["start_frame"])
            stop_f = int(row["stop_frame"])
            act = str(row["action"]).strip().lower()

            # Mask: same agent, same target, frame within range
            mask = (
                (df_pairs["mouse_id"] == agent)
                & (df_pairs["mouse_id_target"] == target)
                & (df_pairs["video_frame"].between(start_f, stop_f))
            )

            df_pairs.loc[mask, "Target"] = act

        return df_pairs

    except Exception as e:
        tqdm.write(f"â�Œ Error reading annotation for {lab_id}/{video_id}: {e}")
        df_pairs["Target"] = np.nan
        return df_pairs


# ============================================================
# 6ï¸�âƒ£ Full pipeline
# ============================================================
def run_train_pipeline(train, body_parts, base_path, annot_base_path, chunk_save_interval=10):
    print("â†’ Preparing metadata...")
    train, final_set = prepare_behavior_sets(train)
    all_results = []
    chunk_idx = 0

    for _, row in tqdm(train.iterrows(), total=len(train), desc="Processing Labs/Videos"):
        df = df_features = df_pairs = None
        try:
            path = f"{base_path}/{row['lab_id']}/{row['video_id']}.parquet"
            if not os.path.exists(path):
                tqdm.write(f"âš ï¸� Missing file: {path}")
                continue

            # 1ï¸�âƒ£ Read tracking
            df = pd.read_parquet(path, columns=["video_frame", "mouse_id", "bodypart", "x", "y"])

            # print(df.shape)
            
            # 2ï¸�âƒ£ Extract scaled features
            df_features = extract_mouse_features(
                df,
                body_parts,
                video_width_pix=row["video_width_pix"],
                video_height_pix=row["video_height_pix"],
                arena_width_cm=row["arena_width_cm"],
                arena_height_cm=row["arena_height_cm"],
            )

            if df_features.empty:
                continue

            # 3ï¸�âƒ£ Pairwise interactions
            df_pairs = compute_pairwise_features(df_features)

            # 4ï¸�âƒ£ Add metadata
            df_pairs = enrich_with_metadata(df_pairs, row)


            # 5ï¸�âƒ£ Attach target annotations
            # df_pairs = attach_annotations(
            #     df_pairs,
            #     lab_id=row["lab_id"],
            #     video_id=row["video_id"],
            #     base_annot_path=annot_base_path,
            # )

            drop_cols = [
                "pix_per_cm_approx",
                "video_width_pix",
                "video_height_pix",
                "arena_width_cm",
                "arena_height_cm",
                "body_parts_tracked",
                "behaviors_labeled",
            ]
            df_pairs = df_pairs.drop(columns=[c for c in drop_cols if c in df_pairs.columns], errors="ignore")

            # Uncomment this to get the concatenated data set
            all_results.append(df_pairs)

            ## Uncomment this to store the data 
            # lab_dir = f"/kaggle/working/Dataset/{row['lab_id']}"
            # os.makedirs(lab_dir, exist_ok=True)
            
            # save_path = f"{lab_dir}/{row['video_id']}.parquet"
            # df_pairs.to_parquet(save_path, compression="snappy")
            
            del df, df_features, df_pairs
            gc.collect()

        except Exception as e:
            tqdm.write(f"â�Œ Error processing {row.get('lab_id')} / {row.get('video_id')}: {e}")
        # finally:
        #     del df, df_features, df_pairs
        #     gc.collect()

    # Final combine after loop
    final_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    print(f"âœ… Completed. Final shape: {final_df.shape}")
    return final_df, final_set


test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
test.drop(columns=['mouse1_id', 'mouse2_id', 'mouse3_id', 'mouse4_id'], inplace=True)

body_parts = ['body_center', 'ear_left', 'ear_right', 'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft',
              'headpiece_bottomfrontright', 'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright',
              'lateral_left', 'lateral_right', 'neck', 'nose', 'tail_base', 'tail_midpoint', 'tail_tip', 'hip_left', 'hip_right', 'head', 
              'forepaw_left', 'forepaw_right', 'hindpaw_left', 'hindpaw_right', 'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2']

final_df, final_set = run_train_pipeline(
    # Uncomment train & comment test_train for complete data (uncomment parquet file saving in run_train_pipeline as well)
    test,
    # train,
    body_parts,
    base_path="/kaggle/input/MABe-mouse-behavior-detection/test_tracking",
    annot_base_path="/kaggle/input/MABe-mouse-behavior-detection/train_annotation"
)



# ============================================================
# ğŸ“¦ Imports
# ============================================================
import pandas as pd
import numpy as np
import os, gc, warnings, joblib, json
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
pd.options.mode.copy_on_write = True


# ============================================================
# 1ï¸�âƒ£ Data Loading
# ============================================================
def load_behavior_data(df_meta, base_dir):
    print(f"ğŸ”¹ Loading data from {base_dir} ...")
    dfs = []
    total_rows = 0

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Loading videos"):
        lab_id, vid = row["lab_id"], row["video_id"]
        fpath = f"{base_dir}/{lab_id}/{vid}.parquet"

        if not os.path.exists(fpath):
            print(f"âš ï¸� Missing file: {fpath}")
            continue

        try:
            df = pd.read_parquet(fpath)
            df["lab_id"] = lab_id
            df["video_id"] = vid

            for col in df.select_dtypes("float").columns:
                df[col] = pd.to_numeric(df[col], downcast="float")
            for col in df.select_dtypes("int").columns:
                df[col] = pd.to_numeric(df[col], downcast="integer")

            df = df[df['Target'].notna()]  # only labeled
            # print(df.columns)
            dfs.append(df)
            total_rows += len(df)

            if len(dfs) >= 10:  # memory control
                dfs = [pd.concat(dfs, ignore_index=True)]
                gc.collect()
        except Exception as e:
            print(f"â�Œ Error loading {fpath}: {e}")
            continue

    if not dfs:
        raise ValueError("â�Œ No data files loaded!")

    full_df = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()
    print(f"âœ… Loaded {len(full_df):,} rows total.")
    return full_df


# ============================================================
# 2ï¸�âƒ£ Feature Preparation
# ============================================================
def prepare_features(df):
    print("ğŸ§© Preparing features...")

    feature_cols = [
        'distance', 'relative_orientation', 'relative_velocity', 'relative_acceleration',
        'COM_x', 'COM_y', 'COM_x_tgt', 'COM_y_tgt',
        'mouse_id', 'mouse_id_target', 'arena_type', 'no_of_mice',
        'mouse_strain', 'target_mouse_strain',
        'mouse_sex', 'target_mouse_sex',
        'mouse_age', 'target_mouse_age',
        'mouse_color', 'target_mouse_color',
        'mouse_condition', 'target_mouse_condition',
        'mouse_condition_time_in_hours', 'target_mouse_condition_time_in_hours',
        'mouse_condition_light_status', 'target_mouse_condition_light_status'
    ]

    keep_cols = [c for c in feature_cols if c in df.columns] + ['Target', 'lab_id', 'video_id']
    if "video_frame" in df.columns:
        keep_cols.append("video_frame")

    df = df[keep_cols].copy()

    categorical_cols = [
        'arena_type', 'mouse_strain', 'target_mouse_strain',
        'mouse_sex', 'target_mouse_sex', 'mouse_age', 'target_mouse_age',
        'mouse_color', 'target_mouse_color', 'mouse_condition', 'target_mouse_condition',
        'mouse_condition_light_status', 'target_mouse_condition_light_status'
    ]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    num_cols = [
        'distance', 'relative_orientation', 'relative_velocity', 'relative_acceleration',
        'COM_x', 'COM_y', 'COM_x_tgt', 'COM_y_tgt', 'no_of_mice',
        'mouse_condition_time_in_hours', 'target_mouse_condition_time_in_hours'
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    # Fill NAs
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown").astype(str)
    for col in num_cols:
        df[col] = df[col].fillna(0)
        df[col] = pd.to_numeric(df[col], downcast="float")

    print("ğŸ”¢ Encoding categoricals...")
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[num_cols])
    df[num_cols] = scaled

    gc.collect()
    feature_cols_final = [c for c in df.columns if c not in ['Target', 'lab_id', 'video_id', 'video_frame']]
    print(f"âœ… Features ready: {len(feature_cols_final)} cols, shape={df.shape}")
    return df, feature_cols_final, le_dict, scaler


# ============================================================
# 3ï¸�âƒ£ Base Training
# ============================================================
def train_initial_model(df, feature_cols):
    print("ğŸš€ Training base RandomForest model...")
    X = df[feature_cols]
    y = df["Target"]

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.1, stratify=y_enc, random_state=42)
    model = RandomForestClassifier(n_estimators=250, max_depth=18, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    print(f"ğŸ“ˆ Base F1: {f1:.4f}")
    print(classification_report(y_val, preds, target_names=le_y.classes_))
    return model, le_y


# ============================================================
# 4ï¸�âƒ£ Semi-supervised Streaming Iterations
# ============================================================
def semi_supervised_iterations_streaming(model, le_y, df_train, df_semi_meta, base_dir, feature_cols, le_dict, scaler,
                                         rounds=3, threshold=0.9):
    print(f"ğŸ”� Starting semi-supervised refinement ({rounds} rounds, threshold={threshold})")

    for epoch in range(rounds):
        print(f"\nğŸŒ€ Round {epoch+1}/{rounds}")
        pseudo_list = []
        total_added = 0

        for _, row in tqdm(df_semi_meta.iterrows(), total=len(df_semi_meta), desc="Video loop"):
            
            lab_id, vid = row["lab_id"], row["video_id"]
            fpath = f"{base_dir}/{lab_id}/{vid}.parquet"

            if not os.path.exists(fpath):
                continue

            try:
                df_vid = pd.read_parquet(fpath)
                df_vid["lab_id"] = lab_id
                df_vid["video_id"] = vid

                # sample 10%
                x = int(df_vid.shape[0] * 0.1)
                df_vid = df_vid.head(x)

                df_vid = preprocess_like_training(df_vid, feature_cols, le_dict, scaler)

                X_vid = df_vid[feature_cols]
                probs = model.predict_proba(X_vid)
                conf = np.max(probs, axis=1)
                preds = np.argmax(probs, axis=1)
                mask = conf >= threshold

                if mask.sum() == 0:
                    continue

                df_high = df_vid.loc[mask, feature_cols + ["lab_id", "video_id", "video_frame"]].copy()
                df_high["Target"] = le_y.inverse_transform(preds[mask])
                df_high["Confidence"] = conf[mask]

                pseudo_list.append(df_high)
                total_added += len(df_high)
                del df_vid, df_high, X_vid, conf, preds, probs, mask
                gc.collect()

            except Exception as e:
                print(f"âš ï¸� {vid} error: {e}")
                continue

        if not pseudo_list:
            print("âš ï¸� No confident pseudo-labels this round â€” stopping early.")
            break

        pseudo_df = pd.concat(pseudo_list, ignore_index=True)
        df_train = pd.concat([df_train, pseudo_df], ignore_index=True)
        del pseudo_list, pseudo_df; gc.collect()

        print(f"âœ… Round {epoch+1}: added {total_added:,} pseudo-labeled samples.")
        print(f"ğŸ”� Retraining model on {len(df_train):,} samples...")

        X_train = df_train[feature_cols]
        y_train = le_y.transform(df_train["Target"])
        model.fit(X_train, y_train)

    print("ğŸ�� Semi-supervised training complete.")
    return model, df_train

def preprocess_like_training(df_vid, feature_cols, le_dict, scaler):
    """
    Apply the same preprocessing used in training (in-place-ish).
    Returns a dataframe where categorical columns are encoded, numeric cols are scaled,
    and all feature_cols are present.
    """
    df = df_vid.copy()

    categorical_cols = list(le_dict.keys())
    # scaler.feature_names_in_ is an array of the numeric columns used at training
    num_cols = list(getattr(scaler, "feature_names_in_", []))

    # Ensure frame_id exists for traceability
    if "video_frame" not in df.columns:
        df["video_frame"] = np.arange(len(df), dtype=np.int64)

    # Handle categorical columns robustly
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            # safer transform: try/except to handle unseen labels
            le = le_dict[col]
            def safe_transform(v):
                try:
                    return le.transform([v])[0]
                except Exception:
                    return -1
            df[col] = df[col].map(safe_transform)
        else:
            df[col] = -1

    # Ensure numeric columns exist and are numeric
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].fillna(0), errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    # If there are no numeric columns scaler may fail; handle that
    if len(num_cols) > 0:
        df[num_cols] = scaler.transform(df[num_cols])

    # Ensure all feature_cols exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    # Keep original ordering (optional)
    return df


# def predict_and_score_unseen(model, le_y, df, base_dir, feature_cols, le_dict, scaler, sample_frac=None):
#     """
#     Streams through unseen video parquet files listed in df_meta (columns: lab_id, video_id),
#     applies the exact same preprocessing as training via preprocess_like_training(),
#     runs model.predict_proba, stores Predicted + Confidence and preserves frame_id in final output.

#     Args:
#       - model, le_y, feature_cols, le_dict, scaler: from training pipeline
#       - df_meta: DataFrame with columns "lab_id" and "video_id" (one row per video)
#       - base_dir: folder root where lab_id/video_id.parquet live
#       - sample_frac: optional float in (0,1] to sample that fraction of each video frames (memory)
#     Returns:
#       - preds_df: DataFrame with columns [lab_id, video_id, frame_id, Predicted, Confidence, (Target if present)]
#       - f1: macro F1 score if Target column present in preds_df else None
#     """
#     results = []
#     # for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Unseen scoring"):
#     # lab_id, vid = row["lab_id"], row["video_id"]
#     # fpath = f"{base_dir}/{lab_id}/{vid}.parquet"
#     # if not os.path.exists(fpath):
#     #     print(f"âš ï¸� Missing file:", fpath)
#     #     continue

#     try:
#         # df_vid = pd.read_parquet(fpath)
#         # attach ids
#         df_vid = df
#         df_vid["lab_id"] = lab_id
#         df_vid["video_id"] = vid

#         # optional sampling to reduce memory/compute
#         if sample_frac is not None and 0 < sample_frac < 1:
#             # sample preserves index; reset index afterwards for safe numeric ops
#             df_vid = df_vid.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

#         # Apply identical preprocessing
#         df_proc = preprocess_like_training(df_vid, feature_cols, le_dict, scaler)

#         # Build feature matrix and predict
#         X_vid = df_proc[feature_cols]
#         probs = model.predict_proba(X_vid)
#         preds_idx = np.argmax(probs, axis=1)
#         conf = np.max(probs, axis=1)

#         # Map back to labels
#         preds_label = le_y.inverse_transform(preds_idx)

#         # Prepare output: keep only needed columns (preserve frame_id)
#         out = pd.DataFrame({
#             "lab_id": df_proc["lab_id"].values,
#             "video_id": df_proc["video_id"].values,
#             "frame_id": df_proc["video_frame"].values,
#             "Mouse_id" : df_proc["mouse_id"].values,
#             "Target_id" : df_proc["mouse_id_target"].values,
#             "Predicted": preds_label,
#             "Confidence": conf
#         })

#         # include Target if present in original df_vid (before preprocess)
#         if "Target" in df_vid.columns:
#             # align by index: df_proc was copy and may have been sampled; df_vid reset_index when sampled
#             # use df_proc index to fetch Target if it existed originally; safer to merge by frame_id if needed
#             if "Target" in df_proc.columns:
#                 out["Target"] = df_proc["Target"].values
#             else:
#                 # if preprocess removed Target, try to get from original df_vid by frame_id
#                 if "video_frame" in df_vid.columns:
#                     tgt_map = df_vid.set_index("video_frame")["Target"] if "Target" in df_vid.columns else None
#                     if tgt_map is not None:
#                         out["Target"] = out["video_frame"].map(tgt_map).values
#                     else:
#                         out["Target"] = np.nan
#                 else:
#                     out["Target"] = np.nan

#         results.append(out)

#         # cleanup per-video
#         del df_vid, df_proc, X_vid, probs, preds_idx, conf, preds_label, out
#         gc.collect()

#     except Exception as e:
#         print(f"â�Œ Error scoring video {vid}: {e}")
#         continue

#     preds_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
#     print(f"âœ… Generated predictions for {len(preds_df):,} frames.")

#     # compute F1 if targets available
#     if "Target" in preds_df.columns and preds_df["Target"].notna().any():
#         f1 = f1_score(preds_df["Target"].fillna("None"), preds_df["Predicted"].fillna("None"),
#                       average="macro", zero_division=0)
#         print(f"ğŸ�� F1 on unseen (if target present): {f1:.4f}")
#     else:
#         f1 = None
#         print("âš ï¸� No Target found in unseen data; F1 not computed.")

#     return preds_df, f1


# ============================================================
# 6ï¸�âƒ£ Full Orchestrator
# ============================================================
def run_two_stage_pipeline(df_train, df_semi, base_dir):
    df_train_full = load_behavior_data(df_train, base_dir)
    # df_train_full = df
    df_train_full, feature_cols, le_dict, scaler = prepare_features(df_train_full)
    model, le_y = train_initial_model(df_train_full, feature_cols)

    model, trained_df = semi_supervised_iterations_streaming(
        model, le_y, df_train_full, df_semi, base_dir,
        feature_cols, le_dict, scaler, rounds=3, threshold=1
    )

    # preds_df, f1 = predict_and_score_unseen(model, le_y, df_unseen, base_dir, feature_cols, le_dict, scaler)

    print("âœ… Full pipeline complete.")
    return model, trained_df, le_y, feature_cols, le_dict, scaler


train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train.drop(columns=['mouse1_id', 'mouse2_id', 'mouse3_id', 'mouse4_id'], inplace=True)

df_semi = train[train['behaviors_labeled'].isna()]
df_train = train[train['behaviors_labeled'].notna()]
df_semi = df_semi.sample(1)

base_dir = "/kaggle/input/mabe-dataset/Dataset"
model, trained_df, le_y, feature_cols, le_dict, scaler = run_two_stage_pipeline(df_train, df_semi, base_dir)



def predict_and_score_unseen(
    model,
    le_y,
    df,
    base_dir,
    feature_cols,
    le_dict,
    scaler,
    sample_frac=None
):

    """
    df â†’ single dataframe containing unseen data for one video.
    Must include frame_id, mouse_id, mouse_id_target and (optionally) Target.
    """

    # -----------------------------
    # 1ï¸�âƒ£ Copy and preserve essentials
    # -----------------------------
    df_vid = df.copy()

    # Preserve raw frame_id
    if "frame_id" not in df_vid.columns:
        raise ValueError("â�Œ df must contain frame_id column.")

    # optional sampling
    if sample_frac is not None and 0 < sample_frac < 1:
        df_vid = df_vid.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # ---------------------------------
    # 2ï¸�âƒ£ Apply identical preprocessing
    # ---------------------------------
    df_proc = preprocess_like_training(df_vid, feature_cols, le_dict, scaler)

    # ---------------------------------
    # 3ï¸�âƒ£ Predict on features
    # ---------------------------------
    X = df_proc[feature_cols]
    probs = model.predict_proba(X)
    pred_idx = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    pred_lbl = le_y.inverse_transform(pred_idx)

    # -----------------------------------------------
    # 4ï¸�âƒ£ Build final output dataframe (with frame_id)
    # -----------------------------------------------
    preds_df = pd.DataFrame({
        "lab_id": df_vid.get("lab_id", None),
        "video_id": df_vid.get("video_id", None),
        "frame_id": df_vid["frame_id"].values,
        "Mouse_id": df_vid["mouse_id"].values,
        "Target_id": df_vid["mouse_id_target"].values,
        "Predicted": pred_lbl,
        "Confidence": conf
    })

    # ---------------------------------
    # 5ï¸�âƒ£ Add Target if exists
    # ---------------------------------
    if "Target" in df_vid.columns:
        preds_df["Target"] = df_vid["Target"].values
    else:
        preds_df["Target"] = None

    # ---------------------------------
    # 6ï¸�âƒ£ Compute F1 if Target exists
    # ---------------------------------
    if preds_df["Target"].notna().any():
        # Fill missing as string "None"
        t = preds_df["Target"].fillna("None")
        p = preds_df["Predicted"].fillna("None")

        f1 = f1_score(t, p, average="macro", zero_division=0)
    else:
        f1 = None

    return preds_df, f1


final_df.rename(columns = {'video_frame':'frame_id'}, inplace =True)
preds_df, f1 = predict_and_score_unseen(
    model=model,
    le_y=le_y,
    df=final_df,
    base_dir=base_dir,
    feature_cols=feature_cols,
    le_dict=le_dict,
    scaler=scaler
)



preds_conf = preds_df[preds_df['Confidence']>0.5]



def convert_frames_to_segments(df):
    """
    Convert frame-level predictions to action segments.
    Also convert numeric Mouse_id and Target_id into mouse1, mouse2, ...
    """

    df = df.sort_values(["video_id", "Mouse_id", "Target_id", "frame_id"]).reset_index(drop=True)

    segments = []

    for (vid, agent, target, action), group in df.groupby(
        ["video_id", "Mouse_id", "Target_id", "Predicted"]
    ):
        group = group.sort_values("frame_id")

        # Detect continuous frame sequences
        diffs = group["frame_id"].diff().fillna(1)
        segment_id = (diffs != 1).cumsum()

        for _, seg_df in group.groupby(segment_id):
            start_f = int(seg_df["frame_id"].iloc[0])
            stop_f  = int(seg_df["frame_id"].iloc[-1])

            segments.append({
                "video_id": vid,
                "agent_id": f"mouse{int(agent)}",
                "target_id": f"mouse{int(target)}",
                "action": action,
                "start_frame": start_f,
                "stop_frame": stop_f,
            })

    seg_df = pd.DataFrame(segments).reset_index().rename(columns={"index": "row_id"})
    return seg_df


predictions = convert_frames_to_segments(preds_conf)

predictions.to_csv('/kaggle/working/submission.csv')


predictions.to_csv('/kaggle/working/submission.csv',index = False)




