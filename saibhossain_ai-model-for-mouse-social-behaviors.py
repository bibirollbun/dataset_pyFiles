import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
         print(os.path.join(dirname, filename))


!pip install numpy pandas matplotlib


!pip install lightgbm


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def inspect_parquet(path, frame_to_plot=0, figsize=(8,6)):
    df = pd.read_parquet(path)
    print("Loaded:", path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nHead:")
    display(df.head())

    # Heuristic: tracking/pose file contains 'x' and 'y' and 'bodypart' columns
    if {'x','y','bodypart','video_frame','mouse_id'}.issubset(df.columns):
        print("\nDetected: POSE / TRACKING file")
        print("Unique mice:", sorted(df['mouse_id'].unique()))
        print("Keypoints:", sorted(df['bodypart'].unique()))
        print("Number of frames:", int(df['video_frame'].nunique()))
        frame = frame_to_plot
        print(f"\nPlotting pose for frame {frame} (frame index may start at 0)")

        plt.figure(figsize=figsize)
        for mid in sorted(df['mouse_id'].unique()):
            sub = df[(df['mouse_id']==mid) & (df['video_frame']==frame)]
            if sub.empty:
                print(f"  Warning: mouse {mid} has no keypoints at frame {frame}")
                continue
            # scatter keypoints
            plt.scatter(sub['x'], sub['y'], label=f"mouse {mid}", s=20)
            # annotate keypoints with their name for clarity
            for _, row in sub.iterrows():
                bp = str(row['bodypart'])
                plt.text(row['x']+2, row['y']+2, bp, fontsize=6)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title(f"Pose at frame {frame}")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.tight_layout()
        plt.show()

        max_frame = int(df['video_frame'].max())
        print("Frame range: 0 ..", max_frame)

    elif {'agent_id','target_id','action','start_frame','stop_frame'}.issubset(df.columns):
        print("\nDetected: ANNOTATION file")
        print("Unique agents:", sorted(df['agent_id'].unique()))
        print("Unique targets:", sorted(df['target_id'].unique()))
        print("Unique actions:", sorted(df['action'].unique()))
        print("Number of annotated segments:", len(df))
        # Convert start/stop to ints (safety)
        df['start_frame'] = df['start_frame'].astype(int)
        df['stop_frame'] = df['stop_frame'].astype(int)

        # Basic statistics
        df['duration'] = df['stop_frame'] - df['start_frame'] + 1
        print("\nPer-action counts and mean durations:")
        print(df.groupby('action')['duration'].agg(['count','mean']).sort_values('count', ascending=False))

        # Gantt-style timeline plot
        agents = sorted(df['agent_id'].unique())
        actions = sorted(df['action'].unique())
        action_to_color = {a: i for i, a in enumerate(actions)} 

        fig, ax = plt.subplots(figsize=(12, max(2, 0.5*len(agents))))
        yticks = []
        yticklabels = []
        for i, agent in enumerate(agents):
            agent_segs = df[df['agent_id']==agent]
            y = i
            yticks.append(y)
            yticklabels.append(f"agent {agent}")
            for _, row in agent_segs.iterrows():
                start = int(row['start_frame'])
                stop = int(row['stop_frame'])
                action = row['action']
                color_idx = action_to_color[action]
                ax.barh(y, width=(stop - start + 1), left=start, height=0.6,
                        color=plt.cm.tab20(color_idx % 20), edgecolor='k', alpha=0.8)
                if (stop - start) > 10:
                    ax.text(start + 1, y, str(action), va='center', ha='left', fontsize=7, color='white')

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("frame")
        ax.set_title("Annotation timeline (per agent)")
        ax.invert_yaxis()
        # legend
        handles = []
        labels = []
        for a, idx in action_to_color.items():
            handles.append(plt.Rectangle((0,0),1,1, color=plt.cm.tab20(idx % 20)))
            labels.append(a)
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()

        print("\nPer-agent annotated time (frames):")
        print(df.groupby('agent_id')['duration'].sum())

    else:
        print("\nUnknown parquet format. Columns are not recognized as pose or annotation.")
        print("Example columns:", list(df.columns))
        print("\nIf this is a tracking file, it should contain at least: x, y, bodypart, video_frame, mouse_id")
        print("If this is an annotation file, it should contain at least: agent_id, target_id, action, start_frame, stop_frame")
    return df

path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/1212811043.parquet"


df = inspect_parquet(path, frame_to_plot=0)


import os
import pandas as pd
from pathlib import Path

# Base dataset path
BASE_PATH = Path("/kaggle/input/MABe-mouse-behavior-detection")

# Function to peek into a parquet file
def peek_parquet(file_path, n_rows=5):
    try:
        df = pd.read_parquet(file_path)
        print(f"\nFile: {file_path.name}")
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))
        display(df.head(n_rows))

        # Check if pose/tracking or annotation file
        if {'x','y','bodypart','video_frame','mouse_id'}.issubset(df.columns):
            print("Type: POSE / TRACKING file")
            print("Unique mice:", df['mouse_id'].unique())
            print("Unique bodyparts:", df['bodypart'].unique())
            print("Number of frames:", df['video_frame'].nunique())
        elif {'agent_id','target_id','action','start_frame','stop_frame'}.issubset(df.columns):
            print("Type: ANNOTATION file")
            print("Unique agents:", df['agent_id'].unique())
            print("Unique targets:", df['target_id'].unique())
            print("Unique actions:", df['action'].unique())
            print("Number of annotated segments:", len(df))
        else:
            print("Unknown file type")
        return df
    except Exception as e:
        print(f"Could not read {file_path.name}: {e}")
        return None

# 1. List all directories and files
print("Dataset structure:")
for root, dirs, files in os.walk(BASE_PATH):
    level = root.replace(str(BASE_PATH), '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files[:5]:  # print first 5 files in each folder
        print(f"{subindent}{f}")
    if len(files) > 5:
        print(f"{subindent}... ({len(files)-5} more files)")

# 2. Count files per type
file_counts = {}
for path in BASE_PATH.rglob("*.parquet"):
    parent = path.parent.name
    file_counts[parent] = file_counts.get(parent, 0) + 1

print("\nNumber of parquet files per folder:")
for k,v in file_counts.items():
    print(f"{k}: {v}")

# 3. Peek into a few files
print("\nPeeking into some train annotation files:")
train_ann_path = BASE_PATH / "train_annotation" / "AdaptableSnail"
train_ann_files = list(train_ann_path.glob("*.parquet"))
for f in train_ann_files[:2]:  # peek first 2 files
    peek_parquet(f)

print("\nPeeking into some test tracking files:")
test_track_path = BASE_PATH / "test_tracking" / "AdaptableSnail"
test_track_files = list(test_track_path.glob("*.parquet"))
for f in test_track_files[:2]:  # peek first 2 files
    peek_parquet(f)

# 4. Optional: summary stats across dataset
print("\nSummary across all train annotation files:")
all_actions = []
for f in train_ann_files:
    df = peek_parquet(f)
    if df is not None and 'action' in df.columns:
        all_actions.extend(df['action'].unique())
print("All unique actions in train:", set(all_actions))


df = pd.read_csv("/kaggle/input/MABe-mouse-behavior-detection/test.csv")
df


df = pd.read_csv("/kaggle/input/MABe-mouse-behavior-detection/train.csv")
df


import os
import glob
import math
import gc
import joblib
import warnings
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG
# ---------------------------
CONFIG = {
    'input_path': '/kaggle/input/MABe-mouse-behavior-detection',  # set to your input dir
    'processed_data_path': 'train_features.parquet',
    'model_save_path': 'lgbm_model.pkl',
    'encoder_save_path': 'label_encoder.pkl',
    'train_limit_per_lab': 30,           # how many videos per lab to use (limits runtime)
    'frame_col': 'video_frame',
    'chunk_prefix': 'chunk_',            # temporary chunk files
    'max_chunks_to_load': 8,             # safety for memory when loading chunk files
    'min_segment_frames': 3,             # min frames to keep a predicted segment
    'smoothing_window': 5,               # majority smoothing window (odd integer)
    'submission_name': 'submission.csv'
}

# ---------------------------
# FEATURE EXTRACTOR (robust to bodypart names)
# ---------------------------
def robust_pivot_tracking(df_tracking):
    """
    Convert tracking dataframe to wide format with columns like:
    <mouseid>_<bodypart>_x, <mouseid>_<bodypart>_y
    Returns df_wide with frame column named CONFIG['frame_col']
    """
    if 'frame' in df_tracking.columns and CONFIG['frame_col'] not in df_tracking.columns:
        df_tracking = df_tracking.rename(columns={'frame': CONFIG['frame_col']})
    # Expect columns at least: [video_frame, mouse_id, bodypart, x, y]
    if not {'mouse_id','bodypart','x','y'}.issubset(set(df_tracking.columns)):
        # Try alternative names
        possible = set(df_tracking.columns)
        # fail gracefully
        return None

    # pivot
    try:
        df_wide = df_tracking.pivot_table(
            index=CONFIG['frame_col'],
            columns=['mouse_id','bodypart'],
            values=['x','y'],
            aggfunc='first'
        )
    except Exception:
        return None

    # flatten
    df_wide.columns = [f"{mid}_{bp}_{coord}" for coord, mid, bp in df_wide.columns]
    df_wide = df_wide.reset_index()
    return df_wide

def find_part_cols(df_wide, mouse, part_names):
    """Return (x_col, y_col) for the first matching part in part_names for given mouse"""
    for part in part_names:
        x_col = f"{mouse}_{part}_x"
        y_col = f"{mouse}_{part}_y"
        if x_col in df_wide.columns and y_col in df_wide.columns:
            return x_col, y_col
    # try fuzzy search
    for c in df_wide.columns:
        if c.startswith(f"{mouse}_") and any(p in c for p in part_names) and c.endswith('_x'):
            return c, c[:-2] + '_y'
    return None, None

def angle_between_points(x1, y1, x2, y2):
    # angle of vector tail->head
    return np.arctan2(y1 - y2, x1 - x2)

def extract_features_from_tracking(df_tracking):
    """
    Input: raw tracking dataframe (per-video, possibly many frames)
    Output: dataframe with frame column and engineered features
    """
    try:
        df_wide = robust_pivot_tracking(df_tracking)
        if df_wide is None:
            return None
    except Exception:
        return None

    # Ensure frame col present
    if CONFIG['frame_col'] not in df_wide.columns:
        return None

    # find mouse ids present
    mice_ids = sorted(set([col.split('_')[0] for col in df_wide.columns if col!=CONFIG['frame_col']]))
    if len(mice_ids) < 2:
        # pad zeros if only one mouse or missing
        # return a df with required frame col and zeros
        out = pd.DataFrame({CONFIG['frame_col']: df_wide[CONFIG['frame_col']]})
        # add placeholder features so pipeline won't crash
        out['dist_m1_m2'] = 0.0
        out['m1_dx'] = out['m1_dy'] = out['m2_dx'] = out['m2_dy'] = 0.0
        out['m1_speed'] = out['m2_speed'] = 0.0
        out['rel_dx'] = out['rel_dy'] = 0.0
        out['m1_angle'] = out['m2_angle'] = 0.0
        out['angle_diff'] = 0.0
        out['m1_acc'] = out['m2_acc'] = out['rel_acc'] = 0.0
        out['ddist'] = 0.0
        out['dist_nose_tail'] = 0.0
        out['dist_head_head'] = 0.0
        return out

    # choose two mice: first two sorted ids
    m1, m2 = mice_ids[0], mice_ids[1]

    # find bodyparts robustly
    m1_head_x, m1_head_y = find_part_cols(df_wide, m1, ['head','nose','snout'])
    m1_tail_x, m1_tail_y = find_part_cols(df_wide, m1, ['tail','tail_base','root','spine','body','center'])
    m2_head_x, m2_head_y = find_part_cols(df_wide, m2, ['head','nose','snout'])
    m2_tail_x, m2_tail_y = find_part_cols(df_wide, m2, ['tail','tail_base','root','spine','body','center'])

    # create output df starting from frame
    out = pd.DataFrame({CONFIG['frame_col']: df_wide[CONFIG['frame_col']]})

    # helper safe column fetch
    def col_or_zero(col):
        return df_wide[col] if (col and col in df_wide.columns) else pd.Series(0.0, index=df_wide.index)

    # compute distances (use head/head fallback)
    h1x = col_or_zero(m1_head_x)
    h1y = col_or_zero(m1_head_y)
    h2x = col_or_zero(m2_head_x)
    h2y = col_or_zero(m2_head_y)

    out['dist_m1_m2'] = np.sqrt((h1x - h2x)**2 + (h1y - h2y)**2).astype('float32')
    out['dist_head_head'] = out['dist_m1_m2'].astype('float32')

    # nose (for nose->tail)
    m1_nose_x, m1_nose_y = find_part_cols(df_wide, m1, ['nose','snout','head'])
    m2_nose_x, m2_nose_y = find_part_cols(df_wide, m2, ['nose','snout','head'])
    m1_nose_x_s = col_or_zero(m1_nose_x)
    m1_nose_y_s = col_or_zero(m1_nose_y)
    m2_tail_x_s = col_or_zero(m2_tail_x)
    m2_tail_y_s = col_or_zero(m2_tail_y)
    out['dist_nose_tail'] = np.sqrt((m1_nose_x_s - m2_tail_x_s)**2 + (m1_nose_y_s - m2_tail_y_s)**2).astype('float32')

    # directional velocities (head position diffs)
    out['m1_dx'] = h1x.diff().fillna(0).astype('float32')
    out['m1_dy'] = h1y.diff().fillna(0).astype('float32')
    out['m2_dx'] = h2x.diff().fillna(0).astype('float32')
    out['m2_dy'] = h2y.diff().fillna(0).astype('float32')

    out['m1_speed'] = np.sqrt(out['m1_dx']**2 + out['m1_dy']**2).astype('float32')
    out['m2_speed'] = np.sqrt(out['m2_dx']**2 + out['m2_dy']**2).astype('float32')

    # relative motion
    out['rel_dx'] = (out['m1_dx'] - out['m2_dx']).astype('float32')
    out['rel_dy'] = (out['m1_dy'] - out['m2_dy']).astype('float32')
    out['rel_speed'] = np.sqrt(out['rel_dx']**2 + out['rel_dy']**2).astype('float32')

    # orientation angles (tail->head)
    t1x = col_or_zero(m1_tail_x)
    t1y = col_or_zero(m1_tail_y)
    t2x = col_or_zero(m2_tail_x)
    t2y = col_or_zero(m2_tail_y)

    out['m1_angle'] = angle_between_points(h1x, h1y, t1x, t1y).astype('float32')
    out['m2_angle'] = angle_between_points(h2x, h2y, t2x, t2y).astype('float32')

    # normalized angle difference to [-pi, pi]
    angle_diff = out['m1_angle'] - out['m2_angle']
    out['angle_diff'] = ((angle_diff + np.pi) % (2 * np.pi) - np.pi).astype('float32')

    # acceleration
    out['m1_acc'] = out['m1_speed'].diff().fillna(0).astype('float32')
    out['m2_acc'] = out['m2_speed'].diff().fillna(0).astype('float32')
    out['rel_acc'] = out['rel_speed'].diff().fillna(0).astype('float32')

    # distance change rate
    out['ddist'] = out['dist_m1_m2'].diff().fillna(0).astype('float32')

    # replace inf/nan
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # keep an ordered feature set
    feature_cols = [
        'dist_m1_m2', 'dist_head_head', 'dist_nose_tail',
        'm1_dx','m1_dy','m2_dx','m2_dy',
        'm1_speed','m2_speed','rel_dx','rel_dy','rel_speed',
        'm1_angle','m2_angle','angle_diff',
        'm1_acc','m2_acc','rel_acc','ddist'
    ]
    # ensure all exist
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[ [CONFIG['frame_col']] + feature_cols ]

# ---------------------------
# CHUNKED PROCESS & SAVE
# ---------------------------
def process_and_save_data():
    print(">>> Phase 1: Processing Data in Chunks...")
    search_path = f"{CONFIG['input_path']}/train_tracking/*/*.parquet"
    all_files = glob.glob(search_path)
    lab_files = {}
    for f in all_files:
        lab_name = os.path.basename(os.path.dirname(f))
        # optional: skip some labs if needed
        if lab_name not in lab_files:
            lab_files[lab_name] = []
        lab_files[lab_name].append(f)

    first_chunk = True
    total_processed = 0
    chunk_idx = 0

    for lab, files in lab_files.items():
        selected_files = files[:CONFIG['train_limit_per_lab']]
        print(f"   Lab {lab}: Processing {len(selected_files)} videos...")
        for trk_path in selected_files:
            ann_path = trk_path.replace('train_tracking', 'train_annotation')
            if not os.path.exists(ann_path):
                continue
            try:
                df_trk = pd.read_parquet(trk_path)
                df_ann = pd.read_parquet(ann_path)
                X_df = extract_features_from_tracking(df_trk)
                if X_df is None or X_df.empty:
                    continue

                # default labels = 'other'
                y = pd.Series(['other'] * len(X_df), index=X_df.index, name='action')
                # assign annotated actions (frame ranges)
                for _, row in df_ann.iterrows():
                    try:
                        start = int(row['start_frame'])
                        stop = int(row['stop_frame'])
                        mask = (X_df[CONFIG['frame_col']] >= start) & (X_df[CONFIG['frame_col']] <= stop)
                        y.loc[mask] = row['action']
                    except Exception:
                        continue

                chunk_df = pd.concat([X_df, y], axis=1)
                # drop frame col before saving to parquet for training accumulation
                to_save = chunk_df.drop(columns=[CONFIG['frame_col']])
                # downcast floats
                float_cols = to_save.select_dtypes(include=['float64']).columns
                to_save[float_cols] = to_save[float_cols].astype('float32')

                if first_chunk:
                    to_save.to_parquet(CONFIG['processed_data_path'], engine='pyarrow', index=False)
                    first_chunk = False
                else:
                    fname = f"{CONFIG['chunk_prefix']}{chunk_idx}.parquet"
                    to_save.to_parquet(fname, index=False)
                    chunk_idx += 1

                total_processed += 1
                del df_trk, df_ann, X_df, chunk_df, to_save, y
                gc.collect()
            except Exception as e:
                # skip problematic video
                # print("skip", trk_path, e)
                continue

    print(f"Processed {total_processed} videos.")
    return total_processed

# ---------------------------
# TRAINING
# ---------------------------
def train_model():
    print(">>> Phase 2: Training & Evaluation...")
    if not os.path.exists(CONFIG['processed_data_path']):
        raise ValueError("No processed data found. Run process_and_save_data() first.")

    df_main = pd.read_parquet(CONFIG['processed_data_path'])
    # load extra chunks
    chunk_files = sorted(glob.glob(f"{CONFIG['chunk_prefix']}*.parquet"))[:CONFIG['max_chunks_to_load']]
    for c in chunk_files:
        temp = pd.read_parquet(c)
        df_main = pd.concat([df_main, temp], axis=0)
        del temp
        gc.collect()

    print(f"Total Dataset Shape: {df_main.shape}")

    # feature / label split
    feature_cols = [c for c in df_main.columns if c != 'action']
    X = df_main[feature_cols]
    y = df_main['action']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2,
                                                      random_state=42, stratify=y_enc)

    print("  Fitting LightGBM with class balancing...")
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        n_jobs=-1,
        class_weight='balanced',
        objective='multiclass'
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(50)
        ]
    )

    # Evaluate
    val_preds = clf.predict(X_val)
    print(f"\nValidation Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    try:
        print(classification_report(y_val, val_preds, target_names=le.classes_))
    except Exception:
        print(classification_report(y_val, val_preds))

    # Save model & encoder
    joblib.dump(clf, CONFIG['model_save_path'])
    joblib.dump(le, CONFIG['encoder_save_path'])
    print(f"Model and encoder saved: {CONFIG['model_save_path']}, {CONFIG['encoder_save_path']}")

    # cleanup chunks and processed file (optional)
    # os.remove(CONFIG['processed_data_path'])
    # for c in glob.glob(f"{CONFIG['chunk_prefix']}*.parquet"): os.remove(c)

    return clf, le, feature_cols

# ---------------------------
# POSTPROCESS helpers
# ---------------------------
def smooth_preds_by_majority(preds, window=5):
    """
    preds: numpy array of integer class indices
    apply majority smoothing with sliding window (odd window recommended)
    """
    if window <= 1:
        return preds
    n = len(preds)
    half = window // 2
    smoothed = np.copy(preds)
    # use deque to keep histogram counts for speed
    from collections import Counter
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        window_vals = preds[a:b]
        # majority
        most = Counter(window_vals).most_common(1)[0][0]
        smoothed[i] = most
    return smoothed

def frames_to_segments(frames, labels, label_map, min_length=3):
    """
    Convert per-frame label list to list of segments [ (label_name, start, stop) ... ]
    filters out 'other' and segments shorter than min_length
    """
    segments = []
    current_label = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append((current_label, start, i-1))
            current_label = labels[i]
            start = i
    segments.append((current_label, start, len(labels)-1))
    # filter
    out = []
    for lab_idx, s, e in segments:
        lab_name = label_map[lab_idx]
        if lab_name == 'other':
            continue
        if (e - s + 1) < min_length:
            continue
        out.append((lab_name, s, e))
    return out

# ---------------------------
# INFERENCE & SUBMISSION BUILD
# ---------------------------
def create_submission(clf, le, feature_cols):
    print(">>> Phase 3: Inference & Submission Building...")
    test_files = sorted(glob.glob(f"{CONFIG['input_path']}/test_tracking/*/*.parquet"))
    submission_rows = []
    row_id = 0

    for test_path in test_files:
        video_id = os.path.basename(test_path).replace('.parquet','')
        try:
            df_test_trk = pd.read_parquet(test_path)
            X_test = extract_features_from_tracking(df_test_trk)
            if X_test is None or X_test.empty:
                continue

            # align features
            for c in feature_cols:
                if c not in X_test.columns:
                    X_test[c] = 0.0
            X_test_model = X_test[feature_cols].astype('float32')

            # predict probabilities to allow smoothing-based decisions
            probs = clf.predict_proba(X_test_model)  # shape (T, n_classes)
            # Optional: boost 'other' slightly to reduce false positives
            try:
                other_idx = list(le.classes_).index('other')
                probs[:, other_idx] *= 1.05
            except Exception:
                pass

            preds = np.argmax(probs, axis=1)
            # smoothing (majority)
            preds_sm = smooth_preds_by_majority(preds, window=CONFIG['smoothing_window'])

            # velocity-based filter: if both mice stationary, set to 'other' for movement actions
            movement_actions = {'chase','escape','follow','attack','approach','chaseattack','attack'}
            # make bool series: both stationary (speed threshold)
            if 'm1_speed' in X_test.columns and 'm2_speed' in X_test.columns:
                is_stationary = (X_test['m1_speed'] < 0.5) & (X_test['m2_speed'] < 0.5)
            else:
                is_stationary = pd.Series([False]*len(X_test))

            # map preds_sm -> label names, but first convert to names array
            label_map = {i:cl for i,cl in enumerate(le.classes_)}

            # apply velocity mask by setting frame preds to index of 'other'
            try:
                other_index = int(list(le.classes_).index('other'))
            except Exception:
                other_index = None

            if other_index is not None:
                for i in range(len(preds_sm)):
                    if is_stationary.iloc[i] and label_map[preds_sm[i]] in movement_actions:
                        preds_sm[i] = other_index

            # convert frame preds to segments
            segments = frames_to_segments(X_test[CONFIG['frame_col']].values, preds_sm, label_map,
                                          min_length=CONFIG['min_segment_frames'])

            # Create rows with agent/target extraction (best-effort)
            # Attempt to obtain mouse ids from original test file
            try:
                if 'mouse_id' in df_test_trk.columns:
                    u_m = df_test_trk['mouse_id'].unique()
                    agent_id = str(u_m[0]) if len(u_m)>0 else 'mouse1'
                    target_id = str(u_m[1]) if len(u_m)>1 else 'mouse2'
                else:
                    agent_id, target_id = 'mouse1', 'mouse2'
            except Exception:
                agent_id, target_id = 'mouse1', 'mouse2'

            for lab_name, s, e in segments:
                submission_rows.append([row_id, video_id, agent_id, target_id, lab_name, int(s), int(e)])
                row_id += 1

            del df_test_trk, X_test, probs, preds, preds_sm
            gc.collect()

        except Exception as e:
            # skip problematic test files but keep running
            # print("skip test", test_path, e)
            continue

    sub_df = pd.DataFrame(submission_rows, columns=['row_id','video_id','agent_id','target_id','action','start_frame','stop_frame'])
    if sub_df.empty:
        # fallback: dummy small submission to avoid empty submission
        sub_df = pd.DataFrame([[0, 'no_video', 'mouse1', 'mouse2', 'other', 0, 10]],
                              columns=['row_id','video_id','agent_id','target_id','action','start_frame','stop_frame'])
    sub_df.to_csv(CONFIG['submission_name'], index=False)
    print(f"Saved submission: {CONFIG['submission_name']} (rows={len(sub_df)})")
    return sub_df

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # 1) Process/train only if run as main
    n_processed = process_and_save_data()
    if n_processed == 0:
        print("No videos processed. Check CONFIG['input_path'] and that parquet files exist.")
    else:
        clf, le, feature_cols = train_model()
        sub = create_submission(clf, le, feature_cols)
        print("Pipeline completed.")



df_sub = pd.read_csv("/kaggle/working/submission.csv")
df_sub


print(df_sub.head())
print("Unique Actions Predicted:", df_sub['action'].unique())


# # =========================================================
# # RUN THIS IN A SEPARATE CELL TO TEST SAVED MODEL
# # =========================================================
# import joblib
# import pandas as pd
# import glob
# import os
# import numpy as np

# def test_saved_model(test_video_path=None):
#     print(">>> Loading Saved Model...")
    
#     # 1. Load Model & Encoder
#     try:
#         clf = joblib.load('/kaggle/working/lgbm_model.pkl')
#         le = joblib.load('/kaggle/working/label_encoder.pkl')
#         print("Model loaded successfully.")
#     except FileNotFoundError:
#         print("Error: Model files not found. Run the training pipeline first!")
#         return

#     # 2. Pick a random test video if none provided
#     if test_video_path is None:
#         possible_files = glob.glob('/kaggle/input/MABe-mouse-behavior-detection/test_tracking/*/*.parquet')
#         if not possible_files:
#             print("No test files found.")
#             return
#         test_video_path = possible_files[0]

#     print(f"Testing on video: {os.path.basename(test_video_path)}")
    
#     # 3. Process Data (Must match training processing exactly!)
#     # We re-define extract_features here briefly or ensure the function above is available
#     df_test = pd.read_parquet(test_video_path)
    
#     # --- Quick Feature Extraction Copy ---
#     # (Assuming extract_features is defined in the notebook environment)
#     X_test = extract_features(df_test) 
    
#     if X_test is None: 
#         print("Feature extraction failed.")
#         return

#     # Align columns with model
#     # LightGBM requires exact column match. We get feature names from the model.
#     model_cols = clf.feature_name_
#     for col in model_cols:
#         if col not in X_test.columns:
#             X_test[col] = 0.0
    
#     # Ensure order matches
#     X_test = X_test[model_cols]
    
#     # 4. Predict
#     preds = clf.predict(X_test)
#     pred_labels = le.inverse_transform(preds)
    
#     # 5. Show Results
#     print("\nPrediction Summary:")
#     unique_actions, counts = np.unique(pred_labels, return_counts=True)
#     for action, count in zip(unique_actions, counts):
#         print(f"  - {action}: {count} frames")
        
#     print("\nFirst 20 Frames:")
#     print(pred_labels[:20])

# # Run the test
# test_saved_model()




