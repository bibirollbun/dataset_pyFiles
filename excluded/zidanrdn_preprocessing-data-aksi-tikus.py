#Preprocessing Preparation
import pandas as pd
import numpy as np
import pyarrow
import random
import glob
import os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)

print("Success")


path_in = "/kaggle/input/MABe-mouse-behavior-detection/"

metadata = os.path.join(path_in, "train.csv")
annotations = os.path.join(path_in, "train_annotation/")
train_tracking = os.path.join(path_in, "train_tracking/")

pattern_all_file = os.path.join(annotations, "*")
content_ann = glob.glob(pattern_all_file)

print("\nHasil Pengecekan Glob")
if len(content_ann) > 0:
    print(f"Brhasil menemukan {len(content_ann)} item: ")
    for item in content_ann[:10]:
        print(item)
else:
    print("Warning: Folder kosong")

#Konfigurasi Kolom
target = 'action'
group = 'video_id'

#Konfirmasi Path
print(f"Path Metadata: {metadata}")
print(f"Path Folder Anotasi: {annotations}")
print(f"Path Folder Train Tracking: {train_tracking}")


#Load Data Metadata
df_meta = pd.read_csv(metadata)
print("Load file metadata (train.csv) ")

print(f"Bentuk data metadata: {df_meta.shape}")
print("Example data Metadata: ")
df_meta.head()


#Load Data Annotations
pattern_subfol = os.path.join(annotations, "*/")
list_subfol = glob.glob(pattern_subfol)

print(f"Found {len(list_subfol)} Sub-Folder Annotations: ")

all_files = []
for folder in list_subfol:
    print(f" - {folder.split('/')[-2]}")


print("\nTake 3 random files from each subfolder.")
for subfol in list_subfol:
    pattern_parquet = os.path.join(subfol, "*.parquet")
    parquet_files = glob.glob(pattern_parquet)

    if len(parquet_files) > 0:
        n_files = min(len(parquet_files), 3)
        sample_files = random.sample(parquet_files, n_files)
        all_files.extend(sample_files)
        print(f" - Take {len(sample_files)} files from {subfol.split('/')[-2]} skip")
    else:
        print(f" - There are no .parquet files in {subfol.splt('/')[-2]}, skip")

print(f"\nTotal files to be loaded from all folders: {len(all_files)}")

list_dfs = []
for f in all_files:
    # list_dfs.append(pd.read_parquet(f))
    df_temp = pd.read_parquet(f)

    name_file = os.path.basename(f)
    video_id = os.path.splitext(name_file)[0]
    df_temp[group] = video_id
    list_dfs.append(df_temp)

if list_dfs:
    df_labels = pd.concat(list_dfs, ignore_index=True)

    print(f"\nCombined annotation data format (target): {df_labels.shape}")
    print("Example data annotations (target) after add 'video_id': ")
    df_labels.head()

    print("\nDistribusi 'action' (target) kita (versi teks): ")
    print(df_labels[target].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
else:
    print("\nERROR: No annotation data loaded")
    print("Check whether the annotation folder contains Parquet files.")


df_process = df_labels.copy()

print("\nCombining metadata (df_meta) into the label data.")

if group not in df_process.columns:
    print(f"ERROR: Column ‘{group}’ (‘video_id’) not found in label data")
    print("Unable to merge metadata. Check your parquet file.")
else:
    meta_features = [
        group,
        'frames_per_second',
        'arena_type',
        'mouse1_sex',
        'arena_shape'
    ]
    df_meta_subset = df_meta[meta_features].drop_duplicates().copy()

    df_process[group] = df_process[group].astype(str)
    df_meta_subset[group] = df_meta_subset[group].astype(str)
    
    df_process = pd.merge(
        df_process,
        df_meta_subset,
        on=group,
        how='left'
    )
    print('Merge Metadata Success')

df_process['duration_frames'] = df_process['stop_frame'] - df_process['start_frame']

#Normalisazi FPS
if 'frames_per_second' in df_process.columns:
    df_process['duration_seconds'] = df_process['duration_frames'] / df_process['frames_per_second']
else:
    print("Warning: The ‘frames_per_second’ column was not found. ‘duration_seconds’ was not created.")

#CHECK NaN
nan_counts = df_process.isnull().sum()
nan_counts_report = nan_counts[nan_counts > 0]
total_nan = nan_counts.sum()
print(f"Total jumlah NaN yang ditemukan: {total_nan}")

if len(nan_counts_report) > 0:
    print("NaN FOUND IN COLUMNS: ")
    print(nan_counts_report)
else:
    print("Check Success: NaN Not Found")

#Missing Values
if 'target_id' in df_process.columns and df_process['target_id'].isnull().any():
    df_process['target_id'] = df_process['target_id'].fillna('None')
    print("NaN in 'target_id' filled with 'None'.")

#Duratioin Seconds
if 'duration_seconds' in df_process.columns and df_process['duration_seconds'].isnull().any():
    median_duration = df_process['duration_seconds'].median()
    df_process['duration_seconds'] = df_process['duration_seconds'].fillna(median_duration)
    print(f"NaN in 'duration_seconds' filled with the median: {median_duration:.2f}")


for col in ['arena_type', 'mouse1_sex', 'arena_shape']:
    if col in df_process.columns and df_process[col].isnull().any():
        mode_value = df_process[col].mode()[0]
        df_process[col] = df_process[col].fillna(mode_value)
        print(f"NaN in '{col}' filled with modus: {mode_value}")

#ENCODING
encoders = {}

#Categori (X)
categorical_features = ['agent_id', 'target_id', 'arena_type', 'mouse1_sex', 'arena_shape']

for col in categorical_features:
    if col in df_process.columns:
        le = LabelEncoder()
        df_process[f'{col}_encoded'] = le.fit_transform(df_process[col])
        encoders[col] = le
        print(f"Columns '{col}' in Encode.")

#Target (Y)
le_action = LabelEncoder()
df_process[f'{target}_encoded'] = le_action.fit_transform(df_process[target])
encoders[target] = le_action
print(f"Column Target '{target}_encoded' in Encode ")

df_process.head()


feature_columns = [
    'start_frame',
    'stop_frame',
    'duration_frames',
    'duration_seconds',
    'agent_id_encoded',
    'target_id_encoded',
    'arena_type_encoded',
    'mouse1_sex_encoded',
    'arena_shape_encoded'
]

final_features = [col for col in feature_columns if col in df_process.columns]

if len(final_features) != len(feature_columns):
    print("Warning: Some feature columns are not found in df_processed ")

print(f"Features to be used (X): {final_features}")

#separate X, Y, and Groups
x = df_process[final_features]
y = df_process[f"{target}_encoded"]
groups = df_process[group]

print(f"\nX Shape (feature): {x.shape}")
print(f"Y Shape (target): {y.shape}")
print(f"Number of unique groups{group}: {groups.nunique()}")

if x.isnull().values.any():
    print("\nWARNING: There are still NaN values in the feature data (X)!.")
    print(x.isnull().sum())
else:
    print("\nCHCEK SUCCESS: There are no NaN values in the feature data (X).")

N_Splits = 5
gkf = GroupKFold(n_splits=N_Splits)

#VERIFIKASI DATA (GROUP K-FOLD)
for fold, (train_idx, val_idx) in enumerate(gkf.split(x, y, groups)):
    print(f"\n=== FOLD {fold+1} ===")
    print(f"Amount of training data: {len(train_idx)}")
    print(f"Amount of validation data: {len(val_idx)}")

    train_videos = groups.iloc[train_idx].unique()
    val_videos = groups.iloc[val_idx].unique()

    overlap = np.intersect1d(train_videos, val_videos) #Search Irisan

    if len(overlap) == 0:
        print(f"CHECK SUCCESS: There are no overlapping '{group}' in fold {fold+1}.")
    else:
        print(f"ERROR: '{GROUP_COL}' tumpang tindih! Cek kembali logika grup.")

    break

