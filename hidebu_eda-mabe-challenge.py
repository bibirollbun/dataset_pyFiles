# ==================
# import libraries
# ==================
import os, gc, glob, pickle, warnings
import random, math, time
import joblib, pickle, itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy as sp
import polars as pl
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


# ===============
# utils
# ===============
def sep(word, num=80):
    print("="*num); print(word); print("="*80)

def show_df(df, num=3, showtail=False):
    print(df.shape)
    display(df.head(num))
    if showtail:
        display(df.tail(num))

def glob_walk(root: Path, glob_str: str) -> list:
    path = Path(root)
    walker = sorted(list(path.glob(glob_str)))
    return walker

def seed_everything(seed, GPU=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if GPU:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
seed_everything(0)


class PATHS:
    base_dir = Path("/kaggle/input/MABe-mouse-behavior-detection")
    train_tracking_dir = Path("/kaggle/input/MABe-mouse-behavior-detection/train_tracking")
    train_annotation_dir = Path("/kaggle/input/MABe-mouse-behavior-detection/train_annotation")

print(glob_walk(PATHS.base_dir, "*"))


# ======================
# Raed data(submission)
# ======================
sub_df = pl.read_csv(PATHS.base_dir/"sample_submission.csv")
sep("sample_submission.csv"); show_df(sub_df, 3)


# ======================
# Raed data(test)
# ======================
test_df = pl.read_csv(PATHS.base_dir/"test.csv")
sep("test.csv"); show_df(test_df, 3)


# ======================
# Raed data
# ======================
train_df = pl.read_csv(PATHS.base_dir/"train.csv")
sep("train.csv"); show_df(train_df, 3), display(train_df.describe())


# =================
# check columns
# =================
sub_cols   = list(sub_df.columns)
test_cols  = list(test_df.columns)
train_cols = list(train_df.columns)

sep("train_cols")
print(train_cols)

sep("cols in train_cols which is not in test_cols")
print([col for col in train_cols if col not in test_cols])

sep("cols in sub_cols which is not in train_cols")
print([col for col in sub_cols if col not in train_cols])


# =================
# lab_id
# =================
unique_labid = list(train_df["lab_id"].unique())

sep("lab_id info")
print(f"n_unique (lab_id): {len(unique_labid)}")
print(unique_labid)


# =====================
# Count per lab_id
# =====================
labid_counts = (
    train_df
    .group_by("lab_id")
    .len()
    .sort("len", descending=True)
)
x = labid_counts["lab_id"].to_list()
y = labid_counts["len"].to_list()

plt.figure(figsize=(12, 5))
bars = plt.bar(x, y)
for bar, count in zip(bars, y):
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        bar.get_height(),                  
        str(count),                        
        ha="center", va="bottom", fontsize=8, rotation=0
    )

plt.xticks(rotation=45)
plt.xlabel("lab_id")
plt.ylabel("count")
plt.title("Lab ID count")
plt.tight_layout()
plt.show()


# =================
# video_id
# =================
unique_videoid = list(train_df["video_id"].unique())

sep("video_id info")
print(f"n_unique (video_id): {len(unique_videoid)}/{len(train_df)}")
print(unique_videoid[:3])


# ==================
# mouse1_strain
# ==================











train_tracking_dirs = glob_walk(PATHS.train_tracking_dir, "*")
print(f"n_train_tracking_dirs/n_unique_labid): {len(train_tracking_dirs)}/{len(unique_labid)}")


# Example
sample = pl.read_parquet("/kaggle/input/MABe-mouse-behavior-detection/train_tracking/MABe22_movies/1000942438.parquet")
sample


sample.filter(pl.col("video_frame")==0)


n_label_track_dict = defaultdict()

for _dir in train_tracking_dirs:
    label_id = str(os.path.basename(_dir))
    _paths = glob_walk(_dir, "*.parquet")
    n_label_track_dict[label_id] = len(_paths)

sorted_dict = dict(sorted(n_label_track_dict.items(), key=lambda x: x[1], reverse=True))
x = list(sorted_dict.keys())
y = list(sorted_dict.values())

plt.figure(figsize=(12, 5))
bars = plt.bar(x, y)
for bar, count in zip(bars, y):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center", va="bottom", fontsize=8
    )

plt.xticks(rotation=45)
plt.xlabel("label_id")
plt.ylabel("count")
plt.title("Count per label_id")
plt.tight_layout()
plt.show()


tracking_cols = list(sample.columns)
print(f"tracking_cols: {tracking_cols}")

length_dict_tracking = defaultdict(list)
for _dir in train_tracking_dirs:
    label_id = str(os.path.basename(_dir))
    _paths = glob_walk(_dir, "*.parquet")
    for _path in tqdm(_paths, desc=label_id, total=len(_paths)):
        _df = pl.read_parquet(_path)
        assert list(_df.columns) == tracking_cols, f"Columns mismatch in {_path}"
        length_dict_tracking[label_id].append(len(_df))

# Convert to DataFrame（label_id, length）
rows = [{"label_id": k, "length": L} 
        for k, lst in length_dict_tracking.items() 
        for L in lst]
len_df = pd.DataFrame(rows)

# Boxplot
data = [len_df.loc[len_df["label_id"] == lab, "length"].values for lab in x]

plt.figure(figsize=(14, 6))
plt.boxplot(
    data,
    labels=x, 
    showfliers=False, 
    vert=True
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("label_id")
plt.ylabel("length")
plt.title("Sequence length distribution per label_id")
plt.tight_layout()
plt.show()


# MABe22_keypoints
print(len(data[0]))
data[0]


# MABe22_movies
print(len(data[1]))
data[1]


# AdaptableShall
print(len(data[-6]))
data[-6]


# VoisterousParrot
print(len(data[-1]))
data[-1]


train_annotation_dirs = glob_walk(PATHS.train_annotation_dir, "*")
print(f"n_train_annotation_dirs/n_unique_labid): {len(train_annotation_dirs)}/{len(unique_labid)}")
print()
sep("not involved labelid in train_annotation_dirs")
train_annotation_labelid = [str(os.path.basename(col)) for col in train_annotation_dirs]
print([col for col in unique_labid if col not in train_annotation_labelid])


# Example
sample = pl.read_parquet("/kaggle/input/MABe-mouse-behavior-detection/train_annotation/CalMS21_supplemental/1006083669.parquet")
sample


n_label_anno_dict = defaultdict()

for _dir in train_annotation_dirs:
    label_id = str(os.path.basename(_dir))
    _paths = glob_walk(_dir, "*.parquet")
    n_label_anno_dict[label_id] = len(_paths)

sorted_dict = dict(sorted(n_label_anno_dict.items(), key=lambda x: x[1], reverse=True))
x = list(sorted_dict.keys())
y = list(sorted_dict.values())

plt.figure(figsize=(12, 5))
bars = plt.bar(x, y)
for bar, count in zip(bars, y):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center", va="bottom", fontsize=8
    )

plt.xticks(rotation=45)
plt.xlabel("label_id")
plt.ylabel("count")
plt.title("Count per label_id")
plt.tight_layout()
plt.show()


anno_cols = list(_df.columns)
print(f"annotation_cols: {anno_cols}")

length_dict_anno = defaultdict(list)
for _dir in train_annotation_dirs:
    label_id = str(os.path.basename(_dir))
    _paths = glob_walk(_dir, "*.parquet")
    for i, _path in enumerate(tqdm(_paths, desc=label_id, total=len(_paths))):
        _df = pl.read_parquet(_path)
        length_dict_anno[label_id].append(len(_df))

# Convert to DataFrame（label_id, length）
rows = [{"label_id": k, "length": L} 
        for k, lst in length_dict_anno.items() 
        for L in lst]
len_df = pd.DataFrame(rows)

# Boxplot
data = [len_df.loc[len_df["label_id"] == lab, "length"].values for lab in x]

plt.figure(figsize=(14, 6))
plt.boxplot(
    data,
    labels=x, 
    showfliers=False, 
    vert=True
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("label_id")
plt.ylabel("length")
plt.title("Sequence length distribution per label_id")
plt.tight_layout()
plt.show()










