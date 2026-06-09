# Core
import os, sys, json, glob, math, random, textwrap, zipfile, gc
from pathlib import Path
from collections import Counter, defaultdict

# Data
import numpy as np
import pandas as pd

# Imaging
from PIL import Image, ImageStat, ImageOps
import cv2

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# ML helpers
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

# Pretty plotting defaults
plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")

# Paths (Kaggle)
INPUT = Path("/kaggle/input")
WORK  = Path("/kaggle/working")
ROOTS = [p for p in INPUT.glob("*cassava*")] or [INPUT]  # be resilient to dataset aliasing
ROOTS



def find_first(*patterns):
    for base in ROOTS:
        for pat in patterns:
            hits = list(base.glob(pat))
            if hits:
                return hits[0]
    return None

# Unzip helper (if images are packed)
def unzip_to_work(zpath: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(outdir)

# Locate canonical files/folders (covers both zipped/unzipped layouts)
csv_path  = find_first("**/train.csv")
json_path = find_first("**/label_num_to_disease_map.json")
train_dir = find_first("**/train_images")  # typical competition layout
test_dir  = find_first("**/test_images")

# If zipped images, unzip to /kaggle/working
if train_dir is None:
    z_train = find_first("**/train_images.zip")
    if z_train: 
        train_dir = WORK/"train_images"
        unzip_to_work(z_train, WORK)
if test_dir is None:
    z_test = find_first("**/test_images.zip")
    if z_test:
        test_dir = WORK/"test_images"
        unzip_to_work(z_test, WORK)

print("csv:", csv_path)
print("json:", json_path)
print("train_dir:", train_dir)
print("test_dir:", test_dir)

# Load metadata
df = pd.read_csv(csv_path)
with open(json_path) as f:
    label_map = json.load(f)

inv_label_map = {v:k for k,v in label_map.items()}  # names -> ids
df["label_name"] = df["label"].map(label_map)

display(df.head())
print("n_train:", len(df), "n_classes:", df['label'].nunique(), "label_map:", label_map)



# Ensure label_name exists robustly
# (handles label_map keys as str or int)
if list(label_map.keys()) and isinstance(next(iter(label_map.keys())), str):
    df["label_name"] = df["label"].map(lambda i: label_map[str(i)])
else:
    df["label_name"] = df["label"].map(label_map)

# Basic sanity
assert train_dir is not None and train_dir.exists(), "Missing train_images"
missing_files = [im for im in df.image_id if not (train_dir/im).exists()]
print("Missing files:", len(missing_files))

# Class balance â€” FIXED
cls_counts = df['label_name'].value_counts()
cls_counts = cls_counts.sort_index(key=lambda idx: idx.map(inv_label_map))  # <â€” use idx, not idx.index

plt.figure(figsize=(7,3))
sns.barplot(x=cls_counts.index, y=cls_counts.values)
plt.title("Class Distribution (train)")
plt.xticks(rotation=20)
plt.ylabel("count"); plt.xlabel("")
plt.show()

for k, v in cls_counts.items():
    print(f"{k:30s} : {v:5d} ({100*v/len(df):.1f}%)")



def show_grid(rows=2, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.4, rows*2.4))
    axes = axes.flatten()
    for i, cls in enumerate(sorted(label_map.values(), key=lambda x: inv_label_map[x])[:rows*cols]):
        sample = df[df.label_name==cls].sample(1).iloc[0]
        img = Image.open(train_dir/sample.image_id).convert("RGB")
        axes[i].imshow(img); axes[i].set_title(cls, fontsize=9); axes[i].axis("off")
    plt.tight_layout(); plt.show()

show_grid()



def image_stats(path: Path):
    # returns dict of width, height, size_kb, brightness, blur, aspect
    with Image.open(path) as im:
        w, h = im.size
        gray = ImageOps.grayscale(im)
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]  # 0..255
        # Blur via Laplacian variance
        gray_cv = np.array(gray)
        blur = cv2.Laplacian(gray_cv, cv2.CV_64F).var()
    size_kb = path.stat().st_size/1024
    aspect = w/h
    return dict(w=w, h=h, size_kb=size_kb, brightness=brightness, blur=blur, aspect=aspect)

# Sample to keep runtime reasonable (use all if you want full audit)
SAMPLE_N = min(4000, len(df))
sample_ids = df.sample(SAMPLE_N, random_state=42).image_id.values

stats = []
for iid in sample_ids:
    stats.append(image_stats(train_dir/iid))
stats_df = pd.DataFrame(stats)

fig, ax = plt.subplots(1,3, figsize=(12,3))
sns.histplot(stats_df['w'], ax=ax[0]); ax[0].set_title("Width")
sns.histplot(stats_df['h'], ax=ax[1]); ax[1].set_title("Height")
sns.histplot(stats_df['brightness'], ax=ax[2]); ax[2].set_title("Brightness")
plt.show()

plt.figure(figsize=(4,3))
sns.histplot(stats_df['blur'])
plt.title("Laplacian Variance (Blur ~ low = blurrier)")
plt.show()

stats_df.describe().T



from collections import Counter

def get_exif_model(path: Path):
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            # 272 (0x110) = Model; 271 = Make
            model = ex.get(272, None)
            make  = ex.get(271, None)
            return f"{make} {model}".strip() if (make or model) else None
    except Exception:
        return None

device_counts = Counter()
for iid in random.sample(list(sample_ids), min(1000, len(sample_ids))):
    device = get_exif_model(train_dir/iid)
    if device: device_counts[device]+=1

device_counts.most_common(10)



# Lightweight pHash (no external deps)
def phash(im: Image.Image, hash_size=8, highfreq_fact=4):
    # https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    img = ImageOps.grayscale(im).resize((hash_size*highfreq_fact, hash_size*highfreq_fact), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)
    dct = cv2.dct(pixels)
    dctlow = dct[:hash_size, :hash_size]
    med = np.median(dctlow)
    diff = dctlow > med
    return "".join("1" if v else "0" for v in diff.flatten())

def hamming(a,b): return sum(c1!=c2 for c1,c2 in zip(a,b))

# Compute hashes for a (stratified) sample
N_HASH = min(5000, len(df))
ids_to_hash = df.groupby("label_name", group_keys=False).apply(lambda g: g.sample(min(len(g), math.ceil(N_HASH/df['label_name'].nunique())), random_state=0)).image_id.tolist()

hashes = {}
for iid in ids_to_hash:
    with Image.open(train_dir/iid) as im:
        hashes[iid] = phash(im)

# Find near-duplicates (Hamming distance <= 5 as heuristic)
pairs = []
ids = list(hashes.keys())
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        d = hamming(hashes[ids[i]], hashes[ids[j]])
        if d <= 5:
            pairs.append((ids[i], ids[j], d))
len(pairs), pairs[:5]



# Build grouping by hash (exact or near-dup clusters). For simplicity, use exact hash groups here.
hash_to_group = {}
for iid in df.image_id:
    hp = None
    try:
        with Image.open(train_dir/iid) as im:
            hp = phash(im)
    except:
        pass
    hash_to_group[iid] = hp or iid  # fallback: unique

df["group"] = df["image_id"].map(hash_to_group)

# GroupKFold split blueprint (store folds to CSV for modeling notebook)
gkf = GroupKFold(n_splits=5)
df["fold"] = -1
for fold, (tr, va) in enumerate(gkf.split(df, y=df["label"], groups=df["group"])):
    df.loc[df.index[va], "fold"] = fold

df.fold.value_counts().sort_index(), df.head()
df.to_csv(WORK/"folds_groupkfold.csv", index=False)


