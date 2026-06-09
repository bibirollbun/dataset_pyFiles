import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import pydicom



# Kaggle input dataset 
DATA_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# Output 
# OUT_DIR = "/kaggle/working/processed_volumes"
OUT_25D = "/kaggle/working/processed_25d"

# os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_25D, exist_ok=True)



series_df = pd.read_csv(
    os.path.join(DATA_DIR, "train_series_descriptions.csv")
)
series_df = series_df[
    series_df.series_description == "Sagittal T2/STIR"
].reset_index(drop=True)

print("Sagittal T2/STIR series:", len(series_df))



def normalize_volume(vol):
    p1, p99 = np.percentile(vol, (1, 99))
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-6)
    return vol#.astype(np.float32)


def fix_depth(vol, target_depth=32):
    d = vol.shape[0]
    if d >= target_depth:
        idx = np.linspace(0, d - 1, target_depth).astype(int)
        return vol[idx]
    pad = target_depth - d
    return np.pad(vol, ((0, pad), (0, 0), (0, 0)), mode="edge")



def make_25d_windows(vol, k=3, max_windows=8):

    half = k // 2

    wins = []
    for i in range(half, vol.shape[0] - half):
        wins.append(vol[i - half:i + half + 1])

    wins = np.stack(wins, axis=0)

    if len(wins) > max_windows:
        idx = np.linspace(0, len(wins) - 1, max_windows).astype(int)
        wins = wins[idx]

    return wins#.astype(np.float32)



def process_series_to_25d(series_path, img_size=256):
    files = glob(os.path.join(series_path, "*.dcm"))
    if len(files) == 0:
        return None

    files = sorted(
        files,
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )

    imgs = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)

            img = cv2.resize(
                img, (img_size, img_size),
                interpolation=cv2.INTER_AREA
            )

            imgs.append(img)

        except Exception:
            continue

    if len(imgs) < 5:
        return None

    vol = np.stack(imgs, axis=0)        
    vol = normalize_volume(vol)
    vol = fix_depth(vol, 32)

    wins = make_25d_windows(vol)       
    return wins.astype(np.float16)     



rows = []

for _, r in tqdm(series_df.iterrows(), total=len(series_df)):
    series_path = os.path.join(
        DATA_DIR,
        "train_images",
        str(int(r.study_id)),
        str(int(r.series_id))
    )

    wins = process_series_to_25d(series_path)
    if wins is None:
        continue

    out_path = os.path.join(
        OUT_25D,
        f"{int(r.study_id)}_{int(r.series_id)}.npy"
    )
    np.save(out_path, wins)

    rows.append({
        "study_id": int(r.study_id),
        "series_id": int(r.series_id),
        "series_description": r.series_description,
        "path_25d": out_path
    })



df_25d = pd.DataFrame(rows)
df_25d.to_csv(
    os.path.join(OUT_25D, "series_25d.csv"),
    index=False
)

print("Saved 2.5D samples:", len(df_25d))
df_25d.head()



train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

series_25d = pd.read_csv(os.path.join(OUT_25D, "series_25d.csv"))

# merge label vào series_25d theo study_id
dataset = series_25d.merge(train_df, on="study_id", how="left")

# sanity: check label missing
missing = dataset.filter(like="stenosis").isna().any(axis=1).sum()
print("Rows with any missing labels:", missing)

dataset.to_csv(os.path.join(OUT_25D, "dataset.csv"), index=False)
print("Saved final dataset:", len(dataset))
dataset.head()



dataset = dataset.dropna().reset_index(drop=True)
print("Final dataset size:", len(dataset))



df = pd.read_csv(os.path.join(OUT_25D, "dataset.csv"))
x = np.load(df.iloc[0].path_25d)

print("X shape:", x.shape, "dtype:", x.dtype)  
print("First row study_id:", df.iloc[0].study_id)
print("Label columns example:", [c for c in df.columns if "stenosis" in c.lower()][:10])



LABEL_MAP = {
    "Normal/Mild": 0,
    "Moderate": 1,
    "Severe": 2
}

meta_cols = [
    "study_id",
    "series_id",
    "series_description",
    "path_25d"
]

label_cols = [c for c in dataset.columns if c not in meta_cols]

def parse_label_column(col):
    """
    Expected patterns:
    spinal_canal_stenosis_L4_L5
    left_neural_foraminal_stenosis_L5_S1
    right_subarticular_stenosis_L3_L4
    """
    parts = col.split("_")

    if parts[0] == "spinal":
        side = "spinal"
        task = "canal"
        level = "_".join(parts[-2:])

    else:
        side = parts[0]                 
        task = parts[1]                 
        level = "_".join(parts[-2:])     

    return side, task, level


records = []

for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
    npy_path = row["path_25d"]

    for col in label_cols:
        label_str = row[col]

        if pd.isna(label_str):
            continue
        if label_str not in LABEL_MAP:
            continue

        side, task, level = parse_label_column(col)

        records.append({
            "npy_path": npy_path,
            "level": level,
            "side": side,
            "task": task,
            "label": LABEL_MAP[label_str]
        })

train_df = pd.DataFrame(records)

TRAIN_DF_PATH = os.path.join(OUT_25D, "train_df.csv")
train_df.to_csv(TRAIN_DF_PATH, index=False)

print("Saved:", TRAIN_DF_PATH)
print("Total training samples:", len(train_df))
train_df.head()





