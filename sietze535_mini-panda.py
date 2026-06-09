import os, shutil
import pandas as pd
import numpy as np

# Config
N_PATIENTS = 1062      # Number of patients you want
N_TILES = 12           # Number of tiles per patient
TILE_DIR = "/kaggle/input/panda-16x128x128-tiles-data/train"
LABEL_CSV = "/kaggle/input/prostate-cancer-grade-assessment/train.csv"
OUT_DIR = "/kaggle/working/mini_dataset"

# Make output directory
os.makedirs(f"{OUT_DIR}/train", exist_ok=True)

# Load original labels
df = pd.read_csv(LABEL_CSV)

# Identify patients that have _0.png through _11.png
tile_files = set(os.listdir(TILE_DIR))
candidate_ids = []

for image_id in df['image_id']:
    has_all_tiles = all(f"{image_id}_{i}.png" in tile_files for i in range(N_TILES))
    if has_all_tiles:
        candidate_ids.append(image_id)

print(f"Found {len(candidate_ids)} patients with exactly {N_TILES} tiles.")

# Check if enough patients
if len(candidate_ids) < N_PATIENTS:
    print(f"Only {len(candidate_ids)} patients available. Reducing N_PATIENTS.")
    N_PATIENTS = len(candidate_ids)

# Sample the patients
sampled_ids = np.random.choice(candidate_ids, N_PATIENTS, replace=False)
df_small = df[df['image_id'].isin(sampled_ids)].reset_index(drop=True)

# Save CSV
df_small.to_csv(f"{OUT_DIR}/train.csv", index=False)

# Copy exactly 12 tiles for each patient
tiles_copied = 0
for pid in sampled_ids:
    for i in range(N_TILES):
        src = os.path.join(TILE_DIR, f"{pid}_{i}.png")
        dst = os.path.join(OUT_DIR, "train", f"{pid}_{i}.png")
        shutil.copy(src, dst)
        tiles_copied += 1

print(f"Copied {tiles_copied} tiles for {N_PATIENTS} patients (each with {N_TILES} tiles).")
print("Mini dataset is ready at:", OUT_DIR)



!head /kaggle/working/mini_dataset/train.csv
!ls /kaggle/working/mini_dataset/train | head



tile_counts = pd.Series([f[:32] for f in os.listdir(TILE_DIR)]).value_counts()
print(tile_counts.describe())
print(tile_counts.value_counts().sort_index())



df = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
print(len(df))


