# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ============================================================
# RSNA Intracranial Aneurysm Detection: Geometry metrics from
# COW NIfTI segmentations + stats by named artery
# ============================================================

# ---------- Cell 0: Install lightweight deps ----------
# (Kaggle usually has numpy/pandas/scipy; nibabel/tqdm added here)
!pip -q install nibabel tqdm

# ---------- Cell 1: Imports & basic setup ----------
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from IPython.display import display, clear_output

# Competition root and standard files
COMP_DIR   = "/kaggle/input/rsna-intracranial-aneurysm-detection"
TRAIN_CSV  = os.path.join(COMP_DIR, "train.csv")   # Adjust if your train file lives elsewhere

# Where to write outputs (persist with notebook versioning)
WORK_DIR   = "/kaggle/working"
OUT_DIR    = os.path.join(WORK_DIR, "aneurysm_metrics_fast")
OUT_CSV    = os.path.join(OUT_DIR, "segment_metrics.csv")

# ------------- Segmentations -------------
# We auto-discover any *_cowseg.nii* anywhere under /kaggle/input.
# If you prefer a fixed folder, set SEG_ROOT and use FILE_GLOB under it.
SEG_ROOT   = "/kaggle/input"           # search base
FILE_GLOB  = "**/*_cowseg.nii*"        # pattern for labeled segmentations

# ------------- Metrics controls -------------
MIN_VOL_MM3 = 5.0     # ignore tiny blobs
SHOW_EVERY  = 1       # how often to refresh console & preview df

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Cell 2: NIfTI I/O and fast surface area ----------
def nii_read(path):
    """
    Load a NIfTI and return (data_uint8, spacing_zyx_mm)
    """
    nii = nib.load(path)
    data = np.asarray(nii.get_fdata(), dtype=np.uint8)
    spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])  # (z, y, x) in mm
    return data, spacing

def fast_surface_area(binary, spacing):
    """
    6-neighborhood exposed-face count * face areas.
    Approximates surface area quickly without marching cubes.
    """
    if binary.dtype != bool:
        binary = binary.astype(bool)
    dz, dy, dx = spacing
    ax = dy * dz  # faces normal to X
    ay = dx * dz  # faces normal to Y
    az = dx * dy  # faces normal to Z

    exposed = 0.0
    # +X
    nb = np.zeros_like(binary); nb[..., :-1] = binary[..., 1:]
    exposed += ax * np.sum(binary & ~nb)
    # -X
    nb = np.zeros_like(binary); nb[..., 1:] = binary[..., :-1]
    exposed += ax * np.sum(binary & ~nb)
    # +Y
    nb = np.zeros_like(binary); nb[:, :-1, :] = binary[:, 1:, :]
    exposed += ay * np.sum(binary & ~nb)
    # -Y
    nb = np.zeros_like(binary); nb[:, 1:, :] = binary[:, :-1, :]
    exposed += ay * np.sum(binary & ~nb)
    # +Z
    nb = np.zeros_like(binary); nb[:-1, :, :] = binary[1:, :, :]
    exposed += az * np.sum(binary & ~nb)
    # -Z
    nb = np.zeros_like(binary); nb[1:, :, :] = binary[:-1, :, :]
    exposed += az * np.sum(binary & ~nb)

    return float(exposed)

def compute_metrics_for_label(mask_bool, spacing):
    """
    Basic 3D metrics for a labeled component:
    - Volume (mm^3)
    - Surface area approx. (mm^2) via exposed faces
    - Max diameter approx. (mm): 2*max distance from centroid
    - Elongation (PCA): sqrt(lambda1/lambda3)
    - Sphericity: (pi^(1/3) * (6V)^(2/3)) / A
    """
    voxvol = spacing[0] * spacing[1] * spacing[2]
    nvox = int(mask_bool.sum())
    volume_mm3 = nvox * voxvol
    if nvox == 0:
        return dict(volume_mm3=0.0, surface_mm2=0.0, max_diameter_mm=0.0,
                    elongation=np.nan, sphericity=np.nan)

    area_mm2 = fast_surface_area(mask_bool, spacing)

    zz, yy, xx = np.where(mask_bool)
    coords_mm = np.vstack([zz*spacing[0], yy*spacing[1], xx*spacing[2]]).T

    center = coords_mm.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(coords_mm - center, axis=1)
    max_diameter_mm = float(dists.max() * 2.0) if dists.size else 0.0

    # PCA-based elongation
    if coords_mm.shape[0] >= 5:
        C = coords_mm - center
        cov = (C.T @ C) / max(len(C)-1, 1)
        w = np.linalg.eigvalsh(cov)
        w = np.sort(np.clip(w, 1e-12, None))[::-1]  # lambda1>=lambda2>=lambda3>0
        elongation = float(np.sqrt(w[0]/w[-1]))
    else:
        elongation = np.nan

    # Sphericity
    if area_mm2 > 0 and volume_mm3 > 0:
        sphericity = float((np.pi**(1/3.0)) * ((6.0*volume_mm3)**(2/3.0)) / area_mm2)
    else:
        sphericity = np.nan

    return dict(volume_mm3=float(volume_mm3),
                surface_mm2=float(area_mm2),
                max_diameter_mm=float(max_diameter_mm),
                elongation=elongation,
                sphericity=sphericity)

# ---------- Cell 3: Discover NIfTI segmentation files ----------
paths = sorted(glob.glob(os.path.join(SEG_ROOT, FILE_GLOB), recursive=True))
print(f"Found {len(paths)} volumes matching pattern: {FILE_GLOB}")

if len(paths) == 0:
    raise FileNotFoundError(
        "No NIfTI segmentations were found. "
        "Make sure you've added a dataset with files named like *_cowseg.nii.gz under /kaggle/input/.\n"
        "You can also change SEG_ROOT/FILE_GLOB near the top of this notebook."
    )

# ---------- Cell 4: Iterate segmentations → per-label metrics CSV ----------
if os.path.exists(OUT_CSV):
    df_all = pd.read_csv(OUT_CSV)
    processed = set((df_all["SeriesInstanceUID"].astype(str) + "_" + df_all["Label"].astype(str)).tolist())
else:
    df_all = pd.DataFrame(columns=[
        "SeriesInstanceUID","Label",
        "Volume_mm3","SurfaceArea_mm2","MaxDiameter_mm",
        "Elongation","Sphericity"
    ])
    processed = set()

for i, nii_path in enumerate(tqdm(paths, desc="Volumes", unit="vol")):
    seg, spacing = nii_read(nii_path)
    uid = Path(nii_path).stem.replace("_cowseg","")
    if uid == Path(nii_path).stem:
        # if suffix not present, fallback to folder name
        uid = Path(nii_path).parent.name

    labels = np.unique(seg)
    labels = labels[labels != 0]
    if labels.size == 0:
        continue

    new_rows = []
    for lbl in labels:
        key = f"{uid}_{int(lbl)}"
        if key in processed:
            continue
        mask = (seg == lbl)

        # quick volume filter
        n_vox = int(mask.sum())
        vol_mm3 = n_vox * spacing[0] * spacing[1] * spacing[2]
        if vol_mm3 < MIN_VOL_MM3:
            continue

        metrics = compute_metrics_for_label(mask, spacing)
        row = {
            "SeriesInstanceUID": uid,
            "Label": int(lbl),
            "Volume_mm3": metrics["volume_mm3"],
            "SurfaceArea_mm2": metrics["surface_mm2"],
            "MaxDiameter_mm": metrics["max_diameter_mm"],
            "Elongation": metrics["elongation"],
            "Sphericity": metrics["sphericity"],
        }
        new_rows.append(row)
        processed.add(key)

    if new_rows:
        df_all = pd.concat([df_all, pd.DataFrame(new_rows)], ignore_index=True)
        Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(OUT_CSV, index=False)

    if ((i+1) % max(1, SHOW_EVERY) == 0) or (i+1 == len(paths)):
        clear_output(wait=True)
        print(f"Processed {i+1}/{len(paths)} volumes. Last UID: {uid}")
        if not df_all.empty:
            display(df_all.tail(8))

print("Saved metrics CSV to:", OUT_CSV)

# ============================================================
# Stats by artery name (merge with train.csv)
# ============================================================

# ---------- Cell 5: Imports for stats ----------
from scipy import stats

# Output folder for stats
STATS_OUT_DIR = os.path.join(WORK_DIR, "segment_stats_out_by_name")
os.makedirs(STATS_OUT_DIR, exist_ok=True)

# Label map (edit if your label indices → artery names differ)
LABELS = {
    1: "Other Posterior Circulation",
    2: "Basilar Tip",
    3: "Right Posterior Communicating Artery",
    4: "Left Posterior Communicating Artery",
    5: "Right Infraclinoid Internal Carotid Artery",
    6: "Left Infraclinoid Internal Carotid Artery",
    7: "Right Supraclinoid Internal Carotid Artery",
    8: "Left Supraclinoid Internal Carotid Artery",
    9: "Right Middle Cerebral Artery",
    10: "Left Middle Cerebral Artery",
    11: "Right Anterior Cerebral Artery",
    12: "Left Anterior Cerebral Artery",
    13: "Anterior Communicating Artery",
}
FEATURES = ['Volume_mm3', 'SurfaceArea_mm2', 'MaxDiameter_mm', 'Elongation', 'Sphericity']

MIN_SAMPLES_PER_GROUP = 5  # per group, per-label
ALPHA = 0.05
EFFECT_MIN = 0.147  # small but noticeable rank-biserial correlation

# ---------- Cell 6: Load CSVs and harmonize UID column ----------
metrics = pd.read_csv(OUT_CSV)

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(
        f"Could not find train.csv at: {TRAIN_CSV}\n"
        "If your train CSV is elsewhere, update TRAIN_CSV near the top."
    )

train = pd.read_csv(TRAIN_CSV)

# Try to find a column equivalent to SeriesInstanceUID
uid_candidates = [
    "SeriesInstanceUID", "StudyInstanceUID", "StudyUID",
    "series_id", "study_id", "SeriesID", "StudyID"
]
uid_found = None
for c in uid_candidates:
    if c in train.columns:
        uid_found = c
        break

if uid_found is None:
    raise ValueError(
        "Could not find a UID column in train.csv. "
        f"Tried: {uid_candidates}. Please rename your UID column to 'SeriesInstanceUID' "
        "or extend uid_candidates above."
    )

if uid_found != "SeriesInstanceUID":
    train = train.rename(columns={uid_found: "SeriesInstanceUID"})

# Identify artery columns by exact name match to LABELS values
artery_cols = [name for name in LABELS.values() if name in train.columns]
if not artery_cols:
    raise ValueError(
        "No artery-name columns were found in train.csv.\n"
        "Expected headers like those in LABELS (e.g., 'Left Middle Cerebral Artery').\n"
        "Please align LABELS values to your train.csv column names."
    )

# Optionally include a global 'Aneurysm Present' if available
cols_to_merge = ['SeriesInstanceUID'] + artery_cols + ([c for c in ['Aneurysm Present'] if c in train.columns])

merged = metrics.merge(train[cols_to_merge], on='SeriesInstanceUID', how='left')
merged['LabelName'] = merged['Label'].map(LABELS)

# Per-row flag: aneurysm present in THIS labeled artery
def row_flag(row, frame_cols):
    ln = row['LabelName']
    if ln in frame_cols:
        val = row[ln]
        try:
            return int(val)
        except Exception:
            return 0
    return np.nan  # column missing — drop later

merged['Aneurysm_In_This_Label'] = merged.apply(lambda r: row_flag(r, merged.columns), axis=1).astype('float')

# Filter valid rows
valid = merged[merged['Aneurysm_In_This_Label'].notna()].copy()
valid['Aneurysm_In_This_Label'] = valid['Aneurysm_In_This_Label'].astype(int)

# ---------- Cell 7: Mann–Whitney utilities ----------
def rank_biserial_from_u(u_stat, n1, n0):
    return (u_stat / (n1 * n0) - 0.5) * 2

def mannwhitney_summary(group1, group0):
    g1 = np.asarray(pd.Series(group1).dropna().values, dtype=float)
    g0 = np.asarray(pd.Series(group0).dropna().values, dtype=float)
    if len(g1)==0 or len(g0)==0:
        return dict(n1=len(g1), n0=len(g0), median1=np.nan, median0=np.nan, U=np.nan, p=np.nan, rbc=np.nan)
    U, p = stats.mannwhitneyu(g1, g0, alternative='two-sided')
    rbc = rank_biserial_from_u(U, len(g1), len(g0))
    return dict(n1=len(g1), n0=len(g0), median1=np.median(g1), median0=np.median(g0), U=U, p=p, rbc=rbc)

# ---------- Cell 8: GLOBAL comparison (flag by specific LabelName) ----------
global_rows = []
for feat in FEATURES:
    res = mannwhitney_summary(
        valid.loc[valid['Aneurysm_In_This_Label']==1, feat],
        valid.loc[valid['Aneurysm_In_This_Label']==0, feat]
    )
    global_rows.append({
        'Feature': feat,
        'Median_Aneurysm(ThisLabel)': res['median1'],
        'Median_Normal(ThisLabel)': res['median0'],
        'U_stat': res['U'],
        'p_value': res['p'],
        'Rank_Biserial_Corr': res['rbc'],
        'N_Aneurysm(ThisLabel)': res['n1'],
        'N_Normal(ThisLabel)': res['n0'],
    })
global_df = pd.DataFrame(global_rows).sort_values('p_value')
global_df.to_csv(Path(STATS_OUT_DIR)/"global_by_labelname_stats.csv", index=False)

# ---------- Cell 9: Per-label (per artery id) comparison ----------
perlabel_rows = []
for lid, lname in LABELS.items():
    sub = valid[valid['Label'] == lid]
    if sub.empty:
        continue
    g1 = sub[sub['Aneurysm_In_This_Label']==1]
    g0 = sub[sub['Aneurysm_In_This_Label']==0]
    if len(g1) < MIN_SAMPLES_PER_GROUP or len(g0) < MIN_SAMPLES_PER_GROUP:
        continue
    for feat in FEATURES:
        res = mannwhitney_summary(g1[feat], g0[feat])
        perlabel_rows.append({
            'Label': lid,
            'LabelName': lname,
            'Feature': feat,
            'Median_Aneurysm(ThisLabel)': res['median1'],
            'Median_Normal(ThisLabel)': res['median0'],
            'U_stat': res['U'],
            'p_value': res['p'],
            'Rank_Biserial_Corr': res['rbc'],
            'N_Aneurysm(ThisLabel)': res['n1'],
            'N_Normal(ThisLabel)': res['n0'],
        })
perlabel_df = pd.DataFrame(perlabel_rows).sort_values(['Label','p_value'])
perlabel_df.to_csv(Path(STATS_OUT_DIR)/"perlabel_by_labelname_stats.csv", index=False)

# ---------- Cell 10: Significance filters & console summary ----------
def mark_sig(df):
    return (df['p_value'] < ALPHA) & (df['Rank_Biserial_Corr'].abs() >= EFFECT_MIN)

sig_global = global_df[mark_sig(global_df)].copy()
sig_perlabel = perlabel_df[mark_sig(perlabel_df)].copy()

sig_global.to_csv(Path(STATS_OUT_DIR)/"global_by_labelname_stats_significant.csv", index=False)
sig_perlabel.to_csv(Path(STATS_OUT_DIR)/"perlabel_by_labelname_stats_significant.csv", index=False)

print(">> GLOBAL by-artery (flag specific to LabelName):")
print(global_df.to_string(index=False))
print("\nSignificant (p < %.3f & |r| >= %.3f):" % (ALPHA, EFFECT_MIN))
print(sig_global.to_string(index=False) if not sig_global.empty else "(none)")

print("\n>> PER-LABEL (only labels with n≥%d per group):" % MIN_SAMPLES_PER_GROUP)
if perlabel_df.empty:
    print("(Not enough samples per label)")
else:
    print(perlabel_df.sort_values('p_value').head(12).to_string(index=False))
    print("\nSignificant (p < %.3f & |r| >= %.3f):" % (ALPHA, EFFECT_MIN))
    print(sig_perlabel.sort_values('p_value').to_string(index=False) if not sig_perlabel.empty else "(none)")

print("\nMetrics CSV:", Path(OUT_CSV).resolve())
print("Stats out dir:", Path(STATS_OUT_DIR).resolve())


