# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# === Kaggle dataset handles ===
DATASET_DIR = "/kaggle/input/grand-xray-slam-division-a"   
TRAIN_CSV   = f"{DATASET_DIR}/train1.csv"         
IMG_DIR     = f"{DATASET_DIR}/train1"     

# Outputs
OUT_DIR   = "/kaggle/working/eda_outputs"  # persisted between cells
SAMPLE_PER_LABEL = 12
SIZE_PROBE_N = 1000
DISPLAY_TOP_N = 30


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

pd.set_option("display.max_columns", 200)
plt.rcParams["figure.figsize"] = (8, 5)

# Ensure output dirs
out_dir = Path(OUT_DIR)
(out_dir / "plots").mkdir(parents=True, exist_ok=True)
(out_dir / "tables").mkdir(exist_ok=True)
(out_dir / "samples").mkdir(exist_ok=True)


PATHOLOGY_COLS = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema',
    'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
    'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
    'Pneumothorax','Support Devices'
]

META_COLS = [
    'Image_name','Patient_ID','Study','Sex','Age','ViewCategory','ViewPosition'
]

EXPECTED_COLS = META_COLS + PATHOLOGY_COLS


df = pd.read_csv(TRAIN_CSV)
print(f"Loaded {len(df)} rows from {TRAIN_CSV}")

# Build filepath if not present and IMG_DIR provided
if "filepath" not in df.columns and "Image_name" in df.columns and IMG_DIR:
    df["filepath"] = df["Image_name"].apply(lambda x: str(Path(IMG_DIR) / x))

df.head()


def save_table(df, name):
    p = Path(OUT_DIR) / "tables" / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"[saved] {p}")
    return p

def barplot_series(s, title, xlabel, ylabel, fname, rotate=45):
    ax = s.plot(kind='bar')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.0f}', label_type='edge', padding=1)
    plt.xticks(rotation=rotate, ha='right')
    plt.tight_layout()
    out = Path(OUT_DIR) / "plots" / f"{fname}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    plt.close()
    print(f"[saved] {out}")
    return out


missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
if missing_cols:
    print("[warn] Missing expected columns:", missing_cols)

info = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(t) for t in df.dtypes.values],
    "non_null": df.notnull().sum().values,
    "nulls": df.isnull().sum().values,
    "per_of_nulls":100*df.isnull().sum().values/df.shape[0],
    "unique": [df[c].nunique() for c in df.columns]
})
save_table(info, "00_dataframe_info")
info


df.Age.plot(kind = 'hist', bins = 30, title = "Age distribution")
plt.xlabel("Age (years)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(Path(OUT_DIR) / "plots" / "01_age_hist.png", dpi=150)
plt.show()


if "Sex" in df.columns:
    sex_counts = df["Sex"].value_counts(dropna=False)
    save_table(sex_counts, "02_sex_counts")
    barplot_series(sex_counts, "Sex distribution", "Sex", "Count", "02_sex_counts")

if "ViewPosition" in df.columns:
    vp_counts = df["ViewPosition"].value_counts(dropna=False)
    save_table(vp_counts, "03_viewposition_counts")
    barplot_series(vp_counts, "ViewPosition distribution", "ViewPosition", "Count", "03_viewposition_counts")

if "ViewCategory" in df.columns:
    vc_counts = df["ViewCategory"].value_counts(dropna=False)
    save_table(vc_counts, "04_viewcategory_counts")
    barplot_series(vc_counts, "ViewCategory distribution", "ViewCategory", "Count", "04_viewcategory_counts")



if "Patient_ID" in df.columns and "Image_name" in df.columns:
    imgs_per_patient = df.groupby("Patient_ID")["Image_name"].count().sort_values(ascending=False)
    save_table(imgs_per_patient.reset_index(name="images"), "05_images_per_patient")
    barplot_series(imgs_per_patient.head(DISPLAY_TOP_N), f"Images per patient (top {DISPLAY_TOP_N})", "Patient_ID", "Images", "05_images_per_patient_top")

if "Study" in df.columns and "Image_name" in df.columns:
    imgs_per_study = df.groupby(["Study", "Patient_ID"])["Image_name"].count().sort_values(ascending=False)
    save_table(imgs_per_study.reset_index(name="images"), "06_images_per_study")
    barplot_series(imgs_per_study.head(DISPLAY_TOP_N), f"Images per study (top {DISPLAY_TOP_N})", "Study", "Images", "06_images_per_study_top")


present_cols = [c for c in PATHOLOGY_COLS if c in df.columns]
label_sums = df[present_cols].fillna(0).sum().sort_values(ascending=False)
save_table(label_sums, "07_label_positives")
barplot_series(label_sums, "Label positives (class imbalance)", "Label", "Positive count", "07_label_positives")



if "No Finding" in present_cols:
    nf = df["No Finding"].fillna(0).astype(int)
    co_counts = {}
    for c in present_cols:
        if c == "No Finding":
            continue
        both = ((df[c].fillna(0).astype(int) == 1) & (nf == 1)).sum()
        co_counts[c] = both
    co_df = pd.Series(co_counts).sort_values(ascending=False).to_frame("co_occurrence_with_NoFinding")
    save_table(co_df.reset_index(names=["label","count"]), "08_no_finding_cooccurrence")
    barplot_series(co_df["co_occurrence_with_NoFinding"], '"No Finding" co-occurrence (should be near 0)', "Label", "Count", "08_no_finding_cooccurrence")


def heatmap(df, title, fname, annotate=False):
    plt.figure(figsize=(10, 8))
    plt.imshow(df.values, aspect='auto', interpolation='nearest')
    plt.xticks(range(df.shape[1]), df.columns, rotation=90)
    plt.yticks(range(df.shape[0]), df.index)
    plt.title(title)
    plt.colorbar()
    if annotate:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                plt.text(j, i, f"{df.iat[i,j]:.2f}", ha='center', va='center', fontsize=7)
    plt.tight_layout()
    out = Path(OUT_DIR) / "plots" / f"{fname}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    plt.close()
    print(f"[saved] {out}")
    return out


bin_df = df[present_cols].fillna(0).astype(int)
co_mat = pd.DataFrame(0, index=present_cols, columns=present_cols, dtype=int)
for a in present_cols:
    for b in present_cols:
        co_mat.loc[a, b] = int(((bin_df[a]==1) & (bin_df[b]==1)).sum())
save_table(co_mat.reset_index().rename(columns={"index":"label"}), "09_cooccurrence_counts")
heatmap(co_mat, "Label co-occurrence (counts)", "09_cooccurrence_counts")


if "ViewPosition" in df.columns:
    rates = (df.groupby("ViewPosition")[present_cols]
               .mean()
               .sort_index())
    save_table(rates.reset_index(), "11_label_rates_by_viewposition")
    heatmap(rates, "Label rates by ViewPosition", "11_label_rates_by_viewposition", annotate=True)


def image_sizes(img_paths, max_n=1000):
    sizes = []
    for i, p in enumerate(img_paths):
        if i >= max_n:
            break
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            continue
    return sizes


if "filepath" in df.columns and IMG_DIR:
    sizes = image_sizes(df["filepath"].values, max_n=SIZE_PROBE_N)
    if sizes:
        wh = pd.DataFrame(sizes, columns=["width","height"])
        save_table(wh.describe().reset_index(), "12_image_size_summary")
        plt.scatter(wh["width"], wh["height"], s=8, alpha=0.6)
        plt.title("Image sizes (sample)")
        plt.xlabel("Width"); plt.ylabel("Height"); plt.tight_layout()
        plt.savefig(Path(OUT_DIR) / "plots" / "12_image_sizes_scatter.png", dpi=150)
        plt.show()


label_sums


N = len(df)
pos = label_sums.reindex(present_cols).astype(float)  # align order
neg = N - pos

# Avoid div by zero
eps = 1e-9
pos_rate = (pos / (N + eps)).rename("pos_rate")
neg_rate = (neg / (N + eps)).rename("neg_rate")

# BCE class weights per label (common choice): w_pos = N/pos, w_neg = N/neg (or normalized variant)
w_pos = (N / (pos + eps)).rename("w_pos")
w_neg = (N / (neg + eps)).rename("w_neg")

# Focal loss alpha suggestion: alpha ~ neg_rate (penalize positives more when rare)
alpha = neg_rate.rename("alpha_focal_suggestion")
gamma = pd.Series(2.0, index=present_cols, name="gamma_default")  # typical

# "Effective number of samples" (Cui et al.) weights
beta = 0.999
eff_num = (1 - beta**pos) / (1 - beta)  # effective samples per class (positives)
w_effective = (1.0 / (eff_num + eps))
w_effective = (w_effective / w_effective.max()).rename("w_effective_norm")

weights_df = pd.concat([pos.astype(int).rename("positives"),
                        neg.astype(int).rename("negatives"),
                        pos_rate, neg_rate, w_pos, w_neg, alpha, gamma, w_effective], axis=1)
save_table(weights_df.reset_index().rename(columns={"index": "label"}), "13_class_weights")

display(weights_df)


print("EDA complete.")
print("Outputs:")
print(" - Tables:", Path(OUT_DIR) / "tables")
print(" - Plots: ", Path(OUT_DIR) / "plots")
print(" - Samples:", Path(OUT_DIR) / "samples")
print(" - Weights:", Path(OUT_DIR) / "tables" / "13_class_weights.csv")

