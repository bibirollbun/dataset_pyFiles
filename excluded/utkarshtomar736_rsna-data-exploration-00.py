import os, glob, json, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pydicom
from pydicom.filereader import dcmread

import nibabel as nib


pd.set_option("display.max_columns", 100)
sns.set_theme(context="notebook", style="whitegrid")
DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection"


train = pd.read_csv(f"{DATA_DIR}/train.csv")
locs = pd.read_csv(f"{DATA_DIR}/train_localizers.csv")

print(train.shape)
print(locs.shape)
train.head(3)


# Identify label columns (13 locations + presence)
location_cols = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
]
presence_col = "Aneurysm Present"
meta_cols = ["SeriesInstanceUID", "Modality", "PatientAge", "PatientSex"]
assert set(location_cols).issubset(train.columns)
assert presence_col in train.columns


series_root = f"{DATA_DIR}/series"
seg_root = f"{DATA_DIR}/segmentations"

series_dirs = set(os.listdir(series_root))
has_series = train["SeriesInstanceUID"].isin(series_dirs)
coverage = has_series.mean()

loc_per_series = locs.groupby("SeriesInstanceUID").size().rename("n_localizers")
train = train.merge(loc_per_series, on="SeriesInstanceUID", how="left").fillna({"n_localizers": 0})

# Check what fraction have segmentations
seg_series = set([f.replace(".nii.gz","").replace(".nii","") for f in os.listdir(seg_root)])
train["has_seg"] = train["SeriesInstanceUID"].isin(seg_series)

coverage, train["has_seg"].mean(), train["n_localizers"].gt(0).mean()


any_location = train[location_cols].sum(axis=1).gt(0).astype(int)
train["presence_from_locations"] = any_location
consistency = (train[presence_col].astype(int) == train["presence_from_locations"]).mean()
incons = train.loc[train[presence_col].astype(int) != train["presence_from_locations"], 
                   meta_cols + [presence_col, "presence_from_locations"]].head(10)
consistency, incons


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4))

# 1) Presence count
sns.countplot(data=train, x=presence_col, ax=ax0)
ax0.set_title("Aneurysm Present")
ax0.set_xlabel("")
ax0.set_ylabel("Count")

# 2) Location positive rates
pos_rates = train[location_cols].mean().sort_values(ascending=True)  # ascending so lowest at bottom
pos_rates.plot(kind="barh", ax=ax1)  # horizontal
ax1.set_title("Location positive rates")
ax1.set_xlabel("Rate")
ax1.set_ylabel("")  # labels are on y as categories
ax1.set_xlim(0, 1)  # rates within [0,1]

# 3) Modality distribution
order = train["Modality"].value_counts().index
sns.countplot(data=train, x="Modality", ax=ax2, order=order)
ax2.set_title("Modality distribution")
ax2.set_xlabel("")
ax2.set_ylabel("Count")
ax2.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()


# Demographics
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left subplot: sex counts
sns.countplot(data=train, x="PatientSex", ax=axes[0])  # Fixed: axes[0]
axes[0].set_title("Sex")                              # Fixed: axes[0]
axes[0].set_xlabel("")                                # Fixed: axes[0]
axes[0].set_ylabel("Count")                           # Fixed: axes[0]

def parse_age(x):
    if pd.isna(x):
        return np.nan
    # typical DICOM age format: '045Y'
    try:
        return float(str(x).strip()[:3])
    except Exception:
        return pd.to_numeric(x, errors="coerce")

age_years = train["PatientAge"].apply(parse_age)

# Right subplot: age histogram
sns.histplot(age_years, bins=20, ax=axes[1])
axes[1].set_title("Age (years, approx)")
axes[1].set_xlabel("Years")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()


locs.head(3), locs["SeriesInstanceUID"].nunique(), locs.shape


# parse coordinate strings if provided as "x y" or "x,y"
def parse_xy(s):
    if pd.isna(s): 
        return (np.nan, np.nan)
    s = str(s).replace(",", " ").split()
    if len(s) >= 2:
        try:
            return float(s[0]), float(s[1])  # Fixed: was float(s), float(s[1])
        except ValueError:
            return (np.nan, np.nan)
    return (np.nan, np.nan)

locs[["x","y"]] = locs["coordinates"].apply(lambda s: pd.Series(parse_xy(s)))
sns.histplot(locs["SeriesInstanceUID"].value_counts(), bins=30)
plt.title("Localizer count per series")
plt.show()

