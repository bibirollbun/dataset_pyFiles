
# 0) Imports & setup
import os, math, random, ast, json, warnings
import numpy as np
import pandas as pd
import ast, json, matplotlib.pyplot as plt
import pydicom
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import seaborn as sns
sns.set_context("notebook")
from IPython.display import display, Markdown as md
import textwrap

display(md("## ğŸ› ï¸� Import and setup"))

display(md("""In this first part, we import all the necessary libraries:

- **os, math, random, ast, json, warnings**: file management, math utilities, and warnings  
- **numpy, pandas**: data manipulation  
- **matplotlib, seaborn**: visualizations  
- **pydicom**: reading medical DICOM images  
- **IPython.display**: for Markdown-formatted display in the notebook  

We also configure **pandas** for a more readable display (number of columns, column width, etc.).  

Finally, two small helper functions are added:  
- `short_uid()`: shortens very long identifiers (SeriesInstanceUID).  
- `hr()`: displays a Markdown separator line (`---`) to structure notebook outputs.  
"""))



# ---------- Pandas display settings ----------
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)

# ---------- Small helpers ----------
def short_uid(x, n=28):
    """Shorten long SeriesInstanceUID for readability"""
    x = str(x)
    return x if len(x) <= n else x[:n] + "â€¦"

def hr():
    display(md("---"))

hr()


# ---------- 1) Dataset folder content ----------
display(md("## ğŸ“� Dataset folder content"))

DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection"
SERIES_DIR = os.path.join(DATA_DIR, "series")

print("Dataset directory set to:", DATA_DIR)
print("Series directory set to:", SERIES_DIR)


if os.path.isdir(DATA_DIR):
    items = sorted(os.listdir(DATA_DIR))
    display(md("\n".join(f"- `{it}`" for it in items)))
else:
    display(md("**âš ï¸� DATA_DIR not found. Did you add the competition dataset?**"))

hr()

# ---------- 2) General dataset info ----------
display(md("## ğŸ“Š General dataset info"))

display(md(""" This section displays general information about the dataset: the size of the train.csv and train_localizers.csv files, 
as well as the complete list of columns in train.csv, allowing you to check the structure and content of the data before any analysis."""))

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
loc_df   = pd.read_csv(os.path.join(DATA_DIR, "train_localizers.csv"))

print("train.csv shape:", train_df.shape)
print("train_localizers.csv shape:", loc_df.shape)

display(md(f"- **Shape of `train.csv`**: `{train_df.shape}`"))
display(md(f"- **Columns (`train.csv`)** ({len(train_df.columns)}):"))
display(md("`" + "`, `".join(train_df.columns.tolist()) + "`"))

hr()

# ---------- 3) Preview of key columns ----------
display(md("## ğŸ‘€ Preview of key columns"))
display(md("""This section provides a quick preview of the key columns (`PatientAge`, `PatientSex`, `Modality`, and `Aneurysm Present`) 
with shortened series identifiers, to easily inspect a few representative rows of the dataset."""))

base_cols = ["SeriesInstanceUID","PatientAge","PatientSex","Modality","Aneurysm Present"]
preview = train_df[base_cols].head(8).copy()
preview.insert(1, "SeriesUID_short", preview["SeriesInstanceUID"].map(short_uid))
preview = preview.drop(columns=["SeriesInstanceUID"])
display(preview.style.set_properties(**{"text-align": "left"}))

hr()

# ---------- 4) DICOM quick look ----------
display(md("## ğŸ–¼ï¸� DICOM quick look"))

display(md("""The dataset contains 188 DICOM series, each identified by a unique `SeriesInstanceUID` (e.g., `1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647`). Each series is composed of multiple images, with the number of slices varying from one series to another.
"""))
if os.path.isdir(SERIES_DIR):
    all_series = sorted(os.listdir(SERIES_DIR))
    if len(all_series) == 0:
        display(md("**âš ï¸� No series found in `series/`.**"))
    else:
        example_series = all_series[0]
        series_path = os.path.join(SERIES_DIR, example_series)
        dicom_files = [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")]
        if not dicom_files:
            dicom_files = os.listdir(series_path)  # fallback

        display(md(f"- **Example series**: `{short_uid(example_series, 40)}`"))
        display(md(f"- **Number of images**: `{len(dicom_files)}`"))

        # Display one DICOM image
        dcm_path = os.path.join(series_path, dicom_files[0])
        dcm = pydicom.dcmread(dcm_path)
        plt.figure(figsize=(5,5))
        plt.imshow(dcm.pixel_array, cmap="gray")
        plt.title(f"Example DICOM â€” {short_uid(example_series, 40)}")
        plt.axis("off")
        plt.show()
else:
    display(md("**âš ï¸� Folder `series/` not found.**"))

hr()

# ---------- 5) Localization columns (13 arteries) ----------
display(md("## ğŸ§  Localization columns (13 arteries)"))

display(md(""" This section displays one representative annotated slice for each of the 13 arterial localizations, 
highlighting the approximate position of aneurysms directly on DICOM images."""))

loc_palette = {
    "Left Infraclinoid Internal Carotid Artery": "tab:blue",
    "Right Infraclinoid Internal Carotid Artery": "tab:orange",
    "Left Supraclinoid Internal Carotid Artery": "tab:green",
    "Right Supraclinoid Internal Carotid Artery": "tab:red",
    "Left Middle Cerebral Artery": "tab:purple",
    "Right Middle Cerebral Artery": "tab:brown",
    "Anterior Communicating Artery": "tab:pink",
    "Left Anterior Cerebral Artery": "tab:orange",
    "Right Anterior Cerebral Artery": "tab:olive",
    "Left Posterior Communicating Artery": "tab:cyan",
    "Right Posterior Communicating Artery": "tab:cyan",
    "Basilar Tip": "tab:green",
    "Other Posterior Circulation": "tab:red",
}

def parse_coords(s):
    if isinstance(s, dict): return s
    for parser in (ast.literal_eval, json.loads):
        try:
            out = parser(s)
            if isinstance(out, dict) and "x" in out and "y" in out:
                return out
        except Exception:
            pass
    return None

def show_localizer_point(row):
    sid = row["SeriesInstanceUID"]; sop = row["SOPInstanceUID"]
    coords = parse_coords(row["coordinates"])
    spath = os.path.join(SERIES_DIR, sid)
    dcm_path = os.path.join(spath, sop + ".dcm")
    if not os.path.exists(dcm_path):
        matches = [f for f in os.listdir(spath) if f.startswith(str(sop))]
        if matches: dcm_path = os.path.join(spath, matches[0])

    if not os.path.exists(dcm_path) or coords is None:
        display(md("**âš ï¸� Cannot display annotated slice (file or coordinates missing).**"))
        return

    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array

    # --- Handle multi-frame DICOMs ---
    if arr.ndim == 3:
        # shape typically (nframes, H, W)
        mid = arr.shape[0] // 2
        img = arr[mid]
    else:
        img = arr

    # --- MONOCHROME1 images (white=low) -> invert to look like MONOCHROME2 ---
    try:
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            # normalize then invert
            img = img.astype("float32")
            img = (img - img.min()) / max(1e-6, (img.max() - img.min()))
            img = 1.0 - img
    except Exception:
        pass

    H, W = img.shape[:2]

    # --- Many localizer coords are in a 512Ã—512 reference; rescale if needed ---
    x, y = float(coords["x"]), float(coords["y"])
    if (W != 512) or (H != 512):
        x = x * (W / 512.0)
        y = y * (H / 512.0)

    # Clamp inside image bounds (avoids warnings if slightly out of range)
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))

    # --- Display ---
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    color = loc_palette.get(row.get("location",""), "red")
    plt.scatter(x, y, s=60, marker="x", c=color)
    plt.axis("off")
    plt.show()

RANDOM_STATE = 42
unique_locs = loc_df["location"].unique()

for loc in unique_locs:
    subset = loc_df[loc_df["location"] == loc]
    if len(subset) == 0:
        continue
    row = subset.sample(1, random_state=RANDOM_STATE).iloc[0]
    display(md(f"ğŸ“� Localization: **{loc}**"))
    show_localizer_point(row)








# --------------------------------------------
# ğŸ§‘â€�âš•ï¸�ğŸ‘©â€�âš•ï¸� Demographics 
# --------------------------------------------
display(md("## ğŸ§‘â€�âš•ï¸�ğŸ‘©â€�âš•ï¸� Demographics"))

# --- Combined figure: Age, Sex, Target ---
fig = plt.figure(figsize=(14,4))

# Age distribution
plt.subplot(1,3,1)
if sns is not None:
    sns.histplot(train_df["PatientAge"], bins=20, kde=True, color="skyblue")
else:
    plt.hist(train_df["PatientAge"], bins=20, color="skyblue")
plt.title("Patient Age Distribution")

# Sex distribution
plt.subplot(1,3,2)
sex_counts = train_df["PatientSex"].value_counts(dropna=False)
plt.bar(sex_counts.index.astype(str), sex_counts.values, color="orange")
plt.title("Patient Sex Distribution")
plt.xlabel("Sex"); plt.ylabel("Count")

# Aneurysm Present distribution
plt.subplot(1,3,3)
target_counts = train_df["Aneurysm Present"].value_counts()
plt.bar(["Absent (0)","Present (1)"],
        [target_counts.get(0,0), target_counts.get(1,0)],
        color=["lightgreen","salmon"])
plt.title("Target: Aneurysm Present")
plt.tight_layout()
plt.show()

# --- Separate barplot for clarity ---
counts = train_df["Aneurysm Present"].value_counts()
plt.figure(figsize=(5,4))
if sns is not None:
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
else:
    plt.bar(counts.index.astype(int), counts.values, color="gray")
plt.title("Distribution of Aneurysm Presence")
plt.xticks([0,1], ["Absent (0)", "Present (1)"])
plt.ylabel("Number of series")
plt.show()

print(f"Number of series without aneurysm: {counts.get(0,0)}")
print(f"Number of series with aneurysm: {counts.get(1,0)}")

# --- Age distribution (again, for clarity) ---
plt.figure(figsize=(6,4))
if sns is not None:
    sns.histplot(train_df["PatientAge"], bins=20, kde=True, color="steelblue")
else:
    plt.hist(train_df["PatientAge"], bins=20, color="steelblue")
plt.title("Age Distribution of Patients")
plt.show()

# --- Sex vs Aneurysm Presence ---
plt.figure(figsize=(6,4))
if sns is not None:
    sns.countplot(x="PatientSex", hue="Aneurysm Present", data=train_df, palette="coolwarm")
else:
    sexes = sorted(train_df["PatientSex"].dropna().unique())
    for i, sex in enumerate(sexes):
        sub = train_df[train_df["PatientSex"]==sex]["Aneurysm Present"].value_counts()
        plt.bar([i-0.2, i+0.2], [sub.get(0,0), sub.get(1,0)], width=0.4)
    plt.xticks(range(len(sexes)), sexes)
plt.title("Sex vs Aneurysm Presence")
plt.show()

# --- Extra stats ---
prevalence = train_df["Aneurysm Present"].mean()
display(md(f"**Overall prevalence of aneurysm in dataset:** {prevalence:.2%}"))

sex_ct = pd.crosstab(train_df["PatientSex"], train_df["Aneurysm Present"], normalize="index")
display(md("**Sex vs Presence (row-wise proportions):**"))
display(sex_ct)

display(md("""Descriptive analysis of the dataset reveals several interesting trends. 
The age distribution shows that the majority of patients are between 50 and 70 years old, with a peak around 60 years old. 
In terms of gender, there are more women than men in the sample (approximately 3,000 versus 1,300). 
The overall prevalence of aneurysms is 42.9%, meaning that nearly one in two patients has a detected abnormality. 
Comparing by gender, we see that women are proportionally more affected (46.8% of aneurysms in female patients versus 34.1% in male patients). 
Finally, the target variable confirms a relative imbalance with 2,484 series without aneurysms and 1,864 positive series."""))



# --------------------------------------------
# ğŸ”¬ Imaging Modality 
# --------------------------------------------
display(md("## ğŸ”¬ Imaging Modality"))

# --- Crosstabs: counts & proportions ---
mod_ct = pd.crosstab(train_df["Modality"], train_df["Aneurysm Present"])
mod_ct_prop = pd.crosstab(train_df["Modality"], train_df["Aneurysm Present"], normalize="index")

display(md("**Counts (by Modality Ã— Target):**"))
display(mod_ct)

display(md("**Row-wise proportions (per modality):**"))
display(mod_ct_prop.round(3))

# --- Visualization ---
plt.figure(figsize=(6,4))
if sns is not None:
    sns.countplot(x="Modality", hue="Aneurysm Present", data=train_df, palette="Set2")
    plt.title("Imaging Modality vs Aneurysm Presence")
    plt.xlabel("Modality"); plt.ylabel("Number of series")
    plt.show()
else:
    modalities = sorted(train_df["Modality"].dropna().unique())
    for k, mod in enumerate(modalities):
        sub = train_df[train_df["Modality"]==mod]["Aneurysm Present"].value_counts()
        plt.bar([k-0.2, k+0.2], [sub.get(0,0), sub.get(1,0)], width=0.4, label=mod)
    plt.xticks(range(len(modalities)), modalities)
    plt.title("Imaging Modality vs Aneurysm Presence")
    plt.xlabel("Modality"); plt.ylabel("Number of series")
    plt.show()

display(md("""Here, we can see the distribution of detected aneurysms according to the medical imaging modality:

- CTA (Computed Tomography Angiography): a CT-based technique, often used in emergencies because it is fast and provides good visualization of the vessels. This modality has the highest proportion of detected aneurysms (53.9%) .
- MRA (Magnetic Resonance Angiography): magnetic resonance imaging without the injection of iodinated contrast medium. It is non-invasive but slightly less sensitive than CTA for small aneurysms. Here, the proportion of aneurysms is 44.3%.
- Post-contrast T1 MRI: T1 MRI sequence performed after contrast injection, useful for visualizing vascular walls and certain abnormalities. Here, it shows a lower proportion of aneurysms (25.2%).
- T2 MRI: MRI sequence sensitive to fluid variations and cerebral hemodynamics, useful in the general evaluation of the brain. It also shows a low proportion of aneurysms (26.2%).

These results suggest that targeted vascular methods (CTA, MRA) detect proportionally more aneurysms than conventional MRI sequences (post-contrast T1, T2), which is consistent with their clinical role."""))



# --------------------------------------------
# ğŸ§  Multi-aneurysm (per series) & Localization Distribution
# --------------------------------------------
display(md("## ğŸ§  Multi-aneurysm (per series) & Localization Distribution"))

# --- 1) Identify localization columns (13 arteries) ---
location_cols = [c for c in train_df.columns 
                 if c not in ["SeriesInstanceUID","PatientAge","PatientSex","Modality","Aneurysm Present"]]

if len(location_cols) == 0:
    display(md("**âš ï¸� No localization columns detected.**"))
else:
    display(md(f"- **Detected localization columns:** {len(location_cols)}"))

    # --- 2) Number of positive localizations per series ---
    train_df = train_df.copy()  # avoid potential SettingWithCopy warnings
    train_df["nb_locations"] = train_df[location_cols].sum(axis=1)

    display(md("**Descriptive stats for the number of positive localizations per series:**"))
    display(train_df["nb_locations"].describe().to_frame().T)

    # Plot 1: distribution of the number of localizations (per series)
    plt.figure(figsize=(6,4))
    if sns is not None:
        sns.countplot(x="nb_locations", data=train_df, palette="viridis")
    else:
        vals = train_df["nb_locations"].value_counts().sort_index()
        plt.bar(vals.index.astype(str), vals.values)
    plt.title("Number of positive localizations per series")
    plt.xlabel("# of localizations"); plt.ylabel("Number of series")
    plt.show()

    # --- 3) Global counts by localization (across all series) ---
    loc_counts = train_df[location_cols].sum().sort_values(ascending=False)
    loc_perc   = 100 * loc_counts / len(train_df)

    summary_loc = pd.DataFrame({
        "count": loc_counts,
        "percentage_of_series": loc_perc.round(2)
    })

    display(md("**Localization frequency across the dataset (sorted):**"))
    display(summary_loc)

    # Plot 2: localization frequency barplot
    plt.figure(figsize=(10,5))
    if sns is not None:
        sns.barplot(x=loc_counts.values, y=loc_counts.index, palette="magma")
    else:
        plt.barh(loc_counts.index, loc_counts.values)
        plt.gca().invert_yaxis()
    plt.title("Localization frequency (series with aneurysm)")
    plt.xlabel("Number of positive series")
    plt.ylabel("Localization")
    plt.show()

display(md("""This analysis focuses on the distribution of aneurysms by anatomical location.
The vast majority of patients have zero positive locations (no aneurysms), but there are also cases with 1 to 5 positive locations per series.
In terms of frequency, the most affected locations are:
the anterior communicating artery (8.35%),
the left supraclinoid internal carotid artery (7.61%),
the right middle cerebral artery (6.76%),
the right supraclinoid internal carotid artery (6.37%).
These areas therefore constitute the main sites of aneurysms in the dataset.
Conversely, certain arteries such as the right anterior cerebral artery (1.29%) or the left infraclinoid internal carotid artery (1.79%) are much less represented.
This distribution suggests that certain anatomical locations are more vulnerable to the development of aneurysms, which is consistent with the medical literature on arterial bifurcation areas and areas of hemodynamic turbulence."""))






# --------------------------------------------
# ğŸ©» DICOM exploration 
# --------------------------------------------

display(md("## ğŸ©» DICOM exploration "))
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


FAST_MODE = True          
MAX_SERIES_FOR_STATS = 80 

def _safe_list_dcms(series_path):
    """Liste les DICOM d'une sÃ©rie. En FAST_MODE: tri simple par nom (pas de header)."""
    files = [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")]
    if not files:
        files = os.listdir(series_path)  
    if FAST_MODE:
        return sorted(files)        
    
    def _key(fname):
        try:
            ds = pydicom.dcmread(os.path.join(series_path, fname), stop_before_pixels=True)
            return getattr(ds, "InstanceNumber", 1e12)
        except Exception:
            return 1e12
    try:
        return sorted(files, key=_key)
    except Exception:
        return sorted(files)

def count_slices_in_series_fast(series_path: str) -> int:
    """Comptage approximatif (rapide) : nombre de fichiers .dcm"""
    files = [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")]
    if not files:
        files = os.listdir(series_path)
    return len(files) if len(files) > 0 else 0

def count_slices_in_series_precise(series_path: str) -> int:
    """Comptage prÃ©cis : somme NumberOfFrames de chaque fichier (lent)."""
    files = [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")]
    if not files:
        files = os.listdir(series_path)
    total = 0
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(series_path, f), stop_before_pixels=True)
            frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
            total += max(frames, 1)
        except Exception:
            total += 1
    return total

def show_series_grid(series_id, n=9):
    """Grille de n slices rÃ©parties dans la sÃ©rie (FAST_MODE = pas de tri par header)."""
    spath = os.path.join(SERIES_DIR, series_id)
    if not os.path.isdir(spath):
        print("âš ï¸� Series not found:", series_id); return
    files = _safe_list_dcms(spath)
    if not files:
        print("âš ï¸� No files found in series:", series_id); return

    idxs = np.linspace(0, len(files)-1, min(n, len(files))).astype(int)
    rows = cols = int(math.ceil(math.sqrt(len(idxs))))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = np.array(axes).reshape(rows, cols)

    for ax, i in zip(axes.flatten(), idxs):
        dcm = pydicom.dcmread(os.path.join(spath, files[i]))
        arr = dcm.pixel_array
        if arr.ndim == 3:   # multi-frame
            arr = arr[arr.shape[0]//2]
        ax.imshow(arr, cmap="gray")
        ax.set_title(f"Slice {i}")
        ax.axis("off")

    for ax in axes.flatten()[len(idxs):]:
        ax.axis("off")
    plt.suptitle(f"Series {series_id} â€” {len(files)} images", y=0.92)
    plt.tight_layout()
    plt.show()

# ---- Slice stats ----
slice_counts = []
series_ids = sorted([d for d in os.listdir(SERIES_DIR) if os.path.isdir(os.path.join(SERIES_DIR, d))])

if MAX_SERIES_FOR_STATS is not None:
    series_ids = series_ids[:MAX_SERIES_FOR_STATS]

for sid in series_ids:
    spath = os.path.join(SERIES_DIR, sid)
    if FAST_MODE:
        n = count_slices_in_series_fast(spath)
    else:
        n = count_slices_in_series_precise(spath)
    slice_counts.append((sid, n))

vals = [v for _, v in slice_counts] or [0]
min_slices = int(np.min(vals)); max_slices = int(np.max(vals))
mean_slices = float(np.mean(vals)); median_slices = float(np.median(vals))
sid_min = next(s for s, v in slice_counts if v == min_slices)
sid_max = next(s for s, v in slice_counts if v == max_slices)

display(md("### ğŸ“Š Slice count per series â€” summary"))
display(md(f"- **Series analysed** : {len(vals)} (FAST_MODE={FAST_MODE})"))
display(md(f"- **Min** : {min_slices} (e.g., `{sid_min}`)"))
display(md(f"- **Max** : {max_slices} (e.g., `{sid_max}`)"))
display(md(f"- **Mean** : {mean_slices:.2f}"))
display(md(f"- **Median** : {median_slices:.0f}"))

plt.figure(figsize=(6,4))
plt.hist(vals, bins=20)
plt.title("Distribution of slices per series")
plt.xlabel("Slices"); plt.ylabel("Series count")
plt.show()

display(md("""Here, we analyze the number of slices per DICOM series. 
There is significant variability between series: the number of slices ranges from 13 to 898, with an average of 195 and a median of 157. 
The distribution shows that the majority of series have fewer than 200 slices, but some longer series have several hundred images."""))

# ---- Show one positive example ----
pos_ids = train_df.loc[train_df["Aneurysm Present"]==1, "SeriesInstanceUID"]
chosen_series = pos_ids.iloc[0] if len(pos_ids) else train_df["SeriesInstanceUID"].iloc[0]

display(md(f"**ğŸ–¼ï¸� Grid of slices for series:** `{chosen_series}`"))
show_series_grid(chosen_series, n=9)




# ------------------------------------------------------------
# âœ… Preprocessing Checklist â€” Step 2
# ------------------------------------------------------------
display(md("## âœ… Preprocessing Checklist (Step 2)"))

checklist = [
    "- [ ] **Intensity normalization** (e.g., min-max or z-score)",
    "- [ ] **Approach choice**: **2D (slice-based)** or **3D (full volume)** models",
    "- [ ] **Class imbalance handling** (positives vs negatives, weighting or oversampling)",
    "- [ ] **Stratified split** train / validation / test (preserve prevalence)",
    "- [ ] **Create a PyTorch Dataset/Dataloader** to load DICOM series",
    "- [ ] **Data augmentations** (rotations, flips, contrast, noise, etc.)",
    "- [ ] **Save preprocessed tensors** for faster training"
]

for item in checklist:
    display(md(item))



import cv2

def preprocess_dcm(path, size=(224,224)):
    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array.astype(np.float32)
    # Normalisation z-score
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    # Resize en 2D
    arr_resized = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    return arr_resized


# SÃ©lectionner une sÃ©rie (par exemple la premiÃ¨re)
example_series = sorted(os.listdir(SERIES_DIR))[0]
series_path = os.path.join(SERIES_DIR, example_series)

# Liste des DICOMs dans cette sÃ©rie
dicom_files = [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")]
if not dicom_files:
    raise ValueError(f"Aucun fichier DICOM trouvÃ© dans {series_path}")

# On prend le premier fichier DICOM
first_file = os.path.join(series_path, dicom_files[0])

# PrÃ©traitement et affichage
img = preprocess_dcm(first_file)
plt.imshow(img, cmap="gray")
plt.title(f"PrÃ©traitement DICOM (224x224) â€” {example_series}")
plt.axis("off")
plt.show()





from sklearn.model_selection import StratifiedGroupKFold

X = train_df["SeriesInstanceUID"]
y = train_df["Aneurysm Present"]

# Ici on utilise SeriesInstanceUID comme fallback (pas de PatientID dispo)
groups = train_df["SeriesInstanceUID"]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    print(f"Fold {fold}: Train {len(train_idx)} | Val {len(val_idx)}")



import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

# Charger ResNet18 prÃ©-entraÃ®nÃ© sur ImageNet
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Modifier la derniÃ¨re couche pour une sortie binaire (prÃ©sence ou non dâ€™anÃ©vrisme)
model.fc = nn.Linear(model.fc.in_features, 1)  # sortie = 1 logit

print(model)

!pip install torchsummary
from torchsummary import summary

device = torch.device("cpu")
model.to(device)

from torchsummary import summary
summary(model, (3, 224, 224), device="cpu")



from graphviz import Digraph
from IPython.display import Image

# CrÃ©ation du graphe
dot = Digraph(comment="ResNet18 Architecture", format="png")
dot.attr(rankdir="TB", size="8")

# EntrÃ©e
dot.node("input", "EntrÃ©e\n(3, 224, 224)", shape="box", style="filled", color="lightblue")

# Blocs principaux
dot.node("conv1", "Conv1 + BN + ReLU\n(64, 112, 112)", shape="box")
dot.node("pool", "MaxPool\n(64, 56, 56)", shape="box", style="filled", color="lightgrey")

dot.node("layer1", "Bloc RÃ©siduel x2\n(64, 56, 56)", shape="box")
dot.node("layer2", "Bloc RÃ©siduel x2\n(128, 28, 28)", shape="box")
dot.node("layer3", "Bloc RÃ©siduel x2\n(256, 14, 14)", shape="box")
dot.node("layer4", "Bloc RÃ©siduel x2\n(512, 7, 7)", shape="box")

# Pooling + FC
dot.node("gap", "Global Avg Pool\n(512, 1, 1)", shape="box", style="filled", color="lightgrey")
dot.node("fc", "Fully Connected\n(512 â†’ 1)", shape="box")
dot.node("sigmoid", "SigmoÃ¯de\nProbabilitÃ©", shape="box", style="filled", color="lightgreen")

# Connexions
dot.edges([("input","conv1"), ("conv1","pool"), ("pool","layer1"),
           ("layer1","layer2"), ("layer2","layer3"), ("layer3","layer4"),
           ("layer4","gap"), ("gap","fc"), ("fc","sigmoid")])

# Sauvegarde et affichage dans le notebook
dot.render("resnet18_architecture", format="png")  
Image(filename="resnet18_architecture.png")


