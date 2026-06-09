# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import pydicom
from tqdm import tqdm



# ==========================================
#             Loading the files
# ==========================================
train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
train_loc = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")
train_df.head()


# ======================================
# Joining train metadada with DICOM
# =======================================

# Root directory for the DICOM series
root_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"

# Get only the SeriesInstanceUIDs present in train.csv
train_uids = set(train_df["SeriesInstanceUID"].unique())

# Limit processing to series used in training
series_dirs = [uid for uid in os.listdir(root_dir) if uid in train_uids]

# Fields to extract from DICOM metadata
selected_keywords = [
    "SeriesInstanceUID", "Modality", "SeriesDescription", "ProtocolName",
    "BodyPartExamined", "Manufacturer", "ManufacturerModelName", "StationName",
    "SoftwareVersions", "SliceThickness", "PixelSpacing", "KVP", "Exposure",
    "ImageType", "StudyDate", "SeriesDate", "PatientAge", "PatientSex",
    "LossyImageCompression", "BurnedInAnnotation"
]

dicom_metadata_list = []

# Loop over DICOM series used in training
for series_uid in tqdm(series_dirs):
    series_path = os.path.join(root_dir, series_uid)
    dicom_files = [f for f in os.listdir(series_path) if f.endswith(".dcm")]
    if not dicom_files:
        continue

    dicom_path = os.path.join(series_path, dicom_files[0])
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        meta = {"SeriesInstanceUID": series_uid}

        for keyword in selected_keywords:
            if keyword == "SeriesInstanceUID":
                continue
            val = ds.get(keyword, None)
            meta[keyword] = str(val) if val not in [None, "", [], {}] else None

        dicom_metadata_list.append(meta)

    except Exception:
        continue

# Convert to DataFrame and drop columns with all missing values
df_filtered = pd.DataFrame(dicom_metadata_list)
df_filtered = df_filtered.dropna(axis=1, how="all")

# Optionally: merge with train_df immediately
train_EDA = train_df.merge(df_filtered, on="SeriesInstanceUID", how="left")

# Preview
train_EDA.head()



# ========================================================
# Let's create a column to mark the low qiality images
# ========================================================

# Garantir que os campos estÃ£o em formato numÃ©rico
train_EDA["SliceThickness"] = pd.to_numeric(train_EDA["SliceThickness"], errors="coerce")
train_EDA["Exposure"] = pd.to_numeric(train_EDA["Exposure"], errors="coerce")

# CritÃ©rio de baixa qualidade
train_EDA["IMG_Low_Quality_CT"] = (train_EDA["SliceThickness"] > 5) | (train_EDA["Exposure"] < 10)




train_EDA.info()


# --- Compute totals and percentages ---
total_cases = len(train_df)

# Aneurysm prevalence
aneurysm_counts = train_df["Aneurysm Present"].value_counts().sort_index()
aneurysm_pct = (aneurysm_counts / total_cases * 100).round(2)

# Sex distribution by aneurysm
sex_counts = train_df.groupby(["PatientSex", "Aneurysm Present"]).size().unstack(fill_value=0)
sex_pct = (sex_counts.div(sex_counts.sum(axis=1), axis=0) * 100).round(2)

# Age distribution summary
age_summary = train_df.groupby("Aneurysm Present")["PatientAge"].describe().round(1)

# --- Charts ---

# Aneurysm prevalence
plt.figure(figsize=(6,4))
ax = sns.countplot(x="Aneurysm Present", data=train_df, palette="Set1")
plt.title("Aneurysm Presence Distribution")
plt.xticks([0,1], ["No Aneurysm", "Aneurysm Present"])
plt.ylabel("Number of Patients")
for p, count, pct in zip(ax.patches, aneurysm_counts, aneurysm_pct):
    ax.annotate(f'{count}\n({pct}%)', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')
plt.show()

# Sex vs Aneurysm
plt.figure(figsize=(8,5))
ax = sns.countplot(x="PatientSex", hue="Aneurysm Present", data=train_df, palette="Set2")
plt.title("Sex Distribution by Aneurysm Presence")
plt.xlabel("Patient Sex")
plt.ylabel("Number of Patients")
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom')
plt.show()

# Age distribution
plt.figure(figsize=(10,6))
sns.histplot(data=train_df, x="PatientAge", hue="Aneurysm Present",
             bins=30, kde=True, palette="husl", element="step", stat="count")
plt.title("Age Distribution by Aneurysm Presence")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()

# --- Textual Summary ---
print("===== ğŸ“Š TEXTUAL SUMMARY =====\n")

print("Aneurysm Presence Distribution:")
for label, count, pct in zip(aneurysm_counts.index, aneurysm_counts, aneurysm_pct):
    label_str = "Aneurysm Present" if label == 1 else "No Aneurysm"
    print(f" - {label_str}: {count} patients ({pct}%)")

print("\nSex Distribution by Aneurysm Presence:")
print(sex_counts)
print("\nPercentages within each sex:")
print(sex_pct)

print("\nAge Distribution by Aneurysm Presence:")
print(age_summary)


# ============================================== #
# List of location-specific aneurysm labels      #
# ============================================== #

location_cols = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation'
]

# Sum of aneurysms per location grouped by sex
sex_location_counts = train_df.groupby("PatientSex")[location_cols].sum().T

# Convert counts into prevalence rates (percentage within sex)
sex_totals = train_df.groupby("PatientSex").size()
sex_location_pct = sex_location_counts.div(sex_totals, axis=1) * 100

# --- Heatmap of prevalence ---
plt.figure(figsize=(12,8))
sns.heatmap(sex_location_pct, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Aneurysm Location Prevalence (%) by Sex")
plt.xlabel("Sex")
plt.ylabel("Artery Location")
plt.show()

# --- Textual summary ---
print("===== ğŸ“Š TEXTUAL SUMMARY: Aneurysm Location Prevalence by Sex =====\n")
print(sex_location_pct.round(2))


"""
STATISTICAL SIGNIFICANCE ANALYSIS: Sex vs. Aneurysm Location

This cell performs chi-square tests to determine if there are statistically
significant relationships between patient sex and aneurysm location prevalence.

ANALYSIS PURPOSE:
- Tests the null hypothesis that sex and aneurysm location are independent
- Identifies if certain locations show gender-based predisposition
- Uses chi-square tests of independence on contingency tables

INTERPRETATION:
- p-value < 0.05: Statistically significant relationship (reject null hypothesis)
- p-value >= 0.05: No significant evidence of relationship
- Chi2 value: Strength of association (higher = stronger relationship)
"""

location_cols = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation'
]

results = []

for col in location_cols:
    # Build contingency table
    contingency = pd.crosstab(train_df["PatientSex"], train_df[col])
    
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    results.append({
        "Artery Location": col,
        "Chi2": chi2,
        "p-value": p,
        "Significant (<0.05)": p < 0.05
    })

# Convert to DataFrame
chi_results = pd.DataFrame(results).sort_values("p-value")

print("===== ğŸ“Š Chi-Square Test Results by Artery Location =====\n")
print(chi_results)


"""
ANALYSIS: Distribution of Imaging Modalities by Aneurysm Label

This cell analyzes the relationship between imaging modalities (CT, MRI, etc.)
and the presence of aneurysms in the training dataset.

WHAT IT DOES:
1. Groups the data by modality and aneurysm label
2. Calculates counts and percentages for each combination
3. Creates a bar plot showing the distribution
4. Annotates bars with both count and percentage values

OUTPUT: Visual comparison of modality usage for positive/negative aneurysm cases
"""

# Group by Modality and Aneurysm label, count occurrences
modality_label_dist = (
    train_EDA.groupby(["Modality_x", "Aneurysm Present"])
    .size()
    .reset_index(name="count")
)

# Calculate percentage within each label group
modality_label_dist["percent"] = (
    modality_label_dist.groupby("Aneurysm Present")["count"]
    .transform(lambda x: (x / x.sum()) * 100)
).round(1)

# Plot
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=modality_label_dist,
    x="Modality_x",
    y="count",
    hue="Aneurysm Present",
    palette="Set2"
)

# Add labels with correct count and percentage values
for i in range(len(modality_label_dist)):
    row = modality_label_dist.iloc[i]
    modality = row["Modality_x"]
    label = row["Aneurysm Present"]
    count = row["count"]
    percent = row["percent"]

    # Find the correct bar
    for patch in barplot.patches:
        if patch.get_height() == count and patch.get_x() < barplot.get_xlim()[1]:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            barplot.annotate(
                f"{count}\n({percent:.1f}%)", 
                (x, y), 
                ha="center", 
                va="bottom", 
                fontsize=9
            )
            break

plt.title("Distribution of Imaging Modalities by Aneurysm Label")
plt.xlabel("Imaging Modality")
plt.ylabel("Number of Exams")
plt.xticks(rotation=45)
plt.legend(title="Aneurysm Present")
plt.tight_layout()
plt.show()









# ============================================
# EDA: SliceThickness â€” histogram + boxplot by manufacturer
# Rule of thumb: thickness > 5 mm â†’ may miss small aneurysms
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) Work on a copy; ensure numeric
# -----------------------------
df = train_EDA.copy()
df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str)
df["SliceThickness"] = pd.to_numeric(df["SliceThickness"], errors="coerce")

# Optional filter: keep only CTA (set to False to use all)
ONLY_CTA = False
mod_col = "Modality_x" if "Modality_x" in df.columns else ("Modality" if "Modality" in df.columns else None)
if ONLY_CTA and mod_col:
    df = df[df[mod_col].astype(str).str.upper() == "CTA"].copy()

# -----------------------------
# 1) Create thickness-based quality flag
# -----------------------------
# Keep this flag separate from any previous IMG_Low_Quality rule
df["IMG_Low_Quality_ST"] = df["SliceThickness"] > 5.0

total = df["SliceThickness"].notna().sum()
n_bad = int(df["IMG_Low_Quality_ST"].sum())
pct_bad = 100.0 * n_bad / max(total, 1)

print(f"Total series with SliceThickness available: {total}")
print(f"Series with SliceThickness > 5 mm: {n_bad} ({pct_bad:.2f}%)")

# -----------------------------
# 2) Manufacturer normalization (collapse vendor variants)
# -----------------------------
def normalize_manufacturer(x: str) -> str:
    if not isinstance(x, str):
        return "OTHER"
    s = x.strip().upper()
    if "SIEMENS" in s:
        return "SIEMENS"
    if "GE" in s:
        return "GE MEDICAL SYSTEMS"
    if "TOSHIBA" in s:
        return "TOSHIBA"
    if "CANON" in s:
        return "CANON"
    if "PHILIPS" in s:
        return "PHILIPS"
    return s if s else "OTHER"

df["Manufacturer_norm"] = df["Manufacturer"].apply(normalize_manufacturer)

# -----------------------------
# 3) Summary table by manufacturer
# -----------------------------
summary = (df.groupby("Manufacturer_norm")
             .agg(
                 n=("SeriesInstanceUID","nunique"),
                 thickness_median=("SliceThickness","median"),
                 thickness_p75=("SliceThickness", lambda s: np.nanpercentile(s.dropna(), 75) if s.notna().any() else np.nan),
                 pct_gt_5mm=("IMG_Low_Quality_ST", "mean")
             )
             .sort_values("pct_gt_5mm", ascending=False))
summary["pct_gt_5mm"] = (summary["pct_gt_5mm"] * 100).round(2)

print("\nSliceThickness summary by Manufacturer (sorted by % > 5 mm):")
display(summary)

# -----------------------------
# 4) Histogram (single axis, matplotlib only)
# -----------------------------
plt.figure(figsize=(8,4))
vals = df["SliceThickness"].dropna().values
bins = np.linspace(max(0, np.nanmin(vals)), min(10, np.nanmax(vals)), 40)  # clamp to 0â€“10 mm for readability
plt.hist(vals, bins=bins)
plt.axvline(5.0, linestyle="--")  # do not set color per toolbox rules
plt.title("Histogram of SliceThickness (mm)")
plt.xlabel("SliceThickness (mm)")
plt.ylabel("Series count")
plt.grid(True)
plt.show()

# -----------------------------
# 5) Boxplot by manufacturer (top-N vendors with enough data)
# -----------------------------
TOP_N = 6
vc = df["Manufacturer_norm"].value_counts()
top_vendors = vc.index[:TOP_N].tolist()

data = []
labels = []
for v in top_vendors:
    arr = df.loc[df["Manufacturer_norm"] == v, "SliceThickness"].dropna().values
    if arr.size >= 5:  # require minimal samples to plot
        data.append(arr)
        labels.append(v)

plt.figure(figsize=(10,4))
plt.boxplot(data, labels=labels, showmeans=True)
plt.title("SliceThickness by Manufacturer (boxplot)")
plt.ylabel("SliceThickness (mm)")
plt.grid(True, axis="y")
plt.show()

# -----------------------------
# 6) Attach the new flag back to train_EDA (optional)
# -----------------------------
train_EDA = train_EDA.merge(
    df[["SeriesInstanceUID","IMG_Low_Quality_ST","Manufacturer_norm","SliceThickness"]],
    on="SeriesInstanceUID", how="left"
)

print("Added columns to train_EDA: IMG_Low_Quality_ST, Manufacturer_norm (and refreshed SliceThickness).")















# Save with the same name as the DataFrame variable
out_path = "/kaggle/working/train_EDA.csv"  # same name
train_EDA.to_csv(out_path, index=False)
print("Saved:", out_path)














# Enriquecendo o train.csv com metadados dicom

import os
import pydicom
import pandas as pd
from tqdm import tqdm

# Root directory for the DICOM series
root_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"

# Get only the SeriesInstanceUIDs present in train.csv
train_uids = set(train_df["SeriesInstanceUID"].unique())

# Limit processing to series used in training
series_dirs = [uid for uid in os.listdir(root_dir) if uid in train_uids]

# Fields to extract from DICOM metadata
selected_keywords = [
    "SeriesInstanceUID", "Modality", "SeriesDescription", "ProtocolName",
    "BodyPartExamined", "Manufacturer", "ManufacturerModelName", "StationName",
    "SoftwareVersions", "SliceThickness", "PixelSpacing", "KVP", "Exposure",
    "ImageType", "StudyDate", "SeriesDate", "PatientAge", "PatientSex",
    "LossyImageCompression", "BurnedInAnnotation"
]

dicom_metadata_list = []

# Loop over DICOM series used in training
for series_uid in tqdm(series_dirs):
    series_path = os.path.join(root_dir, series_uid)
    dicom_files = [f for f in os.listdir(series_path) if f.endswith(".dcm")]
    if not dicom_files:
        continue

    dicom_path = os.path.join(series_path, dicom_files[0])
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        meta = {"SeriesInstanceUID": series_uid}

        for keyword in selected_keywords:
            if keyword == "SeriesInstanceUID":
                continue
            val = ds.get(keyword, None)
            meta[keyword] = str(val) if val not in [None, "", [], {}] else None

        dicom_metadata_list.append(meta)

    except Exception:
        continue

# Convert to DataFrame and drop columns with all missing values
df_filtered = pd.DataFrame(dicom_metadata_list)
df_filtered = df_filtered.dropna(axis=1, how="all")

# Optionally: merge with train_df immediately
train_EDA = train_df.merge(df_filtered, on="SeriesInstanceUID", how="left")

# Preview
train_EDA.head()




#
# Preparing a DF to understand the cliical flow(business Process Approach)
# 

# Define the date fields to evaluate
date_fields = ["StudyDate", "SeriesDate"]

# Count non-null values for each date field
print("ğŸ”� Non-null counts per date field:")
print(train_with_metadata[date_fields].notna().sum())

# Show sample values from each date field
for col in date_fields:
    print(f"\nğŸ“… Sample values from {col}:")
    print(train_with_metadata[col].dropna().unique()[:10])

# Convert to datetime format for further analysis
for col in date_fields:
    if col in train_with_metadata.columns:
        train_with_metadata[col + "_parsed"] = pd.to_datetime(train_with_metadata[col], errors="coerce", format="%Y%m%d")

# Compare StudyDate and SeriesDate to assess consistency
if "StudyDate_parsed" in train_with_metadata.columns and "SeriesDate_parsed" in train_with_metadata.columns:
    train_with_metadata["DateDifferenceDays"] = (
        train_with_metadata["SeriesDate_parsed"] - train_with_metadata["StudyDate_parsed"]
    ).dt.days

    print("\nğŸ“Š Distribution of days between StudyDate and SeriesDate:")
    print(train_with_metadata["DateDifferenceDays"].value_counts().sort_index())






import seaborn as sns
import matplotlib.pyplot as plt

# Agrupa por Modality e Aneurysm label, conta ocorrÃªncias
modality_label_dist = (
    train_EDA.groupby(["Modality_x", "Aneurysm Present"])
    .size()
    .reset_index(name="count")
)

# Calcula o percentual dentro de cada grupo de label
modality_label_dist["percent"] = (
    modality_label_dist.groupby("Aneurysm Present")["count"]
    .transform(lambda x: (x / x.sum()) * 100)
).round(1)

# GrÃ¡fico
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=modality_label_dist,
    x="Modality_x",
    y="count",
    hue="Aneurysm Present",
    palette="Set2"
)

# Adiciona os rÃ³tulos com contagem e percentual corretos
for i in range(len(modality_label_dist)):
    row = modality_label_dist.iloc[i]
    modality = row["Modality_x"]
    label = row["Aneurysm Present"]
    count = row["count"]
    percent = row["percent"]

    # Encontra a barra correta
    for patch in barplot.patches:
        if patch.get_height() == count and patch.get_x() < barplot.get_xlim()[1]:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            barplot.annotate(
                f"{count}\n({percent:.1f}%)", 
                (x, y), 
                ha="center", 
                va="bottom", 
                fontsize=9
            )
            break

plt.title("Distribution of Imaging Modalities by Aneurysm Label")
plt.xlabel("Imaging Modality")
plt.ylabel("Number of Exams")
plt.xticks(rotation=45)
plt.legend(title="Aneurysm Present")
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lista dos 13 labels arteriais
artery_labels = [
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
    "Other Posterior Circulation"
]

# Transforma para formato longo (1 linha por artÃ©ria com aneurisma positivo)
long_df = train_EDA.melt(
    id_vars=["SeriesInstanceUID", "Modality_x"],
    value_vars=artery_labels,
    var_name="Artery",
    value_name="Aneurysm"
)

# Filtra apenas os casos positivos
long_df = long_df[long_df["Aneurysm"] == 1]

# Conta quantidade por Artery + Modality
artery_modality_counts = (
    long_df.groupby(["Artery", "Modality_x"])
    .size()
    .reset_index(name="count")
)

# Calcula % por artÃ©ria
artery_modality_counts["percent"] = (
    artery_modality_counts.groupby("Artery")["count"]
    .transform(lambda x: 100 * x / x.sum())
).round(1)

# GrÃ¡fico de barras
plt.figure(figsize=(14, 8))
barplot = sns.barplot(
    data=artery_modality_counts,
    x="Artery",
    y="count",
    hue="Modality_x",
    palette="Set2"
)

# RÃ³tulos de % nas barras
for container in barplot.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            x = bar.get_x() + bar.get_width() / 2
            idx = barplot.patches.index(bar)
            percent = artery_modality_counts.iloc[idx]["percent"]
            barplot.annotate(f"{percent:.1f}%", (x, height), ha="center", va="bottom", fontsize=8)

plt.title("Distribution of Imaging Modalities per Aneurysm Artery Label")
plt.xlabel("Artery Label")
plt.ylabel("Number of Positive Cases")
plt.xticks(rotation=90)
plt.legend(title="Modality")
plt.tight_layout()
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Filtrar apenas imagens do tipo CTA
cta_df = train_EDA[train_EDA["Modality_x"] == "CTA"].copy()

# Selecionar os campos de interesse
quality_fields = [
    "SliceThickness", "PixelSpacing", "KVP", "Exposure",
    "LossyImageCompression", "BurnedInAnnotation",
    "ImageType", "Manufacturer", "SoftwareVersions"
]

# Converter campos numÃ©ricos que ainda estejam como string
cta_df["SliceThickness"] = pd.to_numeric(cta_df["SliceThickness"], errors="coerce")
cta_df["KVP"] = pd.to_numeric(cta_df["KVP"], errors="coerce")
cta_df["Exposure"] = pd.to_numeric(cta_df["Exposure"], errors="coerce")
cta_df["PixelSpacingX"] = cta_df["PixelSpacing"].apply(lambda x: float(str(x).strip("[]").split(",")[0]) if pd.notnull(x) else None)

# Mostrar estatÃ­sticas descritivas dos campos quantitativos
print("ğŸ“Š Quantitative fields summary:")
print(cta_df[["SliceThickness", "PixelSpacingX", "KVP", "Exposure"]].describe())

# FrequÃªncia dos campos categÃ³ricos
for col in ["LossyImageCompression", "BurnedInAnnotation", "ImageType", "Manufacturer"]:
    print(f"\nğŸ”¸ Value counts for {col}:")
    print(cta_df[col].value_counts(dropna=False))

# VisualizaÃ§Ãµes
plt.figure(figsize=(14, 4))
for i, field in enumerate(["SliceThickness", "PixelSpacingX", "KVP", "Exposure"]):
    plt.subplot(1, 4, i+1)
    sns.histplot(cta_df[field], kde=True, bins=20)
    plt.title(field)
plt.tight_layout()
plt.show()






#
# Count and analyze low-quality CTA images by ImageType
#

import pandas as pd

# Filter only CTA modality exams
cta_df = train_EDA[train_EDA["Modality_x"] == "CTA"].copy()

# Convert SliceThickness and Exposure to numeric (in case they are strings)
cta_df["SliceThickness"] = pd.to_numeric(cta_df["SliceThickness"], errors="coerce")
cta_df["Exposure"] = pd.to_numeric(cta_df["Exposure"], errors="coerce")

# Define low-quality criteria
low_quality_mask = (cta_df["SliceThickness"] > 5) | (cta_df["Exposure"] < 10)
low_quality_cta = cta_df[low_quality_mask]

# Count and percentage of low-quality images
total_cta = len(cta_df)
total_low_quality = len(low_quality_cta)
percent_low_quality = (total_low_quality / total_cta) * 100

# ğŸ“Š Summary
print(f"ğŸ§ª Total CTA exams: {total_cta}")
print(f"âš ï¸� Low-quality CTA exams: {total_low_quality} ({percent_low_quality:.2f}%)")

# ğŸ”� View some examples of low-quality images
display(
    low_quality_cta[["SeriesInstanceUID", "SliceThickness", "Exposure", "ImageType", "Aneurysm Present"]]
    .sort_values(by="SliceThickness", ascending=False)
    .head(10)
)

# ğŸ“Š Distribution of low-quality images by ImageType
# Convert ImageType from string to tuple if needed
import ast
low_quality_cta["ImageType"] = low_quality_cta["ImageType"].apply(lambda x: tuple(ast.literal_eval(x)) if isinstance(x, str) else x)

# Group by ImageType and count
image_type_counts = low_quality_cta.groupby("ImageType").size().sort_values(ascending=False)
print("\nğŸ“Š Low-quality images by ImageType:")
print(image_type_counts)







#
# Let's create a column to mark the low qiality images
#

# Garantir que os campos estÃ£o em formato numÃ©rico
train_EDA["SliceThickness"] = pd.to_numeric(train_EDA["SliceThickness"], errors="coerce")
train_EDA["Exposure"] = pd.to_numeric(train_EDA["Exposure"], errors="coerce")

# CritÃ©rio de baixa qualidade
train_EDA["IMG_Low_Quality"] = (train_EDA["SliceThickness"] > 5) | (train_EDA["Exposure"] < 10)



# Group by Manufacturer and SeriesDescription to analyze low-quality image distribution
low_quality_summary = (
    train_EDA.groupby(["Manufacturer", "SeriesDescription", "IMG_Low_Quality"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Rename columns for clarity
low_quality_summary.columns.name = None
low_quality_summary = low_quality_summary.rename(columns={False: "High_Quality", True: "Low_Quality"})

# Add total and percentage columns
low_quality_summary["Total"] = low_quality_summary["High_Quality"] + low_quality_summary["Low_Quality"]
low_quality_summary["Percent_Low_Quality"] = 100 * low_quality_summary["Low_Quality"] / low_quality_summary["Total"]

# Sort by highest percentage of low-quality images
low_quality_summary = low_quality_summary.sort_values("Percent_Low_Quality", ascending=False)

# Display the most problematic combinations
print("ğŸ“Š Low-quality image distribution by Manufacturer and SeriesDescription:")
display(low_quality_summary.head(20))






import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Root directory
root_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"

# Number of samples to load
N_SAMPLES = 1

# Collect pixel data from N_SAMPLES series
all_hu = []

# Walk through the DICOM series
for patient_id in os.listdir(root_dir)[:N_SAMPLES]:
    patient_dir = os.path.join(root_dir, patient_id)
    if not os.path.isdir(patient_dir):
        continue

    # Load all DICOM slices in the series
    slices = []
    for file in os.listdir(patient_dir):
        path = os.path.join(patient_dir, file)
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(np.int16)

        # Convert to HU using DICOM metadata
        intercept = dicom.RescaleIntercept if "RescaleIntercept" in dicom else 0
        slope = dicom.RescaleSlope if "RescaleSlope" in dicom else 1
        img = img * slope + intercept

        slices.append(img)

    if slices:
        series_hu = np.stack(slices)
        all_hu.append(series_hu)

# Combine and flatten all HU values
all_hu = np.concatenate([vol.ravel() for vol in all_hu])

# Plot histogram
plt.figure(figsize=(12, 6))
plt.hist(all_hu, bins=500, range=(-1024, 2048), color='navy')
plt.title("Histogram of Hounsfield Units (HU) from Sample Series")
plt.xlabel("HU Value")
plt.ylabel("Pixel Count")
plt.grid(True)
plt.show()



import os
import ast
import gc
import numpy as np
import pandas as pd
import pydicom
import traceback
from collections import defaultdict
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
root_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"

# Fixed HU histogram settings (do NOT keep entire volumes in memory)
HU_MIN, HU_MAX = -1024, 2048
N_BINS = 512
BINS = np.linspace(HU_MIN, HU_MAX, N_BINS + 1)
BIN_CENTERS = (BINS[1:] + BINS[:-1]) / 2

# Optional quick filter: process only CTA in metadata table (recommended)
# Expecting train_EDA has SeriesInstanceUID and Modality_x (or Modality)
modality_col = "Modality_x" if "Modality_x" in train_EDA.columns else ("Modality" if "Modality" in train_EDA.columns else None)
if modality_col:
    allowed_cta = set(train_EDA.loc[train_EDA[modality_col] == "CTA", "SeriesInstanceUID"].astype(str))
else:
    allowed_cta = None  # process all if modality column not present

# Ensure numeric types for quality criteria
train_EDA["SliceThickness"] = pd.to_numeric(train_EDA.get("SliceThickness"), errors="coerce")
train_EDA["Exposure"] = pd.to_numeric(train_EDA.get("Exposure"), errors="coerce")

# Create (or reuse) low-quality flag
if "IMG_Low_Quality" not in train_EDA.columns:
    train_EDA["IMG_Low_Quality"] = (train_EDA["SliceThickness"] > 5) | (train_EDA["Exposure"] < 10)

# Keep only columns we need for merges
keep_cols = ["SeriesInstanceUID", "Aneurysm Present", "IMG_Low_Quality", "Manufacturer", "SeriesDescription", "ImageType", "SliceThickness", "Exposure", "KVP"]
meta_lookup = train_EDA[keep_cols].drop_duplicates("SeriesInstanceUID").copy()
meta_lookup["SeriesInstanceUID"] = meta_lookup["SeriesInstanceUID"].astype(str)

# -----------------------------
# Helpers
# -----------------------------
def safe_get(ds, key, default=None):
    """Safely fetch a DICOM attribute (tag) with fallback."""
    try:
        return getattr(ds, key)
    except Exception:
        return default

def parse_imagetype(val):
    """Normalize DICOM ImageType to a tuple for readability."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return tuple(val)
    if isinstance(val, str):
        try:
            got = ast.literal_eval(val)
            if isinstance(got, (list, tuple)):
                return tuple(got)
        except Exception:
            pass
        # Fallback for flattened strings
        if "\\" in val:
            return tuple([x.strip() for x in val.split("\\")])
        if "," in val:
            return tuple([x.strip() for x in val.split(",")])
        return (val,)
    return (str(val),)

def iter_series_dirs(root):
    """Yield series directories under root (one-level)."""
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            yield d, p

def stream_histogram_for_series(series_dir):
    """
    Stream through all DICOM files in a series directory and return:
      - aggregated HU histogram (counts vector)
      - lightweight per-series HU stats (mean, std, percentiles) computed online
    This function NEVER stacks slices -> memory safe.
    """
    counts = np.zeros(N_BINS, dtype=np.int64)
    # Running moments for mean/std (Welford)
    n = 0
    mean = 0.0
    M2 = 0.0
    # Reservoir of samples for percentile estimation (subsample to keep memory low)
    rng = np.random.default_rng(123)
    reservoir = []
    target_reservoir = 200000  # ~200k voxels per series cap

    # Grab minimal DICOM once for meta sanity (optional)
    first_meta = {}

    for fname in os.listdir(series_dir):
        if not fname.lower().endswith(".dcm"):
            continue
        fpath = os.path.join(series_dir, fname)
        try:
            ds = pydicom.dcmread(fpath)
            if not first_meta:
                first_meta = dict(
                    Manufacturer=str(safe_get(ds, "Manufacturer", "")),
                    SeriesDescription=str(safe_get(ds, "SeriesDescription", "")),
                    ImageType=parse_imagetype(safe_get(ds, "ImageType", None)),
                    SliceThickness=pd.to_numeric(safe_get(ds, "SliceThickness", np.nan), errors="coerce"),
                    Exposure=pd.to_numeric(safe_get(ds, "Exposure", np.nan), errors="coerce"),
                    KVP=pd.to_numeric(safe_get(ds, "KVP", np.nan), errors="coerce"),
                )

            arr = ds.pixel_array.astype(np.int16)
            slope = float(safe_get(ds, "RescaleSlope", 1.0))
            intercept = float(safe_get(ds, "RescaleIntercept", 0.0))
            hu = arr * slope + intercept
            # Clip HU for stable histograms
            np.clip(hu, HU_MIN, HU_MAX, out=hu)

            # Update histogram in streaming fashion
            h, _ = np.histogram(hu, bins=BINS)
            counts += h

            # Update running mean/std (Welford)
            flat = hu.ravel()
            n_new = flat.size
            if n_new == 0:
                continue
            n_old = n
            n += n_new
            delta = flat.mean() - mean
            mean += delta * (n_new / n)
            M2 += flat.var() * n_new + (delta**2) * (n_old * n_new / n)

            # Subsample for percentiles
            # Keep at most target_reservoir samples uniformly at random
            need = max(0, target_reservoir - len(reservoir))
            if need > 0:
                take = min(need, flat.size)
                idx = rng.choice(flat.size, size=take, replace=False)
                reservoir.append(flat[idx])
            else:
                # Occasionally replace a portion to remain uniform-ish
                if rng.random() < 0.02:
                    idx = rng.choice(flat.size, size=500, replace=False)
                    replace_idx = rng.choice(len(reservoir), size=500, replace=False)
                    for i, r_i in enumerate(replace_idx):
                        reservoir[r_i] = flat[idx[i]]

            # Free slice ASAP
            del ds, arr, hu, flat, h
        except Exception:
            # Keep going even if one file is corrupt
            traceback.print_exc()
            continue

    if n == 0:
        return None, None, None

    # Finalize stats
    var = M2 / max(1, (n - 1))
    std = np.sqrt(max(var, 0.0))
    if reservoir:
        reservoir = np.concatenate(reservoir)
        p10, p50, p90 = np.percentile(reservoir, [10, 50, 90])
    else:
        p10 = p50 = p90 = np.nan

    stats = dict(hu_mean=float(mean), hu_std=float(std), hu_p10=float(p10), hu_p50=float(p50), hu_p90=float(p90))
    return counts, stats, first_meta

# -----------------------------
# Streaming over ALL series
# -----------------------------
# Global aggregators (histograms)
hist_by_label = {0: np.zeros(N_BINS, dtype=np.int64), 1: np.zeros(N_BINS, dtype=np.int64)}
hist_by_quality = {False: np.zeros(N_BINS, dtype=np.int64), True: np.zeros(N_BINS, dtype=np.int64)}
hist_by_label_quality = defaultdict(lambda: np.zeros(N_BINS, dtype=np.int64))

# Per-series summary for later analysis
series_summaries = []

n_series = 0
for sid, sdir in iter_series_dirs(root_dir):
    # Optional: skip non-CTA if we know modality
    if allowed_cta is not None and sid not in allowed_cta:
        continue

    counts, stats, first_meta = stream_histogram_for_series(sdir)
    if counts is None:
        continue

    # Merge with train_EDA for label and quality
    row = meta_lookup.loc[meta_lookup["SeriesInstanceUID"] == sid]
    if row.empty:
        # If not in the table, still keep stats with unknown label/quality
        label = np.nan
        lowq = np.nan
        manufacturer = first_meta.get("Manufacturer", "")
        series_desc = first_meta.get("SeriesDescription", "")
        img_type = first_meta.get("ImageType", None)
        slice_thk = first_meta.get("SliceThickness", np.nan)
        exposure = first_meta.get("Exposure", np.nan)
        kvp = first_meta.get("KVP", np.nan)
    else:
        r = row.iloc[0]
        label = r.get("Aneurysm Present", np.nan)
        lowq = bool(r.get("IMG_Low_Quality", False))
        manufacturer = r.get("Manufacturer", first_meta.get("Manufacturer", ""))
        series_desc = r.get("SeriesDescription", first_meta.get("SeriesDescription", ""))
        img_type = r.get("ImageType", first_meta.get("ImageType", None))
        slice_thk = r.get("SliceThickness", first_meta.get("SliceThickness", np.nan))
        exposure = r.get("Exposure", first_meta.get("Exposure", np.nan))
        kvp = r.get("KVP", first_meta.get("KVP", np.nan))

    # Update global histograms guardedly
    if label in (0, 1):
        hist_by_label[int(label)] += counts
        if lowq in (True, False):
            hist_by_label_quality[(int(label), bool(lowq))] += counts
    if lowq in (True, False):
        hist_by_quality[bool(lowq)] += counts

    # Save per-series summary row
    series_summaries.append({
        "SeriesInstanceUID": sid,
        "Aneurysm Present": label,
        "IMG_Low_Quality": lowq,
        "Manufacturer": manufacturer,
        "SeriesDescription": series_desc,
        "ImageType": img_type,
        "SliceThickness": slice_thk,
        "Exposure": exposure,
        "KVP": kvp,
        **stats
    })

    n_series += 1
    if n_series % 50 == 0:
        print(f"Processed {n_series} series...")
        gc.collect()

print(f"âœ… Done. Total processed series: {n_series}")

series_df = pd.DataFrame(series_summaries)

# -----------------------------
# Plots (aggregated, still light on memory)
# -----------------------------
# 1) By label
plt.figure(figsize=(12,5))
plt.plot(BIN_CENTERS, hist_by_label[0], label="Label 0 (No Aneurysm)")
plt.plot(BIN_CENTERS, hist_by_label[1], label="Label 1 (Aneurysm)")
plt.title("HU Histogram (streaming, ALL series) by Label")
plt.xlabel("HU")
plt.ylabel("Voxel Count")
plt.legend(); plt.grid(True); plt.show()

# 2) By quality flag
plt.figure(figsize=(12,5))
plt.plot(BIN_CENTERS, hist_by_quality[False], label="High Quality (IMG_Low_Quality=False)")
plt.plot(BIN_CENTERS, hist_by_quality[True],  label="Low Quality (IMG_Low_Quality=True)")
plt.title("HU Histogram (streaming, ALL series) by Quality")
plt.xlabel("HU")
plt.ylabel("Voxel Count")
plt.legend(); plt.grid(True); plt.show()

# 3) By (label Ã— quality)
for k, h in hist_by_label_quality.items():
    lbl, q = k
    plt.plot(BIN_CENTERS, h, label=f"Label={lbl}, LowQ={q}")
plt.title("HU Histogram (streaming) by Label Ã— Quality")
plt.xlabel("HU"); plt.ylabel("Voxel Count")
plt.legend(); plt.grid(True); plt.show()

# -----------------------------
# Useful tables for the notebook
# -----------------------------
print("Per-series HU summary (head):")
display(series_df.head())

print("Quality rate by manufacturer (from processed series):")
tbl = (series_df.groupby(["Manufacturer","IMG_Low_Quality"])
              .size().unstack(fill_value=0))
if True in tbl.columns and False in tbl.columns:
    tbl["Percent_Low_Quality"] = 100 * tbl[True] / (tbl[True] + tbl[False])
display(tbl.sort_values(by=tbl.columns[-1], ascending=False))

print("Correlation between HU stats and quality (sanity check):")
display(series_df[["hu_mean","hu_std","hu_p10","hu_p50","hu_p90","IMG_Low_Quality"]]
        .groupby("IMG_Low_Quality").describe().T)











