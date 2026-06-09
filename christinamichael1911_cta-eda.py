import os
import shutil
import pandas as pd

# Path to competition data
dataset_path = "/kaggle/input/rsna-intracranial-aneurysm-detection"

# Your chosen SeriesInstanceUID
series_uid = "1.2.826.0.1.3680043.8.498.10023411164590664678534044036963716636"

# Copy series images
series_path = f"{dataset_path}/series/{series_uid}"
if os.path.exists(series_path):
    shutil.copytree(series_path, f"/kaggle/working/{series_uid}_images")

# Copy segmentation if available
seg_path = f"{dataset_path}/segmentations/series/{series_uid}"
if os.path.exists(seg_path):
    shutil.copytree(seg_path, f"/kaggle/working/{series_uid}_segmentation")

print("Patient data copied to working directory.")



import os

images_dir = f"/kaggle/working/{series_uid}_images"
print("Number of DICOM files:", len(os.listdir(images_dir)))
print("First 5 files:", os.listdir(images_dir)[:5])



import pydicom

# Pick the first file
dicom_path = os.path.join(images_dir, os.listdir(images_dir)[0])
dcm = pydicom.dcmread(dicom_path)

# Show metadata
print(dcm)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
import os

# Kaggle will automatically have dataset mounted in /kaggle/input
base_path = "/kaggle/input/rsna-intracranial-aneurysm-detection"



train_df = pd.read_csv(f"{base_path}/train.csv")
train_df.head()



print(train_df.shape)     # Rows, columns
print(train_df.info())    # Data types, null values
train_df.describe()       # Numeric column summary



plt.figure(figsize=(10,4))
sns.heatmap(train_df.isnull(), cbar=False)
plt.title("Missing Values Overview")
plt.show()



sns.countplot(data=train_df, x="Aneurysm Present")
plt.title("Aneurysm Presence Distribution")
plt.show()



# Convert PatientAge to numeric
train_df["PatientAge"] = pd.to_numeric(train_df["PatientAge"], errors='coerce')

# Age distribution
plt.hist(train_df["PatientAge"].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Sex distribution
sns.countplot(data=train_df, x="PatientSex")
plt.title("Sex Distribution")
plt.show()



artery_cols = train_df.columns[4:-1]  # skip first 4 cols, last is target

artery_sums = train_df[artery_cols].sum().sort_values(ascending=False)
artery_sums.plot(kind='bar', figsize=(12,5))
plt.title("Aneurysm Count by Artery")
plt.ylabel("Count")
plt.show()



numeric_df = train_df.select_dtypes(include=[np.number])
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()



import os
import pydicom
import matplotlib.pyplot as plt

base_path = "/kaggle/input/rsna-intracranial-aneurysm-detection"

# Look inside the 'series' folder
series_root = os.path.join(base_path, "series")
print(os.listdir(series_root)[:5])  # See first few series IDs

# Match only existing series from train.csv
available_series = set(os.listdir(series_root))
train_df = train_df[train_df["SeriesInstanceUID"].isin(available_series)]

# Pick a sample series
sample_series_id = train_df["SeriesInstanceUID"].iloc[0]
series_path = os.path.join(series_root, sample_series_id)

# Pick one DICOM file from that series
sample_file = os.listdir(series_path)[0]
dcm = pydicom.dcmread(os.path.join(series_path, sample_file))

# Show image
plt.imshow(dcm.pixel_array, cmap=plt.cm.gray)
plt.title(f"Sample Image from Series {sample_series_id}")
plt.axis("off")
plt.show()



available_series = set(os.listdir(series_root))
train_df = train_df[train_df["SeriesInstanceUID"].isin(available_series)]
print(f"After filtering: {len(train_df)} rows")



pos_ratio = train_df["Aneurysm Present"].mean()
print(f"Aneurysm Positive Ratio: {pos_ratio:.2%}")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure PatientAge is numeric
train_df['PatientAge'] = pd.to_numeric(train_df['PatientAge'], errors='coerce')

# 1️⃣ Modality Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Modality', order=train_df['Modality'].value_counts().index, palette='viridis')
plt.title("Distribution of Scan Modalities")
plt.ylabel("Number of Scans")
plt.xlabel("Modality")
plt.xticks(rotation=30)
plt.show()

# 2️⃣ Aneurysm Prevalence by Modality
plt.figure(figsize=(6,4))
sns.barplot(
    x='Modality',
    y='Aneurysm Present',
    data=train_df,
    order=train_df['Modality'].value_counts().index,
    palette='coolwarm'
)
plt.title("Aneurysm Prevalence by Modality")
plt.ylabel("Proportion with Aneurysm")
plt.xlabel("Modality")
plt.xticks(rotation=30)
plt.show()

# 3️⃣ Age Distribution per Modality
plt.figure(figsize=(6,4))
sns.boxplot(
    x='Modality',
    y='PatientAge',
    data=train_df,
    order=train_df['Modality'].value_counts().index,
    palette='Set2'
)
plt.title("Age Distribution per Modality")
plt.xlabel("Modality")
plt.ylabel("Patient Age")
plt.xticks(rotation=30)
plt.show()

# 4️⃣ Crosstab: Modality vs Aneurysm Count
modality_crosstab = pd.crosstab(train_df['Modality'], train_df['Aneurysm Present'])
modality_crosstab['Positive_Ratio'] = modality_crosstab[1] / modality_crosstab.sum(axis=1)
print("\nModality vs Aneurysm Counts & Positive Ratio:")
print(modality_crosstab.sort_values(by='Positive_Ratio', ascending=False))



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from ipywidgets import interact, IntSlider
import ast



BASE_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection/'
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
LOCALIZERS_PATH = os.path.join(BASE_PATH, 'train_localizers.csv')
SERIES_DIR = os.path.join(BASE_PATH, 'series')



# Load CSVs
train_df = pd.read_csv(TRAIN_CSV_PATH)
localizers_df = pd.read_csv(LOCALIZERS_PATH)

# Filter only CTA scans
train_df = train_df[train_df['Modality'] == 'CTA'].copy()
print(f"Filtered to {len(train_df)} CTA scans")
train_df.head()



label_columns = train_df.columns[5:]
train_df['aneurysm_present'] = train_df[label_columns].sum(axis=1) > 0



plt.figure(figsize=(6, 4))
sns.countplot(x='aneurysm_present', data=train_df)
plt.title('Aneurysm Presence (CTA only)')
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(data=train_df, x='PatientAge', hue='aneurysm_present',
             kde=True, ax=axes[0], palette='viridis')
axes[0].set_title('Age Distribution (CTA)')

sns.countplot(data=train_df, x='PatientSex', hue='aneurysm_present',
              ax=axes[1], palette='magma')
axes[1].set_title('Sex Distribution (CTA)')

plt.tight_layout()
plt.show()



label_columns = train_df.columns[5:-1]
location_counts = train_df[label_columns].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=location_counts.values, y=location_counts.index, palette='crest')
plt.title('Aneurysm Frequency by Location (CTA)')
plt.show()



def explore_3d_scan(series_uid):
    series_path = os.path.join(SERIES_DIR, series_uid)
    dicom_files = [pydicom.dcmread(os.path.join(series_path, f)) for f in os.listdir(series_path)]
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=True)

    def show_slice(slice_index):
        plt.imshow(dicom_files[slice_index].pixel_array, cmap=plt.cm.bone)
        plt.title(f"Slice {slice_index} - {series_uid}")
        plt.axis('off')
        plt.show()

    interact(show_slice, slice_index=IntSlider(min=0, max=len(dicom_files)-1, step=1, value=len(dicom_files)//2))



cta_uid = "1.2.826.0.1.3680043.8.498.10030095840917973694487307992374923817"
explore_3d_scan(cta_uid)



def visualize_aneurysm_on_slice(series_uid, sop_uid, coords_str):
    series_path = os.path.join(SERIES_DIR, series_uid)

    target_slice_path = ""
    for fname in os.listdir(series_path):
        dcm_path = os.path.join(series_path, fname)
        with pydicom.dcmread(dcm_path, stop_before_pixels=True) as dcm:
            if dcm.SOPInstanceUID == sop_uid:
                target_slice_path = dcm_path
                break
    
    if not target_slice_path:
        print("SOPInstanceUID not found.")
        return

    target_slice = pydicom.dcmread(target_slice_path)
    coords = ast.literal_eval(coords_str)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(target_slice.pixel_array, cmap=plt.cm.bone)
    axes[0].set_title("Original Slice")
    axes[0].axis('off')

    axes[1].imshow(target_slice.pixel_array, cmap=plt.cm.bone)
    axes[1].scatter([coords['x']], [coords['y']], c='red', s=400, marker='+')
    axes[1].set_title("Aneurysm Marked")
    axes[1].axis('off')

    plt.show()

# Try marking aneurysm for your scan
cta_localizer = localizers_df[localizers_df['SeriesInstanceUID'] == cta_uid]
if not cta_localizer.empty:
    first_row = cta_localizer.iloc[0]
    visualize_aneurysm_on_slice(first_row['SeriesInstanceUID'],
                                first_row['SOPInstanceUID'],
                                first_row['coordinates'])
else:
    print("No aneurysm coordinates found for this scan.")


