import pandas as pd
train_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
train_df.head()


import plotly.express as px

location_cols = [col for col in train_df.columns if col not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
location_counts = train_df[location_cols].sum().sort_values(ascending=False)

fig = px.bar(
    location_counts,
    orientation='v',
    title='ğŸ“Š Aneurysm Count by Location',
    labels={'value': 'Count', 'index': 'Location'},
    color=location_counts.values,
    color_continuous_scale=[[0, '#00BFC4'], [1, '#C77CFF']],
    template='plotly_dark',
    text_auto=True
)
fig.show()



modality_counts = train_df['Modality'].value_counts()

fig = px.pie(
    names=modality_counts.index,
    values=modality_counts.values,
    title='ğŸ§  Imaging Modality Distribution',
    hole=0.4,
    color_discrete_sequence=['#00BFC4', '#C77CFF'],
    template='plotly_dark'
)
fig.update_traces(texttemplate='%{value} (%{percent})', textposition='inside')

fig.show()



import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('dark_background')
plt.figure(figsize=(10,6))
sns.histplot(
    data=train_df,
    x='PatientAge',
    hue='Aneurysm Present',
    bins=30,
    kde=True,
    palette={0: '#00BFC4', 1: '#C77CFF'}
)
plt.title("Age Distribution by Aneurysm Presence", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(alpha=0.2)
plt.show()



pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'], normalize='index') * 100



fig = px.histogram(
    train_df,
    x='Aneurysm Present',
    title='âš–ï¸� Class Imbalance: Any Aneurysm Present',
    color='Aneurysm Present',
    text_auto=True,
    color_discrete_map={0: '#00BFC4', 1: '#C77CFF'},
    template='plotly_dark'
)
fig.update_xaxes(type='category', tickvals=[0, 1], ticktext=["No Aneurysm", "Aneurysm"])
fig.show()



aneurysm_counts = train_df['Aneurysm Present'].value_counts()
print(aneurysm_counts)


total_cases = len(train_df)
print(f"Total cases: {total_cases}")


# Simple % table (can be styled in notebook output)
gender_ct = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'], normalize='index') * 100
gender_ct = gender_ct.rename(columns={0: 'No Aneurysm (%)', 1: 'Aneurysm (%)'})
gender_ct.style.background_gradient(cmap='crest').format("{:.1f}%")



import pandas as pd
import ast

localizers_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')

# Convert coordinate strings to dicts
localizers_df['coords'] = localizers_df['coordinates'].apply(ast.literal_eval)
localizers_df['x'] = localizers_df['coords'].apply(lambda d: d['x'])
localizers_df['y'] = localizers_df['coords'].apply(lambda d: d['y'])

localizers_df.drop(columns=['coordinates', 'coords'], inplace=True)
localizers_df.head()



fig = px.density_heatmap(
    localizers_df,
    x='x',
    y='y',
    nbinsx=50,
    nbinsy=50,
    title='ğŸ§  Heatmap of Aneurysm Locations in Image Space',
    color_continuous_scale='Turbo',
    template='plotly_dark',
)
fig.update_yaxes(autorange="reversed")
fig.show()



fig = px.scatter(
    localizers_df,
    x='x',
    y='y',
    color='location',
    title='ğŸ§  2D Scatter of Aneurysm Coordinates by Location',
    template='plotly_dark',
    color_discrete_sequence=px.colors.qualitative.Dark24
)
fig.update_yaxes(autorange="reversed")
fig.show()



df = train_df
# Identify location columns correctly
location_cols = df.columns[4:-1]  # skip UID, Age, Sex, Modality, and skip final label
location_df = df[location_cols].astype(int)  # just in case they're still object type

# Co-occurrence matrix
co_matrix = location_df.T.dot(location_df)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(co_matrix, cmap="magma", annot=True, fmt=".0f", linewidths=0.5)
plt.title("Aneurysm Co-occurrence Matrix", fontsize=16, color='white')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.gca().set_facecolor('black')
plt.gcf().set_facecolor('#111111')
plt.tight_layout()
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'location_df' is already created and contains the location columns
# location_df = df[location_cols].astype(int)

# 1. Location Count Distribution (number of 1s per row)
location_counts = location_df.sum(axis=1)

# --- Plotting Section ---
plt.figure(figsize=(8, 5))
sns.histplot(location_counts, bins=range(1, location_counts.max()+2), kde=False, color="teal")
plt.title("Distribution of Aneurysm Locations per Patient")
plt.xlabel("Number of Locations with Aneurysm")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Printing the Values Section ---
# Use value_counts() to get the count for each number of locations and sort by the number of locations
value_distribution = location_counts.value_counts().sort_index()

print("Distribution of Aneurysm Locations per Patient:")
print(value_distribution)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is already loaded and preprocessed (i.e., binary columns are int)

# Filter only rows where aneurysm is present
df_pos = df[df["Aneurysm Present"] == 1]

# List of all aneurysm location columns
location_cols = df_pos.columns[4:-1]  # From 'Left Infraclinoid...' to 'Other Posterior Circulation'

# Group by PatientSex and sum each location
sex_location = df_pos.groupby("PatientSex")[location_cols].sum().T

# Reset index for plotting
sex_location = sex_location.reset_index().melt(id_vars="index", var_name="Sex", value_name="Count")
sex_location = sex_location.rename(columns={"index": "Location"})

# Plot
plt.figure(figsize=(16, 6))
sns.barplot(data=sex_location, x="Location", y="Count", hue="Sex")
plt.xticks(rotation=45, ha="right")
plt.title("Aneurysm Location Frequency by Sex")
plt.ylabel("Aneurysm Count")
plt.xlabel("Location")
plt.tight_layout()
plt.show()



# List of location columns
location_cols = df.columns[4:-1]

# Filter only positive cases
df_pos = df[df["Aneurysm Present"] == 1]

# Prepare a DataFrame: for each location, collect patients with that location = 1
age_location = []

for loc in location_cols:
    subset = df_pos[df_pos[loc] == 1][["PatientAge"]].copy()
    subset["Location"] = loc
    age_location.append(subset)

# Combine all into one DataFrame
age_location_df = pd.concat(age_location)

# Plot
plt.figure(figsize=(16, 6))
sns.boxplot(data=age_location_df, x="Location", y="PatientAge", palette="crest")
plt.xticks(rotation=45, ha="right")
plt.title("Patient Age Distribution per Aneurysm Location")
plt.ylabel("Age")
plt.xlabel("Location")
plt.tight_layout()
plt.show()



plt.figure(figsize=(16, 6))
sns.violinplot(data=age_location_df, x="Location", y="PatientAge", palette="crest")
plt.xticks(rotation=45, ha="right")
plt.title("Patient Age Distribution per Aneurysm Location (Violin Plot)")
plt.ylabel("Age")
plt.xlabel("Location")
plt.tight_layout()
plt.show()



# Filter rows with aneurysm present
df_pos = df[df["Aneurysm Present"] == 1].copy()

# Initialize a dictionary to store counts
modality_counts = {}

for loc in location_cols:
    subset = df_pos[df_pos[loc] == 1]
    modality_distribution = subset["Modality"].value_counts()
    modality_counts[loc] = modality_distribution

# Convert to DataFrame and fill missing values with 0
modality_df = pd.DataFrame(modality_counts).T.fillna(0).astype(int)
modality_df = modality_df[["CTA", "MRA"]] if "CTA" in modality_df.columns and "MRA" in modality_df.columns else modality_df

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(modality_df.T, annot=True, cmap="crest", fmt="d")
plt.title("Modality Preference per Aneurysm Location")
plt.xlabel("Location")
plt.ylabel("Modality")
plt.tight_layout()
plt.show()



# Filter rows with aneurysm present
df_pos = df[df["Aneurysm Present"] == 1].copy()

# Initialize a dictionary to store counts
modality_counts = {}

for loc in location_cols:
    subset = df_pos[df_pos[loc] == 1]
    modality_distribution = subset["Modality"].value_counts()
    modality_counts[loc] = modality_distribution

# Convert to DataFrame and fill missing values with 0
modality_df = pd.DataFrame(modality_counts).T.fillna(0).astype(int)
#modality_df = modality_df[["CTA", "MRA"]] if "CTA" in modality_df.columns and "MRA" in modality_df.columns else modality_df

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(modality_df.T, annot=True, cmap="crest", fmt="d")
plt.title("Modality Preference per Aneurysm Location")
plt.xlabel("Location")
plt.ylabel("Modality")
plt.tight_layout()
plt.show()


modality_df_melted = modality_df.reset_index().melt(id_vars="index", value_name="Count", var_name="Modality")
modality_df_melted.rename(columns={"index": "Location"}, inplace=True)

plt.figure(figsize=(14, 6))
sns.barplot(data=modality_df_melted, x="Location", y="Count", hue="Modality", palette="crest")
plt.title("Modality Usage per Aneurysm Location")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Location")
plt.ylabel("Patient Count")
plt.tight_layout()
plt.show()



import ast

train_localizations = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")
# Parse coordinates
train_localizations["coord_dict"] = train_localizations["coordinates"].apply(ast.literal_eval)
train_localizations["x"] = train_localizations["coord_dict"].apply(lambda d: d["x"])
train_localizations["y"] = train_localizations["coord_dict"].apply(lambda d: d["y"])

# Top 4-5 most frequent locations
top_locations = train_localizations["location"].value_counts().head(7).index

# Plot heatmap per location
for loc in top_locations:
    subset = train_localizations[train_localizations["location"] == loc]
    plt.figure(figsize=(6, 5))
    sns.kdeplot(data=subset, x="x", y="y", fill=True, cmap="crest")
    plt.title(f"Spatial Heatmap for {loc}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
label_freq = train[location_cols].sum().sort_values(ascending=False).rename("train.csv")
location_cols = [col for col in train_df.columns if col not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
label_freq = train_df[location_cols].sum().sort_values(ascending=False).rename("train.csv")
localizer_freq = train_localizations["location"].value_counts().rename("train_localizations.csv")
freq_comparison = pd.concat([label_freq, localizer_freq], axis=1).fillna(0).astype(int)
ax = freq_comparison.plot(kind="bar", figsize=(14, 6), color=['#00BFC4', '#C77CFF'])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 5), 
                textcoords='offset points')

plt.title("Frequency Comparison of Aneurysm Locations")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



import os
import pandas as pd

series_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
series_folders = [f for f in os.listdir(series_dir) if os.path.isdir(os.path.join(series_dir, f))]

series_counts = []
for folder in series_folders:
    dcm_files = os.listdir(os.path.join(series_dir, folder))
    series_counts.append({'SeriesInstanceUID': folder, 'NumSlices': len(dcm_files)})

df_series = pd.DataFrame(series_counts)
df_series.describe()


# 


import pydicom

sample_path = os.path.join(series_dir, series_folders[0])
sample_file = os.listdir(sample_path)[0]
dcm = pydicom.dcmread(os.path.join(sample_path, sample_file))

print(f"Orientation: {dcm.ImageOrientationPatient}")
print(f"Voxel spacing: {dcm.PixelSpacing}")
print(f"Slice Thickness: {dcm.SliceThickness}")



import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np

def load_series(series_path):
    files = sorted(os.listdir(series_path), key=lambda x: pydicom.dcmread(os.path.join(series_path, x)).InstanceNumber)
    images = [pydicom.dcmread(os.path.join(series_path, f)).pixel_array for f in files]
    return np.stack(images)

volume = load_series(os.path.join(series_dir, series_folders[0]))

# Scrollable plot
from ipywidgets import interact
@interact(slice=(0, volume.shape[0]-1))
def show_slice(slice=0):
    plt.imshow(volume[slice], cmap='gray')
    plt.axis('off')
    plt.show()



import os
import pydicom

# Count slices and shape per series
dicom_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
series_stats = []

for series_id in os.listdir(dicom_dir)[:50]:  # limit for speed
    series_path = os.path.join(dicom_dir, series_id)
    files = os.listdir(series_path)
    num_slices = len(files)
    sample_dcm = pydicom.dcmread(os.path.join(series_path, files[0]))
    shape = (sample_dcm.Rows, sample_dcm.Columns)
    series_stats.append({"SeriesInstanceUID": series_id, "Slices": num_slices, "Shape": shape})

pd.DataFrame(series_stats).value_counts("Shape").plot(kind="barh", title="Common Image Resolutions")



voxel_data = []

for series_id in os.listdir(dicom_dir)[:50]:
    slices = []
    for f in sorted(os.listdir(os.path.join(dicom_dir, series_id))):
        path = os.path.join(dicom_dir, series_id, f)
        dcm = pydicom.dcmread(path)
        slices.append(dcm)

    try:
        spacing = slices[0].PixelSpacing
        thickness = float(slices[0].SliceThickness)
        voxel_data.append({
            "SeriesInstanceUID": series_id,
            "PixelSpacingX": spacing[0],
            "PixelSpacingY": spacing[1],
            "SliceThickness": thickness,
            "NumSlices": len(slices)
        })
    except:
        continue

voxel_df = pd.DataFrame(voxel_data)
sns.histplot(voxel_df["SliceThickness"], bins=20)
plt.title("Slice Thickness Distribution")



seg_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations"
seg_series = [f.replace(".nii.gz", "") for f in os.listdir(seg_dir)]
print("Segmented Series:", len(seg_series))

# % of series with segmentation
seg_percent = len(seg_series) / len(os.listdir(dicom_dir)) * 100
print(f"{seg_percent:.2f}% of series have vessel segmentation.")


