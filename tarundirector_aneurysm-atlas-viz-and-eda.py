# General-purpose libraries
import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Medical image handling
import pydicom  # for reading DICOM files
import nibabel as nib  # for NIfTI files in segmentations
import cv2  # for basic image operations (optional)

# Display DICOMs in notebooks
from matplotlib.patches import Rectangle

# File navigation
from glob import glob

# Warnings
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
localizers_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")


display(train_df.head(2))


display(localizers_df.head(2))


# Organize RSNA datasets
datasets = {
    "Train Labels": train_df,
    "Aneurysm Localizations": localizers_df,
}

# Print shapes
for name, df in datasets.items():
    num_rows, num_cols = df.shape
    print(f"{name}:")
    print(f"  Number of Rows: {num_rows}")
    print(f"  Number of Columns: {num_cols}\n")


# Count duplicate rows in RSNA train_df
train_duplicates = train_df.duplicated().sum()

# Count duplicate rows in RSNA localizers_df
localizers_duplicates = localizers_df.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_df: {train_duplicates}")
print(f"Number of duplicate rows in localizers_df: {localizers_duplicates}")


def describe_numerical(df, name):
    # Find numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        print(f"\nâš ï¸� No numeric columns found in {name}. Skipping description.")
        return None

    print(f'\nâ�¡ï¸� Description of numerical columns in {name}')
    return df[numeric_cols].describe().T.style.background_gradient(cmap='viridis')

# Apply to both datasets
display(describe_numerical(train_df, "train_df"))


# Column-level summary for RSNA train_df
columns = train_df.columns.tolist()

# Missing value summary
missing_summary = pd.DataFrame({
    'Feature': columns,
    '[TRAIN] Missing Count': train_df.isnull().sum().values,
    '[TRAIN] Missing %': (train_df.isnull().sum().values / len(train_df)) * 100
})

# Unique value summary
unique_summary = pd.DataFrame({
    'Feature': columns,
    'Unique Values [TRAIN]': train_df.nunique().values
})

# Data types
dtypes_summary = pd.DataFrame({
    'Feature': columns,
    'Data Type': train_df.dtypes.values
})

# Merge all summaries
feature_summary = (
    missing_summary
    .merge(unique_summary, on='Feature', how='left')
    .merge(dtypes_summary, on='Feature', how='left')
)

# Display styled DataFrame
feature_summary.fillna(0).style.background_gradient(cmap='viridis')


# Column-level summary for RSNA localizers_df
columns = localizers_df.columns.tolist()

# Missing value summary
missing_summary_loc = pd.DataFrame({
    'Feature': columns,
    '[LOCALIZERS] Missing Count': localizers_df.isnull().sum().values,
    '[LOCALIZERS] Missing %': (localizers_df.isnull().sum().values / len(localizers_df)) * 100
})

# Unique value summary
unique_summary_loc = pd.DataFrame({
    'Feature': columns,
    'Unique Values [LOCALIZERS]': localizers_df.nunique().values
})

# Data types
dtypes_summary_loc = pd.DataFrame({
    'Feature': columns,
    'Data Type': localizers_df.dtypes.values
})

# Merge all summaries
localizer_summary = (
    missing_summary_loc
    .merge(unique_summary_loc, on='Feature', how='left')
    .merge(dtypes_summary_loc, on='Feature', how='left')
)

# Display styled DataFrame
localizer_summary.fillna(0).style.background_gradient(cmap='viridis')


import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Glyph.*missing from current font")

# Custom colors
hist_color = '#3498db'
box_color = '#f39c12'

# Copy and label dataset
train_rsna = train_df.copy()
train_rsna['Dataset'] = 'Train'

# Plotting function for single variable in RSNA
def plot_patient_age_distribution(data, column='PatientAge', dataset_label="RSNA Train Data"):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 6),
        gridspec_kw={'height_ratios': [5, 1.2, 1.5], 'hspace': 0.5}
    )

    # Histogram
    ax_hist = axes[0]
    sns.histplot(data=data, x=column, kde=True, bins=30, color=hist_color, ax=ax_hist, label='Train')
    ax_hist.set_xlabel(column)
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(
        f"Histogram & Box Plot for\n{column} â€” {dataset_label}",
        fontweight='bold',
        fontsize=10
    )
    ax_hist.legend()

    # Boxplot
    ax_box = axes[1]
    sns.boxplot(data=data, x=column, palette=[box_color], ax=ax_box, boxprops=dict(facecolor=box_color, alpha=0.6))
    ax_box.set_xlabel("")
    ax_box.set_ylabel("")
    ax_box.set_title("")

    # Spacer row â€” turn off
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# Run the plot
plot_patient_age_distribution(train_rsna)

# Cleanup
train_rsna.drop('Dataset', axis=1, inplace=True)


import ast

# Step 1: Safe copy
localizers_temp = localizers_df.copy()

# Step 2: Parse stringified dict and extract x and y
localizers_temp['parsed_coordinates'] = localizers_temp['coordinates'].apply(ast.literal_eval)
localizers_temp['x'] = localizers_temp['parsed_coordinates'].apply(lambda d: d.get('x'))
localizers_temp['y'] = localizers_temp['parsed_coordinates'].apply(lambda d: d.get('y'))
localizers_temp['Dataset'] = 'Localizers'

# --- Step 2: Reuse your plotting style ---
def plot_coordinate_distribution(data, column, dataset_label="RSNA Localizers"):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 6),
        gridspec_kw={'height_ratios': [5, 1.2, 1.5], 'hspace': 0.5}
    )

    # Histogram
    ax_hist = axes[0]
    sns.histplot(data=data, x=column, kde=True, bins=30, color=hist_color, ax=ax_hist, label='Localizers')
    ax_hist.set_xlabel(column)
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(
        f"Histogram & Box Plot for\n{column} â€” {dataset_label}",
        fontweight='bold',
        fontsize=10
    )
    ax_hist.legend()

    # Boxplot
    ax_box = axes[1]
    sns.boxplot(data=data, x=column, palette=[box_color], ax=ax_box, boxprops=dict(facecolor=box_color, alpha=0.6))
    ax_box.set_xlabel("")
    ax_box.set_ylabel("")
    ax_box.set_title("")

    # Spacer
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# --- Step 3: Plot without affecting original localizers_df ---
plot_coordinate_distribution(localizers_temp, 'x', "Coordinate X")
plot_coordinate_distribution(localizers_temp, 'y', "Coordinate Y")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# Define palettes
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779',
                     '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']
custom_palette = ['#3498db']  # Only Train

# --- Plotting function (unchanged structure)
def create_categorical_plots(variable, data, source_name):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pie Chart
    plt.subplot(1, 2, 1)
    value_counts = data[variable].value_counts()

    # Collapse small categories
    threshold = 0.05 * value_counts.sum()
    filtered_values = value_counts.copy()
    filtered_values[value_counts < threshold] = 0
    filtered_values = filtered_values[filtered_values > 0]
    other_count = value_counts.sum() - filtered_values.sum()
    if other_count > 0:
        filtered_values['Other'] = other_count

    wedges, texts, autotexts = plt.pie(
        filtered_values,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        colors=pie_chart_palette[:len(filtered_values)],
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=[0.05 if p > 5 else 0 for p in filtered_values],
        textprops={'fontsize': 10}
    )

    plt.title("\n".join(textwrap.wrap(f"Pie Chart for {variable} â€” {source_name}", width=50)), fontweight='bold')
    plt.legend(filtered_values.index, loc="upper left", bbox_to_anchor=(1, 1))

    # Countplot
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=data,
        y=variable,
        hue='dataset',
        palette=custom_palette,
        alpha=0.85
    )
    plt.ylabel(variable)
    plt.xlabel("Count")
    plt.title("\n".join(textwrap.wrap(f"Countplot for {variable} â€” {source_name}", width=50)), fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- Step 1: Prep train_df with categorical columns
train_cats = train_df.copy()
train_cats['dataset'] = 'train'

for var in ['PatientSex', 'Modality', 'Aneurysm Present']:
    create_categorical_plots(var, train_cats, "Train Features")


# --- Step 2: Derive 'Aneurysm Location' column from artery location columns
aneurysm_cols = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation'
]

# Create a long-form dataframe to find where aneurysm is present
aneurysm_location_df = train_df.copy()
aneurysm_location_df['dataset'] = 'train'

# Melt to long form and filter
melted = aneurysm_location_df.melt(
    id_vars=['dataset'],
    value_vars=aneurysm_cols,
    var_name='Aneurysm Location',
    value_name='Presence'
)
melted = melted[melted['Presence'] == 1]  # Only keep positive aneurysm locations

# Plot the derived aneurysm locations
create_categorical_plots('Aneurysm Location', melted, "Derived Aneurysm Locations")


import os
import pydicom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Colors
hist_color = '#3498db'
box_color  = '#f39c12'
sns.set_style("whitegrid")

dicom_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'

# --- Build DataFrame with NumSlices and VolumeDepth_mm
series_stats = []
for sid in os.listdir(dicom_dir):
    series_path = os.path.join(dicom_dir, sid)
    if not os.path.isdir(series_path):
        continue
    files = sorted(os.listdir(series_path))
    if not files:
        continue
    num_slices = len(files)

    # read one DICOM for thickness
    try:
        ds = pydicom.dcmread(os.path.join(series_path, files[0]), stop_before_pixels=True)
        st = float(ds.SliceThickness) if 'SliceThickness' in ds else None
    except:
        st = None

    series_stats.append({
        "NumSlices": num_slices,
        "VolumeDepth_mm": num_slices * st if st else None
    })

stats_df = pd.DataFrame(series_stats).dropna()

# --- Plot both features in a 2Ã—2 grid (no spacer)
fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 8),
    gridspec_kw={'height_ratios': [5, 1.5], 'hspace': 0.4}
)

features = ["NumSlices", "VolumeDepth_mm"]
titles   = ["Series Slice Count", "Volume Depth (mm)"]

for col_idx, feature in enumerate(features):
    # Histogram row
    ax_hist = axes[0, col_idx]
    sns.histplot(
        data=stats_df,
        x=feature,
        kde=True,
        bins=30,
        color=hist_color,
        ax=ax_hist
    )
    ax_hist.set_title(f"Histogram & Box Plot for\n{titles[col_idx]}", fontweight='bold', fontsize=10)
    ax_hist.set_xlabel(feature)
    ax_hist.set_ylabel("Frequency")

    # Boxplot row
    ax_box = axes[1, col_idx]
    sns.boxplot(
        data=stats_df,
        x=feature,
        color=box_color,
        ax=ax_box,
        boxprops=dict(facecolor=box_color, alpha=0.6)
    )
    ax_box.set_xlabel("")
    ax_box.set_ylabel("")

plt.tight_layout()
plt.show()


import os
import pydicom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Color and style
custom_color = '#3498db'
sns.set_style("whitegrid")

dicom_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'

# --- Extract voxel spacing info
voxel_data = []
for sid in os.listdir(dicom_dir):
    series_path = os.path.join(dicom_dir, sid)
    files = sorted(os.listdir(series_path))
    if not files:
        continue

    try:
        sample_dcm = pydicom.dcmread(
            os.path.join(series_path, files[0]),
            stop_before_pixels=True
        )
        ps = sample_dcm.PixelSpacing if 'PixelSpacing' in sample_dcm else [None, None]
        st = float(sample_dcm.SliceThickness) if 'SliceThickness' in sample_dcm else None
        voxel_data.append({
            "PixelSpacingX": float(ps[0]) if ps[0] else None,
            "PixelSpacingY": float(ps[1]) if ps[1] else None,
            "SliceThickness": st
        })
    except:
        continue

voxel_df = pd.DataFrame(voxel_data)

# --- KDE Plots for spacing & thickness
plt.figure(figsize=(14, 5))
for i, col in enumerate(["PixelSpacingX", "PixelSpacingY", "SliceThickness"]):
    plt.subplot(1, 3, i + 1)
    sns.kdeplot(data=voxel_df, x=col, fill=True, color=custom_color)
    plt.title(col, fontweight='bold')

plt.suptitle("Voxel Spacing & Slice Thickness", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Color Setup
custom_color = '#3498db'

# --- Extract DICOM shape info (non-destructive)
shape_data = []
for sid in os.listdir(dicom_dir):
    path = os.path.join(dicom_dir, sid)
    files = os.listdir(path)
    if files:
        dcm = pydicom.dcmread(os.path.join(path, files[0]), stop_before_pixels=True)
        shape_data.append({
            'SeriesInstanceUID': sid,
            'Rows': dcm.Rows,
            'Columns': dcm.Columns,
            'Shape': f"{dcm.Rows}x{dcm.Columns}"
        })

shape_df = pd.DataFrame(shape_data)

# --- Count for top shapes
top_shapes = shape_df['Shape'].value_counts().nlargest(10)

# --- Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ğŸ“ˆ Left: Scatter plot of all (Rows, Columns)
sns.scatterplot(
    data=shape_df,
    x='Columns',
    y='Rows',
    color=custom_color,
    alpha=0.6,
    ax=axes[0]
)
axes[0].set_title("DICOM Image Dimensions (Rows Ã— Columns)", fontweight='bold')
axes[0].set_xlabel("Columns")
axes[0].set_ylabel("Rows")
axes[0].grid(True)

# ğŸ“Š Right: Bar chart of top 6 common resolutions
top_shapes.sort_values().plot(
    kind='barh',
    color=custom_color,
    ax=axes[1]
)
axes[1].set_title("10 Most Common Image Resolutions", fontweight='bold')
axes[1].set_xlabel("Number of Series")
axes[1].set_ylabel("Resolution")

plt.tight_layout()
plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom

# Palette
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779']
custom_color = '#3498db'

# DICOM directory
dicom_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'

# --- Step 1: Gather modality, shape, and slice info
modality_data = []
for sid in os.listdir(dicom_dir):
    path = os.path.join(dicom_dir, sid)
    files = os.listdir(path)
    if files:
        dcm = pydicom.dcmread(os.path.join(path, files[0]), stop_before_pixels=True)
        modality = dcm.Modality if 'Modality' in dcm else 'Unknown'
        shape = f"{dcm.Rows}x{dcm.Columns}"
        modality_data.append({
            'SeriesInstanceUID': sid,
            'Modality': modality,
            'Shape': shape,
            'NumSlices': len(files)
        })

modality_df = pd.DataFrame(modality_data)

# --- Filter: Top 6 shapes only
top_shapes = modality_df['Shape'].value_counts().nlargest(6).index
modality_df_filtered = modality_df[modality_df['Shape'].isin(top_shapes)]

# --- Plotting side-by-side
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# ğŸ“Š Left: Boxplot of NumSlices by Modality (horizontal)
sns.boxplot(
    data=modality_df,
    y='Modality',
    x='NumSlices',
    color=custom_color,
    ax=axes[0]
)
axes[0].set_title("Slice Count Distribution by Modality", fontweight='bold')
axes[0].set_xlabel("Number of Slices")
axes[0].set_ylabel("Modality")

# ğŸ“ˆ Right: Countplot of Modality vs Shape
sns.countplot(
    data=modality_df_filtered,
    x='Modality',
    hue='Shape',
    palette=pie_chart_palette[:len(top_shapes)],
    ax=axes[1]
)
axes[1].set_title("Modality-wise Image Shape Distribution", fontweight='bold')
axes[1].set_xlabel("Modality")
axes[1].set_ylabel("Number of Series")
axes[1].legend(title='Image Shape', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap

# --- Determine segmentation coverage
seg_series = set(f.replace(".nii.gz", "") for f in os.listdir(seg_dir))
voxel_df['Segmented'] = voxel_df['SeriesInstanceUID'].isin(seg_series)

# --- Plot: Pie chart (left) + Countplot (right)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_style('whitegrid')

# ğŸ“Š Pie Chart (left)
value_counts = voxel_df['Segmented'].value_counts()
labels = ['No', 'Yes']
values = [value_counts.get(False, 0), value_counts.get(True, 0)]

wedges, texts, autotexts = axes[0].pie(
    values,
    labels=labels,
    autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
    colors=pie_chart_palette[:2],
    startangle=140,
    wedgeprops=dict(width=0.3),
    explode=[0.05 if p > 5 else 0 for p in values],
    textprops={'fontsize': 10}
)
axes[0].set_title("Segmentation Coverage â€” Pie View", fontweight='bold')

# ğŸ“ˆ Countplot (right)
sns.countplot(
    data=voxel_df,
    x='Segmented',
    palette=[custom_color, '#aaaaaa'],
    ax=axes[1]
)
axes[1].set_title("Segmentation Coverage â€” Count View", fontweight='bold')
axes[1].set_xlabel("Has Segmentation")
axes[1].set_ylabel("Number of Series")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['No', 'Yes'])

plt.tight_layout()
plt.show()


import os, ast, random
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Directories
dicom_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'

# 1) Define the 13 aneurysm location columns
aneurysm_cols = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation'
]

# 2) Pick one random valid example per region
examples = {}
for region in aneurysm_cols:
    sids = train_df.loc[train_df[region] == 1, 'SeriesInstanceUID'].unique().tolist()
    random.shuffle(sids)
    for sid in sids:
        series_path = os.path.join(dicom_dir, sid)
        if not os.path.isdir(series_path):
            continue
        loc = localizers_df[
            (localizers_df.SeriesInstanceUID == sid) &
            (localizers_df.location == region)
        ]
        if loc.empty:
            continue
        examples[region] = {
            'SeriesInstanceUID': sid,
            'localizer': loc.sample(1).iloc[0]
        }
        break

# 3) Setup grid: 2 regions per row â†’ 4 columns
regions = list(examples.items())
n = len(regions)
rows = int(np.ceil(n / 2))

fig, axes = plt.subplots(rows, 4, figsize=(16, 5 * rows))
if rows == 1:
    axes = axes[np.newaxis, :]

for idx, (region, info) in enumerate(regions):
    row = idx // 2
    col_offset = (idx % 2) * 2

    sid     = info['SeriesInstanceUID']
    loc_row = info['localizer']
    coords  = ast.literal_eval(loc_row['coordinates'])
    x, y    = int(coords['x']), int(coords['y'])
    sop_uid = loc_row['SOPInstanceUID']

    # Metadata
    meta = train_df.loc[train_df['SeriesInstanceUID'] == sid].iloc[0]
    age, sex = meta['PatientAge'], meta['PatientSex']

    # Load slice
    files = sorted(os.listdir(os.path.join(dicom_dir, sid)))
    dcm_file = next(f for f in files if f.startswith(sop_uid))
    ds = pydicom.dcmread(os.path.join(dicom_dir, sid, dcm_file))
    img = ds.pixel_array
    if img.ndim == 3:
        img = img[0]
    modality = ds.Modality

    # Compute box
    half = 32
    xmin = max(x - half, 0)
    ymin = max(y - half, 0)
    box_w = min(2*half, img.shape[1] - xmin)
    box_h = min(2*half, img.shape[0] - ymin)

    # Full slice
    ax_full = axes[row, col_offset]
    ax_full.imshow(img, cmap='gray')
    ax_full.add_patch(Rectangle((xmin, ymin), box_w, box_h,
                                edgecolor='red', linewidth=2, fill=False))
    ax_full.set_title(
        f"{region}\nModality: {modality}   Age: {age}   Sex: {sex}",
        fontweight='bold', loc='center'
    )
    ax_full.axis('off')

    # Zoom
    zoom = img[ymin:ymin+box_h, xmin:xmin+box_w]
    ax_zoom = axes[row, col_offset + 1]
    ax_zoom.imshow(zoom, cmap='gray')
    # Ensure border is drawn last
    rect = Rectangle((0, 0), zoom.shape[1], zoom.shape[0],
                     edgecolor='red', linewidth=4, fill=False, zorder=10)
    ax_zoom.add_patch(rect)
    ax_zoom.set_title("Zoomed Aneurysm Region", fontweight='bold', loc='center')
    ax_zoom.axis('off')

# Hide any unused axes
total_plots = rows * 4
for empty_idx in range(n*2, total_plots):
    r = empty_idx // 4
    c = empty_idx % 4
    axes[r, c].axis('off')

plt.tight_layout()
plt.show()


import os
import random
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from IPython.display import HTML

# --- Paths
dicom_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'

# --- Helper to load full DICOM volume
def load_dicom_volume(series_path):
    try:
        files = sorted(os.listdir(series_path), key=lambda f: pydicom.dcmread(os.path.join(series_path, f)).InstanceNumber)
        slices = []
        for f in files:
            dcm = pydicom.dcmread(os.path.join(series_path, f))
            img = dcm.pixel_array
            if img.ndim == 3:  # multi-frame
                img = img[0]
            slices.append(img)
        return np.stack(slices), dcm  # return last ds for metadata
    except Exception as e:
        print(f"Error loading series: {e}")
        return None, None

# --- Select one random valid series
valid_sids = [sid for sid in os.listdir(dicom_dir) if os.path.isdir(os.path.join(dicom_dir, sid))]
random.shuffle(valid_sids)

volume, meta = None, None
for sid in valid_sids:
    try:
        print(f"Trying series: {sid}")
        volume, meta = load_dicom_volume(os.path.join(dicom_dir, sid))
        if volume is not None and volume.shape[0] >= 5:  # filter for actual 3D volumes
            print(f"Successfully loaded series: {sid}")
            break
    except Exception as e:
        print(f"Error with series {sid}: {e}")
        continue

if volume is None:
    print("âš ï¸� No valid DICOM series found.")
else:
    # --- Metadata
    modality = getattr(meta, 'Modality', 'Unknown')
    age = getattr(meta, 'PatientAge', 'N/A')
    sex = getattr(meta, 'PatientSex', 'N/A')
    shape = f"{volume.shape[1]}Ã—{volume.shape[2]}"
    num_slices = volume.shape[0]
    
    print(f"Loaded {modality} volume with {num_slices} slices")
    print(f"Patient Age: {age}, Sex: {sex}")
    
    # --- Normalize pixel values for better visualization
    volume_min = volume.min()
    volume_max = volume.max()
    if volume_max > volume_min:
        normalized_volume = (volume - volume_min) / (volume_max - volume_min)
    else:
        normalized_volume = volume
    
    # --- Create images for each slice and convert to base64
    # Limit to a reasonable number of slices to avoid browser performance issues
    max_slices = min(num_slices, 100)  # Limit to 100 slices max
    step = max(1, num_slices // max_slices)
    
    base64_images = []
    for i in range(0, num_slices, step):
        plt.figure(figsize=(8, 8), dpi=80)
        plt.imshow(normalized_volume[i], cmap='gray')
        plt.axis('off')
        plt.title(f"{modality} | Age: {age} | Sex: {sex} | Shape: {shape} | Slice {i+1}/{num_slices}", 
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        base64_images.append(img_str)
        plt.close()
    
    # --- Create HTML with embedded JavaScript slider
    html_content = f"""
    <div style="text-align:center; max-width:800px; margin:0 auto;">
        <div style="margin-bottom:10px;">
            <img id="sliceImage" src="data:image/png;base64,{base64_images[0]}" 
                 style="max-width:100%; height:auto; border:1px solid #ddd;">
        </div>
        <div style="margin:10px 0;">
            <input type="range" id="sliceSlider" min="0" max="{len(base64_images)-1}" value="0" 
                   style="width:80%; margin:0 auto;">
            <div id="sliceLabel" style="margin-top:5px; font-weight:bold;">
                Slice 1/{len(base64_images)}
            </div>
        </div>
        <div style="margin:10px 0;">
            <button id="playButton" style="padding:5px 15px;">â–¶ï¸� Play</button>
            <button id="stopButton" style="padding:5px 15px; margin-left:10px;">â�¹ï¸� Stop</button>
        </div>
    </div>
    
    <script>
        // Store images in a JavaScript array
        const images = [];
        {' '.join([f"images.push('data:image/png;base64,{img}');" for img in base64_images])}
        
        // Get elements
        const slider = document.getElementById('sliceSlider');
        const image = document.getElementById('sliceImage');
        const label = document.getElementById('sliceLabel');
        const playButton = document.getElementById('playButton');
        const stopButton = document.getElementById('stopButton');
        
        // Update image when slider changes
        slider.oninput = function() {{
            const index = parseInt(this.value);
            image.src = images[index];
            label.textContent = `Slice ${{index*{step}+1}}/${{images.length*{step}}}`;
        }};
        
        // Animation variables
        let animationId = null;
        let currentIndex = 0;
        
        // Play function
        function playSlides() {{
            if (animationId) return; // Already playing
            
            animationId = setInterval(() => {{
                currentIndex = (currentIndex + 1) % images.length;
                slider.value = currentIndex;
                image.src = images[currentIndex];
                label.textContent = `Slice ${{currentIndex*{step}+1}}/${{images.length*{step}}}`;
            }}, 200); // Change slice every 200ms
        }}
        
        // Stop function
        function stopSlides() {{
            if (animationId) {{
                clearInterval(animationId);
                animationId = null;
            }}
        }}
        
        // Add event listeners
        playButton.onclick = playSlides;
        stopButton.onclick = stopSlides;
    </script>
    """
    
    # Display the HTML
    display(HTML(html_content))


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Palette setup
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779',
                     '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']
custom_color = '#3498db'

# DataFrame copy
df = train_df.copy()

# Feature definitions
categorical_features = ['PatientSex', 'Modality']
numeric_feature = 'PatientAge'
aneurysm_cols = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation'
]

sns.set_style("whitegrid")

# --- Categorical features: PatientSex and Modality ---
for feature in categorical_features:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: Aneurysm Present breakdown by feature
    sns.countplot(
        data=df,
        x=feature,
        hue='Aneurysm Present',
        palette=pie_chart_palette[:2],  # 0 and 1
        ax=axes[0]
    )
    axes[0].set_title(f'{feature} vs Aneurysm Presence', fontweight='bold')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Count')

    # Right: Aneurysm locations split by feature
    melted = df.melt(
        id_vars=[feature],
        value_vars=aneurysm_cols,
        var_name='Aneurysm Location',
        value_name='Presence'
    )
    melted = melted[melted['Presence'] == 1]

    unique_vals = df[feature].dropna().unique()
    sns.countplot(
        data=melted,
        y='Aneurysm Location',
        hue=feature,
        palette=pie_chart_palette[:len(unique_vals)],
        ax=axes[1]
    )
    axes[1].set_title(f'Aneurysm Location Breakdown by {feature}', fontweight='bold')
    axes[1].set_xlabel('Count')
    axes[1].set_ylabel('Aneurysm Location')

    plt.tight_layout()
    plt.show()

# --- Numeric feature: PatientAge ---
fig, axes = plt.subplots(1, 2, figsize=(16, 9))  # Height increased here

# Left: Boxplot of Age vs Aneurysm Present
sns.boxplot(
    data=df,
    x='Aneurysm Present',
    y=numeric_feature,
    color=custom_color,
    ax=axes[0]
)
axes[0].set_title('PatientAge vs Aneurysm Present', fontweight='bold')
axes[0].set_xlabel('Aneurysm Present')
axes[0].set_ylabel('PatientAge')

# Right: Boxplot of Age per Aneurysm Location
melted_age = df.melt(
    id_vars=[numeric_feature],
    value_vars=aneurysm_cols,
    var_name='Aneurysm Location',
    value_name='Presence'
)
melted_age = melted_age[melted_age['Presence'] == 1]

sns.boxplot(
    data=melted_age,
    x='Aneurysm Location',
    y=numeric_feature,
    color=custom_color,
    ax=axes[1]
)
axes[1].set_title('PatientAge by Aneurysm Location', fontweight='bold')
axes[1].set_xlabel('Aneurysm Location')
axes[1].set_ylabel('PatientAge')
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Define aneurysm location columns
aneurysm_cols = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation'
]

# Count aneurysm-affected regions per patient
train_df['AneurysmCount'] = train_df[aneurysm_cols].sum(axis=1)

# Filter to only patients with at least 1 aneurysm
aneurysm_counts = train_df[train_df['AneurysmCount'] > 0]['AneurysmCount'].value_counts().sort_index()

# Prepare plot-friendly DataFrame
plot_df = aneurysm_counts.reset_index()
plot_df.columns = ['Aneurysm Sites', 'Patient Count']
plot_df['Aneurysm Sites'] = plot_df['Aneurysm Sites'].astype(int).astype(str) + ' Site' + plot_df['Aneurysm Sites'].apply(lambda x: 's' if x > 1 else '')

# Plot
sns.set_style("whitegrid")
custom_color = '#3498db'

plt.figure(figsize=(8, 5))
sns.barplot(
    data=plot_df,
    y='Aneurysm Sites',
    x='Patient Count',
    color=custom_color
)

plt.title("Patients by Number of Aneurysm-Affected Regions", fontweight='bold')
plt.xlabel("Number of Patients")
plt.ylabel("Aneurysm Site Count")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define the set of aneurysm location columns
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

# 2. Compute the co-occurrence matrix
#    Each cell [i, j] = number of scans in which both location i AND location j have an aneurysm
co_occurrence_matrix = train_df[location_cols].T.dot(train_df[location_cols])

# 3. Plot as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    co_occurrence_matrix,
    annot=True,
    fmt='d',               # integer format
    cmap='viridis',
    linewidths=0.5,
    cbar_kws={'label': 'Co-occurrence Count'}
)
plt.title('Co-occurrence of Intracranial Aneurysm Locations', weight='bold')
plt.xlabel('Aneurysm Location')
plt.ylabel('Aneurysm Location')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

