from glob import glob
from tqdm import tqdm
import pandas as pd
import pydicom
from pydicom.filereader import dcmread
import pyvista as pv
import natsort
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from joblib import Parallel, delayed
import cv2
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
pv.set_jupyter_backend('static')


DATA_PATH = '../data'


print(f"Number of segmentation folders: {len(glob(f'{DATA_PATH}/segmentations/**'))}"
      f"\nNumber of series folders: {len(glob(f'{DATA_PATH}/series/**'))}")


print(f"Number of segmentation files: {len(glob(f'{DATA_PATH}/segmentations/**/*.nii', recursive=True))}")
print(f"Number of series files: {len(glob(f'{DATA_PATH}/series/**/*.dcm', recursive=True))}")


number_of_dicom_per_folder = []
for folder in tqdm(glob(f'{DATA_PATH}/series/**')):
    dicom_files = glob(f"{folder}/**/*.dcm", recursive=True)
    number_of_dicom_per_folder.append(len(dicom_files))

plt.figure(figsize=(10, 5))
plt.hist(number_of_dicom_per_folder, bins=100, color='blue', alpha=0.7)
plt.title('Distribution of DICOM Files per Series Folder')
plt.xlabel('Number of DICOM Files')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


print("Loading DICOM files...")
dicom_files = glob(f'{DATA_PATH}/series/**/*.dcm', recursive=True)


def extract_dicom_metadata(file_path):
    """Extract relevant metadata from a DICOM file"""
    try:
        ds = pydicom.dcmread(file_path, force=True)
        
        metadata = {
            'file_path': file_path,
            'study_date': getattr(ds, 'StudyDate', None),
            'series_description': getattr(ds, 'SeriesDescription', None),
            'manufacturer_model': getattr(ds, 'ManufacturerModelName', None),
            'patient_sex': getattr(ds, 'PatientSex', None),
            'bits_stored': getattr(ds, 'BitsStored', None),
            'patient_weight': getattr(ds, 'PatientWeight', None),
            'slice_thickness': getattr(ds, 'SliceThickness', None),
            'spacing_between_slices': getattr(ds, 'SpacingBetweenSlices', None),
            'modality': getattr(ds, 'Modality', None),
            'manufacturer': getattr(ds, 'Manufacturer', None),
            'study_description': getattr(ds, 'StudyDescription', None),
            'patient_id': getattr(ds, 'PatientID', None),
            'series_number': getattr(ds, 'SeriesNumber', None),
            'instance_number': getattr(ds, 'InstanceNumber', None),
        }
        
        # Extract private tag data if exists
        try:
            # Look for private tags (typically starting with odd group numbers)
            private_tags = []
            for tag in ds.keys():
                if tag.group % 2 == 1:  # Odd group numbers are private
                    try:
                        value = str(ds[tag].value)
                        if len(value) < 100:  # Avoid very long values
                            private_tags.append(value)
                    except:
                        pass
            metadata['private_tag_data'] = ','.join(private_tags) if private_tags else None
        except:
            metadata['private_tag_data'] = None
            
        return metadata
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None


print("Extracting metadata from DICOM files...")
sample_size = len(dicom_files)
sample_files = dicom_files[:sample_size]

n_jobs = -1  # Use all available cores

# Parallel processing
results = Parallel(n_jobs=n_jobs)(
    delayed(extract_dicom_metadata)(file_path)
    for file_path in tqdm(sample_files, desc="Extracting metadata", unit="file")
)

# Filter out None results
metadata_list = [metadata for metadata in results if metadata is not None]


# Create DataFrame
df = pd.DataFrame(metadata_list)


# Convert study_date to datetime
def parse_dicom_date(date_str):
    """Parse DICOM date format YYYYMMDD"""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return datetime.strptime(str(date_str), '%Y%m%d')
    except:
        return None

df['study_date_parsed'] = df['study_date'].apply(parse_dicom_date)

# Convert numeric columns
numeric_cols = ['bits_stored', 'patient_weight', 'slice_thickness', 'spacing_between_slices']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')



# Set up the plotting environment
fig = plt.figure(figsize=(20, 24))


# PLOT 1: Study Date Distribution by Month
print("\n1. Creating Study Date distribution by month...")
plt.subplot(4, 3, 1)

if df['study_date_parsed'].notna().sum() > 0:
    # Extract month-year for grouping
    df['month_year'] = df['study_date_parsed'].dt.to_period('M')
    date_counts = df['month_year'].value_counts().sort_index()
    
    # Convert to datetime for proper x-axis formatting
    dates = [pd.to_datetime(str(period)) for period in date_counts.index]
    
    plt.plot(dates, date_counts.values, marker='o', linewidth=2, markersize=6)
    plt.title('Study Date Distribution by Month', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Studies', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No valid study dates found', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Study Date Distribution by Month', fontsize=14, fontweight='bold')


# PLOT 2: Series Description Distribution (Pie Chart)
print("2. Creating Series Description distribution pie chart...")
plt.subplot(4, 3, 2)

series_desc_counts = df['series_description'].value_counts().head(8)  # Top 8 categories
if len(series_desc_counts) > 0:
    colors = plt.cm.Set3(np.linspace(0, 1, len(series_desc_counts)))
    wedges, texts, autotexts = plt.pie(series_desc_counts.values, 
                                      labels=series_desc_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
    plt.setp(autotexts, size=9, weight="bold")
    plt.setp(texts, size=8)
    plt.title('Series Description Distribution', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No series description data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Series Description Distribution', fontsize=14, fontweight='bold')


# PLOT 3: Manufacturer's Model Name Distribution (Pie Chart)
print("3. Creating Manufacturer's Model Name distribution...")
plt.subplot(4, 3, 3)

model_counts = df['manufacturer_model'].value_counts().head(6)
if len(model_counts) > 0:
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(model_counts)))
    wedges, texts, autotexts = plt.pie(model_counts.values,
                                      labels=model_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=45)
    plt.setp(autotexts, size=9, weight="bold")
    plt.setp(texts, size=8)
    plt.title("Manufacturer's Model Name Distribution", fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No model name data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title("Manufacturer's Model Name Distribution", fontsize=14, fontweight='bold')


# PLOT 4: Patient's Sex Distribution (Pie Chart)
print("4. Creating Patient's Sex distribution...")
plt.subplot(4, 3, 4)

sex_counts = df['patient_sex'].value_counts()
if len(sex_counts) > 0:
    colors = ['#ff9999', '#66b3ff', '#99ff99'][:len(sex_counts)]
    wedges, texts, autotexts = plt.pie(sex_counts.values,
                                      labels=sex_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
    plt.setp(autotexts, size=11, weight="bold")
    plt.setp(texts, size=10)
    plt.title("Patient's Sex Distribution", fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No sex data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title("Patient's Sex Distribution", fontsize=14, fontweight='bold')


# PLOT 5: Bits Stored Distribution (Pie Chart)
print("5. Creating Bits Stored distribution...")
plt.subplot(4, 3, 5)

bits_counts = df['bits_stored'].value_counts()
if len(bits_counts) > 0:
    colors = plt.cm.Set2(np.linspace(0, 1, len(bits_counts)))
    wedges, texts, autotexts = plt.pie(bits_counts.values,
                                      labels=[f'{int(x)} bits' for x in bits_counts.index],
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=0)
    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, size=9)
    plt.title('Bits Stored Distribution', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No bits stored data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Bits Stored Distribution', fontsize=14, fontweight='bold')


# PLOT 6: Patient's Weight Distribution (Histogram)
print("6. Creating Patient's Weight histogram...")
plt.subplot(4, 3, 6)

weight_data = df['patient_weight'].dropna()
if len(weight_data) > 0:
    plt.hist(weight_data, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Patient's Weight Distribution", fontsize=14, fontweight='bold')
    plt.xlabel('Weight (kg)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_weight = weight_data.mean()
    plt.axvline(mean_weight, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_weight:.1f} kg')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'No weight data available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title("Patient's Weight Distribution", fontsize=14, fontweight='bold')


# PLOT 7: Slice Thickness Distribution (Histogram)
print("7. Creating Slice Thickness histogram...")
plt.subplot(4, 3, 7)

thickness_data = df['slice_thickness'].dropna()
# Filter out outliers > 5 mm
thickness_data = thickness_data[thickness_data <= 5]

if len(thickness_data) > 0:
    plt.hist(thickness_data, bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Slice Thickness Distribution (â‰¤ 5 mm)', fontsize=14, fontweight='bold')
    plt.xlabel('Slice Thickness (mm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_thickness = thickness_data.mean()
    plt.axvline(mean_thickness, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_thickness:.2f} mm')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'No slice thickness data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Slice Thickness Distribution', fontsize=14, fontweight='bold')


# PLOT 8: Spacing Between Slices Distribution (Histogram)
print("8. Creating Spacing Between Slices histogram...")
plt.subplot(4, 3, 8)

spacing_data = df['spacing_between_slices'].dropna()
# Filter out outliers > 5 mm
spacing_data = spacing_data[spacing_data <= 5]

if len(spacing_data) > 0:
    plt.hist(spacing_data, bins=40, alpha=0.7, color='coral', edgecolor='black')
    plt.title('Spacing Between Slices Distribution (â‰¤ 5 mm)', fontsize=14, fontweight='bold')
    plt.xlabel('Spacing (mm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_spacing = spacing_data.mean()
    plt.axvline(mean_spacing, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_spacing:.2f} mm')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'No spacing data available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Spacing Between Slices Distribution', fontsize=14, fontweight='bold')


# PLOT 9: Private Tag Data Analysis
print("9. Creating Private Tag Data analysis...")
plt.subplot(4, 3, 9)

# Analyze private tag data frequencies (top 10)
private_data = df['private_tag_data'].dropna()
if len(private_data) > 0:
    # Get top 10 most frequent values
    top_counts = private_data.value_counts().head(10)

    # Shorten labels to first 4 characters + '...'
    short_labels = [str(val)[:30] + '...' for val in top_counts.index]

    plt.bar(short_labels, top_counts.values, color='mediumpurple', edgecolor='black')
    plt.title('Top 10 Most Frequent Private Tags', fontsize=14, fontweight='bold')
    plt.xlabel('Private Tag (truncated)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.grid(axis='y', alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No private tag data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Top 10 Most Frequent Private Tags', fontsize=14, fontweight='bold')

# PLOT 10: Modality Distribution
print("10. Creating Modality distribution...")
plt.subplot(4, 3, 10)

modality_counts = df['modality'].value_counts()
if len(modality_counts) > 0:
    plt.bar(modality_counts.index, modality_counts.values, color='gold', alpha=0.8, edgecolor='black')
    plt.title('Imaging Modality Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Modality', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No modality data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Imaging Modality Distribution', fontsize=14, fontweight='bold')


# PLOT 11: Manufacturer Distribution
print("11. Creating Manufacturer distribution...")
plt.subplot(4, 3, 11)

mfr_counts = df['manufacturer'].value_counts().head(5)
if len(mfr_counts) > 0:
    colors = plt.cm.viridis(np.linspace(0, 1, len(mfr_counts)))
    bars = plt.bar(range(len(mfr_counts)), mfr_counts.values, color=colors, alpha=0.8, edgecolor='black')
    plt.title('Manufacturer Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Manufacturer', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(mfr_counts)), [mfr[:15] + '...' if len(mfr) > 15 else mfr for mfr in mfr_counts.index], rotation=45)
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No manufacturer data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Manufacturer Distribution', fontsize=14, fontweight='bold')


# PLOT 12: Series Number vs Instance Number Scatter
print("12. Creating Series vs Instance Number scatter plot...")
plt.subplot(4, 3, 12)

series_data = df[['series_number', 'instance_number']].dropna()
if len(series_data) > 0:
    plt.scatter(series_data['series_number'], series_data['instance_number'], 
               alpha=0.6, color='darkblue', s=30)
    plt.title('Series Number vs Instance Number', fontsize=14, fontweight='bold')
    plt.xlabel('Series Number', fontsize=12)
    plt.ylabel('Instance Number', fontsize=12)
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No series/instance data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Series Number vs Instance Number', fontsize=14, fontweight='bold')

plt.tight_layout(pad=3.0)
plt.show()


# Summary Statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

print(f"\nDataset Overview:")
print(f"Total DICOM files processed: {len(df)}")

print(f"\nMissing Data Analysis:")
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percentage
}).sort_values('Missing Percentage', ascending=False)
print(missing_df)

print(f"\nNumerical Statistics:")
numerical_cols = ['patient_weight', 'slice_thickness', 'spacing_between_slices', 'bits_stored']
for col in numerical_cols:
    if df[col].notna().sum() > 0:
        print(f"\n{col.upper()}:")
        print(f"  Mean: {df[col].mean():.3f}")
        print(f"  Median: {df[col].median():.3f}")
        print(f"  Std: {df[col].std():.3f}")
        print(f"  Min: {df[col].min():.3f}")
        print(f"  Max: {df[col].max():.3f}")

print(f"\nCategorical Data Summary:")
categorical_cols = ['series_description', 'manufacturer_model', 'patient_sex', 'modality', 'manufacturer']
for col in categorical_cols:
    unique_count = df[col].nunique()
    if unique_count > 0:
        print(f"\n{col.upper()}: {unique_count} unique values")
        print(f"  Top values: {list(df[col].value_counts().head(4).index)}")

print(f"\nDate Range Analysis:")
if df['study_date_parsed'].notna().sum() > 0:
    date_range = df['study_date_parsed'].dropna()
    print(f"  Earliest study: {date_range.min().strftime('%Y-%m-%d')}")
    print(f"  Latest study: {date_range.max().strftime('%Y-%m-%d')}")
    print(f"  Date range: {(date_range.max() - date_range.min()).days} days")



print(f"Studies from the future (after Aug 2025): {df['month_year'].gt('2025-08').sum()}")
slice_thickness_data = df['slice_thickness'].dropna()
# outlier slice thickness (>5mm)
if len(slice_thickness_data[slice_thickness_data > 5]) > 0:
    print(f"Outlier slice thickness (>5mm): {len(slice_thickness_data[slice_thickness_data > 5])} instances")
# outlier spacing between slices (>5mm)
spacing_data = df['spacing_between_slices'].dropna()
if len(spacing_data[spacing_data > 5]) > 0:
    print(f"Outlier spacing between slices (>5mm): {len(spacing_data[spacing_data > 5])} instances")


df = pd.read_csv(f'{DATA_PATH}/train.csv')
df.head(5)


meta_cols = ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality']

# Artery-level aneurysm labels (13 columns)
artery_cols = [
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
    'Other Posterior Circulation',
]

# Target label
target_col = 'Aneurysm Present'


# Preprocess
df['PatientAge'] = pd.to_numeric(df['PatientAge'], errors='coerce')

# Define column groups
meta_cols = ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality']
target_col = 'Aneurysm Present'
artery_cols = df.columns[4:-1].tolist()  # All artery-specific binary labels

# Initialize figure
plt.figure(figsize=(18, 16))
plt.suptitle("Exploratory Data Analysis - Aneurysm Detection Dataset", fontsize=18, fontweight='bold')

# 1. Target distribution
plt.subplot(3, 3, 1)
df[target_col].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Aneurysm Presence')
plt.xlabel('Label')
plt.ylabel('Count')

# 2. Patient Age distribution
plt.subplot(3, 3, 2)
df['PatientAge'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Patient Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

# 3. Patient Sex distribution
plt.subplot(3, 3, 3)
df['PatientSex'].value_counts().plot(kind='bar', color='plum', edgecolor='black')
plt.title('Patient Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')

# 4. Imaging Modality
plt.subplot(3, 3, 4)
df['Modality'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Modality Distribution')
plt.xlabel('Modality')
plt.ylabel('Count')

# 5. Aneurysm count per artery
plt.subplot(3, 3, 5)
df[artery_cols].sum().sort_values(ascending=False).plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Aneurysm Count per Artery')
plt.xticks(rotation=90)
plt.ylabel('Count')

# 6. Artery involvement when Aneurysm Present = 1
plt.subplot(3, 3, 6)
df[df[target_col] == 1][artery_cols].sum().sort_values(ascending=False).plot(kind='bar', color='tomato', edgecolor='black')
plt.title('Artery Involvement (when Aneurysm Present = 1)')
plt.xticks(rotation=90)
plt.ylabel('Count')



plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Count how many artery columns are labeled 1 per scan
df['n_arteries_positive'] = df[artery_cols].sum(axis=1)

# Plot distribution
plt.figure(figsize=(6, 4))
df['n_arteries_positive'].value_counts().sort_index().plot(kind='bar', color='indigo', edgecolor='black')
plt.title('Number of Arteries with Aneurysms per Scan', fontsize=14)
plt.xlabel('Number of Positive Artery Labels')
plt.ylabel('Count of Scans')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


multi_artery_cases = df[df['Aneurysm Present'] == 1]['n_arteries_positive']
multi_artery_rate = (multi_artery_cases > 1).mean()

print(f"ğŸ‘‰ {multi_artery_rate:.1%} of aneurysm-positive scans involve more than one artery.")


# Compute aneurysm counts and percentage (w.r.t aneurysm-positive cases)
artery_counts = df[artery_cols].sum().sort_values(ascending=False)
total_positive = df['Aneurysm Present'].sum()
artery_percent = (artery_counts / total_positive * 100).round(1)

# Create combined DataFrame
artery_stats = pd.DataFrame({
    'Aneurysm Count': artery_counts,
    'Percent of Aneurysm-Positive Scans': artery_percent
})

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(artery_stats.index, artery_stats['Aneurysm Count'], color='cornflowerblue', edgecolor='black')
plt.title('Aneurysm Distribution per Artery', fontsize=14, fontweight='bold')
plt.ylabel('Aneurysm Count')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add percentage labels above bars
for bar, pct in zip(bars, artery_stats['Percent of Aneurysm-Positive Scans']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{pct}%", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


localizer_df = pd.read_csv(f"{DATA_PATH}/train_localizers.csv")


import ast

# Parse coordinate strings
localizer_df['coordinates'] = localizer_df['coordinates'].apply(ast.literal_eval)

# Separate x and y
localizer_df['x'] = localizer_df['coordinates'].apply(lambda c: c['x'])
localizer_df['y'] = localizer_df['coordinates'].apply(lambda c: c['y'])

# Quick stats
print(localizer_df[['x', 'y']].describe())


plt.figure(figsize=(6, 6))
plt.scatter(localizer_df['x'], localizer_df['y'], alpha=0.4, s=10, c='crimson')
plt.title('Aneurysm Coordinates Distribution (in-plane)', fontsize=14)
plt.xlabel('X position (pixels)')
plt.ylabel('Y position (pixels)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()


df_modality = df[['SeriesInstanceUID', 'Modality']]
modality_counts = pd.merge(localizer_df[['SeriesInstanceUID']], df_modality, on='SeriesInstanceUID', how='left')
print(modality_counts['Modality'].value_counts())


plt.figure(figsize=(8, 4))
sns.countplot(y='location', data=localizer_df, order=localizer_df['location'].value_counts().index)
plt.title('Aneurysm Location Frequency', fontsize=14)
plt.xlabel('Count')
plt.ylabel('Artery Location')
plt.tight_layout()
plt.show()


# https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/593948 @umarali1
def load_sorted_dicom_series(series_path):
    '''
    Since the dicom series path files are not in spatial order by file name, we have
    to use the metadata contained within the slices to proper sort them
    '''
    dcm_files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')]
    slices = [pydicom.dcmread(f) for f in dcm_files]

    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, IndexError):
        slices.sort(key=lambda s: int(s.InstanceNumber))
    return slices


slices = load_sorted_dicom_series(f'{DATA_PATH}/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647')
# Ensure consistent spacing (check pixel spacing and slice thickness)
pixel_spacing = slices[0].PixelSpacing  # [row, col]
slice_thickness = float(slices[0].SliceThickness)
spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])

# Stack into 3D volume
image_3d = np.stack([s.pixel_array for s in slices]).astype(np.int16)

# Flip if needed (some modalities store slices reversed)
image_3d = np.flip(image_3d, axis=0)

print(f"Volume shape: {image_3d.shape}, spacing: {spacing}")


Z, Y, X = image_3d.shape
# The number of slices to display
num_slices = image_3d.shape[0] # Number of slices along the Z-axis

# Determine the grid size for the subplots
# We'll use a square grid to make it look neat
cols = 8 # Number of columns in the subplot grid
rows = (num_slices + cols - 1) // cols # Calculate required number of rows

# Create a figure and a grid of subplots
fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
fig.suptitle(f'All {num_slices} Slices of 3D Volume (Z-axis)', fontsize=16)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop through each slice and plot it
for i in range(num_slices):
    ax = axes[i]
    # The slice is a 2D array of shape (Y, X)
    slice_data = image_3d[i, :, :]

    # Display the slice. The `extent` argument can be used to
    # correctly scale the axes based on your spacing.
    # The spacing is (Z, Y, X), so spacing[1] and spacing[2] correspond to Y and X axes.
    extent = [0, X * spacing[2], 0, Y * spacing[1]]
    ax.imshow(slice_data, cmap='gray', origin='lower', extent=extent)

    ax.set_title(f'Slice {i+1}/{num_slices}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Turn off the axis ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

# Hide any unused subplots
for i in range(num_slices, len(axes)):
    axes[i].axis('off')

# Adjust the layout to prevent titles from overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# You can still use the trame backend for interactive viewing
# pv.set_jupyter_backend('trame')  # for local Jupyter Notebook
volume = pv.wrap(image_3d)
volume.spacing = spacing[::-1]
volume.origin = (0, 0, 0)

p = pv.Plotter(notebook=True)
p.add_volume(volume, cmap='viridis')
p.add_mesh(volume.outline(), color='gray')
p.add_axes()
p.show_grid()

p.show()


import ipywidgets as widgets
from IPython.display import display

def plot_slice(slice_index):
    fig, ax = plt.subplots(figsize=(8, 8))
    slice_data = image_3d[slice_index, :, :]
    extent = [0, X * spacing[2], 0, Y * spacing[1]]
    ax.imshow(slice_data, cmap='gray', origin='lower', extent=extent)
    ax.set_title(f'Interactive Slice {slice_index+1}/{num_slices}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.show()

# Create a slider widget
slice_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=num_slices - 1,
    step=1,
    description='Slice Index:',
    continuous_update=False
)

# Use `ipywidgets.interactive` to connect the slider and the function
interactive_plot = widgets.interactive(plot_slice, slice_index=slice_slider)
display(interactive_plot)



unique_locs = localizer_df.groupby("location").first().reset_index()

n = len(unique_locs)
cols = 3
rows = (n + cols - 1) // cols
plt.figure(figsize=(5 * cols, 5 * rows))

for i, row in unique_locs.iterrows():
    try:
        series_uid = row["SeriesInstanceUID"]
        sop_uid = row["SOPInstanceUID"]
        coords = row["coordinates"]

        # Load DICOM
        dcm_path = ff"{DATA_PATH}/series/{series_uid}/{sop_uid}.dcm"
        if not os.path.exists(dcm_path):
            print(f"Missing DICOM: {dcm_path}")
            continue

        dcm = pydicom.dcmread(dcm_path)
        image = dcm.pixel_array

        # Normalize
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = (image * 255).astype(np.uint8)

        # Draw marker
        x, y = int(coords["x"]), int(coords["y"])
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.circle(img_rgb, (x, y), radius=10, color=(255, 0, 0), thickness=2)

        # Plot
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.title(row["location"], fontsize=10)
        plt.axis("off")
    except Exception as e:
        print(f"Error at index {i}: {e}")

plt.tight_layout()
plt.show()


seg_dirs = sorted(glob(f"{DATA_PATH}/segmentations/*/"))
print(f"Found {len(seg_dirs)} segmentation folders")

def show_segmentation_overlay(id_path):
    # Find both files
    image_path = glob(os.path.join(id_path, "*.nii"))[1]            # vessel image
    mask_path = glob(os.path.join(id_path, "*_cowseg.nii"))[0]       # segmentation mask

    # Load NIfTI files
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    # Pick a mid slice (sagittal/axial/coronal)
    slice_idx = img.shape[2] // 2
    img_slice = img[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]

    # Normalize image for display
    img_slice = (img_slice - img_slice.min()) / np.ptp(img_slice)

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_slice, cmap='gray')
    plt.title('Vessel Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_slice, cmap='gray')
    plt.imshow(mask_slice, cmap='Reds', alpha=0.5)
    plt.title('Overlay: Vessel + Segmentation')
    plt.axis('off')

    plt.suptitle(f"Segmentation from: {os.path.basename(id_path.strip('/'))}")
    plt.tight_layout()
    plt.show()

for path in seg_dirs[:3]:  # show first 4
    show_segmentation_overlay(path)




