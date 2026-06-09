!pip install ultralytics
!pip uninstall albumentations -y


# Cell 1: Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# YOLO imports
from ultralytics import YOLO
import yaml

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")


class AneurysmDataExplorer:
    """Class to handle data exploration for the brain aneurysm detection challenge"""
    
    def __init__(self, data_path: str = "/kaggle/input/rsna-intracranial-aneurysm-detection"):
        self.data_path = Path(data_path)
        self.train_df = None
        self.localizer_df = None
        self._load_data()
        
    def _load_data(self):
        """Load training and localizer datasets"""
        self.train_df = pd.read_csv(self.data_path / "train.csv")
        self.localizer_df = pd.read_csv(self.data_path / "train_localizers.csv")
        
    def get_basic_stats(self):
        """Return basic dataset statistics"""
        stats = {
            'total_series': len(self.train_df),
            'total_localizations': len(self.localizer_df),
            'aneurysm_positive': self.train_df['Aneurysm Present'].sum(),
            'modalities': self.train_df['Modality'].value_counts().to_dict(),
            'age_stats': {
                'mean': self.train_df['PatientAge'].mean(),
                'std': self.train_df['PatientAge'].std()
            }
        }
        return stats
        
    def get_location_analysis(self):
        """Analyze aneurysm locations and return sorted counts"""
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
        
        location_counts = {col: self.train_df[col].sum() for col in location_cols}
        return location_counts, location_cols
    
    def plot_data_overview(self):
        """Create comprehensive data visualization"""
        location_counts, location_cols = self.get_location_analysis()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Modality distribution
        modality_counts = self.train_df['Modality'].value_counts()
        axes[0,0].pie(modality_counts.values, labels=modality_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Imaging Modality Distribution')
        
        # Age distribution
        axes[0,1].hist(self.train_df['PatientAge'].dropna(), bins=30, alpha=0.7, color='skyblue')
        axes[0,1].set_xlabel('Age (years)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Patient Age Distribution')
        
        # Aneurysm presence
        aneurysm_counts = self.train_df['Aneurysm Present'].value_counts()
        axes[1,0].bar(['No Aneurysm', 'Aneurysm Present'], aneurysm_counts.values, 
                     color=['lightcoral', 'lightgreen'])
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Aneurysm Presence Distribution')
        
        # Top aneurysm locations
        sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        locations, counts = zip(*sorted_locations)
        
        axes[1,1].barh(range(len(locations)), counts)
        axes[1,1].set_yticks(range(len(locations)))
        axes[1,1].set_yticklabels([loc.replace(' Artery', '') for loc in locations], fontsize=8)
        axes[1,1].set_xlabel('Count')
        axes[1,1].set_title('Most Common Aneurysm Locations')
        
        plt.tight_layout()
        plt.show()
        
        return location_counts, location_cols

explorer = AneurysmDataExplorer()
stats = explorer.get_basic_stats()
location_counts, location_cols = explorer.plot_data_overview()

print(f"Dataset: {stats['total_series']} series, {stats['aneurysm_positive']} positive cases")
print(f"Positive rate: {stats['aneurysm_positive']/stats['total_series']*100:.1f}%")


data_path = Path("/kaggle/input/rsna-intracranial-aneurysm-detection")
print("=== DATA STRUCTURE DIAGNOSTIC ===")
print(f"Data path exists: {data_path.exists()}")

if data_path.exists():
    print(f"Contents: {list(data_path.iterdir())}")
    
    # Check for series folder
    series_path = data_path / "series"
    print(f"Series path exists: {series_path.exists()}")
    
    if series_path.exists():
        series_folders = list(series_path.iterdir())
        print(f"Number of series folders: {len(series_folders)}")
        print(f"First few series: {[f.name for f in series_folders[:3]]}")
        
        # Check a specific series folder
        if series_folders:
            sample_series = series_folders[0]
            dcm_files = list(sample_series.glob("*.dcm"))
            print(f"Sample series {sample_series.name}: {len(dcm_files)} DICOM files")

# Check if our selected series exist
print(f"\n=== SERIES VERIFICATION ===")
if len(explorer.train_df) > 0:
    # Get series IDs
    positive_cases = explorer.train_df[explorer.train_df['Aneurysm Present'] == 1]
    negative_cases = explorer.train_df[explorer.train_df['Aneurysm Present'] == 0]
    
    print(f"Positive cases: {len(positive_cases)}")
    print(f"Negative cases: {len(negative_cases)}")
    
    if len(positive_cases) > 0 and len(negative_cases) > 0:
        pos_series_id = positive_cases['SeriesInstanceUID'].iloc[0]
        neg_series_id = negative_cases['SeriesInstanceUID'].iloc[0]
        
        print(f"Selected positive series: {pos_series_id}")
        print(f"Selected negative series: {neg_series_id}")
        
        # Check if these series folders exist
        pos_series_path = data_path / "series" / pos_series_id
        neg_series_path = data_path / "series" / neg_series_id
        
        print(f"Positive series folder exists: {pos_series_path.exists()}")
        print(f"Negative series folder exists: {neg_series_path.exists()}")
        
        # If folders exist, check DICOM files
        if pos_series_path.exists():
            pos_dcm_files = list(pos_series_path.glob("*.dcm"))
            print(f"Positive series DICOM files: {len(pos_dcm_files)}")
        
        if neg_series_path.exists():
            neg_dcm_files = list(neg_series_path.glob("*.dcm"))
            print(f"Negative series DICOM files: {len(neg_dcm_files)}")



# Cell 3F: Updated DICOM Processing with Correct Path
class WorkingDICOMProcessor:
    """DICOM processor with the correct data path"""
    
    def __init__(self, data_path: str = "/kaggle/input/rsna-intracranial-aneurysm-detection"):
        self.data_path = Path(data_path)
        self.series_path = self.data_path / "series"
        
    def load_dicom_series(self, series_id: str, max_slices: int = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """Load DICOM files for a series"""
        series_folder = self.series_path / series_id
        
        if not series_folder.exists():
            return [], []
            
        dicom_files = list(series_folder.glob("*.dcm"))
        
        # Sort files by instance number if possible
        dicom_files_with_info = []
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                instance_num = getattr(ds, 'InstanceNumber', 0)
                dicom_files_with_info.append((instance_num, dcm_file))
            except:
                dicom_files_with_info.append((0, dcm_file))
        
        # Sort by instance number
        dicom_files_with_info.sort(key=lambda x: x[0])
        dicom_files = [x[1] for x in dicom_files_with_info]
        
        if max_slices:
            # Take evenly spaced slices
            total_files = len(dicom_files)
            if total_files > max_slices:
                indices = np.linspace(0, total_files-1, max_slices, dtype=int)
                dicom_files = [dicom_files[i] for i in indices]
            
        images, metadata = [], []
        
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                
                # Extract pixel array
                img = ds.pixel_array.astype(np.float32)
                
                # Apply rescaling if available
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * ds.RescaleSlope + ds.RescaleIntercept
                
                images.append(img)
                
                # Store metadata
                meta = {
                    'SOPInstanceUID': ds.SOPInstanceUID,
                    'InstanceNumber': getattr(ds, 'InstanceNumber', len(images)),
                    'SliceThickness': getattr(ds, 'SliceThickness', None),
                    'PixelSpacing': getattr(ds, 'PixelSpacing', None),
                    'Modality': getattr(ds, 'Modality', 'Unknown')
                }
                metadata.append(meta)
                
            except Exception as e:
                continue
                
        return images, metadata
    
    def normalize_image(self, img: np.ndarray, window_center: float = None, window_width: float = None) -> np.ndarray:
        """Normalize image with optional windowing"""
        if img.size == 0:
            return img
            
        if window_center is not None and window_width is not None:
            # Apply medical imaging windowing
            img_windowed = np.clip(img, 
                                 window_center - window_width/2, 
                                 window_center + window_width/2)
        else:
            # Use percentile-based normalization
            p1, p99 = np.percentile(img, [1, 99])
            img_windowed = np.clip(img, p1, p99)
        
        # Normalize to 0-1
        img_range = img_windowed.max() - img_windowed.min()
        if img_range > 0:
            img_norm = (img_windowed - img_windowed.min()) / img_range
        else:
            img_norm = np.zeros_like(img_windowed)
        
        return img_norm
    
    def visualize_series_with_aneurysms(self, series_id: str, localizer_df: pd.DataFrame, max_slices: int = 12):
        """Visualize series and highlight aneurysm locations if available"""
        
        images, metadata = self.load_dicom_series(series_id, max_slices)
        
        if not images:
            return None, None
        
        # Get aneurysm localizations for this series
        series_localizations = localizer_df[localizer_df['SeriesInstanceUID'] == series_id]
        
        # Create mapping of SOPInstanceUID to localization
        localization_map = {}
        for _, loc in series_localizations.iterrows():
            sop_uid = loc['SOPInstanceUID']
            coords = loc['coordinates']
            location = loc['location']
            
            if sop_uid not in localization_map:
                localization_map[sop_uid] = []
            localization_map[sop_uid].append({
                'coords': coords,
                'location': location
            })
        
        # Display images
        n_images = len(images)
        cols = 4
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif n_images == 1:
            axes = np.array([[axes]])
        
        for i, (img, meta) in enumerate(zip(images, metadata)):
            row, col = i // cols, i % cols
            
            # Normalize image
            img_norm = self.normalize_image(img)
            
            # Display image
            axes[row, col].imshow(img_norm, cmap='gray')
            
            # Check if this image has aneurysm annotations
            sop_uid = meta['SOPInstanceUID']
            title = f"Slice {meta.get('InstanceNumber', i+1)}"
            
            if sop_uid in localization_map:
                title += f" âš ï¸� ({len(localization_map[sop_uid])} aneurysms)"
                
                # Plot aneurysm locations
                for loc_info in localization_map[sop_uid]:
                    coords_str = loc_info['coords']
                    try:
                        # Parse coordinates
                        coords = coords_str.strip('()[]').split(',')
                        x, y = float(coords[0]), float(coords[1])
                        
                        # Plot marker
                        axes[row, col].plot(x, y, 'r+', markersize=10, markeredgewidth=2)
                        axes[row, col].plot(x, y, 'ro', markersize=15, alpha=0.3)
                        
                    except:
                        pass
            
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return images, metadata

# Initialize the working processor
processor = WorkingDICOMProcessor()

# Load the data correctly
explorer = AneurysmDataExplorer("/kaggle/input/rsna-intracranial-aneurysm-detection")
stats = explorer.get_basic_stats()
location_counts, location_cols = explorer.get_location_analysis()

print(f"âœ… Successfully loaded competition data:")
print(f"   Total series: {stats['total_series']}")
print(f"   Positive cases: {stats['aneurysm_positive']} ({stats['aneurysm_positive']/stats['total_series']*100:.1f}%)")
print(f"   Localizations: {len(explorer.localizer_df)}")

# Now visualize actual cases with aneurysms
print(f"\nğŸ�¯ Visualizing actual positive case with aneurysms:")
positive_series = explorer.train_df[explorer.train_df['Aneurysm Present'] == 1]['SeriesInstanceUID'].iloc[0]
series_info = explorer.train_df[explorer.train_df['SeriesInstanceUID'] == positive_series].iloc[0]

print(f"Series: {positive_series}")
print(f"Modality: {series_info['Modality']}")
print(f"Patient Age: {series_info['PatientAge']}")

# Visualize with aneurysm markers
pos_images, pos_meta = processor.visualize_series_with_aneurysms(
    positive_series, explorer.localizer_df, max_slices=12
)

print(f"\nğŸ”� Visualizing negative case (no aneurysms):")
negative_series = explorer.train_df[explorer.train_df['Aneurysm Present'] == 0]['SeriesInstanceUID'].iloc[0]
neg_series_info = explorer.train_df[explorer.train_df['SeriesInstanceUID'] == negative_series].iloc[0]

print(f"Series: {negative_series}")
print(f"Modality: {neg_series_info['Modality']}")
print(f"Patient Age: {neg_series_info['PatientAge']}")

# Visualize negative case
neg_images, neg_meta = processor.visualize_series_with_aneurysms(
    negative_series, explorer.localizer_df, max_slices=12
)


# Cell 3G: Enhanced Data Analysis with Real Data
def analyze_real_competition_data():
    """Comprehensive analysis of the actual competition dataset"""
    
    # Basic statistics
    print("=== COMPREHENSIVE DATA ANALYSIS ===")
    train_df = explorer.train_df
    localizer_df = explorer.localizer_df
    
    # Dataset overview
    print(f"ğŸ“Š Dataset Overview:")
    print(f"   Total series: {len(train_df):,}")
    print(f"   Positive cases: {train_df['Aneurysm Present'].sum():,}")
    print(f"   Negative cases: {(train_df['Aneurysm Present'] == 0).sum():,}")
    print(f"   Positive rate: {train_df['Aneurysm Present'].mean()*100:.1f}%")
    
    # Modality analysis
    print(f"\nğŸ”¬ Modality Distribution:")
    modality_stats = train_df['Modality'].value_counts()
    for modality, count in modality_stats.items():
        pos_rate = train_df[train_df['Modality'] == modality]['Aneurysm Present'].mean() * 100
        print(f"   {modality}: {count:,} series ({count/len(train_df)*100:.1f}%) - {pos_rate:.1f}% positive")
    
    # Age analysis
    print(f"\nğŸ‘¥ Demographics:")
    print(f"   Age: {train_df['PatientAge'].mean():.1f} Â± {train_df['PatientAge'].std():.1f} years")
    print(f"   Age range: {train_df['PatientAge'].min():.0f} - {train_df['PatientAge'].max():.0f} years")
    
    # Gender analysis
    gender_stats = train_df['PatientSex'].value_counts()
    for gender, count in gender_stats.items():
        pos_rate = train_df[train_df['PatientSex'] == gender]['Aneurysm Present'].mean() * 100
        print(f"   {gender}: {count:,} ({count/len(train_df)*100:.1f}%) - {pos_rate:.1f}% positive")
    
    # Aneurysm location analysis
    print(f"\nğŸ§­ Aneurysm Location Analysis:")
    print(f"   Total localizations: {len(localizer_df):,}")
    
    # Most common locations
    location_text_stats = localizer_df['location'].value_counts().head(10)
    print(f"   Most common locations:")
    for location, count in location_text_stats.items():
        print(f"     {location}: {count} ({count/len(localizer_df)*100:.1f}%)")
    
    # Series with multiple aneurysms
    series_aneurysm_counts = localizer_df['SeriesInstanceUID'].value_counts()
    multiple_aneurysms = (series_aneurysm_counts > 1).sum()
    max_aneurysms = series_aneurysm_counts.max()
    
    print(f"\nğŸ“ˆ Aneurysm Distribution:")
    print(f"   Series with multiple aneurysms: {multiple_aneurysms}")
    print(f"   Maximum aneurysms in one series: {max_aneurysms}")
    print(f"   Average aneurysms per positive series: {len(localizer_df)/train_df['Aneurysm Present'].sum():.2f}")
    
    # Sample series analysis
    print(f"\nğŸ”� Sample Series Analysis:")
    sample_series = train_df['SeriesInstanceUID'].head(10).tolist()
    
    series_dicom_counts = []
    for series_id in sample_series:
        series_path = Path("/kaggle/input/rsna-intracranial-aneurysm-detection/series") / series_id
        if series_path.exists():
            dcm_count = len(list(series_path.glob("*.dcm")))
            series_dicom_counts.append(dcm_count)
    
    if series_dicom_counts:
        print(f"   DICOM files per series (sample): {np.mean(series_dicom_counts):.1f} Â± {np.std(series_dicom_counts):.1f}")
        print(f"   Range: {min(series_dicom_counts)} - {max(series_dicom_counts)} files")

# Run comprehensive analysis
analyze_real_competition_data()

# Create enhanced visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Modality distribution
modality_counts = explorer.train_df['Modality'].value_counts()
axes[0,0].pie(modality_counts.values, labels=modality_counts.index, autopct='%1.1f%%')
axes[0,0].set_title('Imaging Modality Distribution')

# 2. Age distribution by aneurysm presence
positive_ages = explorer.train_df[explorer.train_df['Aneurysm Present'] == 1]['PatientAge'].dropna()
negative_ages = explorer.train_df[explorer.train_df['Aneurysm Present'] == 0]['PatientAge'].dropna()

axes[0,1].hist(negative_ages, bins=30, alpha=0.7, label='No Aneurysm', color='lightblue')
axes[0,1].hist(positive_ages, bins=30, alpha=0.7, label='Aneurysm Present', color='lightcoral')
axes[0,1].set_xlabel('Age (years)')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Age Distribution by Aneurysm Presence')
axes[0,1].legend()

# 3. Aneurysm presence by modality
modality_aneurysm = explorer.train_df.groupby('Modality')['Aneurysm Present'].agg(['count', 'sum']).reset_index()
modality_aneurysm['rate'] = modality_aneurysm['sum'] / modality_aneurysm['count'] * 100

axes[0,2].bar(modality_aneurysm['Modality'], modality_aneurysm['rate'], color='lightgreen')
axes[0,2].set_ylabel('Positive Rate (%)')
axes[0,2].set_title('Aneurysm Rate by Modality')
axes[0,2].tick_params(axis='x', rotation=45)

# 4. Top aneurysm locations (from localization data)
location_text_counts = explorer.localizer_df['location'].value_counts().head(8)
axes[1,0].barh(range(len(location_text_counts)), location_text_counts.values)
axes[1,0].set_yticks(range(len(location_text_counts)))
axes[1,0].set_yticklabels(location_text_counts.index, fontsize=8)
axes[1,0].set_xlabel('Count')
axes[1,0].set_title('Most Common Aneurysm Locations')

# 5. Aneurysms per series distribution
series_aneurysm_counts = explorer.localizer_df['SeriesInstanceUID'].value_counts()
axes[1,1].hist(series_aneurysm_counts.values, bins=range(1, series_aneurysm_counts.max()+2), 
               alpha=0.7, color='gold', edgecolor='black')
axes[1,1].set_xlabel('Number of Aneurysms per Series')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Aneurysms per Series')

# 6. Gender distribution with aneurysm rates
gender_stats = explorer.train_df.groupby('PatientSex')['Aneurysm Present'].agg(['count', 'sum']).reset_index()
gender_stats['rate'] = gender_stats['sum'] / gender_stats['count'] * 100

x_pos = range(len(gender_stats))
axes[1,2].bar([p - 0.2 for p in x_pos], gender_stats['count'], 0.4, label='Total', alpha=0.7)
ax2 = axes[1,2].twinx()
ax2.bar([p + 0.2 for p in x_pos], gender_stats['rate'], 0.4, label='Positive Rate (%)', 
        color='orange', alpha=0.7)

axes[1,2].set_xlabel('Gender')
axes[1,2].set_ylabel('Count')
ax2.set_ylabel('Positive Rate (%)')
axes[1,2].set_title('Gender Distribution and Aneurysm Rates')
axes[1,2].set_xticks(x_pos)
axes[1,2].set_xticklabels(gender_stats['PatientSex'])
axes[1,2].legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("\nâœ… Enhanced data analysis and visualization complete!")


