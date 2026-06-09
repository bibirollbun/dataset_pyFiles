import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from IPython.display import display, HTML


# Set up paths - adjust these to match your directory structure
BASE_PATH = Path('/kaggle/input/rsna-intracranial-aneurysm-detection')

# Load the train labels CSV
train_df = pd.read_csv(BASE_PATH / 'train.csv')


print(f"Dataset shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")
print("\nFirst few rows:")
display(train_df.head())


# Verify the correct directory structure
print("Verifying RSNA dataset structure...")
print("=" * 60)

# Check for the series directory
series_dir = BASE_PATH / 'series'
if series_dir.exists():
    series_folders = list(series_dir.iterdir())
    print(f"Found 'series' directory with {len(series_folders)} series folders")
    
    # Show a few examples
    print(f"\nExample series folders:")
    for folder in series_folders[:3]:
        if folder.is_dir():
            dcm_files = list(folder.glob('*.dcm'))
            print(f"  ğŸ“� {folder.name[:50]}...")
            print(f"     â””â”€â”€ Contains {len(dcm_files)} DICOM files")
else:
    print("'series' directory not found")

print("=" * 60)


# Cell 5: Updated DICOM loader with correct path structure
def load_dicom_series(series_uid, base_path=BASE_PATH):
    """Load all DICOM files for a given SeriesInstanceUID."""
    
    # Construct the correct path
    series_path = base_path / 'series' / series_uid
    
    if not series_path.exists():
        print(f"Series not found at: {series_path}")
        return []
    
    # Get all DICOM files in the series directory
    dcm_files = sorted(series_path.glob('*.dcm'))
    
    print(f"Found {len(dcm_files)} DICOM files for series {series_uid[:30]}...")
    
    # Load the DICOM files
    dicoms = []
    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_file)
            dicoms.append(ds)
        except Exception as e:
            print(f"Error reading {dcm_file.name}: {e}")
    
    print(f"Successfully loaded {len(dicoms)} DICOM files")
    return dicoms


# Test with the first series from our CSV
sample_series_uid = train_df['SeriesInstanceUID'].iloc[0]
print(f"Loading series: {sample_series_uid}")
sample_dicoms = load_dicom_series(sample_series_uid)


# Cell 6: Display DICOM metadata and basic info
def analyze_dicom_series(dicoms):
    """Analyze and display information about a DICOM series."""
    
    if not dicoms:
        print("No DICOM files to analyze")
        return
    
    # Get metadata from first DICOM
    first_dcm = dicoms[0]
    
    print("=" * 60)
    print("DICOM SERIES INFORMATION")
    print("=" * 60)
    
    # Basic metadata
    metadata = {
        'Number of slices': len(dicoms),
        'Patient ID': getattr(first_dcm, 'PatientID', 'N/A'),
        'Patient Age': getattr(first_dcm, 'PatientAge', 'N/A'),
        'Patient Sex': getattr(first_dcm, 'PatientSex', 'N/A'),
        'Study Date': getattr(first_dcm, 'StudyDate', 'N/A'),
        'Modality': getattr(first_dcm, 'Modality', 'N/A'),
        'Manufacturer': getattr(first_dcm, 'Manufacturer', 'N/A'),
        'Slice Thickness': f"{getattr(first_dcm, 'SliceThickness', 'N/A')} mm",
        'Image Size': f"{getattr(first_dcm, 'Rows', 'N/A')} x {getattr(first_dcm, 'Columns', 'N/A')}",
    }
    
    for key, value in metadata.items():
        print(f"{key:20}: {value}")
    
    # Window settings
    window_center = getattr(first_dcm, 'WindowCenter', None)
    window_width = getattr(first_dcm, 'WindowWidth', None)
    
    if window_center is not None and window_width is not None:
        # Handle cases where these might be multi-valued
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = window_center[0]
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = window_width[0]
        
        print(f"Window Center       : {window_center}")
        print(f"Window Width        : {window_width}")
    
    # Pixel array info
    try:
        pixel_array = first_dcm.pixel_array
        print(f"\nPixel Array Info:")
        print(f"  Shape             : {pixel_array.shape}")
        print(f"  Data type         : {pixel_array.dtype}")
        print(f"  Min value         : {pixel_array.min()}")
        print(f"  Max value         : {pixel_array.max()}")
    except Exception as e:
        print(f"\nCould not access pixel array: {e}")
    
    return metadata


metadata = analyze_dicom_series(sample_dicoms)


# Cell 7: Visualization functions for DICOM images
def apply_windowing(img, window_center, window_width):
    """Apply windowing to improve contrast for visualization."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    img_windowed = img.copy()
    img_windowed[img_windowed < img_min] = img_min
    img_windowed[img_windowed > img_max] = img_max
    
    # Normalize to 0-255
    img_windowed = ((img_windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    return img_windowed

def get_pixel_array_hu(dicom_dataset):
    """Convert pixel array to Hounsfield Units and apply windowing."""
    # Get pixel array
    image = dicom_dataset.pixel_array.astype(float)
    
    # Convert to Hounsfield Units (HU)
    rescale_slope = getattr(dicom_dataset, 'RescaleSlope', 1)
    rescale_intercept = getattr(dicom_dataset, 'RescaleIntercept', 0)
    
    if rescale_slope != 1 or rescale_intercept != 0:
        image = image * rescale_slope + rescale_intercept
    
    # Get window settings (use brain window as default)
    window_center = getattr(dicom_dataset, 'WindowCenter', 40)
    window_width = getattr(dicom_dataset, 'WindowWidth', 80)
    
    # Handle multi-valued window settings
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = float(window_center[0])
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = float(window_width[0])
    
    # Apply windowing
    img_windowed = apply_windowing(image, window_center, window_width)
    
    return img_windowed, image  # Return both windowed and HU values


# Test visualization with a single slice
if sample_dicoms:
    middle_idx = len(sample_dicoms) // 2
    img_windowed, img_hu = get_pixel_array_hu(sample_dicoms[middle_idx])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Windowed image
    axes[0].imshow(img_windowed, cmap='gray')
    axes[0].set_title(f'Windowed Image (Slice {middle_idx + 1}/{len(sample_dicoms)})')
    axes[0].axis('off')
    
    # Histogram of HU values
    axes[1].hist(img_hu.flatten(), bins=50, edgecolor='black')
    axes[1].set_xlabel('Hounsfield Units (HU)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of HU Values')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Cell 8: Display multiple slices from a series
def plot_series_slices(dicoms, num_slices=16, cols=4):
    """Display multiple slices from a DICOM series."""
    
    if not dicoms:
        print("No DICOM files to display")
        return
    
    num_slices = min(num_slices, len(dicoms))
    rows = (num_slices + cols - 1) // cols
    
    # Sample evenly across the series
    indices = np.linspace(0, len(dicoms) - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_slices > 1 else [axes]
    
    for i, idx in enumerate(indices):
        if i < len(axes):
            try:
                img_windowed, _ = get_pixel_array_hu(dicoms[idx])
                axes[i].imshow(img_windowed, cmap='gray')
                axes[i].set_title(f'Slice {idx + 1}/{len(dicoms)}')
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)[:30]}', 
                           ha='center', va='center')
                axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')
    
    # Add series info to the title
    series_uid = train_df['SeriesInstanceUID'].iloc[0]
    series_info = train_df[train_df['SeriesInstanceUID'] == series_uid].iloc[0]
    aneurysm_status = "WITH Aneurysm" if series_info['Aneurysm Present'] else "NO Aneurysm"
    
    plt.suptitle(f'Series: {series_uid[:30]}... - {aneurysm_status}\n'
                 f'Modality: {series_info["Modality"]} | '
                 f'Age: {series_info["PatientAge"]} | '
                 f'Sex: {series_info["PatientSex"]}',
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig('multi_slice_analysis.png')
    plt.show()



plot_series_slices(sample_dicoms, num_slices=16)


# Cell 9: Compare positive and negative cases
def compare_aneurysm_cases(train_df, num_slices_per_series=4):
    """Display side-by-side comparison of cases with and without aneurysms."""
    
    # Get one positive and one negative case
    positive_series = train_df[train_df['Aneurysm Present'] == 1]['SeriesInstanceUID'].iloc[0]
    negative_series = train_df[train_df['Aneurysm Present'] == 0]['SeriesInstanceUID'].iloc[0]
    
    print(f"Loading positive case: {positive_series[:30]}...")
    pos_dicoms = load_dicom_series(positive_series)
    
    print(f"Loading negative case: {negative_series[:30]}...")
    neg_dicoms = load_dicom_series(negative_series)
    
    if not pos_dicoms or not neg_dicoms:
        print("Could not load both series for comparison")
        return
    
    # Create comparison figure
    fig, axes = plt.subplots(2, num_slices_per_series, figsize=(num_slices_per_series * 3, 6))
    
    # Sample slices evenly through each series
    pos_indices = np.linspace(0, len(pos_dicoms) - 1, num_slices_per_series, dtype=int)
    neg_indices = np.linspace(0, len(neg_dicoms) - 1, num_slices_per_series, dtype=int)
    
    # Display positive case slices
    for i, idx in enumerate(pos_indices):
        img, _ = get_pixel_array_hu(pos_dicoms[idx])
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Slice {idx + 1}/{len(pos_dicoms)}')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('WITH\nAneurysm', rotation=0, ha='right', va='center')
    
    # Display negative case slices
    for i, idx in enumerate(neg_indices):
        img, _ = get_pixel_array_hu(neg_dicoms[idx])
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Slice {idx + 1}/{len(neg_dicoms)}')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('NO\nAneurysm', rotation=0, ha='right', va='center')
    
    # Get additional information
    pos_info = train_df[train_df['SeriesInstanceUID'] == positive_series].iloc[0]
    neg_info = train_df[train_df['SeriesInstanceUID'] == negative_series].iloc[0]
    
    # Find affected arteries for positive case
    artery_cols = [col for col in train_df.columns if col not in 
                   ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
    affected = [col for col in artery_cols if pos_info[col] == 1]
    
    plt.suptitle(f'Comparison: Aneurysm Cases\n'
                 f'Positive: {pos_info["Modality"]}, Age {pos_info["PatientAge"]}, {pos_info["PatientSex"]} '
                 f'- Affected: {", ".join(affected[:2]) if affected else "Unknown"}\n'
                 f'Negative: {neg_info["Modality"]}, Age {neg_info["PatientAge"]}, {neg_info["PatientSex"]}',
                 fontsize=11)
    
    plt.tight_layout()
    plt.savefig('aneurysm_case_comparison.png')
    plt.show()


compare_aneurysm_cases(train_df)


# Cell 10: Analyze series with multiple aneurysm locations
def analyze_multi_location_cases(train_df):
    """Find and analyze cases with aneurysms in multiple locations."""
    
    # Get artery columns
    artery_cols = [col for col in train_df.columns if col not in 
                   ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
    
    # Count aneurysm locations per case
    positive_cases = train_df[train_df['Aneurysm Present'] == 1].copy()
    positive_cases['num_locations'] = positive_cases[artery_cols].sum(axis=1)
    
    # Find cases with multiple locations
    multi_location = positive_cases[positive_cases['num_locations'] > 1]
    
    print(f"Cases with multiple aneurysm locations: {len(multi_location)} out of {len(positive_cases)} positive cases")
    print(f"Percentage: {len(multi_location) / len(positive_cases) * 100:.1f}%")
    
    # Display distribution
    location_counts = positive_cases['num_locations'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 5))
    
    # Bar chart of location counts
    plt.subplot(1, 2, 1)
    location_counts.plot(kind='bar', color='mediumpurple')
    plt.xlabel('Number of Aneurysm Locations')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Aneurysm Location Counts')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Show an example case with multiple locations
    if len(multi_location) > 0:
        example = multi_location.iloc[0]
        affected = [col for col in artery_cols if example[col] == 1]
        
        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.9, f"Example Multi-Location Case:", fontsize=12, fontweight='bold')
        plt.text(0.1, 0.8, f"Series: {example['SeriesInstanceUID'][:40]}...", fontsize=10)
        plt.text(0.1, 0.7, f"Age: {example['PatientAge']}, Sex: {example['PatientSex']}", fontsize=10)
        plt.text(0.1, 0.6, f"Modality: {example['Modality']}", fontsize=10)
        plt.text(0.1, 0.5, f"Number of locations: {len(affected)}", fontsize=10)
        
        plt.text(0.1, 0.35, "Affected Arteries:", fontsize=11, fontweight='bold')
        for i, artery in enumerate(affected):
            plt.text(0.1, 0.25 - i*0.08, f"  â€¢ {artery}", fontsize=9)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('analyze_multi_location_cases.png')
    plt.show()
    
    return multi_location


multi_location_cases = analyze_multi_location_cases(train_df)


# Cell 11: Dataset summary statistics and visualizations
def create_comprehensive_analysis(train_df):
    """Create a comprehensive analysis dashboard."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Aneurysm prevalence
    ax1 = plt.subplot(3, 4, 1)
    aneurysm_counts = train_df['Aneurysm Present'].value_counts()
    colors = ['#3498db', '#e74c3c']
    ax1.pie(aneurysm_counts.values, labels=['No Aneurysm', 'Aneurysm'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Overall Aneurysm Prevalence', fontsize=11, fontweight='bold')
    
    # 2. Age distribution
    ax2 = plt.subplot(3, 4, 2)
    train_df['PatientAge'].hist(bins=20, color='#2ecc71', edgecolor='black', ax=ax2)
    ax2.set_xlabel('Age (years)')
    ax2.set_ylabel('Count')
    ax2.set_title('Age Distribution', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sex distribution
    ax3 = plt.subplot(3, 4, 3)
    sex_counts = train_df['PatientSex'].value_counts()
    sex_counts.plot(kind='bar', color=['#9b59b6', '#f39c12'], ax=ax3)
    ax3.set_xlabel('Sex')
    ax3.set_ylabel('Count')
    ax3.set_title('Sex Distribution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Modality distribution
    ax4 = plt.subplot(3, 4, 4)
    modality_counts = train_df['Modality'].value_counts()
    modality_counts.plot(kind='bar', color=['#1abc9c', '#34495e'], ax=ax4)
    ax4.set_xlabel('Modality')
    ax4.set_ylabel('Count')
    ax4.set_title('Imaging Modality', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Age by aneurysm status
    ax5 = plt.subplot(3, 4, 5)
    train_df[train_df['Aneurysm Present']==0]['PatientAge'].hist(bins=15, alpha=0.6, 
                                                                  label='No Aneurysm', 
                                                                  color='#3498db', ax=ax5)
    train_df[train_df['Aneurysm Present']==1]['PatientAge'].hist(bins=15, alpha=0.6, 
                                                                  label='Aneurysm', 
                                                                  color='#e74c3c', ax=ax5)
    ax5.set_xlabel('Age (years)')
    ax5.set_ylabel('Count')
    ax5.set_title('Age by Aneurysm Status', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Top aneurysm locations
    ax6 = plt.subplot(3, 4, (6, 10))  # Span multiple cells
    artery_cols = [col for col in train_df.columns if col not in 
                   ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
    
    location_counts = {}
    for col in artery_cols:
        count = train_df[col].sum()
        if count > 0:
            # Shorten names for display
            short_name = col.replace('Internal Carotid Artery', 'ICA')\
                           .replace('Middle Cerebral Artery', 'MCA')\
                           .replace('Anterior Cerebral Artery', 'ACA')\
                           .replace('Posterior Communicating Artery', 'PComA')\
                           .replace('Anterior Communicating Artery', 'AComA')
            location_counts[short_name] = count
    
    locations = list(location_counts.keys())
    counts = list(location_counts.values())
    
    y_pos = np.arange(len(locations))
    ax6.barh(y_pos, counts, color='#8e44ad')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(locations, fontsize=9)
    ax6.set_xlabel('Number of Cases')
    ax6.set_title('Aneurysm Locations', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. Sex vs Aneurysm
    ax7 = plt.subplot(3, 4, 7)
    sex_aneurysm = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'], normalize='index') * 100
    sex_aneurysm.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'], ax=ax7)
    ax7.set_xlabel('Sex')
    ax7.set_ylabel('Percentage (%)')
    ax7.set_title('Aneurysm Rate by Sex', fontsize=11, fontweight='bold')
    ax7.legend(['No Aneurysm', 'Aneurysm'], loc='upper right')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Modality vs Aneurysm
    ax8 = plt.subplot(3, 4, 8)
    mod_aneurysm = pd.crosstab(train_df['Modality'], train_df['Aneurysm Present'], normalize='index') * 100
    mod_aneurysm.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'], ax=ax8)
    ax8.set_xlabel('Modality')
    ax8.set_ylabel('Percentage (%)')
    ax8.set_title('Aneurysm Rate by Modality', fontsize=11, fontweight='bold')
    ax8.legend(['No Aneurysm', 'Aneurysm'], loc='upper right')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add summary statistics as text
    ax9 = plt.subplot(3, 4, (11, 12))
    ax9.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    ==================
    Total Series: {len(train_df):,}
    Positive Cases: {train_df['Aneurysm Present'].sum():,} ({train_df['Aneurysm Present'].mean()*100:.1f}%)
    
    Age Range: {train_df['PatientAge'].min()}-{train_df['PatientAge'].max()} years
    Mean Age: {train_df['PatientAge'].mean():.1f} Â± {train_df['PatientAge'].std():.1f}
    
    Most Common Location:
    {artery_cols[train_df[artery_cols].sum().argmax()]}
    ({train_df[artery_cols].sum().max()} cases)
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', va='center')
    
    plt.suptitle('RSNA Intracranial Aneurysm Detection - Dataset Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    plt.show()


create_comprehensive_analysis(train_df)

