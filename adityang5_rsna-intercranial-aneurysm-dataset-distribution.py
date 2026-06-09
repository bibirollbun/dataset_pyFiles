# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# image
import pydicom as dicom

# other stuff
import warnings
warnings.filterwarnings('ignore')

import ast


# configs
pd.set_option('display.max_columns', None) # we want to display all columns in this notebook
pd.set_option('display.max_rows', 100) # increase number of displayed rows
pd.set_option('max_colwidth', None) # make full cells content visible

# random seed
my_random_seed = 123

# aesthetics
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'


# ===============================================================================
# DATA LOADING
# ===============================================================================

print("="*80)
print("RSNA INTRACRANIAL ANEURYSM DETECTION - DATASET ANALYSIS")
print("="*80)

t1 = time.time()
df_train = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
df_train_local = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
t2 = time.time()
print(f'Data loading time: {np.round(t2-t1,4)} seconds')


# ===============================================================================
# VISUALIZATIONS
# ===============================================================================

print("\n" + "="*50)
print("PART 1: VISUAL ANALYSIS")
print("="*50)

# Dataset Overview
print(f"\nDATASET SHAPES:")
print(f"Training data: {df_train.shape}")
print(f"Localizer data: {df_train_local.shape}")

# Basic statistics visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Age distribution
axes[0,0].hist(df_train['PatientAge'], bins=25, color=default_color_1, alpha=0.7)
axes[0,0].set_title('Patient Age Distribution')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Frequency')
axes[0,0].grid(True, alpha=0.3)

# 2. Sex distribution
sex_counts = df_train['PatientSex'].value_counts()
axes[0,1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightpink'])
axes[0,1].set_title('Patient Sex Distribution')

# 3. Modality distribution
modality_counts = df_train['Modality'].value_counts()
axes[1,0].bar(modality_counts.index, modality_counts.values, color=default_color_2)
axes[1,0].set_title('Imaging Modality Distribution')
axes[1,0].set_xlabel('Modality')
axes[1,0].set_ylabel('Count')
axes[1,0].tick_params(axis='x', rotation=45)
for i, v in enumerate(modality_counts.values):
    axes[1,0].text(i, v + 10, str(v), ha='center')

# 4. Aneurysm presence
aneurysm_counts = df_train['Aneurysm Present'].value_counts()
axes[1,1].bar(['No Aneurysm', 'Aneurysm Present'], aneurysm_counts.values, color=[default_color_3, 'orange'])
axes[1,1].set_title('Aneurysm Presence Distribution')
axes[1,1].set_ylabel('Count')
for i, v in enumerate(aneurysm_counts.values):
    axes[1,1].text(i, v + 20, str(v), ha='center')

plt.tight_layout()
plt.show()

# Anatomical location analysis
location_columns = [
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

# Location frequency (only positive cases)
location_sums = df_train[location_columns].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(location_sums)), location_sums.values, color='steelblue')
plt.title('Aneurysm Frequency by Anatomical Location')
plt.xlabel('Anatomical Location')
plt.ylabel('Number of Aneurysms')
plt.xticks(range(len(location_sums)), location_sums.index, rotation=45, ha='right')

# Add value labels on bars
for bar, value in zip(bars, location_sums.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(value), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Modality vs Aneurysm presence
plt.figure(figsize=(10, 6))
modality_aneurysm = pd.crosstab(df_train['Modality'], df_train['Aneurysm Present'])
modality_aneurysm_pct = modality_aneurysm.div(modality_aneurysm.sum(axis=1), axis=0) * 100

ax = modality_aneurysm_pct.plot(kind='bar', stacked=True, 
                                color=['lightcoral', 'lightgreen'],
                                figsize=(10, 6))
plt.title('Aneurysm Presence Rate by Imaging Modality')
plt.xlabel('Modality')
plt.ylabel('Percentage')
plt.legend(['No Aneurysm', 'Aneurysm Present'])
plt.xticks(rotation=45)

# Add percentage labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='center')

plt.tight_layout()
plt.show()

# Localizer data analysis
if not df_train_local.empty:
    # Convert coordinates to dictionary if needed
    if isinstance(df_train_local['coordinates'].iloc[0], str):
        df_train_local['coordinates'] = df_train_local['coordinates'].map(ast.literal_eval)
    
    # Extract coordinates
    df_train_local['x'] = df_train_local['coordinates'].map(lambda d: d['x'])
    df_train_local['y'] = df_train_local['coordinates'].map(lambda d: d['y'])
    
    # Location distribution in localizer data
    plt.figure(figsize=(12, 8))
    location_local_counts = df_train_local['location'].value_counts()
    bars = plt.bar(range(len(location_local_counts)), location_local_counts.values, color='darkorange')
    plt.title('Localized Aneurysm Distribution by Anatomical Location')
    plt.xlabel('Anatomical Location')
    plt.ylabel('Count')
    plt.xticks(range(len(location_local_counts)), location_local_counts.index, rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars, location_local_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Spatial distribution of aneurysms
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(df_train_local['x'], df_train_local['y'], 
                         c=df_train_local['location'].astype('category').cat.codes, 
                         cmap='tab20', alpha=0.7, s=50)
    plt.title('Spatial Distribution of Aneurysms (X-Y Coordinates)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Create custom legend
    unique_locations = df_train_local['location'].unique()
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab20(i/len(unique_locations)), 
                                 markersize=8, label=loc) 
                      for i, loc in enumerate(unique_locations)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

        # Bounding box position analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # X-coordinate distribution
    axes[0,0].hist(df_train_local['x'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Distribution of Aneurysm X-Coordinates')
    axes[0,0].set_xlabel('X Coordinate (pixels)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    
    # Y-coordinate distribution
    axes[0,1].hist(df_train_local['y'], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[0,1].set_title('Distribution of Aneurysm Y-Coordinates')
    axes[0,1].set_xlabel('Y Coordinate (pixels)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].grid(True, alpha=0.3)
    
    # X-coordinate by location (box plot)
    unique_locations_short = df_train_local['location'].unique()[:8]  # Limit for readability
    data_subset = df_train_local[df_train_local['location'].isin(unique_locations_short)]
    axes[1,0].boxplot([data_subset[data_subset['location'] == loc]['x'].values 
                       for loc in unique_locations_short], 
                      labels=[loc[:20] + '...' if len(loc) > 20 else loc 
                             for loc in unique_locations_short])
    axes[1,0].set_title('X-Coordinate Distribution by Location (Top 8)')
    axes[1,0].set_xlabel('Anatomical Location')
    axes[1,0].set_ylabel('X Coordinate')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Y-coordinate by location (box plot)
    axes[1,1].boxplot([data_subset[data_subset['location'] == loc]['y'].values 
                       for loc in unique_locations_short], 
                      labels=[loc[:20] + '...' if len(loc) > 20 else loc 
                             for loc in unique_locations_short])
    axes[1,1].set_title('Y-Coordinate Distribution by Location (Top 8)')
    axes[1,1].set_xlabel('Anatomical Location')
    axes[1,1].set_ylabel('Y Coordinate')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Heatmap of aneurysm positions
    plt.figure(figsize=(12, 8))
    
    # Create 2D histogram
    x_bins = 20
    y_bins = 20
    hist, x_edges, y_edges = np.histogram2d(df_train_local['x'], df_train_local['y'], 
                                           bins=[x_bins, y_bins])
    
    # Plot heatmap
    plt.imshow(hist.T, origin='lower', cmap='YlOrRd', aspect='auto',
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    plt.colorbar(label='Aneurysm Count')
    plt.title('Spatial Density Heatmap of Aneurysm Locations')
    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.show()


print("\nVisual analysis complete!")


# ===============================================================================
# TEXT SUMMARY
# ===============================================================================

print("\n" + "="*80)
print("PART 2: TEXT SUMMARY")
print("="*80)

def generate_dataset_summary():
    """Generate a comprehensive text summary for LLM consumption"""
    
    summary = []
    summary.append("RSNA INTRACRANIAL ANEURYSM DETECTION - DATASET SUMMARY")
    summary.append("=" * 60)
    
    # Basic dataset info
    summary.append(f"\nDATASET OVERVIEW:")
    summary.append(f"- Training samples: {len(df_train):,}")
    summary.append(f"- Localizer samples: {len(df_train_local):,}")
    summary.append(f"- Features: {len(df_train.columns)}")
    summary.append(f"- Target variable: Aneurysm Present (binary)")
    
    # Patient demographics
    summary.append(f"\nPATIENT DEMOGRAPHICS:")
    summary.append(f"- Age range: {df_train['PatientAge'].min()}-{df_train['PatientAge'].max()} years")
    summary.append(f"- Mean age: {df_train['PatientAge'].mean():.1f} Â± {df_train['PatientAge'].std():.1f} years")
    summary.append(f"- Median age: {df_train['PatientAge'].median():.0f} years")
    
    sex_counts = df_train['PatientSex'].value_counts()
    for sex, count in sex_counts.items():
        pct = (count / len(df_train)) * 100
        summary.append(f"- {sex}: {count:,} ({pct:.1f}%)")
    
    # Imaging modalities
    summary.append(f"\nIMAGING MODALITIES:")
    modality_counts = df_train['Modality'].value_counts().sort_values(ascending=False)
    for modality, count in modality_counts.items():
        pct = (count / len(df_train)) * 100
        summary.append(f"- {modality}: {count:,} ({pct:.1f}%)")
    
    # Target distribution
    summary.append(f"\nTARGET DISTRIBUTION:")
    aneurysm_counts = df_train['Aneurysm Present'].value_counts()
    no_aneurysm = aneurysm_counts[0]
    aneurysm = aneurysm_counts[1]
    summary.append(f"- No aneurysm: {no_aneurysm:,} ({(no_aneurysm/len(df_train)*100):.1f}%)")
    summary.append(f"- Aneurysm present: {aneurysm:,} ({(aneurysm/len(df_train)*100):.1f}%)")
    summary.append(f"- Class balance ratio: {no_aneurysm/aneurysm:.2f}:1 (negative:positive)")
    
    # Anatomical locations
    summary.append(f"\nANATOMICAL LOCATION FREQUENCY:")
    location_sums = df_train[location_columns].sum().sort_values(ascending=False)
    for location, count in location_sums.items():
        pct_of_total = (count / len(df_train)) * 100
        pct_of_positive = (count / aneurysm) * 100 if aneurysm > 0 else 0
        summary.append(f"- {location}: {count} ({pct_of_total:.2f}% of all, {pct_of_positive:.1f}% of positive cases)")
    
    # Modality vs aneurysm analysis
    summary.append(f"\nMODALITY vs ANEURYSM PRESENCE:")
    for modality in df_train['Modality'].unique():
        modality_data = df_train[df_train['Modality'] == modality]
        total_modality = len(modality_data)
        aneurysm_modality = modality_data['Aneurysm Present'].sum()
        aneurysm_rate = (aneurysm_modality / total_modality) * 100
        summary.append(f"- {modality}: {aneurysm_modality}/{total_modality} positive ({aneurysm_rate:.1f}%)")
    
    # Localizer data analysis
    if not df_train_local.empty:
        summary.append(f"\nLOCALIZER DATA ANALYSIS:")
        summary.append(f"- Total localized aneurysms: {len(df_train_local):,}")
        summary.append(f"- Unique series with localizations: {df_train_local['SeriesInstanceUID'].nunique():,}")
        
        # Convert coordinates if needed
        if isinstance(df_train_local['coordinates'].iloc[0], str):
            df_train_local_temp = df_train_local.copy()
            df_train_local_temp['coordinates'] = df_train_local_temp['coordinates'].map(ast.literal_eval)
            df_train_local_temp['x'] = df_train_local_temp['coordinates'].map(lambda d: d['x'])
            df_train_local_temp['y'] = df_train_local_temp['coordinates'].map(lambda d: d['y'])
        else:
            df_train_local_temp = df_train_local.copy()
        
        summary.append(f"- X-coordinate range: {df_train_local_temp['x'].min():.1f} to {df_train_local_temp['x'].max():.1f}")
        summary.append(f"- Y-coordinate range: {df_train_local_temp['y'].min():.1f} to {df_train_local_temp['y'].max():.1f}")
        
        summary.append(f"\nLOCALIZED ANEURYSM DISTRIBUTION:")
        location_local_counts = df_train_local['location'].value_counts().sort_values(ascending=False)
        for location, count in location_local_counts.items():
            pct = (count / len(df_train_local)) * 100
            summary.append(f"- {location}: {count} ({pct:.1f}%)")
    
    # Data quality insights
    summary.append(f"\nDATA QUALITY INSIGHTS:")
    summary.append(f"- Missing values in training data: {df_train.isnull().sum().sum()}")
    summary.append(f"- Missing values in localizer data: {df_train_local.isnull().sum().sum()}")
    
    # Multiple aneurysms analysis
    aneurysm_per_case = df_train[location_columns].sum(axis=1)
    multiple_aneurysms = (aneurysm_per_case > 1).sum()
    summary.append(f"- Cases with multiple aneurysms: {multiple_aneurysms} ({(multiple_aneurysms/aneurysm*100):.1f}% of positive cases)")
    summary.append(f"- Maximum aneurysms per case: {aneurysm_per_case.max()}")
    
    # Coverage analysis
    total_positive_cases = df_train['Aneurysm Present'].sum()
    cases_with_localization = df_train[df_train['SeriesInstanceUID'].isin(df_train_local['SeriesInstanceUID'].unique())]['Aneurysm Present'].sum()
    coverage = (cases_with_localization / total_positive_cases) * 100 if total_positive_cases > 0 else 0
    
    summary.append(f"\nLOCALIZATION COVERAGE:")
    summary.append(f"- Positive cases with localization data: {cases_with_localization}/{total_positive_cases} ({coverage:.1f}%)")

        # Bounding box position statistics
    summary.append(f"\nBOUNDING BOX POSITION STATISTICS:")
    summary.append(f"- X-coordinate statistics:")
    summary.append(f"  * Mean: {df_train_local_temp['x'].mean():.1f} Â± {df_train_local_temp['x'].std():.1f}")
    summary.append(f"  * Median: {df_train_local_temp['x'].median():.1f}")
    summary.append(f"  * Range: {df_train_local_temp['x'].min():.1f} - {df_train_local_temp['x'].max():.1f}")
    summary.append(f"  * IQR: {df_train_local_temp['x'].quantile(0.25):.1f} - {df_train_local_temp['x'].quantile(0.75):.1f}")
    
    summary.append(f"- Y-coordinate statistics:")
    summary.append(f"  * Mean: {df_train_local_temp['y'].mean():.1f} Â± {df_train_local_temp['y'].std():.1f}")
    summary.append(f"  * Median: {df_train_local_temp['y'].median():.1f}")
    summary.append(f"  * Range: {df_train_local_temp['y'].min():.1f} - {df_train_local_temp['y'].max():.1f}")
    summary.append(f"  * IQR: {df_train_local_temp['y'].quantile(0.25):.1f} - {df_train_local_temp['y'].quantile(0.75):.1f}")
    
    # Position analysis by anatomical location
    summary.append(f"\nPOSITION ANALYSIS BY ANATOMICAL LOCATION:")
    for location in df_train_local['location'].value_counts().head(5).index:  # Top 5 locations
        location_data = df_train_local_temp[df_train_local_temp['location'] == location]
        x_mean = location_data['x'].mean()
        x_std = location_data['x'].std()
        y_mean = location_data['y'].mean()
        y_std = location_data['y'].std()
        summary.append(f"- {location}:")
        summary.append(f"  * X: {x_mean:.1f} Â± {x_std:.1f}, Y: {y_mean:.1f} Â± {y_std:.1f}")
        summary.append(f"  * Sample size: {len(location_data)}")
    
    # Spatial clustering analysis
    x_clusters = []
    y_clusters = []
    
    # Simple quartile-based clustering
    x_q1, x_q3 = df_train_local_temp['x'].quantile([0.25, 0.75])
    y_q1, y_q3 = df_train_local_temp['y'].quantile([0.25, 0.75])
    
    # Count aneurysms in different quadrants
    left_upper = len(df_train_local_temp[(df_train_local_temp['x'] <= x_q1) & (df_train_local_temp['y'] >= y_q3)])
    right_upper = len(df_train_local_temp[(df_train_local_temp['x'] >= x_q3) & (df_train_local_temp['y'] >= y_q3)])
    left_lower = len(df_train_local_temp[(df_train_local_temp['x'] <= x_q1) & (df_train_local_temp['y'] <= y_q1)])
    right_lower = len(df_train_local_temp[(df_train_local_temp['x'] >= x_q3) & (df_train_local_temp['y'] <= y_q1)])
    center = len(df_train_local_temp) - (left_upper + right_upper + left_lower + right_lower)
    
    summary.append(f"\nSPATIAL DISTRIBUTION QUADRANTS:")
    summary.append(f"- Left-Upper: {left_upper} ({left_upper/len(df_train_local_temp)*100:.1f}%)")
    summary.append(f"- Right-Upper: {right_upper} ({right_upper/len(df_train_local_temp)*100:.1f}%)")
    summary.append(f"- Left-Lower: {left_lower} ({left_lower/len(df_train_local_temp)*100:.1f}%)")
    summary.append(f"- Right-Lower: {right_lower} ({right_lower/len(df_train_local_temp)*100:.1f}%)")
    summary.append(f"- Center: {center} ({center/len(df_train_local_temp)*100:.1f}%)")

    
    return "\n".join(summary)

# Generate and print the summary
dataset_summary = generate_dataset_summary()
print(dataset_summary)

# Save summary to text file (optional)
# with open('dataset_summary.txt', 'w') as f:
#     f.write(dataset_summary)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")





