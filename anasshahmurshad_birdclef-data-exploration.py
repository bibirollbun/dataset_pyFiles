# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import librosa
import torchaudio
import torch
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("ğŸ“Š BirdCLEF 2025 Dataset Analysis")
print("="*50)


# Load the main datasets
train = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
taxonomy = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')

print(f"âœ… Training data loaded: {train.shape[0]:,} records, {train.shape[1]} columns")
print(f"âœ… Taxonomy data loaded: {taxonomy.shape[0]:,} species, {taxonomy.shape[1]} columns")

# Display basic information
print("\nğŸ“‹ DATASET STRUCTURE")
print("-" * 30)
print("Training Data Columns:", list(train.columns))
print("Taxonomy Data Columns:", list(taxonomy.columns))


# Display first few rows
print("ğŸ”� Training Data Sample:")
display(train.head())

print("\nğŸ”� Taxonomy Data Sample:")
display(taxonomy.head())


# Check for missing values
print("â�Œ MISSING VALUES CHECK")
print("-" * 30)
print("Training Data:")
print(train.isnull().sum())
print("\nTaxonomy Data:")
print(taxonomy.isnull().sum())


# Merge datasets for comprehensive analysis
merged_df = pd.merge(train, taxonomy, on="primary_label", how="left")

# Basic statistics
total_recordings = len(merged_df)
unique_species = train["primary_label"].nunique()
unique_classes = taxonomy["class_name"].nunique()

print("ğŸ�¦ SPECIES DIVERSITY ANALYSIS")
print("="*40)
print(f"ğŸ“Š Total Recordings: {total_recordings:,}")
print(f"ğŸ”¢ Unique Species: {unique_species}")
print(f"ğŸ“‚ Unique Classes: {unique_classes}")

# Class distribution
class_counts = merged_df["class_name"].value_counts()
print(f"\nğŸ“ˆ RECORDINGS PER CLASS:")
for class_name, count in class_counts.items():
    percentage = (count / total_recordings) * 100
    print(f"  {class_name}: {count:,} ({percentage:.1f}%)")


# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Class Distribution
ax1 = plt.subplot(3, 3, 1)
class_counts.plot(kind='bar', color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
plt.title('ğŸ”¢ Recordings per Class', fontsize=14, fontweight='bold')
plt.xlabel('Class Name')
plt.ylabel('Number of Recordings')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# 2. Species per Class
ax2 = plt.subplot(3, 3, 2)
species_per_class = merged_df.groupby("class_name")["primary_label"].nunique()
species_per_class.plot(kind='bar', color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
plt.title('ğŸ�¦ Unique Species per Class', fontsize=14, fontweight='bold')
plt.xlabel('Class Name')
plt.ylabel('Number of Species')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# 3. Collection Source Distribution
ax3 = plt.subplot(3, 3, 3)
collection_counts = merged_df['collection'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
plt.pie(collection_counts.values, labels=collection_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('ğŸ“š Data Sources Distribution', fontsize=14, fontweight='bold')

# 4. Rating Distribution (XC only)
ax4 = plt.subplot(3, 3, 4)
xc_data = merged_df[merged_df['collection'] == 'XC']
xc_ratings = xc_data[xc_data['rating'] > 0]['rating']
plt.hist(xc_ratings, bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black')
plt.title('â­� Rating Distribution (XC Collection)', fontsize=14, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)

# 5. Top 20 Species by Recording Count
ax5 = plt.subplot(3, 3, 5)
top_species = train.groupby('primary_label')['filename'].count().sort_values(ascending=False).head(20)
top_species.plot(kind='barh', color='#4ECDC4')
plt.title('ğŸ�† Top 20 Species by Recording Count', fontsize=14, fontweight='bold')
plt.xlabel('Number of Recordings')

# 6. Long Tail Distribution
ax6 = plt.subplot(3, 3, 6)
species_counts = train.groupby('primary_label')['filename'].count().sort_values(ascending=False)
plt.loglog(range(1, len(species_counts) + 1), species_counts.values, 'bo-', alpha=0.7)
plt.title('ğŸ“ˆ Long Tail Distribution (Log Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Species Rank')
plt.ylabel('Number of Recordings')
plt.grid(True, alpha=0.3)

# 7. Geographical Distribution
ax7 = plt.subplot(3, 3, (7, 8))
# Sample data for faster plotting
sample_size = min(5000, len(merged_df))
geo_sample = merged_df.sample(n=sample_size, random_state=42)

scatter = plt.scatter(geo_sample['longitude'], geo_sample['latitude'], 
                     c=pd.Categorical(geo_sample['class_name']).codes, 
                     cmap='viridis', alpha=0.6, s=10)
plt.title('ğŸŒ� Global Distribution of Recordings', fontsize=14, fontweight='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Class Type')

# 8. Average Rating by Class
ax8 = plt.subplot(3, 3, 9)
avg_rating_by_class = merged_df[merged_df['rating'] > 0].groupby('class_name')['rating'].mean()
avg_rating_by_class.plot(kind='bar', color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
plt.title('â­� Average Rating by Class', fontsize=14, fontweight='bold')
plt.xlabel('Class Name')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


classes = ['Aves', 'Amphibia', 'Mammalia', 'Insecta']
class_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for i, class_name in enumerate(classes):
    class_data = merged_df[merged_df['class_name'] == class_name]
    species_counts = class_data.groupby('primary_label')['filename'].count().sort_values(ascending=False)
    
    # Plot top 20 species for each class
    top_20 = species_counts.head(20)
    axes[i].barh(range(len(top_20)), top_20.values, color=class_colors[i], alpha=0.8)
    axes[i].set_yticks(range(len(top_20)))
    axes[i].set_yticklabels(top_20.index, fontsize=8)
    axes[i].set_xlabel('Number of Recordings')
    axes[i].set_title(f'ğŸ”� Top Species in {class_name} Class', fontweight='bold')
    axes[i].grid(axis='x', alpha=0.3)
    
    # Add summary statistics
    total_recordings = len(class_data)
    unique_species = class_data['primary_label'].nunique()
    axes[i].text(0.02, 0.98, f'Total: {total_recordings:,}\nSpecies: {unique_species}', 
                transform=axes[i].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()


# Geographic statistics by class
geo_stats = merged_df.groupby('class_name').agg({
    'latitude': ['min', 'max', 'mean'],
    'longitude': ['min', 'max', 'mean'],
    'primary_label': 'count'
}).round(2)

print("ğŸŒ� GEOGRAPHIC ANALYSIS")
print("="*30)
print("Geographic Distribution Statistics:")
display(geo_stats)


# Load the world map from Geopandas
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# Create individual geographic plots for each class
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.ravel()

for i, class_name in enumerate(classes):
    class_data = merged_df[merged_df["class_name"] == class_name]
    
    # Plot the world map background
    world.plot(ax=axes[i], color="lightgray", edgecolor="black", alpha=0.5)
    
    # Scatter plot for species locations
    axes[i].scatter(class_data["longitude"], class_data["latitude"], 
                   alpha=0.6, s=10, color=class_colors[i])
    
    axes[i].set_title(f'ğŸŒ� {class_name} Geographic Distribution', fontweight="bold")
    axes[i].set_xlabel("Longitude")
    axes[i].set_ylabel("Latitude")
    axes[i].grid(True, alpha=0.3)
    
    # Add count annotation
    count = len(class_data)
    axes[i].text(0.02, 0.98, f'Records: {count:,}', 
                 transform=axes[i].transAxes, fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()


# Analyze train_soundscapes directory
soundscape_path = "/kaggle/input/birdclef-2025/train_soundscapes/"
soundscape_files = [f for f in os.listdir(soundscape_path) if f.endswith('.ogg')]

print("ğŸ”Š AUDIO DATA ANALYSIS")
print("="*30)
print(f"ğŸ“� Soundscape Files: {len(soundscape_files)}")

# Sample analysis of audio characteristics
sample_size = min(2000, len(soundscape_files))
sampled_files = random.sample(soundscape_files, sample_size)

print(f"ğŸ�µ Analyzing {sample_size} sample files for audio characteristics...")


durations = []
rms_values = []

for file in tqdm(sampled_files, desc="Processing audio"):
    try:
        audio_path = os.path.join(soundscape_path, file)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Calculate duration
        duration = waveform.shape[1] / sample_rate
        durations.append(duration)
        
        # Calculate RMS energy (noise level indicator)
        rms_energy = torch.sqrt(torch.mean(waveform ** 2)).item()
        rms_values.append(rms_energy)
        
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

print(f"âœ… Successfully processed {len(durations)} audio files")


# Plot audio characteristics
if durations and rms_values:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Duration distribution
    ax1.hist(durations, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('ğŸ•� Audio Duration Distribution')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvline(np.mean(durations), color='red', linestyle='--', 
               label=f'Mean: {np.mean(durations):.1f}s')
    ax1.legend()
    
    # RMS energy distribution
    ax2.hist(rms_values, bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('RMS Energy')
    ax2.set_ylabel('Frequency')
    ax2.set_title('ğŸ“Š Background Noise Level Distribution')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(np.mean(rms_values), color='red', linestyle='--', 
               label=f'Mean: {np.mean(rms_values):.4f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"ğŸ“Š Audio Statistics:")
    print(f"   Duration: {np.mean(durations):.1f}Â±{np.std(durations):.1f} seconds")
    print(f"   RMS Energy: {np.mean(rms_values):.4f}Â±{np.std(rms_values):.4f}")


# Data quality checks
print("ğŸ”� DATA QUALITY ASSESSMENT")
print("="*40)

# Check for species with very few recordings
species_counts = train.groupby('primary_label')['filename'].count()
low_count_species = species_counts[species_counts < 5]

print(f"ğŸ“Š Recording Distribution:")
print(f"   Total species: {len(species_counts)}")
print(f"   Species with <5 recordings: {len(low_count_species)} ({len(low_count_species)/len(species_counts)*100:.1f}%)")
print(f"   Species with <10 recordings: {len(species_counts[species_counts < 10])} ({len(species_counts[species_counts < 10])/len(species_counts)*100:.1f}%)")

# Rating quality for XC collection
xc_data = merged_df[merged_df['collection'] == 'XC']
rated_recordings = len(xc_data[xc_data['rating'] > 0])
total_xc = len(xc_data)

print(f"\nâ­� Rating Coverage (XC Collection):")
print(f"   Rated recordings: {rated_recordings:,}/{total_xc:,} ({rated_recordings/total_xc*100:.1f}%)")
print(f"   Average rating: {xc_data[xc_data['rating'] > 0]['rating'].mean():.2f}")


print("ğŸ“Š KEY INSIGHTS SUMMARY")
print("="*50)

print("ğŸ�¯ DATASET CHARACTERISTICS:")
print(f"   â€¢ Total recordings: {total_recordings:,}")
print(f"   â€¢ Unique species: {unique_species}")
print(f"   â€¢ Data sources: {', '.join(collection_counts.index)}")
print(f"   â€¢ Dominant class: {class_counts.index[0]} ({class_counts.iloc[0]:,} recordings)")

print("\nğŸ”� DATA DISTRIBUTION:")
print("   â€¢ Highly imbalanced dataset with long-tail distribution")
print("   â€¢ Aves (birds) dominate with 97% of recordings")
print("   â€¢ Geographic concentration in Americas and Europe")

print("\nâš ï¸�  MODELING CONSIDERATIONS:")
print("   â€¢ Class imbalance requires careful sampling strategies")
print("   â€¢ Long-tail distribution suggests need for data augmentation")
print("   â€¢ Geographic bias may affect model generalization")
print("   â€¢ Multi-source data requires consistent preprocessing")

print("\nğŸ’¡ RECOMMENDATIONS:")
print("   â€¢ Use stratified sampling for train/validation splits")
print("   â€¢ Apply class weighting or focal loss for imbalanced classes")
print("   â€¢ Consider geographic stratification for robust evaluation")
print("   â€¢ Implement data augmentation for rare species")
print("   â€¢ Filter low-quality recordings based on ratings")

print("\nâœ… Analysis Complete! Ready for modeling phase.")

