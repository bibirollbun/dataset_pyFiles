import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
import librosa.display
from matplotlib import cm
from collections import Counter
import glob
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

# Set plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Güncellenmiş path'ler
BASE_PATH = '/kaggle/input/birdclef-2025/'
OUTPUT_PATH = '/kaggle/working/'

# Taxonomy verisini yükle
taxonomy_df = pd.read_csv(f'{BASE_PATH}taxonomy.csv')

# Train verisini yükle
try:
    train_df = pd.read_csv(f'{BASE_PATH}train.csv', nrows=None)
except:
    train_df = pd.read_csv(f'{BASE_PATH}train.csv', nrows=10000)
    print("Loaded subset due to size constraints")

# Çıktı dizini
output_dir = os.path.join(OUTPUT_PATH, 'data_visualizations')
os.makedirs(output_dir, exist_ok=True)

# 1. Analyze bird species distribution
def analyze_species_distribution():
    print("Analyzing species distribution...")
    
    # Train verisinden tür dağılımını al
    if not train_df.empty:
        # Türleri say
        species_counts = train_df['primary_label'].value_counts()
        total_samples = len(train_df)
        
        # Tür dağılım grafiği
        plt.figure(figsize=(14, 8))
        species_counts[:20].plot(kind='bar')
        plt.title(f'Top 20 Bird Species (Total Samples: {total_samples})')
        plt.xlabel('Species Code')
        plt.ylabel('Number of Recordings')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'species_distribution.png'))
        plt.show()  # Grafiği göster
        plt.close()
        
        return species_counts
    else:
        print("Train data not available for species distribution analysis")
        return None

# 2. Collection Analysis - YENİ EKLENEN
def analyze_collections():
    print("\nAnalyzing data collections...")
    if 'collection' in train_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collection dağılımı
        collection_counts = train_df['collection'].value_counts()
        axes[0,0].pie(collection_counts.values, labels=collection_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Data Collection Distribution')
        
        # Collection bazında rating dağılımı
        if 'rating' in train_df.columns:
            sns.boxplot(data=train_df, x='collection', y='rating', ax=axes[0,1])
            axes[0,1].set_title('Rating Distribution by Collection')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Collection bazında tür sayısı
        collection_species = train_df.groupby('collection')['primary_label'].nunique()
        collection_species.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Number of Species by Collection')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Collection bazında total kayıt sayısı
        collection_records = train_df['collection'].value_counts()
        collection_records.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Number of Records by Collection')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'collection_analysis.png'))
        plt.show()  # Grafiği göster
        plt.close()
    else:
        print("Collection column not found")

# 3. Taxonomy Analysis - YENİ EKLENEN
def analyze_taxonomy():
    print("\nAnalyzing taxonomy (class distribution)...")
    if not taxonomy_df.empty:
        # Merge train data with taxonomy for class analysis
        train_with_taxonomy = train_df.merge(taxonomy_df, on='primary_label', how='left')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Class dağılımı (Aves, Amphibia, Insecta, etc.)
        class_counts = train_with_taxonomy['class_name'].value_counts()
        axes[0,0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Class Distribution in Training Data')
        
        # Class bazında tür sayısı
        class_species = train_with_taxonomy.groupby('class_name')['primary_label'].nunique()
        class_species.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Number of Species per Class')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Class bazında total kayıt sayısı
        class_records = train_with_taxonomy['class_name'].value_counts()
        class_records.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Number of Records per Class')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Top 10 tür her class için
        if 'Aves' in class_counts.index:
            birds_only = train_with_taxonomy[train_with_taxonomy['class_name'] == 'Aves']
            top_birds = birds_only['primary_label'].value_counts()[:10]
            top_birds.plot(kind='barh', ax=axes[1,1])
            axes[1,1].set_title('Top 10 Bird Species')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'taxonomy_analysis.png'))
        plt.show()  # Grafiği göster
        plt.close()

# 4. Geographic Analysis - YENİ EKLENEN
def analyze_geographic_distribution():
    print("\nAnalyzing geographic distribution...")
    if all(col in train_df.columns for col in ['longitude', 'latitude']):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Geographic scatter plot
        scatter = axes[0,0].scatter(train_df['longitude'], train_df['latitude'], 
                                   alpha=0.6, s=1)
        axes[0,0].set_xlabel('Longitude')
        axes[0,0].set_ylabel('Latitude')
        axes[0,0].set_title('Recording Locations')
        
        # Latitude distribution
        axes[0,1].hist(train_df['latitude'].dropna(), bins=50, alpha=0.7)
        axes[0,1].set_xlabel('Latitude')
        axes[0,1].set_ylabel('Number of Recordings')
        axes[0,1].set_title('Latitude Distribution')
        
        # Longitude distribution
        axes[1,0].hist(train_df['longitude'].dropna(), bins=50, alpha=0.7)
        axes[1,0].set_xlabel('Longitude')
        axes[1,0].set_ylabel('Number of Recordings')
        axes[1,0].set_title('Longitude Distribution')
        
        # Geographic diversity per coordinate
        if 'collection' in train_df.columns:
            for collection in train_df['collection'].unique():
                if pd.notna(collection):
                    subset = train_df[train_df['collection'] == collection]
                    axes[1,1].scatter(subset['longitude'], subset['latitude'], 
                                    label=collection, alpha=0.6, s=10)
            axes[1,1].legend()
            axes[1,1].set_xlabel('Longitude')
            axes[1,1].set_ylabel('Latitude')
            axes[1,1].set_title('Recording Locations by Collection')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'geographic_analysis.png'))
        plt.show()  # Grafiği göster
        plt.close()

# 5. Secondary Labels Analysis - YENİ EKLENEN
def analyze_secondary_labels():
    print("\nAnalyzing secondary labels...")
    if 'secondary_labels' in train_df.columns:
        # Secondary label varlığı
        has_secondary = train_df['secondary_labels'].notna() & (train_df['secondary_labels'] != "['']")
        secondary_stats = has_secondary.value_counts()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Secondary label varlık grafiği
        axes[0,0].pie(secondary_stats.values, 
                     labels=['No Secondary Labels', 'Has Secondary Labels'], 
                     autopct='%1.1f%%')
        axes[0,0].set_title('Records with Secondary Labels')
        
        # Primary vs secondary label count
        train_df['has_secondary'] = has_secondary
        label_counts = train_df.groupby('primary_label')['has_secondary'].sum().sort_values(ascending=False)
        label_counts[:15].plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Top 15 Species with Most Secondary Labels')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Collection vs secondary labels
        if 'collection' in train_df.columns:
            collection_secondary = train_df.groupby('collection')['has_secondary'].agg(['sum', 'count'])
            collection_secondary['ratio'] = collection_secondary['sum'] / collection_secondary['count']
            collection_secondary['ratio'].plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('Secondary Label Ratio by Collection')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Secondary label length distribution (for records that have them)
        secondary_data = train_df[train_df['has_secondary']]
        if not secondary_data.empty:
            try:
                secondary_lengths = secondary_data['secondary_labels'].apply(lambda x: len(eval(x)) if pd.notna(x) and x != "['']" else 0)
                axes[1,1].hist(secondary_lengths, bins=20, alpha=0.7)
                axes[1,1].set_xlabel('Number of Secondary Labels')
                axes[1,1].set_ylabel('Count')
                axes[1,1].set_title('Distribution of Secondary Label Counts')
            except:
                axes[1,1].text(0.5, 0.5, 'Could not parse secondary labels', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'secondary_labels_analysis.png'))
        plt.show()  # Grafiği göster
        plt.close()

# 6. Quality and Rating Analysis - YENİ EKLENEN
def analyze_quality_ratings():
    print("\nAnalyzing quality ratings...")
    if 'rating' in train_df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Rating distribution
        rating_counts = train_df['rating'].value_counts().sort_index()
        axes[0,0].bar(rating_counts.index, rating_counts.values)
        axes[0,0].set_xlabel('Rating')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_title('Rating Distribution')
        
        # Rating by collection
        if 'collection' in train_df.columns:
            sns.boxplot(data=train_df, x='collection', y='rating', ax=axes[0,1])
            axes[0,1].set_title('Rating Distribution by Collection')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Average rating by species (top 20)
        species_ratings = train_df.groupby('primary_label')['rating'].mean().sort_values(ascending=False)
        species_ratings[:20].plot(kind='bar', ax=axes[0,2])
        axes[0,2].set_title('Top 20 Species by Average Rating')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Rating vs record count correlation
        species_stats = train_df.groupby('primary_label').agg({
            'rating': 'mean',
            'primary_label': 'count'
        }).rename(columns={'primary_label': 'count'})
        
        axes[1,0].scatter(species_stats['count'], species_stats['rating'], alpha=0.6)
        axes[1,0].set_xlabel('Number of Records')
        axes[1,0].set_ylabel('Average Rating')
        axes[1,0].set_title('Species Record Count vs Average Rating')
        
        # Low quality recordings analysis
        low_quality = train_df[train_df['rating'] <= 2]
        if not low_quality.empty:
            low_q_species = low_quality['primary_label'].value_counts()[:15]
            low_q_species.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Top 15 Species with Low Quality Recordings (≤2)')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        # Rating distribution over time (if we can extract from filename)
        try:
            # Extract year from filename if possible
            train_df['year'] = train_df['filename'].str.extract(r'(\d{4})')
            if train_df['year'].notna().any():
                yearly_ratings = train_df.groupby('year')['rating'].mean()
                yearly_ratings.plot(kind='line', ax=axes[1,2], marker='o')
                axes[1,2].set_title('Average Rating by Year')
                axes[1,2].set_xlabel('Year')
                axes[1,2].set_ylabel('Average Rating')
            else:
                axes[1,2].text(0.5, 0.5, 'No year data found in filenames', 
                              ha='center', va='center', transform=axes[1,2].transAxes)
        except:
            axes[1,2].text(0.5, 0.5, 'Could not extract year data', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_ratings_analysis.png'))
        plt.show()  # Grafiği göster
        plt.close()

# 7. Author and License Analysis - YENİ EKLENEN
def analyze_authors_licenses():
    print("\nAnalyzing authors and licenses...")
    if all(col in train_df.columns for col in ['author', 'license']):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top contributing authors
        author_counts = train_df['author'].value_counts()[:15]
        author_counts.plot(kind='barh', ax=axes[0,0])
        axes[0,0].set_title('Top 15 Contributing Authors')
        
        # License distribution
        license_counts = train_df['license'].value_counts()
        axes[0,1].pie(license_counts.values, labels=license_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('License Distribution')
        
        # Author diversity (number of species per author)
        author_species = train_df.groupby('author')['primary_label'].nunique().sort_values(ascending=False)[:15]
        author_species.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Top 15 Authors by Species Diversity')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # License vs quality relationship
        if 'rating' in train_df.columns:
            license_quality = train_df.groupby('license')['rating'].mean().sort_values(ascending=False)
            license_quality.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Average Quality by License Type')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'authors_licenses_analysis.png'))
        plt.show()  # Grafiği göster
        plt.close()

# 8. Audio File Structure Analysis - YENİ EKLENEN
def analyze_audio_structure():
    print("\nAnalyzing audio file structure...")
    if not train_df.empty and 'filename' in train_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # File format distribution
        train_df['file_format'] = train_df['filename'].str.split('.').str[-1]
        format_counts = train_df['file_format'].value_counts()
        axes[0,0].pie(format_counts.values, labels=format_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Audio File Format Distribution')
        
        # Files per species
        files_per_species = train_df['primary_label'].value_counts()
        axes[0,1].hist(files_per_species.values, bins=50, alpha=0.7)
        axes[0,1].set_xlabel('Number of Files per Species')
        axes[0,1].set_ylabel('Number of Species')
        axes[0,1].set_title('Distribution of Files per Species')
        
        # Directory structure analysis
        train_df['dir_depth'] = train_df['filename'].str.count('/')
        depth_counts = train_df['dir_depth'].value_counts().sort_index()
        axes[1,0].bar(depth_counts.index, depth_counts.values)
        axes[1,0].set_xlabel('Directory Depth')
        axes[1,0].set_ylabel('Number of Files')
        axes[1,0].set_title('File Directory Depth Distribution')
        
        # Filename length analysis
        train_df['filename_length'] = train_df['filename'].str.len()
        axes[1,1].hist(train_df['filename_length'], bins=30, alpha=0.7)
        axes[1,1].set_xlabel('Filename Length')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Filename Length Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'audio_structure_analysis.png'))
        plt.show()  # Grafiği göster
        plt.close()

# 3. Audio analysis - if audio files are available
def analyze_audio_samples():
    print("Analyzing audio samples...")
    
    audio_dir = os.path.join(BASE_PATH, 'train_audio')
    if not os.path.exists(audio_dir):
        print(f"Audio directory {audio_dir} not found. Skipping audio analysis.")
        return
    
    # List all audio files
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.ogg') or file.endswith('.wav') or file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    
    # If there are too many files, sample a subset
    if len(audio_files) > 100:
        import random
        audio_files = random.sample(audio_files, 100)
    
    # Extract audio features
    durations = []
    sample_rates = []
    species = []
    
    for audio_file in tqdm(audio_files[:20], desc="Processing audio files"):  # Process a subset
        try:
            y, sr = librosa.load(audio_file, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            durations.append(duration)
            sample_rates.append(sr)
            
            # Extract species from filename
            species_name = os.path.basename(os.path.dirname(audio_file))
            species.append(species_name)
            
            # Generate and save spectrogram for a few samples
            if len(durations) <= 5:  # Only create spectrograms for the first 5 files
                plt.figure(figsize=(10, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrogram: {species_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'spectrogram_{species_name}_{len(durations)}.png'))
                plt.close()
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    # Plot distribution of audio durations
    plt.figure(figsize=(10, 6))
    sns.histplot(durations, bins=30, kde=True)
    plt.title('Distribution of Audio Recording Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'audio_durations.png'))
    plt.show()  # Grafiği göster
    plt.close()
    
    # Plot species vs duration
    plt.figure(figsize=(12, 8))
    species_df = pd.DataFrame({'species': species, 'duration': durations})
    sns.boxplot(x='species', y='duration', data=species_df)
    plt.title('Audio Duration by Species')
    plt.xlabel('Species')
    plt.ylabel('Duration (seconds)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_by_species.png'))
    plt.show()  # Grafiği göster
    plt.close()
    
    return durations, species

def analyze_spectrograms():
    print("\nSpectrogram directory not available - skipping spectrogram analysis...")
    return None, None

def analyze_ratings():
    print("\nAnalyzing recording quality ratings...")
    if 'rating' in train_df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=train_df, x="rating")
        plt.title("Distribution of Recording Quality Ratings (1-5)")
        plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
        plt.show()  # Grafiği göster
        plt.close()
    else:
        print("Rating column not found in training data")

def analyze_locations():
    print("\nAnalyzing recording locations...")
    if all(col in train_df.columns for col in ['longitude', 'latitude', 'collection']):
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=train_df, x="longitude", y="latitude", 
                        alpha=0.5, hue="collection", palette="viridis")
        plt.title("Recording Locations (XC vs. iNat vs. CSA)")
        plt.savefig(os.path.join(output_dir, 'recording_locations.png'))
        plt.show()  # Grafiği göster
        plt.close()
    else:
        print("Location data columns missing")

def analyze_soundscapes():
    print("\nAnalyzing sample soundscapes...")
    soundscape_path = os.path.join(BASE_PATH, 'train_soundscapes/H02_20230420_074000.ogg')
    
    if os.path.exists(soundscape_path):
        try:
            soundscape, sr = librosa.load(soundscape_path, sr=32000)
            
            plt.figure(figsize=(12, 8))
            
            # Waveform
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(soundscape, sr=sr)
            plt.title("Soundscape Waveform (Background Noise)")
            
            # Spectrogram
            plt.subplot(2, 1, 2)
            S_soundscape = librosa.feature.melspectrogram(y=soundscape, sr=sr)
            librosa.display.specshow(librosa.power_to_db(S_soundscape), 
                                   x_axis="time", y_axis="mel")
            plt.colorbar(format='%+2.0f dB')
            plt.title("Soundscape Spectrogram")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'soundscape_analysis.png'))
            plt.show()  # Grafiği göster
            plt.close()
            
        except Exception as e:
            print(f"Soundscape analysis failed: {str(e)}")
    else:
        print("Soundscape file not found")

# Main execution
if __name__ == "__main__":
    print("Starting BirdClef data visualization and analysis...")
    
    # 1. Tür dağılımı
    species_counts = analyze_species_distribution()
    
    # 2. YENİ ANALİZLER
    analyze_collections()
    analyze_taxonomy()
    analyze_geographic_distribution() 
    analyze_secondary_labels()
    analyze_quality_ratings()
    analyze_authors_licenses()
    analyze_audio_structure()
    
    # 3. Ses analizi
    if not train_df.empty:
        durations, species = analyze_audio_samples()
    else:
        print("Skipping audio analysis due to missing data")
    
    # 4. Eski analizler
    analyze_ratings()
    analyze_locations()
    analyze_soundscapes()
    
    # 5. Spectrogram analizi (şu an mevcut değil)
    # spec_count, spec_samples = analyze_spectrograms()
    # if spec_count:
    #     print(f"Analyzed {spec_count} spectrogram samples")
    
    print(f"\nAll visualizations saved to: {output_dir}") 
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"✓ Species distribution analysis")
    print(f"✓ Collection analysis (data sources)")
    print(f"✓ Taxonomy analysis (Aves, Amphibia, Insecta, etc.)")
    print(f"✓ Geographic distribution")
    print(f"✓ Secondary labels analysis")
    print(f"✓ Quality ratings analysis")
    print(f"✓ Authors and licenses analysis")
    print(f"✓ Audio file structure analysis")
    print(f"✓ Audio samples analysis")
    # print(f"✓ Spectrograms analysis")  # Şu an mevcut değil
    print("✓ Soundscapes analysis")
    print("✓ All visualizations saved to output directory")
    print("✓ All visualizations saved to output directory")
    print("✓ All visualizations saved to output directory")
    print("✓ All visualizations saved to output directory")
    print("✓ All visualizations saved to output directory")
    print("✓ All visualizations saved to output directory")





