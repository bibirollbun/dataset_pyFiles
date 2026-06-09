import pandas as pd
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import warnings
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
from scipy import stats
from scipy.signal import periodogram
import gc  # For garbage collection
import json
import time

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# For interactive plots in notebook
%matplotlib inline


class Config:
    # Debug options
    DEBUG_MODE = False  # Set to False for full dataset processing
    VISUALIZE = True   # Whether to create visualization plots
    SAMPLE_SIZE = 100  # Number of files to analyze in debug mode
    
    # Paths
    BASE_DIR = '/kaggle/input/birdclef-2025/'  # Base directory for dataset
    TRAIN_AUDIO_DIR = os.path.join(BASE_DIR, 'train_audio')
    TRAIN_SOUNDSCAPES_DIR = os.path.join(BASE_DIR, 'train_soundscapes')
    TEST_SOUNDSCAPES_DIR = os.path.join(BASE_DIR, 'test_soundscapes')
    OUTPUT_DIR = "/kaggle/working/"
    
    # Audio parameters
    SR = 32000  # Sampling rate
    
    def __init__(self):
        # Adjust paths for local environment
        if not os.path.exists(self.BASE_DIR):
            print("Adjusting paths for local environment...")
            self.BASE_DIR = "./data"
            self.TRAIN_AUDIO_DIR = "./data/train_audio"
            self.TRAIN_SOUNDSCAPES_DIR = "./data/train_soundscapes"
            self.TEST_SOUNDSCAPES_DIR = "./data/test_soundscapes" 
            self.OUTPUT_DIR = "./output"
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)

# Initialize configuration
cfg = Config()

# Print paths
print(f"Base directory: {cfg.BASE_DIR}")
print(f"Train audio directory: {cfg.TRAIN_AUDIO_DIR}")
print(f"Output directory: {cfg.OUTPUT_DIR}")


# Function to load audio file and return the waveform
def load_audio_file(file_path, sr=Config.SR):
    """
    Load audio file and return the waveform
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to assess audio quality
def assess_audio_quality(y):
    """
    Assess the quality of audio based on signal statistics
    Returns a quality score between 0 and 1
    """
    if y is None or len(y) == 0:
        return 0
    
    # Calculate signal statistics
    signal_std = np.std(y)
    signal_var = np.var(y)
    signal_rms = np.sqrt(np.mean(np.square(y)))
    signal_pwr = np.mean(np.square(y))
    
    # T = std + var + rms + pwr
    t_stat = signal_std + signal_var + signal_rms + signal_pwr
    
    # Normalize to 0-1 range (approximately)
    quality_score = min(1.0, t_stat * 100)
    
    return quality_score

# Function to extract frequency characteristics
def extract_frequency_characteristics(y, sr=Config.SR):
    """
    Extract frequency domain characteristics from audio
    """
    if y is None or len(y) == 0:
        return None
    
    # Calculate power spectral density
    freqs, psd = periodogram(y, fs=sr)
    
    # Find dominant frequency
    dominant_freq = freqs[np.argmax(psd)]
    
    # Calculate frequency statistics (only for audible range: 20Hz-20kHz)
    idx_range = np.where((freqs >= 20) & (freqs <= 20000))[0]
    
    if len(idx_range) > 0:
        audible_freqs = freqs[idx_range]
        audible_psd = psd[idx_range]
        
        # Calculate spectral centroid (weighted average of frequencies)
        if np.sum(audible_psd) > 0:
            spectral_centroid = np.sum(audible_freqs * audible_psd) / np.sum(audible_psd)
        else:
            spectral_centroid = 0
            
        # Calculate spectral bandwidth (spread of frequencies)
        if np.sum(audible_psd) > 0:
            spectral_bandwidth = np.sqrt(np.sum(((audible_freqs - spectral_centroid) ** 2) * audible_psd) / np.sum(audible_psd))
        else:
            spectral_bandwidth = 0
            
        # Calculate spectral flatness (geometric mean / arithmetic mean)
        # This tells us how noise-like (1) vs. tonal (0) the sound is
        audible_psd_nonzero = audible_psd[audible_psd > 0]
        if len(audible_psd_nonzero) > 0:
            spectral_flatness = stats.mstats.gmean(audible_psd_nonzero) / np.mean(audible_psd_nonzero)
        else:
            spectral_flatness = 0
    else:
        spectral_centroid = 0
        spectral_bandwidth = 0
        spectral_flatness = 0
    
    return {
        'dominant_frequency': dominant_freq,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_flatness': spectral_flatness
    }

# Function to extract call type from filename
def extract_call_type_from_filename(filename):
    """
    Extract call type from filename if available
    Common naming pattern: XC######-{call_type}.ogg
    """
    call_types = [
        'song', 'call', 'alarm', 'flight', 'drum', 'display', 'duet', 
        'juvenile', 'begging', 'contact', 'excited', 'warn', 'wing',
        'dawn', 'growl', 'chatter', 'trill', 'whistle'
    ]
    
    filename_lower = filename.lower()
    
    for call_type in call_types:
        if call_type in filename_lower:
            return call_type
    
    # If no specific call type found
    if 'song' in filename_lower:
        return 'song'
    elif 'call' in filename_lower:
        return 'call'
    else:
        return 'unknown'


# Try to load metadata
def load_metadata(cfg):
    try:
        metadata_path = os.path.join(cfg.BASE_DIR, 'train.csv')
        if os.path.exists(metadata_path):
            metadata_df = pd.read_csv(metadata_path)
            print(f"Loaded metadata with {len(metadata_df)} entries")
            return metadata_df
        else:
            print("Metadata file not found, proceeding with file system analysis")
            return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

metadata_df = load_metadata(cfg)

# Show a sample of the metadata if available
if metadata_df is not None:
    print("\nMetadata columns:", metadata_df.columns.tolist())
    print("\nSample of metadata:")
    display(metadata_df.head())
    
    # Count unique species
    unique_species = metadata_df['primary_label'].nunique()
    print(f"\nNumber of unique species: {unique_species}")


# Try to load taxonomy data
def load_taxonomy(cfg):
    try:
        taxonomy_path = os.path.join(cfg.BASE_DIR, 'taxonomy.csv')
        if os.path.exists(taxonomy_path):
            taxonomy_df = pd.read_csv(taxonomy_path)
            print(f"Loaded taxonomy data with {len(taxonomy_df)} entries")
            return taxonomy_df
        else:
            print("Taxonomy file not found")
            return None
    except Exception as e:
        print(f"Error loading taxonomy: {e}")
        return None

taxonomy_df = load_taxonomy(cfg)

# Show a sample of the taxonomy if available
if taxonomy_df is not None:
    print("\nTaxonomy columns:", taxonomy_df.columns.tolist())
    print("\nSample of taxonomy:")
    display(taxonomy_df.head())
    
    # Count species by class
    if 'class' in taxonomy_df.columns:
        class_counts = taxonomy_df['class'].value_counts()
        print("\nSpecies count by class:")
        display(class_counts)


def get_species_from_directory(cfg):
    """Get list of species from directory structure"""
    if os.path.exists(cfg.TRAIN_AUDIO_DIR):
        species_list = [d for d in os.listdir(cfg.TRAIN_AUDIO_DIR) 
                      if os.path.isdir(os.path.join(cfg.TRAIN_AUDIO_DIR, d))]
        print(f"Found {len(species_list)} species directories")
        return species_list
    else:
        print(f"Training audio directory not found: {cfg.TRAIN_AUDIO_DIR}")
        return []

# Get species from metadata or directory structure
if metadata_df is not None:
    species_list = metadata_df['primary_label'].unique().tolist()
    print(f"Found {len(species_list)} species in metadata")
else:
    species_list = get_species_from_directory(cfg)

# Create species directory mapping
species_dir_mapping = {}
if os.path.exists(cfg.TRAIN_AUDIO_DIR):
    for species in species_list:
        species_dir = os.path.join(cfg.TRAIN_AUDIO_DIR, species)
        if os.path.isdir(species_dir):
            species_dir_mapping[species] = species_dir
    
    print(f"Mapped {len(species_dir_mapping)} species to directories")

# Calculate samples per species
samples_per_species = {}
for species, species_dir in species_dir_mapping.items():
    audio_files = [f for f in os.listdir(species_dir) 
                 if f.endswith(('.mp3', '.ogg', '.wav'))]
    samples_per_species[species] = len(audio_files)

# Convert to DataFrame for easier analysis
samples_df = pd.DataFrame({
    'species': list(samples_per_species.keys()),
    'sample_count': list(samples_per_species.values())
}).sort_values('sample_count', ascending=False)

# Print statistics
print("\nSamples per species statistics:")
print(f"- Min samples: {samples_df['sample_count'].min()}")
print(f"- Max samples: {samples_df['sample_count'].max()}")
print(f"- Average samples: {samples_df['sample_count'].mean():.1f}")
print(f"- Median samples: {samples_df['sample_count'].median():.1f}")
print(f"- Total samples: {samples_df['sample_count'].sum()}")

# Show species with fewest samples
print("\nSpecies with fewest samples:")
display(samples_df.nsmallest(10, 'sample_count'))


# Plot distribution of samples per species
plt.figure(figsize=(12, 6))
plt.hist(samples_df['sample_count'], bins=50, edgecolor='black')
plt.title('Distribution of Samples per Species')
plt.xlabel('Number of Audio Samples')
plt.ylabel('Number of Species')
plt.grid(True, alpha=0.3)
plt.show()

# Plot top 30 species with most samples
plt.figure(figsize=(14, 10))
top_species = samples_df.head(30)
sns.barplot(x='sample_count', y='species', data=top_species)
plt.title('Top 30 Species by Number of Samples')
plt.xlabel('Number of Audio Files')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


def analyze_audio_length(species_list, species_dir_mapping, cfg):
    """Analyze the length of audio files"""
    results = {
        'audio_lengths': [],
        'species_lengths': {}
    }
    
    # Select a subset of species for analysis if in debug mode
    if cfg.DEBUG_MODE:
        np.random.seed(42)
        if len(species_list) > 20:  # Analyze up to 20 species in debug mode
            species_subset = np.random.choice(species_list, 20, replace=False).tolist()
        else:
            species_subset = species_list
    else:
        species_subset = species_list
    
    print(f"Analyzing audio lengths for {len(species_subset)} species")
    
    for species_idx, species in enumerate(species_subset):
        # Get species directory
        species_dir = species_dir_mapping.get(species)
        if not species_dir or not os.path.exists(species_dir):
            continue
        
        # Get audio files for this species
        audio_files = [f for f in os.listdir(species_dir) 
                     if f.endswith(('.mp3', '.ogg', '.wav'))]
        
        # If in debug mode, only process a sample
        if cfg.DEBUG_MODE:
            if len(audio_files) > cfg.SAMPLE_SIZE:
                audio_files = np.random.choice(audio_files, cfg.SAMPLE_SIZE, replace=False).tolist()
        
        species_lengths = []
        
        # Process audio files
        for file_idx, filename in enumerate(tqdm(audio_files, desc=f"Species {species_idx+1}/{len(species_subset)}")):
            file_path = os.path.join(species_dir, filename)
            
            # Load and analyze audio
            audio = load_audio_file(file_path, sr=cfg.SR)
            
            if audio is not None:
                # Calculate audio length in seconds
                audio_length = len(audio) / cfg.SR
                results['audio_lengths'].append(audio_length)
                species_lengths.append(audio_length)
        
        # Store species-level results
        if species_lengths:
            results['species_lengths'][species] = species_lengths
            
    return results

# Run the audio length analysis
audio_length_results = analyze_audio_length(species_list, species_dir_mapping, cfg)

if audio_length_results['audio_lengths']:
    # Calculate statistics
    audio_lengths = np.array(audio_length_results['audio_lengths'])
    
    print("\nAudio Length Statistics:")
    print(f"- Number of files analyzed: {len(audio_lengths)}")
    print(f"- Average length: {np.mean(audio_lengths):.2f} seconds")
    print(f"- Min length: {np.min(audio_lengths):.2f} seconds")
    print(f"- Max length: {np.max(audio_lengths):.2f} seconds")
    print(f"- Median length: {np.median(audio_lengths):.2f} seconds")
    
    # Plot histogram of audio lengths
    plt.figure(figsize=(12, 6))
    plt.hist(audio_lengths, bins=50, edgecolor='black')
    plt.axvline(np.median(audio_lengths), color='red', linestyle='dashed', 
                linewidth=1, label=f'Median: {np.median(audio_lengths):.2f}s')
    plt.axvline(np.mean(audio_lengths), color='green', linestyle='dashed', 
                linewidth=1, label=f'Mean: {np.mean(audio_lengths):.2f}s')
    plt.title('Distribution of Audio File Lengths')
    plt.xlabel('Length (seconds)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
else:
    print("No audio files were analyzed for length")


# Calculate average length per species
if audio_length_results['species_lengths']:
    species_avg_lengths = {
        species: np.mean(lengths)
        for species, lengths in audio_length_results['species_lengths'].items()
        if lengths
    }
    
    # Convert to DataFrame
    species_length_df = pd.DataFrame({
        'species': list(species_avg_lengths.keys()),
        'avg_length': list(species_avg_lengths.values())
    }).sort_values('avg_length', ascending=False)
    
    # Display species with longest and shortest average audio lengths
    print("Species with longest average audio:")
    display(species_length_df.head(10))
    
    print("Species with shortest average audio:")
    display(species_length_df.tail(10))
    
    # Plot average audio length by species
    plt.figure(figsize=(14, 10))
    sns.barplot(x='avg_length', y='species', data=species_length_df.head(20))
    plt.title('Average Audio Length by Species (Top 20)')
    plt.xlabel('Average Length (seconds)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_call_types(species_list, species_dir_mapping, cfg):
    """Analyze call types from filenames"""
    results = {
        'call_types': {},
        'overall_counts': Counter()
    }
    
    # Select a subset of species for analysis if in debug mode
    if cfg.DEBUG_MODE:
        if len(species_list) > 30:  # Analyze up to 30 species in debug mode
            species_subset = np.random.choice(species_list, 30, replace=False).tolist()
        else:
            species_subset = species_list
    else:
        species_subset = species_list
    
    print(f"Analyzing call types for {len(species_subset)} species")
    
    for species_idx, species in enumerate(species_subset):
        # Get species directory
        species_dir = species_dir_mapping.get(species)
        if not species_dir or not os.path.exists(species_dir):
            continue
        
        # Get audio files for this species
        audio_files = [f for f in os.listdir(species_dir) 
                     if f.endswith(('.mp3', '.ogg', '.wav'))]
        
        species_call_types = Counter()
        
        # Process audio files
        for filename in audio_files:
            # Extract call type from filename
            call_type = extract_call_type_from_filename(filename)
            species_call_types[call_type] += 1
            results['overall_counts'][call_type] += 1
        
        # Store species-level results
        if species_call_types:
            results['call_types'][species] = dict(species_call_types)
    
    return results

# Run the call type analysis
call_type_results = analyze_call_types(species_list, species_dir_mapping, cfg)

if call_type_results['overall_counts']:
    # Create DataFrame for overall counts
    call_counts = pd.DataFrame({
        'call_type': list(call_type_results['overall_counts'].keys()),
        'count': list(call_type_results['overall_counts'].values())
    }).sort_values('count', ascending=False)
    
    print("\nCall Type Distribution:")
    display(call_counts)
    
    # Plot call type distribution
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='call_type', data=call_counts)
    plt.title('Distribution of Bird Call Types')
    plt.xlabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Create pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(call_counts['count'], labels=call_counts['call_type'], 
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Bird Call Types')
    plt.tight_layout()
    plt.show()
else:
    print("No call types were analyzed")


# Prepare data for sunburst chart
if call_type_results['call_types']:
    # Create a list of dictionaries for the sunburst
    sunburst_data = []
    
    for species, call_types in call_type_results['call_types'].items():
        for call_type, count in call_types.items():
            sunburst_data.append({
                'species': species,
                'call_type': call_type,
                'count': count
            })
    
    # Convert to DataFrame
    sunburst_df = pd.DataFrame(sunburst_data)
    
    # Create sunburst chart
    fig = px.sunburst(
        sunburst_df,
        path=['call_type', 'species'],
        values='count',
        title='Sunburst Chart of Call Types by Species',
        width=800,
        height=800
    )
    
    fig.update_layout(margin=dict(t=30, b=30, l=0, r=0))
    fig.show()
else:
    print("No call type data available for sunburst visualization")


def analyze_quality_ratings(metadata_df):
    """Analyze quality ratings from metadata"""
    if metadata_df is None or 'rating' not in metadata_df.columns:
        print("No quality ratings available in metadata")
        return None
    
    # Calculate statistics
    quality_stats = metadata_df['rating'].describe()
    print("\nQuality Rating Statistics:")
    display(quality_stats)
    
    # Count of each rating
    rating_counts = metadata_df['rating'].value_counts().sort_index()
    print("\nCount of each rating value:")
    display(rating_counts)
    
    # Plot distribution of ratings
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=metadata_df, palette='viridis')
    plt.title('Distribution of Audio Quality Ratings')
    plt.xlabel('Rating (0=unknown, 1=low, 5=high)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Average rating by species
    species_ratings = metadata_df.groupby('primary_label')['rating'].agg(['mean', 'count']).reset_index()
    species_ratings = species_ratings.sort_values('mean', ascending=False)
    
    print("\nSpecies with highest average quality rating:")
    display(species_ratings.head(10))
    
    print("\nSpecies with lowest average quality rating:")
    display(species_ratings.tail(10))
    
    return species_ratings

# Run quality rating analysis if metadata is available
species_ratings = None
if metadata_df is not None:
    species_ratings = analyze_quality_ratings(metadata_df)

# Function to compute our own quality assessment on a sample of files
def analyze_computed_quality(species_list, species_dir_mapping, cfg):
    """Compute our own quality assessment on audio files"""
    results = {
        'quality_scores': [],
        'species_scores': {}
    }
    
    # Select a subset of species for analysis if in debug mode
    if cfg.DEBUG_MODE:
        if len(species_list) > 15:  # Analyze up to 15 species in debug mode
            species_subset = np.random.choice(species_list, 15, replace=False).tolist()
        else:
            species_subset = species_list
    else:
        species_subset = species_list
    
    print(f"Analyzing computed quality for {len(species_subset)} species")
    
    for species_idx, species in enumerate(species_subset):
        # Get species directory
        species_dir = species_dir_mapping.get(species)
        if not species_dir or not os.path.exists(species_dir):
            continue
        
        # Get audio files for this species
        audio_files = [f for f in os.listdir(species_dir) 
                     if f.endswith(('.mp3', '.ogg', '.wav'))]
        
        # If in debug mode, only process a sample
        if cfg.DEBUG_MODE:
            if len(audio_files) > cfg.SAMPLE_SIZE:
                audio_files = np.random.choice(audio_files, cfg.SAMPLE_SIZE, replace=False).tolist()
        
        species_scores = []
        
        # Process audio files
        for filename in tqdm(audio_files, desc=f"Computing quality for {species}"):
            file_path = os.path.join(species_dir, filename)
            
            # Load and assess audio quality
            audio = load_audio_file(file_path, sr=cfg.SR)
            
            if audio is not None:
                # Compute quality score
                quality = assess_audio_quality(audio)
                results['quality_scores'].append((species, filename, quality))
                species_scores.append(quality)
        
        # Store species-level results
        if species_scores:
            results['species_scores'][species] = species_scores
    
    return results

# Run computed quality analysis
computed_quality_results = analyze_computed_quality(species_list, species_dir_mapping, cfg)

if computed_quality_results['quality_scores']:
    # Extract quality scores
    quality_scores = [score for _, _, score in computed_quality_results['quality_scores']]
    
    print("\nComputed Quality Score Statistics:")
    print(f"- Average quality: {np.mean(quality_scores):.4f}")
    print(f"- Min quality: {np.min(quality_scores):.4f}")
    print(f"- Max quality: {np.max(quality_scores):.4f}")
    
    # Plot histogram of quality scores
    plt.figure(figsize=(12, 6))
    plt.hist(quality_scores, bins=50, edgecolor='black')
    plt.title('Distribution of Computed Audio Quality Scores')
    plt.xlabel('Quality Score (0-1)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate average quality score per species
    species_quality = {}
    for species, scores in computed_quality_results['species_scores'].items():
        if scores:
            species_quality[species] = np.mean(scores)
    
    # Convert to DataFrame
    species_quality_df = pd.DataFrame({
        'species': list(species_quality.keys()),
        'avg_quality': list(species_quality.values())
    }).sort_values('avg_quality', ascending=False)
    
    # Plot average quality by species
    plt.figure(figsize=(14, 10))
    sns.barplot(x='avg_quality', y='species', data=species_quality_df)
    plt.title('Average Computed Quality Score by Species')
    plt.xlabel('Average Quality Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_frequency_patterns(species_list, species_dir_mapping, cfg):
    """Analyze frequency patterns in audio files"""
    results = {
        'frequency_data': {}
    }
    
    # Select a subset of species for analysis if in debug mode
    if cfg.DEBUG_MODE:
        if len(species_list) > 10:  # Analyze up to 10 species in debug mode
            species_subset = np.random.choice(species_list, 10, replace=False).tolist()
        else:
            species_subset = species_list
    else:
        species_subset = species_list
    
    print(f"Analyzing frequency patterns for {len(species_subset)} species")
    
    for species_idx, species in enumerate(species_subset):
        # Get species directory
        species_dir = species_dir_mapping.get(species)
        if not species_dir or not os.path.exists(species_dir):
            continue
        
        # Get audio files for this species
        audio_files = [f for f in os.listdir(species_dir) 
                     if f.endswith(('.mp3', '.ogg', '.wav'))]
        
        # If in debug mode, only process a sample
        if cfg.DEBUG_MODE:
            if len(audio_files) > cfg.SAMPLE_SIZE:
                audio_files = np.random.choice(audio_files, cfg.SAMPLE_SIZE // 2, replace=False).tolist()
        
        species_freq_data = []
        
        # Process audio files
        for filename in tqdm(audio_files, desc=f"Analyzing frequencies for {species}"):
            file_path = os.path.join(species_dir, filename)
            
            # Load and analyze audio
            audio = load_audio_file(file_path, sr=cfg.SR)
            
            if audio is not None:
                # Extract frequency characteristics
                freq_data = extract_frequency_characteristics(audio, sr=cfg.SR)
                if freq_data:
                    freq_data['species'] = species
                    freq_data['filename'] = filename
                    species_freq_data.append(freq_data)
        
        # Store species-level results
        if species_freq_data:
            results['frequency_data'][species] = species_freq_data
    
    return results

# Run frequency pattern analysis
freq_pattern_results = analyze_frequency_patterns(species_list, species_dir_mapping, cfg)

if freq_pattern_results['frequency_data']:
    # Extract frequency characteristics for each species
    species_freq_data = {}
    for species, freq_list in freq_pattern_results['frequency_data'].items():
        if freq_list:
            species_freq_data[species] = {
                'dominant_frequency': [item['dominant_frequency'] for item in freq_list],
                'spectral_centroid': [item['spectral_centroid'] for item in freq_list],
                'spectral_bandwidth': [item['spectral_bandwidth'] for item in freq_list],
                'spectral_flatness': [item['spectral_flatness'] for item in freq_list]
            }
    
    # Calculate median values for each characteristic by species
    median_values = {}
    for species, freq_data in species_freq_data.items():
        median_values[species] = {
            'dominant_frequency': np.median(freq_data['dominant_frequency']),
            'spectral_centroid': np.median(freq_data['spectral_centroid']),
            'spectral_bandwidth': np.median(freq_data['spectral_bandwidth']),
            'spectral_flatness': np.median(freq_data['spectral_flatness'])
        }
    
    # Convert to DataFrame
    freq_df = pd.DataFrame.from_dict(median_values, orient='index')
    
    print("\nFrequency Characteristics by Species:")
    display(freq_df.head())
    
    # Plot frequency characteristics
    for characteristic in ['dominant_frequency', 'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness']:
        plt.figure(figsize=(14, 8))
        sorted_df = freq_df.sort_values(characteristic, ascending=False)
        plt.barh(range(len(sorted_df)), sorted_df[characteristic])
        plt.yticks(range(len(sorted_df)), sorted_df.index)
        plt.title(f'Median {characteristic.replace("_", " ").title()} by Species')
        plt.xlabel('Value (Hz for frequencies)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def analyze_time_series(species_list, species_dir_mapping, cfg):
    """Analyze time series patterns in bird calls"""
    # Select one species for detailed analysis
    if species_list and species_dir_mapping:
        # Try to find a species with known song patterns
        target_species = None
        for species in ['norspe1', 'comyel', 'rebnut', 'amerob', 'houspa']:
            if species in species_dir_mapping:
                target_species = species
                break
        
        if target_species is None and species_list:
            target_species = species_list[0]
        
        species_dir = species_dir_mapping.get(target_species)
        if not species_dir or not os.path.exists(species_dir):
            print(f"Directory not found for species {target_species}")
            return
        
        print(f"Performing time series analysis for {target_species}")
        
        # Get audio files
        audio_files = [f for f in os.listdir(species_dir) 
                     if f.endswith(('.mp3', '.ogg', '.wav'))]
        
        if not audio_files:
            print("No audio files found")
            return
        
        # Select one file for analysis
        filename = audio_files[0]
        file_path = os.path.join(species_dir, filename)
        
        # Load audio
        audio = load_audio_file(file_path, sr=cfg.SR)
        
        if audio is None:
            print(f"Failed to load audio file: {file_path}")
            return
        
        # Plot waveform
        plt.figure(figsize=(14, 5))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(audio)) / cfg.SR, audio)
        plt.title(f'Waveform for {target_species} - {filename}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Power spectral density
        plt.subplot(2, 1, 2)
        freqs, psd = periodogram(audio, fs=cfg.SR)
        plt.semilogy(freqs, psd)
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Extract envelope as a time series
        def extract_envelope(signal, frame_size=1024, hop_length=512):
            energy = []
            for i in range(0, len(signal) - frame_size, hop_length):
                chunk = signal[i:i + frame_size]
                energy.append(np.sum(chunk ** 2) / frame_size)
            return np.array(energy)
        
        # Get envelope
        envelope = extract_envelope(audio)
        
        # Plot envelope
        plt.figure(figsize=(14, 5))
        plt.plot(np.arange(len(envelope)) * (512 / cfg.SR), envelope)
        plt.title(f'Energy Envelope for {target_species} - {filename}')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Try fitting an ARIMA model to the envelope
        try:
            # Use a small sample to make computation faster
            sample_size = min(len(envelope), 1000)
            sample_envelope = envelope[:sample_size]
            
            # Fit ARIMA model
            model = ARIMA(sample_envelope, order=(5, 1, 0))
            model_fit = model.fit()
            
            # Print summary
            print("\nARIMA Model Summary:")
            print(model_fit.summary())
            
            # Plot forecast vs actual
            plt.figure(figsize=(14, 5))
            plt.plot(sample_envelope, label='Actual')
            plt.plot(model_fit.fittedvalues, color='red', label='Fitted')
            plt.title(f'ARIMA Model Fit for {target_species}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # Forecast next 100 points
            forecast = model_fit.forecast(steps=100)
            
            # Plot forecast
            plt.figure(figsize=(14, 5))
            plt.plot(range(len(sample_envelope)), sample_envelope, label='Actual')
            plt.plot(range(len(sample_envelope), len(sample_envelope) + 100), forecast, color='red', label='Forecast')
            plt.title(f'ARIMA Forecast for {target_species}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")

# Run time series analysis
analyze_time_series(species_list, species_dir_mapping, cfg)







