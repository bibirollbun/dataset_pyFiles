# Basic utilities
import os
import math
import time
import random
import gc
import logging
import warnings
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
# Data handling

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Audio processing
import librosa

# Visualization
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm  # for progress bars



class Config:
    # Define paths for the dataset
    OUTPUT_DIR = '/kaggle/working/'
    DATA_ROOT = '/kaggle/input/birdclef-2025'  # Path to the dataset

    # Audio settings
    FS = 32000  # Sampling rate (audio)

    # Mel spectrogram parameters (for converting audio to image)
    N_FFT = 1024       # FFT window size
    HOP_LENGTH = 512   # Step size for each frame
    N_MELS = 128       # Number of mel bands
    FMIN = 50          # Minimum Mel frequency
    FMAX = 14000       # Maximum Mel frequency

    # Parameters for audio duration and spectrogram size
    TARGET_DURATION = 5.0  # Length of each audio (in seconds)
    TARGET_SHAPE = (256, 256)  # Size of the spectrogram image

    # No limit on the number of samples during training (full dataset)
    N_MAX = None  

    # flag for training mode
    TRAINING_MODE = True  
    
    # Additional training-specific configurations
    EPOCHS = 10  
    BATCH_SIZE = 32  
    LEARNING_RATE = 0.001  

# Create the config object
config = Config()




# Load taxonomy data (bird species details)
taxonomy_df = pd.read_csv(f'{config.DATA_ROOT}/taxonomy.csv')
print("taxonomy data loaded")

# Create mapping from bird ID to class name
species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))

# Load training metadata
train_df = pd.read_csv(f'{config.DATA_ROOT}/train.csv')
print("training metadata loaded ")


print("="*40)
print(f"ğŸ“¦ Train Data Shape: {train_df.shape}")
print(f"ğŸ“š Taxonomy Data Shape: {taxonomy_df.shape}")
print("="*40)

### ğŸ”¹ 1. Columns in Each File
print("\nğŸ”� Columns in Train Data:", train_df.columns.tolist())
print("ğŸ”� Columns in Taxonomy Data:", taxonomy_df.columns.tolist())

### ğŸ”¹ 2. Data Types
print("\nğŸ“Š Train Data Types:")
print(train_df.info())

print("\nğŸ“Š Taxonomy Data Types:")
print(taxonomy_df.info())

### ğŸ”¹ 3. Basic Descriptive Statistics
print("\nğŸ“ˆ Basic Stats - Train Data")
display(train_df.describe(include='all').T)  # works well in Jupyter

print("\nğŸ“ˆ Basic Stats - Taxonomy Data")
display(taxonomy_df.describe(include='all').T)

### ğŸ”¹ 4. Missing Values Check
print("\nâ�Œ Missing Values in Train Data:")
print(train_df.isnull().sum())

print("\nâ�Œ Missing Values in Taxonomy Data:")
print(taxonomy_df.isnull().sum())

### ğŸ”¹ 5. Random Sample Rows for Quick Glance
print("\nğŸ”¹ Sample Rows from Train Data:")
display(train_df.sample(5))

print("\nğŸ”¹ Sample Rows from Taxonomy Data:")
display(taxonomy_df.sample(5))



# Unique bird species ke labels ko list me conversion
label_list = sorted(train_df['primary_label'].unique())  # sorting unique labels
label_id_list = list(range(len(label_list)))  # Har label ka ek ID number banaya

# Dictionary banayi: label se id aur id se label
label2id = dict(zip(label_list, label_id_list))
id2label = dict(zip(label_id_list, label_list))

print(f'Found {len(label_list)} unique species')  # Total species print ki

# Training data ka kaam karne ke liye naya dataframe banaya
working_df = train_df[['primary_label', 'rating', 'filename']].copy()

# Har label ko uski ID 
working_df['target'] = working_df.primary_label.map(label2id)

# File ka full path
working_df['filepath'] = config.DATA_ROOT + '/train_audio/' + working_df.filename

# Sample name banaya: foldername-filename (extension ke bina)
working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

# Har primary label se uski class name 
working_df['class'] = working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))

# Sirf itne samples process karne hain jitne DEBUG mode ke liye allowed hain
total_samples = min(len(working_df), config.N_MAX or len(working_df))

print(f'Total samples to process: {total_samples} out of {len(working_df)} available')




print("="*50)
print(f"âœ… Total samples to process: {total_samples} out of {len(working_df)} available")
print("="*50)

# ğŸ”¢ Class-wise sample count
print("\nğŸ“Š Samples per Bird Class:")
print(working_df['class'].value_counts().to_string())

# ğŸ§¾ Basic Info of DataFrame
print("\nâ„¹ï¸� DataFrame Info:")
working_df.info()

# ğŸ“ˆ Basic Statistics (numerical columns like rating, target)
print("\nğŸ“Š Descriptive Statistics:")
print(working_df.describe().T)

# ğŸ”� Missing Values Check
print("\nâ�Œ Missing Values Summary:")
print(working_df.isnull().sum())



# Class-wise samples count ko plot karte hain
plt.figure(figsize=(10, 6))
sns.countplot(data=working_df, x='class', order=working_df['class'].value_counts().index)

# Plot ki customization (optional)
plt.title('Sample Distribution by Class')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Sample Distribution by Class.png')
# Plot 
plt.show()



# Count total unique classes
print("Total classes:", working_df['primary_label'].nunique())

# Count samples per class
print("Samples per class:")
print(working_df['primary_label'].value_counts().describe())

# Visualize class imbalance
plt.figure(figsize=(10, 6))
working_df['primary_label'].value_counts().plot(kind='hist', bins=50, edgecolor='black')
plt.title('Distribution of Sample Counts per Bird Species')
plt.xlabel('Number of Samples per specie')
plt.ylabel('Number of Classes')
plt.tight_layout()

# Save the plot
plt.savefig('class_distribution_histogram.png')
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=working_df)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.grid(False)
plt.savefig('Distribution of Ratings.png')
plt.show()



print(f"Average rating: {working_df['rating'].mean():.2f}")
print(f"Median rating: {working_df['rating'].median()}")


plt.figure(figsize=(14, 6))
sns.boxplot(x='class', y='rating', data=working_df)
plt.title('Rating Distribution by Bird Class')
plt.xlabel('Bird Class')
plt.ylabel('Rating')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('Rating Distribution by Bird Class.png')
plt.show()


# Count primary labels in working_df
primary_label_counts = working_df['primary_label'].value_counts()

# Plot distribution of all primary labels (horizontal bar plot)
plt.figure(figsize=(15, 30))  # Adjusting the figure size to fit all labels
ax = sns.barplot(y=primary_label_counts.index, x=primary_label_counts.values)
plt.title('Distribution of All Primary Labels (Horizontal)', fontsize=16)
plt.ylabel('Primary Label', fontsize=14)
plt.xlabel('Count', fontsize=14)
plt.tight_layout()
plt.show()

# Show statistics about class imbalance
print(f"Most common species: {primary_label_counts.index[0]} with {primary_label_counts.values[0]} samples")
print(f"Least common species: {primary_label_counts.index[-1]} with {primary_label_counts.values[-1]} samples")
print(f"Imbalance ratio (most common / least common): {primary_label_counts.values[0] / primary_label_counts.values[-1]:.2f}")

# Analyze the long tail
plt.figure(figsize=(12, 6))
plt.plot(range(len(primary_label_counts)), sorted(primary_label_counts.values, reverse=True))
plt.title('Species Sample Count (Sorted)', fontsize=16)
plt.xlabel('Species Rank', fontsize=14)
plt.ylabel('Number of Samples', fontsize=14)
plt.grid(True)
plt.show()

# Determine rare classes (< 10 samples)
rare_classes = primary_label_counts[primary_label_counts < 10]
print(f"Number of rare classes (<10 samples): {len(rare_classes)}")
print(f"Rare classes: {rare_classes.to_dict()}")



display(working_df)


import ast

# Step 1: Define function to parse string to list
def parse_secondary_labels(label_str):
    if pd.isna(label_str):
        return []
    try:
        return ast.literal_eval(label_str)
    except:
        return []

# Step 2: Parse 'secondary_labels' column in train_df
train_df['parsed_secondary_labels'] = train_df['secondary_labels'].apply(parse_secondary_labels)

# Step 3: Merge parsed secondary labels into working_df
working_df = working_df.merge(
    train_df[['filename', 'parsed_secondary_labels']],
    on='filename',
    how='left'
)

# Step 4: Rename for clarity
working_df.rename(columns={'parsed_secondary_labels': 'secondary_labels'}, inplace=True)

# Step 5: Create unique ID mapping for secondary labels
secondary_label_list = sorted(set([label for sublist in working_df['secondary_labels'] for label in sublist]))
secondary_label2id = {label: idx for idx, label in enumerate(secondary_label_list)}
id2secondary_label = {idx: label for label, idx in secondary_label2id.items()}

# Step 6: Map secondary labels to ID targets
working_df['secondary_target'] = working_df['secondary_labels'].apply(
    lambda x: [secondary_label2id.get(label, -1) for label in x]
)

# Step 7: Check stats
has_secondary = working_df['secondary_labels'].apply(lambda x: len(x) > 0)
print(f"\nâœ… Recordings with secondary labels: {has_secondary.sum()} ({has_secondary.sum()/len(working_df)*100:.2f}%)")



plt.figure(figsize=(10, 6))
sns.countplot(x=working_df['secondary_labels'].apply(len))
plt.title('Number of Secondary Labels per Recording')
plt.xlabel('Count of Secondary Labels')
plt.ylabel('Number of Recordings')
plt.savefig('Number of Secondary Labels per Recording.png')
plt.show()



display(working_df.sample(1))


# Function to get audio duration
def get_audio_duration(file_path, sr=32000):  # SAMPLE_RATE jo b rakhna ho
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        return len(audio) / sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Calculate durations for all files in working_df (or sample agar zyada ho)
durations = []
filepaths = working_df['filepath'].tolist()

print(f"Calculating durations for {len(filepaths)} audio files...")

for fp in tqdm(filepaths):
    duration = get_audio_duration(fp)
    if duration is not None:
        durations.append(duration)
    else:
        durations.append(np.nan)  # handle missing if error

# Add durations to working_df
working_df['duration'] = durations

# Plot duration distribution
plt.figure(figsize=(12, 6))
plt.hist(working_df['duration'].dropna(), bins=50, color='skyblue')
plt.title('Distribution of Audio Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.savefig("Distribution of Audio Durations.png")
plt.show()

# Print some stats
print(f"Duration stats:")
print(f"Mean: {np.nanmean(working_df['duration']):.2f} sec")
print(f"Median: {np.nanmedian(working_df['duration']):.2f} sec")
print(f"Min: {np.nanmin(working_df['duration']):.2f} sec")
print(f"Max: {np.nanmax(working_df['duration']):.2f} sec")

# Check short and long recordings count
short_count = (working_df['duration'] < 1).sum()
long_count = (working_df['duration'] > 60).sum()
total = working_df.shape[0]

print(f"Very short recordings (<1s): {short_count} ({short_count/total*100:.2f}%)")
print(f"Long recordings (>60s): {long_count} ({long_count/total*100:.2f}%)")



# Number of waveforms to plot
num_samples = 5

# Randomly sample files from working_df
sampled_df = working_df.sample(n=num_samples, random_state=42).reset_index(drop=True)

plt.figure(figsize=(15, 3 * num_samples))

for i, row in sampled_df.iterrows():
    file_path = row['filepath']
    label = row['primary_label']
    duration = row['duration']

    # Load audio
    audio, sr = librosa.load(file_path, sr=None)  # Use native sample rate

    plt.subplot(num_samples, 1, i+1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"Waveform of {label} (Duration: {duration:.2f}s)")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

plt.tight_layout()

# Save the figure before showing
plt.savefig('waveform_samples.png', dpi=300)

plt.show()



# Function jo audio ko mel spectrogram me conversion
def audio2melspec(audio_data):
    # if Nan usko remove krty hain
    if np.isnan(audio_data).any():
        mean_val = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_val)

    # Mel spectrogram 
    mel = librosa.feature.melspectrogram(
        y=audio_data,
        sr=config.FS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0
    )

    # Usko decibels me conversion
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalization
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return mel_db



def prepare_audio(audio, target_len):
    #if audio choti ho to usko repeat karo
    while len(audio) < target_len:
        audio = np.concatenate([audio, audio])

    # Center se target length ka audio
    start = max(0, len(audio) // 2 - target_len // 2)
    audio = audio[start:start + target_len]

    # if audio bhi choti ho to padding
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    
    return audio



print("ğŸ”„ lets start audio processing...")
start_time = time.time()

all_bird_data = {}
errors = []
target_len = int(config.TARGET_DURATION * config.FS)

for i, row in tqdm(working_df.iterrows(), total=total_samples):
    if config.N_MAX and i >= config.N_MAX:
        break
    try:
        # Load karo audio
        audio, _ = librosa.load(row.filepath, sr=config.FS)

        # Audio ko prepare karo
        audio = prepare_audio(audio, target_len)

        # Mel spectrogram banao
        mel = audio2melspec(audio)

        # Agar shape match nahi karta to resize karo
        if mel.shape != config.TARGET_SHAPE:
            mel = cv2.resize(mel, config.TARGET_SHAPE)

        # Dictionary me save karo
        all_bird_data[row.samplename] = mel.astype(np.float32)

    except Exception as e:
        print(f"â�Œ Error in {row.filepath}")
        errors.append((row.filepath, str(e)))

end_time = time.time()

print(f"âœ… Done in {end_time - start_time:.1f} seconds")
print(f"ğŸŸ¢ Processed: {len(all_bird_data)} files")
print(f"ğŸ”´ Failed: {len(errors)} files")

print("saving the numpy file")
# Save the dictionary as a NumPy compressed file (.npz)
np.savez_compressed('all_bird_data.npz', **all_bird_data)

print("âœ… Data saved as all_bird_data.npz")


working_df.to_csv('working_df.csv', index=False)
print("Saved the workin df as csv")



# Simple list to store sample data and a set to track displayed classes
samples = []
displayed_classes = set()

# Limit the number of samples to display
max_samples = 4

# Iterate through the dataframe
for i, row in working_df.iterrows():
    if len(samples) >= max_samples:  # Stop once we've selected enough samples
        break

    if row['samplename'] in all_bird_data:
        # If class not already displayed, add sample
        if row['class'] not in displayed_classes:
            samples.append((row['samplename'], row['class'], row['primary_label']))
            displayed_classes.add(row['class'])

# Plotting the spectrograms
if samples:
    plt.figure(figsize=(16, 12))
    
    # Display the spectrogram for each sample
    for i, (samplename, class_name, species) in enumerate(samples):
        plt.subplot(2, 2, i+1)  # Create 2x2 grid
        plt.imshow(all_bird_data[samplename], aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"{class_name}: {species}")
        plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('melspec_examples.png')  # Save plot as an image
    plt.show()  # Display the plot



shapes = [mel.shape for mel in all_bird_data.values()]
unique_shapes = set(shapes)
print(f"Unique shapes: {unique_shapes}")



# Create directory to save plots if it doesn't exist
os.makedirs("melspec_samples", exist_ok=True)

sample_keys = random.sample(list(all_bird_data.keys()), 5)

for key in sample_keys:
    plt.figure(figsize=(10, 4))
    plt.imshow(all_bird_data[key], aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Sample: {key}")
    plt.colorbar()
    plt.tight_layout()
    
    # Save each image with a unique filename
    filename = f"melspec_samples/{key.replace('/', '_').replace(':', '_')}.png"
    plt.savefig(filename)
    plt.show()



all_values = np.concatenate([mel.flatten() for mel in all_bird_data.values()])
# Save the plot directly without specifying the filename variable
plt.hist(all_values, bins=50, color='skyblue')
plt.title("Distribution of Mel Spectrogram Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)

# Save the plot directly
plt.savefig("mel_spectrogram_distribution.png")
plt.close()



energies = [np.mean(mel) for mel in all_bird_data.values()]
plt.hist(energies, bins=50, color='orange')
plt.title("Mean Energy of Spectrograms")
plt.xlabel("Mean Value")
plt.ylabel("Count")
plt.savefig("Mean Energy of Spectrograms.png")
plt.show()





