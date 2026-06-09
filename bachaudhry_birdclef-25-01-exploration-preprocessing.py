import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc, random 
from pathlib import Path
from tqdm.notebook import tqdm
import IPython.display as ipd
from IPython.display import display, clear_output
import ipywidgets as widgets

import librosa
import librosa.display
import soundfile as sf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# Initialize and set basic configuration
cfg = Config(SEED=42, SAMPLE_RATE=32000,
             DATA_PATH=Path("/kaggle/input/birdclef-2025"))

# Verifying changes
print(cfg.__dict__)


# Function to seed everything to ensure reproducibility
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Change to true if input sizes are kept constant

seed_everything(cfg.SEED)


# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Files in the base data path
print(f"Files in the base data path include: {os.listdir(cfg.DATA_PATH)}")


# Taking a closer look at the meta data
metadata_path = cfg.DATA_PATH / "train.csv"

if metadata_path.exists():
    train_df = pd.read_csv(metadata_path)
    print(train_df.head(15))
    print("\nMetadata Columns:", train_df.columns)
    print("\nTraining Samples:", len(train_df))
    print("\nUnique Species:", train_df['primary_label'].nunique())
    print("\nSecondary Species Labels(Recordist Marked):", train_df['secondary_labels'].nunique())
    # Key distributions
    print("\nSpecies distribution (top 10):")
    print(train_df[['primary_label', 'scientific_name']].value_counts().head(10))
    print("\nSpecies distribution (bottom 10):")
    print(train_df[['primary_label', 'scientific_name']].value_counts().tail(10))
else:
    print(f"Metadata file not found at {meta_datapath}. Check path!")


train_df.info()


train_df.describe(include=[object])


train_df.describe(include=[np.number])


# Update config
cfg.N_MELS = 128           # number of MEL bands(can be adjusted after experimentation)
cfg.N_FFT = 2048           # window size for fast fourier transform (FFT)
cfg.HOP_LENGTH = 512       # number of samples b/w successive frames
cfg.FMIN = 50              # minimum frequency
cfg.FMAX = 14000           # maximum frequency (relevant for bird calls)
# New Clip Params
cfg.TARGET_DURATION_S = 5  # setting at 5 secs to make it easier to hear context
cfg.TARGET_SAMPLES = cfg.TARGET_DURATION_S * cfg.SAMPLE_RATE


# Verifying changes
print(cfg.__dict__)


# Helper function to plot waveform and spectrograms
def plot_spectrogram(waveform, sr, title="Waveform and Mel Spectrogram"):
    """Plots waveform and Mel spectrogram for a given audio signal"""
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot waveform
    librosa.display.waveshow(waveform, sr=sr, ax=axs[0])
    axs[0].set_title('Waveform')
    axs[0].set_ylabel('Amplitude')

    # Generate and plot Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr,
                                                     n_fft=cfg.N_FFT,
                                                     hop_length=cfg.HOP_LENGTH,
                                                     n_mels=cfg.N_MELS,
                                                     fmin=cfg.FMIN,
                                                     fmax=cfg.FMAX)
    # Using power_to_db converts amplitude spectrogram to dB scale to improve visuals
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    img = librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=cfg.HOP_LENGTH,
                                   x_axis='time', y_axis='mel', ax=axs[1],
                                   fmin=cfg.FMIN, fmax=cfg.FMAX)
    axs[1].set_title('Mel Spectrogram (dB)')
    axs[1].set_ylabel('Mel Frequency')
    axs[1].set_xlabel('Time (s)')
    fig.colorbar(img, ax=axs[1], format='%+2.0f dB')

    plt.suptitle(title, fontsize=17)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
    return mel_spectrogram_db


# Load and visualize a few random samples from the training data
if 'train_df' in globals() and not train_df.empty: # check of DF exists and isn't empty
    num_samples = 5
    sample_files = train_df.sample(num_samples, random_state=cfg.SEED)

    for index, row in sample_files.iterrows():
        filename = row['filename']
        label = row['primary_label']
        file_path = cfg.DATA_PATH/"train_audio"/filename

        if file_path.exists():
            print(f"Loading: {filename}; Label: {label}")
            try:
                # Load audio file and use duration=None initially to load the full file
                waveform, sr = librosa.load(file_path, sr=cfg.SAMPLE_RATE, duration=None)
                print(f"    Original Duration: {len(waveform) / sr:.2f} seconds.")

                # Play 10 sec snippet
                display_duration = min(10.0, len(waveform) / sr)
                print(f"   Playing first {display_duration:.2f} seconds:")
                ipd.display(ipd.Audio(waveform[:int(display_duration*sr)], rate=sr))

                # Plot waveform and spectrogram of the first chunk (e.g. 5 seconds)
                # Use the whole file if shorter than 5 secs
                plot_waveform, _ = librosa.load(file_path, sr=cfg.SAMPLE_RATE, duration=cfg.TARGET_DURATION_S)
                _ = plot_spectrogram(plot_waveform, cfg.SAMPLE_RATE, 
                                     title=f"{filename}; (Label: {label}); First {cfg.TARGET_DURATION_S}s")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            print("-" * 30)
        else:
            print(f"File not found: {file_path}")
else:
    print("Training metadata DataFrame ('train_df') not found or is empty! Skipping sample loading.")
    print("Try to manually specify some audio file paths for exploration.")
                


# Analyze audio durations
if 'train_df' in globals() and not train_df.empty:
    print("Analyzing audio durations...")
    durations = []
    pbar = tqdm(train_df['filename'].tolist(), desc="Calculating durations")
    for filename in pbar:
        file_path = cfg.DATA_PATH/"train_audio"/filename
        if file_path.exists():
            try:
                # Efficient approach to get duration with loading the whole file
                info = sf.info(file_path)
                durations.append(info.duration)
            except Exception as e:
                print(f"Could not get info for {filename}: {e}") #Comment / uncomment for debugging
                durations.append(np.nan) # mark errors
        else:
            durations.append(np.nan)


train_df['duration'] = durations # new column for durations
#train_df.dropna(subset=['duration'], inplace=True) # remove rows where duration couldn't be calculated
train_df['duration'].describe()


train_df['duration'].isnull().sum()


plt.figure(figsize=(12, 8))
sns.histplot(train_df['duration'], bins=100)
plt.title('Distribution of Audio File Durations (s)')
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.show()

print(train_df['duration'].describe())
print(f"\nNumber of clips if using {cfg.TARGET_DURATION_S}s segments (approx.):")
# calculate total duration / target segment duration
total_segments = np.ceil(train_df['duration'] / cfg.TARGET_DURATION_S).sum()
print(f"  ~ {int(total_segments):,} segments")


# Function to create fixed length clips
def create_clips(waveform, sr, clip_length_samples, overlap_samples=0, end_behavior='pad'):
    """ Splits or pads a waveform into fixed length clips"""
    clips = []
    total_samples = len(waveform)
    step = clip_length_samples - overlap_samples
    current_pos = 0

    while current_pos < total_samples:
        end_pos = current_pos + clip_length_samples
        clip = waveform[current_pos:end_pos]

        # Handle end of the waveform
        if len(clip) < clip_length_samples:
            if end_behavior == 'pad':
                padding_needed = clip_length_samples - len(clip)
                clip = np.pad(clip, (0, padding_needed), 'constant')
                clips.append(clip)
            elif end_behavior == 'truncate':
                pass # discard if the remaining part is shorter than the clip length
            elif end_behavior == 'variable':
                # Add shorter clip - requires downstream logic to handle variable sizes or padding later
                if len(clip) > 0: # Avoid empty clips if overlap > step
                    clips.append(clip)
            else:
                raise ValueError(f"Unknown end_behavior: {end_behavior}")
            break # End of waveform
        else:
            clips.append(clip)

        if step <= 0: # Avoid infinite loop if overlap >= clip_length
            raise ValueError("Overlap must be less than clip length")
        current_pos += step

        # Ensure we don't go past the end if step makes us jumpover the last samples
        # mostly relevant if overlap > 0
        if current_pos >= total_samples and end_behavior != 'variable':
            break # Prevents adding a fully paddedd clip unnecessarily when using 'pad'
    
    return clips
      


if 'waveform' in globals(): # last loaded waveform 
    print(f"\nSplitting example waveform (duration: {len(waveform) / cfg.SAMPLE_RATE:.2f}s) into\
    {cfg.TARGET_DURATION_S}s clips...")
    example_clips = create_clips(waveform, cfg.SAMPLE_RATE, cfg.TARGET_SAMPLES, end_behavior='pad')
    print(f"Generated {len(example_clips)} clips.")

    # Visualizing the first clip
    if example_clips:
        print("Visualizing the first generated clip:")
        _ = plot_spectrogram(example_clips[0], cfg.SAMPLE_RATE, title="First 5s Clip")
    else:
        print("No clips generated (original file may be too short).")


# Setting up widget functionality while modifying the code we used to generate mel-spectrograms and waveforms
if 'train_df' in globals() and not train_df.empty:
    # Ensure duration column exists from previous steps
    if 'duration' not in train_df.columns:
        print("Warning: `duration` column not found in train_df.")
        # Limiting the size of the interactive_df for performance.
        interactive_df = train_df.sample(n=50, random_state=42) # Reduce to 50 to improve performance
    else:
        # Selecting files with a reasonable duration
        interactive_df = train_df[(train_df['duration'] >= cfg.TARGET_DURATION_S)].copy()

    if not interactive_df.empty:
        # Create drop down menu items from filenames and labels
        options = [
            (f"{row['filename']} ({row['primary_label']})", idx)
            for idx, row in interactive_df.sample(min(50, len(interactive_df)), random_state=cfg.SEED).iterrows() # Reduce to 50
        ]
        options.sort() # alphabetical sorting

        file_dropdown = widgets.Dropdown(options=options, description='Select File:')
        output_area = widgets.Output()

        def on_file_change(change):
            with output_area:
                clear_output(wait=True) # Clear previous output
                if change['new'] is not None:
                    selected_index = change['new']
                    row = interactive_df.loc[selected_index]
                    filename = row['filename']
                    label = row['primary_label']
                    file_path = cfg.DATA_PATH/"train_audio"/filename

                    if file_path.exists():
                        print(f"Loading: {filename} (Label: {label})")
                        try:
                            # Increased target duration for more context
                            waveform, sr = librosa.load(file_path, sr=cfg.SAMPLE_RATE, duration=cfg.TARGET_DURATION_S*2)
                            display(ipd.Audio(waveform[:cfg.TARGET_SAMPLES], rate=sr)) # Play first segment
                            _ = plot_spectrogram(waveform[:cfg.TARGET_SAMPLES], sr, 
                                                 title=f"{filename} ({label} - First {cfg.TARGET_DURATION_S}s)")
                        except Exception as e:
                            print(f"      Error processing {filename}: {e}")
                    else:
                        print("File not found: {file_path}")

        file_dropdown.observe(on_file_change, names='value')

        print("Interactive Spectrogram Viewer:")
        display(file_dropdown)
        display(output_area)
        
        # Trigger initial loadd for default selection
        on_file_change({'new': file_dropdown.value})
    
    else:
        print("No suitable files found for interactive exploration (duration >= 5s).")
else:
    print("Train metadata DataFrame ('train_df') not found or empty. Skipping interactive viewer.")
                            




