# Audio Processing for BirdCLEF 2025
# Slicing audio into 10-second segments and creating mel-spectrograms with parallel processing

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import glob
import concurrent.futures
import multiprocessing
import tqdm
tqdm.tqdm = tqdm.tqdm_notebook = tqdm.tqdm

# Path configuration
BASE_PATH = '/kaggle/input/birdclef-2025'
TRAIN_AUDIO_PATH = os.path.join(BASE_PATH, 'train_audio')

# Number of workers for parallel processing
# Using 80% of available CPUs is usually a good balance
NUM_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.8))
print(f"Using {NUM_WORKERS} workers for parallel processing")

# Create destination directories
def create_output_dirs():
    # For 10-second audio
    os.makedirs('audio_10sec', exist_ok=True)
    os.makedirs('specs_10sec', exist_ok=True)
    
    print("Output directories created")

# Function to slice audio into segments
def segment_audio(audio, sr, duration):
    """
    Slices audio into segments of specified duration
    
    Parameters:
    -----------
    audio : np.array
        Audio data
    sr : int
        Sample rate
    duration : int
        Segment duration in seconds
    
    Returns:
    --------
    list
        List of audio segments
    """
    # Calculate number of samples for the desired duration
    samples_per_segment = int(sr * duration)
    
    # Calculate number of full segments
    num_segments = len(audio) // samples_per_segment
    
    segments = []
    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = audio[start:end]
        segments.append(segment)
    
    return segments

# Function to create mel-spectrogram
def create_melspectrogram(audio, sr, size=(224, 224)):
    """
    Creates a mel-spectrogram from audio data
    
    Parameters:
    -----------
    audio : np.array
        Audio data
    sr : int
        Sample rate
    size : tuple
        Output image size (width, height)
    
    Returns:
    --------
    PIL.Image
        Mel-spectrogram image of size (width, height)
    """
    # Calculate mel-spectrogram
    melspec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=128,
        fmax=sr/2
    )
    
    # Convert to decibels
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    # Create figure of required size
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Display spectrogram
    img = librosa.display.specshow(
        melspec_db, 
        x_axis='time', 
        y_axis='mel', 
        sr=sr,
        cmap=cm.viridis,
        ax=ax
    )
    
    # Remove axes and padding
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Use a unique temporary filename for parallel processing
    import uuid
    temp_file = f'temp_spec_{uuid.uuid4()}.png'
    plt.savefig(temp_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Open and resize the image
    img = Image.open(temp_file)
    img = img.resize(size)
    
    # Remove temporary file
    os.remove(temp_file)
    
    return img

# Function to process a single audio file
def process_audio_file(args):
    """
    Process a single audio file - create segments and spectrograms
    
    Parameters:
    -----------
    args : tuple
        (bird_folder, audio_file, bird_folder_path)
    
    Returns:
    --------
    dict
        Results statistics
    """
    bird_folder, audio_file, bird_folder_path = args
    
    # Initialize statistics
    stats = {
        'audio_10sec': 0,
        'specs_10sec': 0,
        'errors': 0
    }
    
    # Load audio
    audio_path = os.path.join(bird_folder_path, audio_file)
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Base filename without extension
        base_filename = os.path.splitext(audio_file)[0]
        
        # Slice into 10-second segments
        segments_10sec = segment_audio(audio, sr, 10)
        
        # Process 10-second segments
        for i, segment in enumerate(segments_10sec):
            # Create filename with index
            segment_filename = f"{base_filename}_{i+1:02d}.ogg"
            
            # Save audio segment
            sf.write(f'audio_10sec/{bird_folder}/{segment_filename}', segment, sr)
            stats['audio_10sec'] += 1
            
            # Create and save mel-spectrogram
            spec_img = create_melspectrogram(segment, sr)
            spec_img.save(f'specs_10sec/{bird_folder}/{os.path.splitext(segment_filename)[0]}.png')
            stats['specs_10sec'] += 1
                
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        stats['errors'] += 1
    
    return stats

# Main processing function with parallel execution
def process_bird_audio_parallel():
    # Create output directories
    create_output_dirs()
    
    # Get list of audio folders (bird species IDs)
    bird_folders = [f for f in os.listdir(TRAIN_AUDIO_PATH) if os.path.isdir(os.path.join(TRAIN_AUDIO_PATH, f))]
    
    # Initialize overall statistics
    total_stats = {
        'audio_10sec': 0,
        'specs_10sec': 0,
        'errors': 0
    }
    
    # For each bird species folder
    print(f"Processing {len(bird_folders)} bird species folders...")
    for bird_folder in bird_folders:
        # Create output folders for current species
        os.makedirs(f'audio_10sec/{bird_folder}', exist_ok=True)
        os.makedirs(f'specs_10sec/{bird_folder}', exist_ok=True)
        
        # Get list of audio files
        bird_folder_path = os.path.join(TRAIN_AUDIO_PATH, bird_folder)
        audio_files = [f for f in os.listdir(bird_folder_path) if f.endswith(('.ogg', '.mp3', '.wav'))]
        
        # Prepare arguments for parallel processing
        process_args = [(bird_folder, audio_file, bird_folder_path) for audio_file in audio_files]
        
        # Process files in parallel
        print(f"Processing {len(process_args)} files for {bird_folder}...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks without tqdm (to avoid ipywidgets dependency)
            results = list(executor.map(process_audio_file, process_args))
        
        # Aggregate statistics
        for result in results:
            for key in total_stats:
                total_stats[key] += result[key]
    
    return total_stats

# Function to display examples of created mel-spectrograms
def visualize_examples():
    # Find random examples of spectrograms
    spec_10sec_paths = glob.glob('specs_10sec/**/*.png', recursive=True)
    
    # If examples found, select the first few
    spec_10sec_examples = spec_10sec_paths[:4] if spec_10sec_paths else []
    
    # Create figure for display
    n_examples = len(spec_10sec_examples)
    if n_examples == 0:
        print("No examples to display")
        return
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4*n_examples))
    
    # If only one example, axes won't be an array
    if n_examples == 1:
        axes = [axes]
    
    # Display examples of 10-second spectrograms
    for i, path in enumerate(spec_10sec_examples):
        img = Image.open(path)
        axes[i].imshow(np.array(img))
        axes[i].set_title(f"10-second spectrogram: {os.path.basename(path)}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Get statistics about processed files
def get_processing_stats():
    # Count number of created files
    audio_10sec_count = sum(len(files) for _, _, files in os.walk('audio_10sec'))
    specs_10sec_count = sum(len(files) for _, _, files in os.walk('specs_10sec'))
    
    print(f"Processing statistics:")
    print(f"- Created {audio_10sec_count} 10-second audio segments")
    print(f"- Created {specs_10sec_count} mel-spectrograms for 10-second segments")

# Run the main parallel process
if __name__ == "__main__":
    print("Starting parallel audio processing...")
    
    # Important: matplotlib needs to be configured for non-interactive backend in parallel processing
    plt.switch_backend('agg')
    
    # Process files in parallel
    stats = process_bird_audio_parallel()
    
    print("Processing completed!")
    print(f"Processing statistics from parallel processing:")
    print(f"- Created {stats['audio_10sec']} 10-second audio segments")
    print(f"- Created {stats['specs_10sec']} mel-spectrograms for 10-second segments")
    print(f"- Encountered {stats['errors']} errors during processing")
    
    # Double-check with filesystem statistics
    print("\nVerifying with filesystem statistics:")
    get_processing_stats()
    
    # Display spectrogram examples
    visualize_examples()

