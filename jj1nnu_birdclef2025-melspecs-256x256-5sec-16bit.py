# Audio Processing for BirdCLEF 2025
# Creating mel-spectrograms from 5-second segments with parallel processing (without saving audio)
import os
import numpy as np
import pandas as pd
import librosa
import scipy.ndimage
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import glob
import concurrent.futures
import multiprocessing
import tqdm
tqdm.tqdm = tqdm.tqdm_notebook = tqdm.tqdm
import scipy.ndimage
import random
import gc
    
all_bird_data = {}
errors = []

# Path configuration
BASE_PATH = '/kaggle/input/birdclef-2025'
TRAIN_AUDIO_PATH = os.path.join(BASE_PATH, 'train_audio')
TRAIN_SOUNDSCAPES_PATH = os.path.join(BASE_PATH, 'train_soundscapes')

# random seed
random.seed(42)

# fabio.csv
FABIO_CSV_PATH = '/kaggle/input/fabio-csv1/fabio.csv'
fabio_df = pd.read_csv(FABIO_CSV_PATH)
fabio_segments = {
    row['filename']: (float(row['start']), float(row['stop']))
    for _, row in fabio_df.iterrows()
}

# Number of workers for parallel processing
# Using 80% of available CPUs is usually a good balance
NUM_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.8))
print(f"Using {NUM_WORKERS} workers for parallel processing")

# Create destination directories
def create_output_dirs():
    # Only for spectrograms
    os.makedirs('specs_5sec', exist_ok=True)
    
    print("Output directory created")

def extend_audio_to_target_length(audio, sr, target_duration=5.0):
    """
    ì˜¤ë””ì˜¤ë¥¼ ëª©í‘œ ê¸¸ì�´ë¡œ í™•ì�¥ (ë°˜ë³µ íŒ¨ë”© ì‚¬ìš©)
    """
    target_length = int(sr * target_duration)
    
    if len(audio) >= target_length:
        return audio
    
    # ë°˜ë³µ íŒ¨ë”©ìœ¼ë¡œ ëª©í‘œ ê¸¸ì�´ ì±„ìš°ê¸°
    repeat_count = int(np.ceil(target_length / len(audio)))
    extended_audio = np.tile(audio, repeat_count)[:target_length]
    
    return extended_audio

def get_random_20sec_segment(audio, sr, max_total_sec=20):
    """
    ì˜¤ë””ì˜¤ì—�ì„œ ë�œë�¤í•œ ìœ„ì¹˜ì—�ì„œ ì‹œì�‘í•˜ëŠ” 20ì´ˆ êµ¬ê°„ì�„ ì¶”ì¶œ
    """
    max_samples = int(max_total_sec * sr)
    total_samples = len(audio)
    if total_samples <= max_samples:
        # ì˜¤ë””ì˜¤ê°€ 20ì´ˆ ì�´í•˜ë�¼ë©´ ì „ì²´ ë°˜í™˜
        return audio
    else:
        # ë�œë�¤ ì‹œì�‘ì � ì„ íƒ�
        max_start = total_samples - max_samples
        start_sample = random.randint(0, max_start)
        end_sample = start_sample + max_samples
        return audio[start_sample:end_sample]

def segment_audio_interval(audio, sr, start_sec, stop_sec, duration, max_total_sec=20):
    """
    start_sec~stop_sec êµ¬ê°„ë§Œ 5ì´ˆì”© ë¶„í• , 20ì´ˆ ì�´ìƒ�ì�´ë©´ ë�œë�¤ 20ì´ˆë§Œ ì‚¬ìš©
    """
    start_sample = int(start_sec * sr)
    stop_sample = int(stop_sec * sr)
    audio_interval = audio[start_sample:stop_sample]
    
    # 5ì´ˆ ë¯¸ë§Œì�´ë©´ ë°˜ë³µ íŒ¨ë”©ìœ¼ë¡œ í™•ì�¥
    audio_interval = extend_audio_to_target_length(audio_interval, sr, duration)
    
    audio_interval = get_random_20sec_segment(audio_interval, sr, max_total_sec)
    samples_per_segment = int(sr * duration)
    num_segments = len(audio_interval) // samples_per_segment
    segments = []
    for i in range(num_segments):
        seg_start = i * samples_per_segment
        seg_end = seg_start + samples_per_segment
        segment = audio_interval[seg_start:seg_end]
        segments.append(segment)
    return segments

# Function to slice audio into segments
def segment_audio(audio, sr, duration, max_total_sec=20):
    """
    ì˜¤ë””ì˜¤ ì „ì²´ë¥¼ 5ì´ˆì”© ë¶„í• , 20ì´ˆ ì�´ìƒ�ì�´ë©´ ë�œë�¤ 20ì´ˆë§Œ ì‚¬ìš©
    """
    # 5ì´ˆ ë¯¸ë§Œì�´ë©´ ë°˜ë³µ íŒ¨ë”©ìœ¼ë¡œ í™•ì�¥
    audio = extend_audio_to_target_length(audio, sr, duration)
    
    audio = get_random_20sec_segment(audio, sr, max_total_sec)
    samples_per_segment = int(sr * duration)
    num_segments = len(audio) // samples_per_segment
    segments = []
    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = audio[start:end]
        segments.append(segment)
    return segments

# Function to create mel-spectrogram (uint16 ë²„ì „)
def create_melspectrogram(audio, sr, size=(256, 256), db_min=-80, db_max=0):
    melspec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=size[0],  
        fmax=sr/2
    )
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    if melspec_db.shape[1] != size[1]:
        melspec_db = scipy.ndimage.zoom(melspec_db, (1, size[1] / melspec_db.shape[1]), order=1)
    melspec_db = np.clip(melspec_db, db_min, db_max)
    melspec_norm = (melspec_db - db_min) / (db_max - db_min)
    
    # uint16ìœ¼ë¡œ ë³€í™˜ (0-65535 ë²”ìœ„)
    melspec_uint16 = (melspec_norm * 65535).astype(np.uint16)
    return melspec_uint16

# Function to process a single audio file
def process_audio_file(args):
    bird_folder, audio_file, bird_folder_path = args
    stats = {'specs_5sec': 0, 'errors': 0, 'extended': 0}
    audio_path = os.path.join(bird_folder_path, audio_file)
    
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        base_filename = os.path.splitext(audio_file)[0]
        rel_path = os.path.join(bird_folder, audio_file)
        
        # 5ì´ˆ ë¯¸ë§Œ ì²´í�¬ ë°� í™•ì�¥
        original_duration = len(audio) / sr
        if original_duration < 5.0:
            stats['extended'] += 1
        
        if rel_path in fabio_segments:
            start_sec, stop_sec = fabio_segments[rel_path]
            segments_5sec = segment_audio_interval(audio, sr, start_sec, stop_sec, 5, max_total_sec=20)
        else:
            segments_5sec = segment_audio(audio, sr, 5, max_total_sec=20)
            
        for i, segment in enumerate(segments_5sec):
            segment_filename = f"{bird_folder}-{base_filename}_{i+1:02d}"
            spec_img = create_melspectrogram(segment, sr)
            all_bird_data[segment_filename] = spec_img
            stats['specs_5sec'] += 1
            
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        stats['errors'] += 1
        errors.append((audio_file, str(e)))
        
    return stats

# ìƒˆë¡œ ì¶”ê°€: soundscape íŒŒì�¼ ì²˜ë¦¬ í•¨ìˆ˜ (ê°œì„ ë�¨)
def process_soundscape_file(audio_file):
    """
    soundscape íŒŒì�¼ì—�ì„œ ë§¨ ì•� 5ì´ˆë§Œ ì¶”ì¶œí•˜ì—¬ negative ìƒ˜í”Œë¡œ ë³€í™˜
    5ì´ˆ ë¯¸ë§Œì�´ë©´ ë°˜ë³µ íŒ¨ë”©ìœ¼ë¡œ í™•ì�¥
    """
    stats = {'specs_5sec': 0, 'errors': 0, 'extended': 0}
    audio_path = os.path.join(TRAIN_SOUNDSCAPES_PATH, audio_file)
    
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        base_filename = os.path.splitext(audio_file)[0]
        
        # 5ì´ˆ ê¸¸ì�´ ì„¤ì •
        target_length = int(sr * 5)
        
        if len(audio) >= target_length:
            # 5ì´ˆ ì�´ìƒ�ì�´ë©´ ë§¨ ì•� 5ì´ˆë§Œ ì‚¬ìš©
            segment = audio[:target_length]
        else:
            # 5ì´ˆ ë¯¸ë§Œì�´ë©´ ë°˜ë³µ íŒ¨ë”©ìœ¼ë¡œ 5ì´ˆ ì±„ìš°ê¸°
            segment = extend_audio_to_target_length(audio, sr, 5.0)
            stats['extended'] += 1
        
        segment_filename = f"negative-{base_filename}"
        spec_img = create_melspectrogram(segment, sr)
        all_bird_data[segment_filename] = spec_img
        stats['specs_5sec'] += 1
            
    except Exception as e:
        print(f"Error processing soundscape {audio_file}: {e}")
        stats['errors'] += 1
        errors.append((audio_file, str(e)))
        
    return stats

def process_soundscapes():
    """
    ëª¨ë“  soundscape íŒŒì�¼ì�„ ì²˜ë¦¬í•˜ì—¬ negative ìƒ˜í”Œ ìƒ�ì„±
    """
    if not os.path.exists(TRAIN_SOUNDSCAPES_PATH):
        print(f"Warning: {TRAIN_SOUNDSCAPES_PATH} does not exist, skipping soundscape processing")
        return {'specs_5sec': 0, 'errors': 0, 'extended': 0}
    
    soundscape_files = [f for f in os.listdir(TRAIN_SOUNDSCAPES_PATH) if f.endswith('.ogg')]
    print(f"Processing {len(soundscape_files)} soundscape files for negative samples...")
    
    total_stats = {'specs_5sec': 0, 'errors': 0, 'extended': 0}
    
    for audio_file in soundscape_files:
        stats = process_soundscape_file(audio_file)
        for key in total_stats:
            total_stats[key] += stats[key]
    
    return total_stats

# Main processing function with parallel execution
def process_bird_audio_parallel():
    # Create output directories
    create_output_dirs()
    
    # Get list of audio folders (bird species IDs)
    bird_folders = [f for f in os.listdir(TRAIN_AUDIO_PATH) if os.path.isdir(os.path.join(TRAIN_AUDIO_PATH, f))]
    
    # Initialize overall statistics
    total_stats = {
        'specs_5sec': 0,
        'errors': 0,
        'extended': 0
    }

    # Process bird audio files (positive samples)
    print(f"Processing {len(bird_folders)} bird species folders...")
    for bird_folder in bird_folders:
        bird_folder_path = os.path.join(TRAIN_AUDIO_PATH, bird_folder)
        audio_files = [f for f in os.listdir(bird_folder_path) if f.endswith(('.ogg', '.mp3', '.wav'))]
        for audio_file in audio_files:
            stats = process_audio_file((bird_folder, audio_file, bird_folder_path))
            for key in total_stats:
                total_stats[key] += stats[key]

    # Process soundscape files (negative samples)
    soundscape_stats = process_soundscapes()
    for key in total_stats:
        total_stats[key] += soundscape_stats[key]

    return total_stats

# ë°°ì¹˜ ì €ì�¥ í•¨ìˆ˜ ì¶”ê°€
def save_spectrograms_in_batches(all_bird_data, batch_size=5000):
    """ë°°ì¹˜ ë‹¨ìœ„ë¡œ ìŠ¤í�™íŠ¸ë¡œê·¸ë�¨ ì €ì�¥ (ë©”ëª¨ë¦¬ íš¨ìœ¨ì �)"""
    keys = list(all_bird_data.keys())
    num_batches = len(keys) // batch_size + (1 if len(keys) % batch_size else 0)
    
    print(f"Saving {len(keys)} spectrograms in {num_batches} batches...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(keys))
        batch_keys = keys[start_idx:end_idx]
        
        batch_data = {key: all_bird_data[key] for key in batch_keys}
        
        filename = f'melspecs_uint16_batch_{i:03d}.npy'
        np.save(filename, batch_data)
        print(f"Saved batch {i+1}/{num_batches}: {filename} ({len(batch_keys)} samples)")
        
        # ë©”ëª¨ë¦¬ ì •ë¦¬
        del batch_data
        gc.collect()
    
    print("All spectrograms saved successfully!")

# Function to display examples of created mel-spectrograms
def visualize_examples_from_dict(all_bird_data):
    import matplotlib.pyplot as plt
    keys = list(all_bird_data.keys())[:4]
    for i, k in enumerate(keys):
        plt.subplot(1, 4, i+1)
        # uint16ì�„ floatë¡œ ë³€í™˜í•˜ì—¬ ì‹œê°�í™”
        spec_float = all_bird_data[k].astype(np.float32) / 65535.0
        plt.imshow(spec_float, aspect='auto', origin='lower', cmap='viridis')
        plt.title(k)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Get statistics about processed files
def get_processing_stats():
    # Count number of created files
    specs_5sec_count = sum(len(files) for _, _, files in os.walk('specs_5sec'))
    
    print(f"Processing statistics:")
    print(f"- Created {specs_5sec_count} mel-spectrograms for 5-second segments")

# Run the main process
if __name__ == "__main__":
    print("Starting parallel audio processing...")
    print("Using uint16 format for memory efficiency...")
    print("Short audio files (<5s) will be extended using repeat padding...")
    
    # Important: matplotlib needs to be configured for non-interactive backend in parallel processing
    plt.switch_backend('agg')
    
    # Process files in parallel
    stats = process_bird_audio_parallel()
    
    print("Processing completed!")
    print(f"Processing statistics from parallel processing:")
    print(f"- Created {stats['specs_5sec']} mel-spectrograms for 5-second segments")
    print(f"- Extended {stats['extended']} short audio files (<5s) using repeat padding")
    print(f"- Encountered {stats['errors']} errors during processing")
    
    # ì¶”ê°€: positive/negative ìƒ˜í”Œ ìˆ˜ í™•ì�¸
    positive_samples = len([k for k in all_bird_data.keys() if not k.startswith('negative-')])
    negative_samples = len([k for k in all_bird_data.keys() if k.startswith('negative-')])
    print(f"- Positive samples: {positive_samples}")
    print(f"- Negative samples: {negative_samples}")
    print(f"- Negative ratio: {negative_samples/(positive_samples+negative_samples)*100:.1f}%")

    # ë©”ëª¨ë¦¬ íš¨ìœ¨ì �ì�¸ ë°°ì¹˜ ì €ì�¥
    print("\nSaving spectrograms in batches to avoid memory issues...")
    save_spectrograms_in_batches(all_bird_data, batch_size=5000)
    
    # ë©”ëª¨ë¦¬ ì‚¬ìš©ëŸ‰ ì¶”ì •
    total_samples = len(all_bird_data)
    memory_mb = total_samples * 256 * 256 * 2 / (1024 * 1024)  # uint16 = 2 bytes
    print(f"\nMemory usage estimate: {memory_mb:.1f} MB ({memory_mb/1024:.1f} GB)")
    
    # Double-check with filesystem statistics
    print("\nVerifying with filesystem statistics:")
    get_processing_stats()


