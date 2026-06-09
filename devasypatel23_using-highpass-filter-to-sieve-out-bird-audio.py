import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import torch
import matplotlib.pyplot as plt
import soundfile as sf # Silero VAD uses soundfile

class CFG:
    # Paths (ADJUST THESE if necessary)
    train_csv = r'/kaggle/input/birdclef-2025/train.csv' # Example path, adjust!
    train_datadir = r'/kaggle/input/birdclef-2025/train_audio' # Example path, adjust!

    # Audio parameters (Copy from your baseline script)
    FS = 32000
    TARGET_DURATION = 5.0 # Duration used for feature extraction
    TARGET_SHAPE = (256, 256) # Target shape for resizing

    N_FFT = 1024
    HOP_LENGTH = 500 # Adjust if different in your processing
    N_MELS = 128
    FMIN = 40
    FMAX = 15000

    # VAD parameters
    VAD_THRESHOLD = 0.4 # Threshold mentioned in discussion (default is 0.5)

cfg = CFG()

# --- Helper Functions (Adapted from baseline) ---

def audio_to_melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize for visualization (optional, but often helpful)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_db # Return dB for better visualization range

def get_vad_timestamps(audio_path, vad_model_utils, cfg):
    """Get speech timestamps using Silero VAD"""
    try:
        # Silero VAD expects 16kHz, read_audio handles resampling
        wav = vad_model_utils['read_audio'](audio_path, sampling_rate=16000)
        speech_timestamps = vad_model_utils['get_speech_timestamps'](
            wav,
            vad_model_utils['model'],
            threshold=cfg.VAD_THRESHOLD,
            sampling_rate=16000 # Ensure VAD uses 16kHz
        )
        return speech_timestamps, 16000 # Return timestamps and the sample rate used by VAD
    except Exception as e:
        print(f"  Error getting VAD timestamps for {os.path.basename(audio_path)}: {e}")
        return [], 16000

def plot_spectrogram_with_vad(spec, vad_timestamps, vad_sr, title, cfg):
    """Plots spectrogram and overlays VAD segments"""
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(spec, sr=cfg.FS, hop_length=cfg.HOP_LENGTH,
                             x_axis='time', y_axis='mel', fmin=cfg.FMIN, fmax=cfg.FMAX)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)

    # Overlay VAD timestamps
    # Convert VAD sample indices (at vad_sr) to time (seconds)
    vad_times = [(ts['start'] / vad_sr, ts['end'] / vad_sr) for ts in vad_timestamps]

    for start_time, end_time in vad_times:
        plt.axvspan(start_time, end_time, color='red', alpha=0.3, label='Human Voice (VAD)' if 'Human Voice (VAD)' not in plt.gca().get_legend_handles_labels()[1] else "")

    if vad_times:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# --- Main Analysis Logic ---

# 1. List of problematic samples from your log output
problematic_samples = [
    "greegr-XC490733", "cinbec1-XC389682", "whtdov-iNat863170",
    "bkmtou1-iNat1270065", "speowl1-XC719818", "mastit1-iNat227747",
    "amekes-iNat863228", "amekes-iNat522505", "saffin-iNat157525",
    "grekis-XC360620"
    # Add more if needed, removed duplicates from your log example
]
print(f"Analyzing {len(problematic_samples)} problematic samples...")

# 2. Load metadata
try:
    train_df = pd.read_csv(cfg.train_csv)
    # Create samplename if it doesn't exist (match training script logic)
    if 'samplename' not in train_df.columns:
         train_df['samplename'] = train_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    # Create filepath if it doesn't exist
    if 'filepath' not in train_df.columns:
        train_df['filepath'] = train_df['filename'].apply(lambda x: os.path.join(cfg.train_datadir, x))

    # Create mapping for quick lookup
    samplename_to_filepath = pd.Series(train_df.filepath.values, index=train_df.samplename).to_dict()
    print(f"Loaded metadata for {len(train_df)} samples.")
except FileNotFoundError:
    print(f"Error: Could not find train.csv at {cfg.train_csv}")
    exit()
except Exception as e:
    print(f"Error loading metadata: {e}")
    exit()


# 3. Load VAD model
try:
    # Use force_reload=True if you update the library or encounter issues
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False) # Set to True if needed
    vad_model_utils = {
        'model': model,
        'get_speech_timestamps': utils[0],
        'save_audio': utils[1],
        'read_audio': utils[2],
        'VADIterator': utils[3],
        'collect_chunks': utils[4]
    }
    print("Silero VAD model loaded successfully.")
except Exception as e:
    print(f"Error loading Silero VAD model: {e}")
    print("Please ensure you have internet connectivity and the model can be downloaded.")
    exit()

# 4. Analyze each sample
for sample in problematic_samples:
    print(f"\n--- Processing: {sample} ---")
    if sample not in samplename_to_filepath:
        print(f"  Warning: Samplename '{sample}' not found in train.csv metadata.")
        continue

    audio_path = samplename_to_filepath[sample]

    if not os.path.exists(audio_path):
        print(f"  Error: Audio file not found at '{audio_path}'")
        continue

    try:
        # Load full audio first
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS, mono=True)
        print(f"  Loaded audio: Duration={librosa.get_duration(y=audio_data, sr=cfg.FS):.2f}s")

        # Generate spectrogram for the *entire* audio for context
        # (Your processing likely took a 5s chunk, but seeing the whole helps)
        full_spec = audio_to_melspec(audio_data, cfg)

        # Get VAD timestamps for the full audio
        vad_timestamps, vad_sr = get_vad_timestamps(audio_path, vad_model_utils, cfg)
        print(f"  Detected {len(vad_timestamps)} potential speech segments.")

        # Plot
        plot_spectrogram_with_vad(full_spec, vad_timestamps, vad_sr,
                                  f"Full Spectrogram & VAD: {sample}", cfg)

    except Exception as e:
        print(f"  Error processing sample {sample}: {e}")

print("\nAnalysis complete.")


import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# --- Configuration ---
class CFG:
    # Paths (ADJUST THESE)
    train_csv = r'/kaggle/input/birdclef-2025/train.csv' # Path to your train CSV
    train_datadir = r'/kaggle/input/birdclef-2025/train_audio' # Path to your train audio folder
    output_dir = r'/kaggle/working/' # Where to save filtered audio

    # Audio parameters
    FS = 32000 # Sample rate

    # Filtering parameters
    filter_type = 'highpass' # 'highpass', 'lowpass', 'bandstop'
    cutoff_freq_highpass = 1500 # Hz - Frequencies below this will be attenuated
    # cutoff_freq_lowpass = 800 # Hz - Frequencies above this will be attenuated
    # bandstop_low = 80 # Hz
    # bandstop_high = 1100 # Hz

    # Sample to process (Choose one from your problematic list)
    sample_filename = 'speowl1/XC719818.ogg' # Example using greegr-XC490733 bkmtou1-iNat1270065 speowl1-XC719818

cfg = CFG()

# --- Helper Functions ---
def apply_fft_filter(y, sr, filter_type, cutoff_low=None, cutoff_high=None):
    """Applies a filter in the frequency domain using FFT."""
    # Compute the Short-Time Fourier Transform (STFT)
    stft_result = librosa.stft(y)
    # Get the frequency bins corresponding to the STFT
    freqs = librosa.fft_frequencies(sr=sr) # Get frequencies for each bin

    # Create a copy to modify
    stft_filtered = stft_result.copy()

    print(f"Applying {filter_type} filter...")
    if filter_type == 'highpass':
        if cutoff_low is None:
            raise ValueError("cutoff_low must be specified for highpass filter")
        print(f"  Attenuating frequencies below {cutoff_low} Hz")
        # Zero out bins corresponding to frequencies below the cutoff
        stft_filtered[freqs < cutoff_low, :] = 0
    elif filter_type == 'lowpass':
        if cutoff_high is None:
            raise ValueError("cutoff_high must be specified for lowpass filter")
        print(f"  Attenuating frequencies above {cutoff_high} Hz")
        # Zero out bins corresponding to frequencies above the cutoff
        stft_filtered[freqs > cutoff_high, :] = 0
    elif filter_type == 'bandstop':
        if cutoff_low is None or cutoff_high is None:
            raise ValueError("cutoff_low and cutoff_high must be specified for bandstop filter")
        print(f"  Attenuating frequencies between {cutoff_low} Hz and {cutoff_high} Hz")
        # Find indices for frequencies within the band
        band_indices = np.where((freqs >= cutoff_low) & (freqs <= cutoff_high))[0]
        # Zero out bins within the band
        stft_filtered[band_indices, :] = 0
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    # Compute the Inverse Short-Time Fourier Transform (ISTFT)
    y_filtered = librosa.istft(stft_filtered, length=len(y)) # Ensure output length matches input

    return y_filtered

def plot_spectrogram(y, sr, title, ax):
    """Helper to plot a spectrogram."""
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title)
    return img

# --- Main Script ---
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Construct full path to the audio file
    audio_path = os.path.join(cfg.train_datadir, cfg.sample_filename)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        exit()

    print(f"Loading audio: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=cfg.FS)
        print(f"Audio loaded successfully. Duration: {librosa.get_duration(y=y, sr=sr):.2f}s")

        # --- Apply the filter ---
        y_filtered = apply_fft_filter(
            y, sr,
            filter_type=cfg.filter_type,
            cutoff_low=cfg.cutoff_freq_highpass if cfg.filter_type in ['highpass', 'bandstop'] else None,
            cutoff_high=None # Adjust if using lowpass or bandstop
            # cutoff_high=cfg.cutoff_freq_lowpass if cfg.filter_type == 'lowpass' else (cfg.bandstop_high if cfg.filter_type == 'bandstop' else None) # More general
        )
        print("Filtering complete.")

        # --- Save audio files ---
        original_filename = f"original_{os.path.splitext(os.path.basename(audio_path))[0]}.wav"
        filtered_filename = f"filtered_{cfg.filter_type}{cfg.cutoff_freq_highpass}_{os.path.splitext(os.path.basename(audio_path))[0]}.wav" # Example naming

        original_save_path = os.path.join(cfg.output_dir, original_filename)
        filtered_save_path = os.path.join(cfg.output_dir, filtered_filename)

        print(f"Saving original audio to: {original_save_path}")
        sf.write(original_save_path, y, sr)

        print(f"Saving filtered audio to: {filtered_save_path}")
        sf.write(filtered_save_path, y_filtered, sr)

        # --- Plot Spectrograms for Comparison ---
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

        plot_spectrogram(y, sr, "Original Spectrogram", axs[0])
        img = plot_spectrogram(y_filtered, sr, f"Filtered Spectrogram ({cfg.filter_type} @ {cfg.cutoff_freq_highpass}Hz)", axs[1])

        fig.colorbar(img, ax=axs, format="%+2.0f dB")
        plt.tight_layout()
        plot_save_path = os.path.join(cfg.output_dir, f"spectrogram_comparison_{os.path.splitext(os.path.basename(audio_path))[0]}.png")
        plt.savefig(plot_save_path)
        print(f"Saved spectrogram comparison plot to: {plot_save_path}")
        plt.show()

        print("\nFinished. Please listen to the saved .wav files in:")
        print(f"{cfg.output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")





