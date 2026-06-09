!pip install pyrootutils
!pip install lightning
!pip install torch-audiomentations
!pip install noisereduce pedalboard
!pip install nvidia-dali-cuda120
# !pip download --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly nvidia-dali-nightly-cuda120
# !ls /kaggle/working |grep nvidia_dali_nightly_cuda120 |xargs pip install 
# !rm -rf /kaggle/working/*
import sys
from datasets import load_dataset
import os
import numpy as np
import pandas as pd
import pickle
import ast
import math
import time
import random
import gc
import cv2
from pathlib import Path
from tqdm.notebook import tqdm
import librosa
import matplotlib.pyplot as plt

import torch
import torchaudio
from torchaudio import transforms

import IPython.display as ipd
import sys
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator
ipd.clear_output()
print("Import finished")


from pathlib import Path
class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# Initialize and set basic configuration
cfg = Config(
    SEED=42, 
    SAMPLE_RATE=32000,
    META_DATA_PATH = Path("/kaggle/input/birdclef-2025/train.csv"),
    DATA_PATH=Path("/kaggle/input/birdclef-2025/train_audio"),
    NOISE_DATA_PATH=Path("/kaggle/input/esc50-environmental-for-aumentationmissing-class/no_call_augmented"),
    OUTPUT_FOLDER =Path("/kaggle/working/256_2048/train_Mel_spec"),
    HUMAN_VOICE_PKL = Path('/kaggle/input/bc25-human-detect-sound/train_voice_data.pkl'),
    MEL_COMBINATION = Path('/kaggle/input/data-augmentation-part4/mel_combination.csv'),
    CLASS_NAME = np.load('/kaggle/input/metadate-bc25/class_names.npy', allow_pickle=True).tolist(),
    NUM_CLASSES = 206,
    CLASS2IDX={},
    MEL_COMBINATION_INDEX = 16, # domain [0,15]
    COLOR_MAP =['inferno'],
    OUTPUT_METADATA="/kaggle/working/spec_img_meta.csv",
    WINDOW="hann",
    NFILTER_MEL=128,
    WINDOW_LENGTH= 1024,
    WINDOW_STEP= 512,
    FREQ_HIGH=14000,
    FREQ_LOW=150,
    CUT_OFF_DB= -80,
    TARGET_DURATION_S = 5,
    TARGET_SAMPLES = 5*32000,
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    BATCH_SIZE = 32,
    NUM_WORKERS = 32,
    )
cfg.CLASS2IDX = {cls: idx for idx, cls in enumerate(cfg.CLASS_NAME)}
# Function to seed everything to ensure reproducibility
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Change to true if input sizes are kept constant

seed_everything(cfg.SEED)
# Verifying changes
# Device check
print(f"Using device: {cfg.DEVICE}")


# ============================
# 1. HÃ m load vÃ  tiá»�n xá»­ lÃ½ audio
# ============================
def load_and_preprocess(audio_path, sr):
    """
    - Load file .ogg báº±ng librosa.
    """
    samples, sr = librosa.load(audio_path, sr=sr)
    samples = samples / samples.abs().max()
    return samples, sr

# ============================
# 2. HÃ m save Mel_specs thÃ nh file .npz
# ============================

def save_mel_to_npz(mel_spectrogram: np.ndarray, output_path: str):
    """
    Parameters:
        mel_spectrogram (np.ndarray): 2D array of shape (n_mels, time_frames), dtype=float32/float64.
        output_path (str): Path to save the .npz file (should end with .npz).
    """
    if not isinstance(mel_spectrogram, np.ndarray):
        raise ValueError("mel_spectrogram must be a NumPy array")

    if mel_spectrogram.ndim != 2:
        raise ValueError("mel_spectrogram must be a 2D array (n_mels, time_frames)")

    # Convert to float16 if not already
    mel_spectrogram = mel_spectrogram.astype(np.float16)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to compressed .npz
    np.savez_compressed(output_path, mel=mel_spectrogram)


# ============================
# 3. HÃ m show Mel_spec
# ============================

def load_and_plot_mel_npz(npz_path: Path, title: str = "Mel Spectrogram"):
    """
    Load a Mel spectrogram from a .npz file and plot it.

    Parameters:
        npz_path (Path or str): Path to the .npz file.
        title (str): Plot title.
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")
    
    data = np.load(npz_path)
    if "mel" not in data:
        raise KeyError(f"'mel' key not found in {npz_path}")
    
    mel = data["mel"]

    if mel.ndim != 2:
        raise ValueError("Loaded mel spectrogram must be 2D")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bands")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.show()


#=============================
# 4. Metadata function handler
# ====================
def to_idx_list(lbl_list: str, class2idx: dict[int,int]) -> list[int]:
    """
    Convert a string representation of a list of labels (e.g. "['123','456']")
    into a list of integer indices using class2idx mapping.
    """
    labels = ast.literal_eval(lbl_list)
    return [class2idx[l] for l in labels if l and l in class2idx]

def load_train_df(cfg, rating_threshold: float = 0.0) -> pd.DataFrame | None:
    """
    Load metadata CSV, filter by rating, and prepare:
      - 'label' (int)
      - 'secondary_label_idx' (list[int])
      - 'ogg_path' (str)
      - 'npz_path' (str)
    """
    meta_path = Path(cfg.META_DATA_PATH)
    if not meta_path.exists():
        return None

    # 1) Load and filter
    df = pd.read_csv(meta_path)
    train_df = df[df['rating'] > rating_threshold].copy()

    # 2) Drop unused cols (if present)
    drop_cols = [
        'url','license','common_name','collection','author',
        'type','latitude','longitude','scientific_name'
    ]
    train_df.drop(columns=[c for c in drop_cols if c in train_df], inplace=True)

    # 3) Primary label â†’ integer
    train_df['label'] = train_df['primary_label'].map(cfg.CLASS2IDX)

    # 4) Secondary labels â†’ list of ints
    train_df['secondary_label_idx'] = train_df['secondary_labels']\
        .apply(lambda s: to_idx_list(s, cfg.CLASS2IDX))

    # 5) /kaggle/input/birdclef-2025/train_audio/XXX.ogg
    train_df['ogg_path'] = train_df['filename'].apply(
        lambda fn: str(Path("/kaggle/input/birdclef-2025/train_audio") / fn)
    )

    # 7) Summary
    print("\nColumns:", train_df.columns.tolist())
    print("Total species:", df['primary_label'].nunique())
    print("Filtered species:", train_df['primary_label'].nunique())
    print("Secondary-label counts:", train_df['secondary_labels'].nunique())
    print("Top-10 distribution:\n", train_df['primary_label'].value_counts().head(10))

    return train_df
    
def load_human_voice(pkl_path):
    with open(pkl_path, "rb") as f:
        human_voice_dict = pickle.load(f)
    return human_voice_dict



#=========================================
# STEP 1 Remove human voice and silence segment  
#=========================================

from pydub import AudioSegment
from pydub.silence import split_on_silence
from torch.utils.data import Dataset, DataLoader

# ======== PROCESS FUNCTION ========
    
def remove_human_voice(samples, sr, human_voice_intervals, pad_duration=0.5, save_human_call_data=False):
    """
    Cáº¯t bá»� cÃ¡c Ä‘oáº¡n audio trÃ¹ng vá»›i tiáº¿ng ngÆ°á»�i vÃ  ná»‘i láº¡i cÃ¡c Ä‘oáº¡n cÃ²n láº¡i.
    THAY Ä�á»”I SO Vá»šI BAN Ä�áº¦U: thÃªm offset 0.5s á»Ÿ start vÃ  end time.
    Args:
        samples (np.ndarray): máº£ng 1 chiá»�u waveform.
        sr (int): táº§n sá»‘ láº¥y máº«u (sample rate).
        human_voice_intervals (list of dict): [{'start': float, 'end': float}], Ä‘Æ¡n vá»‹ giÃ¢y.
        pad_duration (float): thá»�i gian (giÃ¢y) chÃ¨n padding (im láº·ng) giá»¯a cÃ¡c Ä‘oáº¡n.
        save_human_call_data (bool): náº¿u True, sáº½ lÆ°u cÃ¡c Ä‘oáº¡n chá»©a tiáº¿ng ngÆ°á»�i vÃ o folder `human_call_data`.

    Returns:
        np.ndarray: máº£ng audio má»›i Ä‘Ã£ loáº¡i bá»� tiáº¿ng ngÆ°á»�i.
    """
    total_duration = len(samples) / sr
    result = []

    # Sort intervals by start time
    intervals = sorted(human_voice_intervals, key=lambda x: x['start'])
    current_pos = 0

    pad_samples = int(pad_duration * sr)
    silence_pad = np.zeros(pad_samples, dtype=samples.dtype)

    # Prepare folder to save clips of human voice if needed
    if save_human_call_data:
        out_dir = Path("human_call_data")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, interval in enumerate(intervals):
        start_sample = int(max(0,interval['start'] - 0.5) * sr) #offset 0.5 s
        end_sample = int(min(total_duration, interval['end'] + 0.5) * sr)  #offset 0.5 s

        # LÆ°u Ä‘oáº¡n cÃ³ tiáº¿ng ngÆ°á»�i
        if save_human_call_data:
            segment = samples[start_sample:end_sample]
            filename = out_dir / f"human_{timestamp}_part{idx}_from_{interval['start']:.2f}s_to_{interval['end']:.2f}s.wav"
            sf.write(filename, segment, sr)

        # Láº¥y pháº§n trÆ°á»›c Ä‘oáº¡n tiáº¿ng ngÆ°á»�i
        if start_sample > current_pos:
            segment = samples[current_pos:start_sample]
            result.append(segment)
            if pad_duration > 0:
                result.append(silence_pad)

        current_pos = max(current_pos, end_sample)

    # Láº¥y pháº§n cuá»‘i sau Ä‘oáº¡n tiáº¿ng ngÆ°á»�i
    if current_pos < len(samples):
        result.append(samples[current_pos:])

    if result:
        output = np.concatenate(result)
        if len(output) > sr:
            return output
    return samples  # fallback: toÃ n bá»™ lÃ  tiáº¿ng ngÆ°á»�i thÃ¬ giá»¯ nguyÃªn samples
    
from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silience(samples, sr):
    """
    Cáº¯t bá»� nhá»¯ng pháº§n Ã¢m thanh yÃªn láº·ng cÃ³ Ä‘á»™ dÃ i lá»›n hÆ¡n 1 giÃ¢y
    """
    try:
        # Convert to int16 for AudioSegment
        samples_int16 = (samples * 32767).astype(np.int16)

        audio = AudioSegment(
            samples_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,  # int16 = 2 bytes
            channels=1
        )

        # Split on silence
        audio_chunks = split_on_silence(audio,
                                        min_silence_len=1000,
                                        silence_thresh=-45,
                                        keep_silence=100)

        # Reconstruct
        if not audio_chunks:
            combined = audio  # fallback
        else:
            combined = AudioSegment.empty()
            for chunk in audio_chunks:
                combined += chunk

        out_samples = np.array(combined.get_array_of_samples()).astype(np.float32)
        
        if len(out_samples) < sr:
            out_samples = samples_int16
        return out_samples / 32767.0

    except Exception as e:
        print(f"â�Œ Error in remove_silience: {e}")
        return samples  # fallback: tráº£ láº¡i input gá»‘c


#=========================================
# STEP 2 no_call_augmented folder containing 
#=========================================
def save_esc50_dataset(saved_path):
    # Sáº½ cÃ³ tá»•ng cá»™ng 800 file audio 5s liÃªn quan tá»›i Ã¢m thanh vá»� thiÃªn nhiÃªn hoáº·c Ä‘á»“ váº­t Ä‘Æ°á»£c lÆ°u láº¡i
    # 1) Ä�á»�c metadata vÃ  Ä‘á»‹nh nghÄ©a danh sÃ¡ch cÃ¡c category cáº§n augment
    esc50 = pd.read_csv("/kaggle/input/environmental-sound-classification-50/esc50.csv")
    augmented_category = [
        'chainsaw', 'vacuum_cleaner', 'door_wood_knock', 'can_opening', 'crow', 'clapping',
        'pouring_water', 'water_drops', 'church_bells', 'keyboard_typing', 'wind', 'footsteps',
        'brushing_teeth', 'crackling_fire', 'drinking_sipping', 'snoring', 'washing_machine',
        'clock_tick', 'door_wood_creaks', 'sea_waves'
    ]
    
    # 2) ThÆ° má»¥c gá»‘c chá»©a file .wav
    src_root = Path("/kaggle/input/environmental-sound-classification-50/audio/audio/44100")
    
    # 3) ThÆ° má»¥c Ä‘Ã­ch Ä‘á»ƒ lÆ°u cÃ¡c file Ä‘Ã£ chá»�n
    dest_root = Path("/kaggle/working/no_call_augmented/noise")
    dest_root.mkdir(parents=True, exist_ok=True)
    
    # 4) Lá»�c DataFrame
    df_sel = esc50[esc50["category"].isin(augmented_category)]
    
    # 5) Copy tá»«ng file
    for _, row in df_sel.iterrows():
        filename = row["filename"]          # vÃ­ dá»¥ "1-100032-A-0.wav"
        cat      = row["category"]
        
        src_path  = src_root / filename
        dest_path = dest_root / filename
    
        # copy file
        if src_path.exists():
            shutil.copy(src_path, dest_path)
        else:
            print(f"WARNING: khÃ´ng tÃ¬m tháº¥y {src_path}")
    
    print("Done saving unrelevant data from esc50 process!")

#===================================
# STEP 3 AUGMENTATION 
#====================================
from torch_audiomentations import Compose, Gain, PolarityInversion, AddBackgroundNoise, AddColoredNoise

def build_torch_audio_augment_pipeline(
    background_noise_path,
    sample_rate=32000,
    min_snr_db=5,
    max_snr_db=20
):
    augment = Compose(
        transforms=[
            Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.5, output_type="tensor"),
            PolarityInversion(p=0.3, output_type="tensor"),
            AddBackgroundNoise(
                background_paths=background_noise_path,
                sample_rate=sample_rate,
                min_snr_in_db=min_snr_db,
                max_snr_in_db=max_snr_db,
                p=0.7,
                output_type="tensor"
            ),
            AddColoredNoise(
                min_snr_in_db=min_snr_db,
                max_snr_in_db=max_snr_db,
                min_f_decay=-2.0,
                max_f_decay=2.0,
                sample_rate=sample_rate,
                p=0.5,
                output_type="tensor"
            )
        ],
        output_type="tensor"  # Cáº¥u hÃ¬nh cho toÃ n bá»™ pipeline
    )
    print("Finished build augmentation pipeline")
    return augment

# ============================
# STEP 4. HÃ m Táº¡o Mel Spectrogram
# ============================
import torchaudio
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

    
# Khá»Ÿi táº¡o bá»™ chuyá»ƒn Ä‘á»•i (chá»‰ cáº§n táº¡o 1 láº§n)
def compute_mel_spectrogram(audio_segment, cfg):
    audio_data = np.array(audio_segment, dtype=np.float32)

    @pipeline_def
    def mel_spectrogram_pipe(nfft, window_length, window_step, sample_rate, nfilter, freq_high, freq_low, device="gpu"):
        audio = types.Constant(device=device, value=audio_data)
        spectrogram = fn.spectrogram(
            audio,
            device=device,
            nfft=nfft,
            window_length=window_length,
            window_step=window_step,
        )
        mel_spectrogram = fn.mel_filter_bank(
            spectrogram,
            device=device,
            sample_rate=sample_rate,
            nfilter=nfilter,
            freq_high=freq_high,
            freq_low=freq_low
        )
        mel_spectrogram_dB = fn.to_decibels(
            mel_spectrogram,
            device=device,
            multiplier=10.0,
            cutoff_db=-80
        )
        return mel_spectrogram_dB

    pipe = mel_spectrogram_pipe(
        device="gpu",
        batch_size=1,
        num_threads=1,
        nfft=cfg.WINDOW_LENGTH,
        window_length=cfg.WINDOW_LENGTH,
        window_step=cfg.WINDOW_STEP,
        sample_rate=cfg.SAMPLE_RATE,
        nfilter=cfg.NFILTER_MEL,
        freq_high=cfg.FREQ_HIGH,
        freq_low=cfg.FREQ_LOW,
    )

    pipe.build()
    outputs = pipe.run()
    mel_spectrogram_dali_db = np.array(outputs[0][0].as_cpu())  
    return mel_spectrogram_dali_db

def compute_mel_spectrogram_pytorch(cfg):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.SAMPLE_RATE,
        n_fft=cfg.WINDOW_LENGTH,
        win_length=cfg.WINDOW_LENGTH,
        hop_length=cfg.WINDOW_STEP,
        n_mels=cfg.NFILTER_MEL,
        f_min=cfg.FREQ_LOW,
        f_max=cfg.FREQ_HIGH,
        power = 2.0,
        
    ).to(cfg.DEVICE)
    
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=-cfg.CUT_OFF_DB).to(cfg.DEVICE)
    print("Finished build mel spectrograme transform pipeline")
    return mel_transform, db_transform  

#==============================
# STEP 5: Full pipeline
#==============================
import numpy as np
import torchaudio
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torchaudio
from pathlib import Path
from tqdm import tqdm

def create_new_dataset_pipe(
    cfg,
    df,
    input_dir,
    out_dir,
    sr,
    mel_transform = None,
    db_transform = None,
    augment_pipeline=None,
    human_voice_dict=None,
    reduce_human_noise=True,
    reduce_silence=True,
    use_augmentation=False
):
    """
    Xá»­ lÃ½ tá»«ng file Ã¢m thanh, tÃ­nh mel spectrogram, lÆ°u má»—i file thÃ nh .npz vÃ o Ä‘Ãºng folder theo class.

    Args:
        df (pd.DataFrame): DataFrame chá»©a Ã­t nháº¥t cá»™t 'path' hoáº·c 'ogg_path' vÃ  'label' (class)
        input_dir (str | Path): thÆ° má»¥c chá»©a file Ã¢m thanh
        out_dir (str | Path): thÆ° má»¥c gá»‘c Ä‘á»ƒ lÆ°u output (sáº½ táº¡o subfolder theo label)
        sr (int): sample rate chuáº©n hÃ³a
        mel_transform, db_transform: transform mel vÃ  dB
        augment_pipeline: torch_audiomentations pipeline
        human_voice_dict: dict chá»©a tiáº¿ng ngÆ°á»�i
        reduce_human_noise: loáº¡i bá»� tiáº¿ng ngÆ°á»�i náº¿u cÃ³
        reduce_silence: loáº¡i bá»� Ä‘oáº¡n yÃªn láº·ng náº¿u cÃ³
        use_augmentation: Ã¡p dá»¥ng augment náº¿u cÃ³
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    augment_pipeline = augment_pipeline.to(cfg.DEVICE)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="ğŸ�¼ Processing dataset"):
        rel_path = row.get("path", row.get("ogg_path", None))
        class_label = row.get("primary_label", "unknown")

        if rel_path is None or class_label is None:
            continue

        audio_path = input_dir / rel_path
        if not audio_path.exists():
            print(f"âš ï¸� Not found: {audio_path}")
            continue

        try:
            waveform, orig_sr = torchaudio.load(audio_path)
            waveform = waveform.mean(dim=0, keepdim=True)
            waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=sr)
            waveform = waveform / waveform.abs().max()
    
            samples = waveform.squeeze().cpu().numpy()
            if len(samples) > 600*sr:
                samples = samples[:600*sr]
                
            # Remove human voice
            
            if reduce_human_noise and human_voice_dict:
                intervals = human_voice_dict.get(str(audio_path), [])
                samples = remove_human_voice(samples, sr, intervals)
                
            # Remove silence
            if reduce_silence:
                samples = remove_silience(samples, sr)            
    
            # Apply augmentation
            waveform = torch.tensor(samples, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0).unsqueeze(0)
            if use_augmentation and augment_pipeline:
                with torch.no_grad():
                    waveform = augment_pipeline(waveform)
            
            # TÃ­nh Mel spectrogram
            if mel_transform and db_transform:
                mel = mel_transform(waveform)
                mel_db = db_transform(mel).squeeze().cpu().numpy()
            else:
                waveform = waveform.squeeze().cpu().numpy()
                mel_db = compute_mel_spectrogram(waveform,cfg)
            # mel_db = compute_mel_dali(waveform, mel_pipe)
            mel_db = mel_db.astype(np.float16)
    
            # LÆ°u vÃ o Ä‘Ãºng folder class
            out_class_dir = out_dir / str(class_label)
            out_class_dir.mkdir(parents=True, exist_ok=True)
            add_ons_name = ''
            if reduce_human_noise:
                add_ons_name += "_rh"
            if reduce_silence:
                add_ons_name += "_rs"
            if use_augmentation:
                add_ons_name += "_au"
            fname = Path(rel_path).stem + add_ons_name + ".npz"
            out_path = out_class_dir / fname
            np.savez_compressed(out_path, mel=mel_db)
            
            # Step 6: Clear memory
            del waveform, mel_db, samples
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"â�Œ Error processing {audio_path}: {e}")




# save_esc50_dataset(cfg.NOISE_DATA_PATH)
augment_transform = build_torch_audio_augment_pipeline(background_noise_path = str(cfg.NOISE_DATA_PATH), 
                                                        sample_rate=32000,
                                                        min_snr_db=2,
                                                        max_snr_db=15)
mel_transform, db_transform = compute_mel_spectrogram_pytorch(cfg)
import time
start = time.time()
create_new_dataset_pipe(
    cfg,
    df=load_train_df(cfg),
    input_dir=cfg.DATA_PATH,
    out_dir=cfg.OUTPUT_FOLDER,
    sr=cfg.SAMPLE_RATE,
    augment_pipeline=augment_transform,
    human_voice_dict=load_human_voice(cfg.HUMAN_VOICE_PKL),
    reduce_human_noise=True,
    reduce_silence=True,
    use_augmentation=True
)
print("total time using dali =", time.time() - start)


# start = time.time()
# create_new_dataset_pipe(
#     cfg,
#     df=load_train_df(cfg).iloc[:10],
#     input_dir=cfg.DATA_PATH,
#     out_dir=cfg.OUTPUT_FOLDER,
#     sr=cfg.SAMPLE_RATE,
#     augment_pipeline=augment_transform,
#     mel_transform=mel_transform,
#     db_transform=db_transform,
#     human_voice_dict=load_human_voice(cfg.HUMAN_VOICE_PKL),
#     reduce_human_noise=True,
#     reduce_silence=True,
#     use_augmentation=False
# )
# print("total time use torch=", time.time() - start)







