!pip install pyrootutils
# !pip install lightning
!pip install torch-audiomentations
!pip install noisereduce pedalboard
!pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120


from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
import os
import torch
import pandas as pd
import pickle
from typing import Union
import ast
import soundfile as sf
from datetime import datetime
import random



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
    OUTPUT_FOLDER =Path("train_Mel_spec"),
    HUMAN_VOICE_PKL = Path('/kaggle/input/bc25-human-detect-sound/train_voice_data.pkl'),
    CLASS_NAME = np.load('/kaggle/input/metadate-bc25/class_names.npy', allow_pickle=True).tolist(),
    NUM_CLASSES = 206,
    CLASS2IDX={},
    MEL_COMBINATION_INDEX = 16, # domain [0,15]
    COLOR_MAP =['inferno'],
    OUTPUT_METADATA="spec_img_meta.csv",
    WINDOW="hann",
    NFILTER_MEL=256,
    WINDOW_LENGTH= 2048,
    WINDOW_STEP= 1024,
    FREQ_HIGH=14000,
    FREQ_LOW=100,
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
print(f"Using device: {cfg.DEVICE}")





#=============================
# 4. Metadata function handler
# ====================
from typing import Dict, List

def to_idx_list(lbl_list: str, class2idx: Dict[int, int]) -> List[int]:
    """
    Convert a string representation of a list of labels (e.g. "['123','456']")
    into a list of integer indices using class2idx mapping.
    """
    labels = ast.literal_eval(lbl_list)
    return [class2idx[l] for l in labels if l and l in class2idx]



def load_train_df(cfg, rating_threshold: float = 0.0) -> Union[pd.DataFrame, None]:

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
    train_df = df[df['rating'] >= rating_threshold].copy()

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
        lambda fn: str(cfg.DATA_PATH / fn)
    )
    #6) drop non-existing files
    # train_df = train_df[train_df['ogg_path'].apply(lambda x: Path(x).exists())]
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





# ======== PROCESS FUNCTION ========
    
def remove_human_voice(samples, sr, human_voice_intervals, pad_duration=0.5, save_human_call_data=False):
    """
    XÃ³a bá»� tiáº¿ng ngÆ°á»�i khá»�i tÃ­n hiá»‡u Ã¢m thanh, hoáº¡t Ä‘á»™ng vá»›i cáº£ NumPy vÃ  PyTorch.

    Args:
        samples (np.ndarray or torch.Tensor): waveform 1 chiá»�u.
        sr (int): sample rate.
        human_voice_intervals (list of dict): [{'start': float, 'end': float}], Ä‘Æ¡n vá»‹ giÃ¢y.
        pad_duration (float): thá»�i gian im láº·ng chÃ¨n giá»¯a cÃ¡c Ä‘oáº¡n.
        save_human_call_data (bool): lÆ°u tiáº¿ng ngÆ°á»�i náº¿u True.

    Returns:
        cÃ¹ng kiá»ƒu vá»›i `samples`: audio Ä‘Ã£ loáº¡i tiáº¿ng ngÆ°á»�i.
    """
    is_torch = isinstance(samples, torch.Tensor)
    if is_torch:
        samples_np = samples.cpu().numpy()
    else:
        samples_np = samples

    total_duration = len(samples_np) / sr
    result = []

    # Sort intervals
    intervals = sorted(human_voice_intervals, key=lambda x: x['start'])
    current_pos = 0
    pad_samples = int(pad_duration * sr)
    silence_pad = np.zeros(pad_samples, dtype=samples_np.dtype)

    # Táº¡o folder lÆ°u náº¿u cáº§n
    if save_human_call_data:
        out_dir = Path("human_call_data")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, interval in enumerate(intervals):
        start_sample = int(max(0, interval['start'] - 0.5) * sr)
        end_sample = int(min(total_duration, interval['end'] + 0.5) * sr)

        if save_human_call_data:
            segment = samples_np[start_sample:end_sample]
            filename = out_dir / f"human_{timestamp}_part{idx}_from_{interval['start']:.2f}s_to_{interval['end']:.2f}s.wav"
            sf.write(filename, segment, sr)

        if start_sample > current_pos:
            result.append(samples_np[current_pos:start_sample])
            if pad_duration > 0:
                result.append(silence_pad)
        current_pos = max(current_pos, end_sample)

    if current_pos < len(samples_np):
        result.append(samples_np[current_pos:])

    if result:
        output_np = np.concatenate(result)
        if len(output_np) > sr:
            return torch.from_numpy(output_np) if is_torch else output_np

    return samples  # giá»¯ nguyÃªn náº¿u toÃ n bá»™ lÃ  tiáº¿ng ngÆ°á»�i

def remove_silence(samples, sr, silence_thresh_db=-45.0, min_silence_len=1.0, keep_silence=0.1):
    """
    XÃ³a cÃ¡c Ä‘oáº¡n yÃªn láº·ng dÃ i hÆ¡n `min_silence_len` giÃ¢y.
    Há»— trá»£ cáº£ numpy vÃ  PyTorch.

    Args:
        samples (np.ndarray or torch.Tensor): waveform 1 chiá»�u.
        sr (int): sample rate.
        silence_thresh_db (float): ngÆ°á»¡ng dBFS Ä‘á»ƒ xem lÃ  yÃªn láº·ng.
        min_silence_len (float): Ä‘á»™ dÃ i tá»‘i thiá»ƒu (giÃ¢y) Ä‘á»ƒ tÃ­nh lÃ  yÃªn láº·ng.
        keep_silence (float): thá»�i lÆ°á»£ng (giÃ¢y) Ä‘Æ°á»£c giá»¯ láº¡i á»Ÿ Ä‘áº§u/cuá»‘i má»—i Ä‘oáº¡n khÃ´ng yÃªn láº·ng.

    Returns:
        cÃ¹ng kiá»ƒu vá»›i `samples`: audio Ä‘Ã£ loáº¡i Ä‘oáº¡n yÃªn láº·ng.
    """
    is_torch = isinstance(samples, torch.Tensor)
    if is_torch:
        samples_np = samples.cpu().numpy()
    else:
        samples_np = samples

    samples_np = samples_np.astype(np.float32)
    win_size = int(0.02 * sr)  # 20ms window
    hop_size = int(0.01 * sr)  # 10ms hop
    min_silence_samples = int(min_silence_len * sr)
    keep_samples = int(keep_silence * sr)

    # Compute short-time RMS
    rms = np.array([
        np.sqrt(np.mean(samples_np[i:i + win_size] ** 2))
        for i in range(0, len(samples_np) - win_size + 1, hop_size)
    ])

    # Convert to dBa
    eps = 1e-10
    rms_db = 20 * np.log10(rms + eps)
    silence_mask = rms_db < silence_thresh_db

    # Expand frame-wise mask to sample-wise
    frame_mask = np.repeat(silence_mask, hop_size)
    frame_mask = np.pad(frame_mask, (0, len(samples_np) - len(frame_mask)), constant_values=True)

    # Invert to find non-silent regions
    non_silent_mask = ~frame_mask
    change_points = np.diff(non_silent_mask.astype(np.int8), prepend=0)
    start_indices = np.where(change_points == 1)[0]
    end_indices = np.where(change_points == -1)[0]

    if len(end_indices) < len(start_indices):
        end_indices = np.append(end_indices, len(samples_np))

    result = []
    for start, end in zip(start_indices, end_indices):
        if end - start >= min_silence_samples:
            s = max(0, start - keep_samples)
            e = min(len(samples_np), end + keep_samples)
            result.append(samples_np[s:e])

    if result:
        output_np = np.concatenate(result)
        if len(output_np) > sr:
            return torch.from_numpy(output_np) if is_torch else output_np

    return samples  # fallback: giá»¯ nguyÃªn náº¿u toÃ n bá»™ lÃ  yÃªn láº·ng

#=========================
# 5. DALI pipeline
# =========================

class ExternalInputIterator:
    def __init__(self, df, cfg, reduce_human_voice=True, reduce_silence=True, shuffle =False):
        self.reduce_human_voice = reduce_human_voice
        self.reduce_silence = reduce_silence
        self.batch_size = cfg.BATCH_SIZE
        self.sample_rate = cfg.SAMPLE_RATE
        self.human_voice_dict = load_human_voice(cfg.HUMAN_VOICE_PKL)
        self.df = df
        # if shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.i = 0
        self.n = len(self.df)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n:
            self.i = 0
            raise StopIteration
        batch_audio = []
        batch_labels = []
        batch_idx = []
        # self.current_filenames = []

        for _ in range(self.batch_size):
            if self.i >= self.n:
                break
            row = self.df.iloc[self.i]
            audio_path = row['ogg_path']
            label = row['label']
            print(f"Processing {self.i+1}/{self.n}: {audio_path}")
            samples, sr = librosa.load(audio_path, sr=self.sample_rate)

            if self.batch_size == 1: # batch size 1 dÃ¹ng cho táº¡o dataset
                if len(samples) > 600 * self.sample_rate: # Giá»›i háº¡n Ä‘á»™ dÃ i tá»‘i Ä‘a lÃ  600s
                    samples = samples[:600 * self.sample_rate]

            if self.reduce_human_voice:
                remap_path = Path('/kaggle/input/birdclef-2025/train_audio/')/row['filename']
                human_voice_intervals = self.human_voice_dict.get(str(remap_path), [])
                if human_voice_intervals:
                    samples = remove_human_voice(samples, sr, human_voice_intervals)
            
            if self.reduce_silence:
                samples = remove_silence(samples, sr, silence_thresh_db=-45.0, min_silence_len=1.0, keep_silence=0.1)

            samples = samples.astype(np.float32)
            batch_audio.append(samples)
            batch_labels.append(np.array([label], dtype=np.float32))
            # self.current_filenames.append(Path(audio_path).name)
            batch_idx.append(np.array([self.i], dtype=np.int32))
            self.i += 1
        return batch_audio, batch_labels, batch_idx




class AudioDaliProcessor:
    def __init__(self, df, audio_dir, cfg, device_id=0, max_duration=10, output_dir=None, reduce_human_voice=True, reduce_silence=True):
        self.max_samples = max_duration * cfg.SAMPLE_RATE
        self.audio_dir = Path(audio_dir)
        self.cfg = cfg
        self.df = df
        self.reduce_human_voice = reduce_human_voice
        self.reduce_silence = reduce_silence
        self.batch_size = cfg.BATCH_SIZE
        self.num_threads = cfg.NUM_THREAD
        self.num_worker = cfg.NUM_WORKERS
        self.device_id = device_id
        self.output_dir = Path(output_dir) if output_dir else None

        self.pipe = self._build_pipeline()
        self.pipe.build()

        self.iterator = DALIGenericIterator(
            pipelines=self.pipe,
            output_map=["mel","label", "id"],
            auto_reset=True
        )

    @pipeline_def
    def _pipeline(self):
        audio, label, id = fn.external_source(
            source = ExternalInputIterator(
                df = self.df,
                cfg = self.cfg,
                reduce_human_voice=self.reduce_human_voice,
                reduce_silence=self.reduce_silence
            ),
            num_outputs=3,
            dtype=[types.FLOAT, types.FLOAT, types.INT32],
            batch = True,
            # parallel =True,
        )
        audio = audio.gpu()

        if self.batch_size > 1:
            audio = fn.slice(
                audio,
                start=0,
                shape=self.max_samples,
                axes=[0],
                out_of_bounds_policy="pad",
                fill_values=0.0
            )

        audio = fn.normalize(audio, device="gpu")

        spectrogram = fn.spectrogram(
            audio,
            device="gpu",
            nfft=self.cfg.WINDOW_LENGTH,
            window_length=self.cfg.WINDOW_LENGTH,
            window_step=self.cfg.WINDOW_STEP,
        )

        mel = fn.mel_filter_bank(
            spectrogram,
            device="gpu",
            sample_rate=self.cfg.SAMPLE_RATE,
            nfilter=self.cfg.NFILTER_MEL,
            freq_low=self.cfg.FREQ_LOW,
            freq_high=self.cfg.FREQ_HIGH,
        )

        mel_db = fn.to_decibels(
            mel,
            device="gpu",
            multiplier=10.0,
            cutoff_db=-80
        )

        return mel_db, label, id

    def _build_pipeline(self):
        return self._pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            # py_num_workers=self.num_worker,
            # py_start_method="spawn",
        )

    def run(self, show_first=False):
        all_mels = []
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        print("ğŸ”„ Processing batches:")
        for batch in self.iterator:
            mels = batch[0]["mel"].to("cpu").numpy()
            labels = batch[0]["label"].to("cpu").numpy()
            ids = batch[0]["id"].to("cpu").numpy()
            for mel, label, id in zip(mels, labels, ids):
                row = self.df.iloc[id[0]]
                fname = Path(row['ogg_path']).name
                label_name = self.cfg.CLASS_NAME[int(label[0])]  # láº¥y label chÃ­nh xÃ¡c tá»« df
                mel = mel.astype(np.float16)
                all_mels.append(mel)

                if self.output_dir:
                    out_path = self.output_dir / label_name / (Path(fname).stem + ".npz")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out_path, mel=mel)

                if show_first:
                    self._plot_mel(mel)
                    show_first = False

        return all_mels

    def _plot_mel(self, mel):
        plt.figure(figsize=(10, 4))
        plt.imshow(mel, aspect='auto', origin='lower', cmap='magma')
        plt.title("Mel Spectrogram")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Bands")
        plt.colorbar(label="dB")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Load metadata
    start_ = datetime.now()
    df = load_train_df(cfg, rating_threshold=0.0)
    if df is None:
        print("No metadata found.")
        exit(1)

    # Load human voice data
    human_voice_dict = load_human_voice(cfg.HUMAN_VOICE_PKL)
    # Set up configuration for DALI processing
    cfg.WINDOW_LENGTH = 2048
    cfg.WINDOW_STEP = 1024
    cfg.NFILTER_MEL = 256
    cfg.FREQ_LOW = 150
    cfg.FREQ_HIGH = 14000
    cfg.BATCH_SIZE = 1
    cfg.NUM_THREAD = 6
    # cfg.NUM_WORKERS = 4 # only for training or loading

    processor = AudioDaliProcessor(
        df,
        cfg=cfg,
        audio_dir=cfg.DATA_PATH,
        output_dir=cfg.OUTPUT_FOLDER,
        reduce_human_voice=False,
        reduce_silence=False,
    )

    mel_list = processor.run(show_first=True)
    #save thá»�i gian cháº¡y vÃ o file log.txt
    with open("log.txt", "a") as f:
        f.write(f"Start time: {start_}\n")
        f.write(f"End time: {datetime.now()}\n")
        f.write(f"Total spectrograms processed: {len(mel_list)}\n")
    print(f"âœ… Total spectrograms processed: {len(mel_list)}")
    # os.system("sudo shutdown now")




