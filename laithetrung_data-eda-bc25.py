!nvidia-smi


!pip install noisereduce pedalboard
!pip download --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly nvidia-dali-nightly-cuda120
!ls /kaggle/working |grep nvidia_dali_nightly_cuda120 |xargs pip install 
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
import os, gc, random 
import pandas as pd
import pickle
from pathlib import Path
from tqdm.notebook import tqdm
import IPython.display as ipd
from IPython.display import display, clear_output
import ipywidgets as widgets
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
from pedalboard import Pedalboard, Gain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali

!rm -rf /kaggle/working/
clear_output()
print("Install and Import DONE")


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
    DATA_PATH=Path("/kaggle/input/birdclef-2025/train_audio"),
    OUTPUT_FOLDER =Path("/kaggle/working/"),
    COLOR_MAP =['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    OUTPUT_METADATA="spec_img_meta.csv",
    WINDOW="hann",
    NFILTER_MEL=128,
    WINDOW_LENGTH= 1024,
    WINDOW_STEP= 512,
    FREQ_HIGH=14000,
    TARGET_DURATION_S = 5,
    TARGET_SAMPLES = 5*32000,
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
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
print(cfg.__dict__)
# Device check
print(f"Using device: {cfg.DEVICE}")


# Taking a closer look at the meta data
metadata_path = Path("/kaggle/input/birdclef-2025/train.csv")

if metadata_path.exists():
    train_df = pd.read_csv(metadata_path)
    train_df = train_df.drop(columns=['url', 'license', 'common_name','collection','author','type'])
    print(train_df.head(15))
    print("\nMetadata Columns:", train_df.columns)
    print("\nTraining Samples:", len(train_df))
    print("\nUnique Species:", train_df['primary_label'].nunique())
    print("\nSecondary Species Labels(Recordist Marked):", train_df['secondary_labels'].nunique())
    # Key distributions
    print("\nSpecies distribution (top 10):")
    print(train_df['primary_label'].value_counts().head(10))
else:
    print(f"Metadata file not found at {meta_datapath}. Check path!")


train_df.describe(include=[np.number])


scientific_name_counts = train_df['scientific_name'].value_counts()
print(scientific_name_counts)
overlap_map = {name: 0.8 if count < 50 else 0.2 for name, count in scientific_name_counts.items()}
print("dataframe mới")
train_df['overlap-percent'] = train_df['scientific_name'].map(overlap_map)
print(train_df.head())



print("Analyzing audio durations...")
durations = []
pbar = tqdm(train_df['filename'].tolist(), desc="Calculating durations")
for filename in pbar:
    file_path = cfg.DATA_PATH/filename
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


train_df["duration"]= durations
train_df.head(5)


duration_by_class = train_df.groupby('primary_label')['duration'].sum()
duration_by_class = duration_by_class.reset_index().sort_values(by=['duration']).reset_index()
# Hiển thị dữ liệu tổng hợp (có thể in ra để kiểm tra)
print(duration_by_class)

# Vẽ biểu đồ cột:
plt.figure(figsize=(10, 6))
plt.bar(duration_by_class['primary_label'].astype(str), duration_by_class['duration'], color='skyblue')
plt.xlabel('Primary Label')
plt.ylabel('total time (second)')
plt.title('Total Time Per Class (primary_label)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================
# 1. Hàm load và tiền xử lý audio
# ============================
def load_and_preprocess(audio_path, sr):
    """
    - Load file .ogg bằng librosa.
    - Giảm nhiễu bằng noisereduce.
    - Tăng âm lượng bằng pedalboard.
    """
    samples, sr = librosa.load(audio_path, sr=sr)
    samples_nr = nr.reduce_noise(y=samples, sr=sr)
    board = Pedalboard([Gain(gain_db=10)])
    samples_proc = board(samples_nr, sr)
    return samples_proc, sr

# ============================
# 2. Hàm tạo sliding window (5 giây, bước nhảy 0.5 giây)
# ============================
def get_sliding_windows(audio, sr, segment_duration=5.0, overlap_percent=0.5):
    """
    Trả về danh sách các tuple (start_sample, end_sample) cho mỗi window.
    """
    step = segment_duration*(1-overlap_percent)
    seg_samples = int(segment_duration * sr)
    step_samples = int(step * sr)
    windows = []
    for start in range(0, len(audio) - seg_samples + 1, step_samples):
        end = start + seg_samples
        windows.append((start, end))
    return windows
    
# ============================
# 3. Hàm loại bỏ đoạn có tiếng người và padding
# ============================
def remove_human_voice(segment_audio, sr, window_start, human_voice_intervals, seg_duration=5.0):
    """
    - segment_audio: mảng audio của 1 window (5 giây)
    - window_start: thời gian bắt đầu (giây) của window trong file gốc.
    - human_voice_intervals: danh sách dict {'start': ..., 'end': ...} (giây) đã phát hiện tiếng người trong file gốc.
    
    Hàm này sẽ loại bỏ (cut) các đoạn trùng với tiếng người, nối các đoạn còn lại lại,
    rồi thêm padding (âm im) ở 2 đầu để độ dài về 5 giây.
    """
    seg_len = len(segment_audio)
    
    # Lọc các interval có giao với window hiện tại
    intervals = [iv for iv in human_voice_intervals 
                 if iv['start'] < window_start + seg_duration and iv['end'] > window_start]
    intervals = sorted(intervals, key=lambda x: x['start'])
    
    # Nếu không có tiếng người trong window -> return segment ban đầu
    if not intervals:
        return segment_audio
    
    nonhuman_parts = []
    current_sample = 0
    # Xử lý các interval theo thứ tự tăng dần
    for iv in intervals:
        # Chuyển đổi thời gian của interval về thời gian tương đối trong window
        start_relative = max(iv['start'] - window_start, 0)
        end_relative = min(iv['end'] - window_start, seg_duration)
        start_idx = int(start_relative * sr)
        end_idx = int(end_relative * sr)
        # Nếu có đoạn không có tiếng người trước interval hiện tại, lấy nó
        if start_idx > current_sample:
            nonhuman_parts.append(segment_audio[current_sample:start_idx])
        # Cập nhật vị trí hiện tại (bỏ qua phần có tiếng người)
        current_sample = end_idx
    # Lấy phần sau interval cuối cùng
    if current_sample < seg_len:
        nonhuman_parts.append(np.zeros(seg_len -current_sample))
        nonhuman_parts.append(segment_audio[current_sample:])
    
    if nonhuman_parts:
        nonhuman_audio = np.concatenate(nonhuman_parts)
    else:
        nonhuman_audio = np.array([], dtype=segment_audio.dtype)
    
    # Padding thêm silence để có đủ số mẫu ban đầu (5 giây)
    desired_length = seg_len
    current_length = len(nonhuman_audio)
    if current_length < desired_length:
        pad_total = desired_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        nonhuman_audio = np.concatenate([
            np.zeros(pad_left, dtype=segment_audio.dtype), 
            nonhuman_audio, 
            np.zeros(pad_right, dtype=segment_audio.dtype)
        ])
    else:
        nonhuman_audio = nonhuman_audio[:desired_length]
    
    return nonhuman_audio

# ============================
# 4. Lưu ảnh Mel Spectrogram
# ============================
from PIL import Image
import matplotlib.cm as cm


def save_spectrogram_image(mel_spec, cfg, output_dir, output_filename, output_size=(1280, 720)):
    # Normalize mel spectrogram to 0–1
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
    # List of colormaps
    colormaps = cfg.COLOR_MAP

    for cmap_name in colormaps:
        out_path =  cfg.OUTPUT_FOLDER /cmap_name /output_dir
        out_path.mkdir(parents=True, exist_ok=True)
        output_file = out_path / output_filename
        
        cmap = cm.get_cmap(cmap_name)
        rgba_img = cmap(mel_spec)  # shape: (H, W, 4), values in [0, 1]
        rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb_img)

        # Flip vertically (to match origin='lower' behavior)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_resized = img.resize(output_size, Image.LANCZOS)

        # Save final image
        img_resized.save(output_file)


def show_spectrogram(spec, title, sr, hop_length, y_axis="log", x_axis="time"):
    librosa.display.specshow(
        spec, sr=sr, y_axis=y_axis, x_axis=x_axis, hop_length=hop_length
    )
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


#======================================================================================
# ********************************Test output 1 file***********************************
#=====================================================================================
# y, sr = librosa.load("/kaggle/input/birdclef-2025/train_audio/1139490/CSA36385.ogg")
# y = y[:32000*5]
y,sr = load_and_preprocess("/kaggle/input/birdclef-2025/train_audio/1139490/CSA36385.ogg", 32000)
y = y[32000*0:32000*5]
audio_data = np.array(y, dtype=np.float32)

@pipeline_def
def mel_spectrogram_pipe(nfft, window_length, window_step,sample_rate,nfilter,freq_high,device="gpu"):
    audio = types.Constant(device=device, value=audio_data)
    spectrogram = fn.spectrogram(
        audio,
        device=device,
        nfft=nfft,
        window_length=window_length,
        window_step=window_step,
    )
    mel_spectrogram = fn.mel_filter_bank(
        spectrogram, sample_rate=sr, nfilter=nfilter, freq_high=freq_high
    )
    mel_spectrogram_dB = fn.to_decibels(
        mel_spectrogram, multiplier=10.0, cutoff_db=-80
    )
    return mel_spectrogram_dB
    
pipe = mel_spectrogram_pipe(
    device="gpu",
    batch_size=1,
    num_threads=3,
    device_id=0,
    nfft=cfg.WINDOW_LENGTH,
    window_length=cfg.WINDOW_LENGTH,
    window_step=cfg.WINDOW_STEP,
    sample_rate=cfg.SAMPLE_RATE,
    nfilter=cfg.NFILTER_MEL,
    freq_high=cfg.FREQ_HIGH,
)
pipe.build()
outputs = pipe.run()
mel_spectrogram_dali_db = np.array(outputs[0][0].as_cpu())

plt.imshow(mel_spectrogram_dali_db)

print(mel_spectrogram_dali_db.shape)




# -------------------------
# 4. DALI Pipeline: Tạo Mel Spectrogram
# -------------------------

def compute_mel_spectrogram(audio_segment, cfg):
    audio_data = np.array(audio_segment, dtype=np.float32)
    @pipeline_def
    def mel_spectrogram_pipe(nfft, window_length, window_step,sample_rate,nfilter,freq_high,device="gpu"):
        audio = types.Constant(device=device, value=audio_data)
        spectrogram = fn.spectrogram(
            audio,
            device=device,
            nfft=nfft,
            window_length=window_length,
            window_step=window_step,
        )
        mel_spectrogram = fn.mel_filter_bank(
            spectrogram, sample_rate=sr, nfilter=nfilter, freq_high=freq_high
        )
        mel_spectrogram_dB = fn.to_decibels(
            mel_spectrogram, multiplier=10.0, cutoff_db=-80
        )
        return mel_spectrogram_dB
        
    pipe = mel_spectrogram_pipe(
        device="gpu",
        batch_size=1,
        num_threads=3,
        device_id=0,
        nfft=cfg.WINDOW_LENGTH,
        window_length=cfg.WINDOW_LENGTH,
        window_step=cfg.WINDOW_STEP,
        sample_rate=cfg.SAMPLE_RATE,
        nfilter=cfg.NFILTER_MEL,
        freq_high=cfg.FREQ_HIGH,
    )
    pipe.build()
    outputs = pipe.run()
    mel_spectrogram_dali_db = np.array(outputs[0][0].as_cpu())

    return mel_spectrogram_dali_db



def process_audio_file(audio_path, human_voice_dict, output_folder, cfg, overlap_percent):
    samples_proc, sr = load_and_preprocess(audio_path, cfg.SAMPLE_RATE)
    # Tạo sliding windows, mỗi window TARGET_DURATION_S giây, bước 0.5 giây
    windows = get_sliding_windows(samples_proc, sr, cfg.TARGET_DURATION_S, overlap_percent = overlap_percent)
    
    # Lấy human voice intervals nếu tồn tại cho file này (dùng đường dẫn tương đối so với DATA_PATH)
    human_intervals = human_voice_dict.get(audio_path, [])
    for start_sample, end_sample in windows:
        window_start_time = start_sample / sr
        segment_audio = samples_proc[start_sample:end_sample]
        # Loại bỏ tiếng người và padding lại thành đủ TARGET_DURATION_S giây
        segment_clean = remove_human_voice(segment_audio, sr, window_start_time, human_intervals, cfg.TARGET_DURATION_S)
        # Tính Mel Spectrogram sử dụng NVIDIA DALI
        mel_spec = compute_mel_spectrogram(segment_clean, cfg)
        # Tạo tên file: AudioFileName_start_end.jpg
        base_name = os.path.splitext(Path(audio_path).name)[0]
        output_filename = f"{base_name}_{window_start_time:.1f}_{(window_start_time+cfg.TARGET_DURATION_S):.1f}.jpg"
        #Save into virisdis format
        save_spectrogram_image(mel_spec, cfg, output_folder, output_filename, output_size=(1280, 720))
        # save_spectrogram_image(mel_spec, str(output_file), cmap='gray')
        print(f"Saved {output_filename}")



with open("/kaggle/input/bc25-human-detect-sound/train_voice_data.pkl", "rb") as f:
    human_voice_dict = pickle.load(f)
        
    # Lặp qua từng dòng trong DataFrame và xử lý file audio
os.makedirs(cfg.OUTPUT_FOLDER, exist_ok=True)
for cmap_name in cfg.COLOR_MAP:
    path =  cfg.OUTPUT_FOLDER /cmap_name 
    path.mkdir(parents=True, exist_ok=True)
for idx, row in train_df.iterrows():
    output_folder =  row['primary_label']
    os.makedirs(output_folder, exist_ok=True)
    rel_path = row['filename']  # đường dẫn tương đối
    overlap_percent = row['overlap-percent']
    audio_path = cfg.DATA_PATH / rel_path
    print(f"Processing {audio_path} ...")
    # try:
    process_audio_file(str(audio_path), human_voice_dict, output_folder, cfg, overlap_percent)
    break
    # except Exception as e:
    #     print(f"Error processing {audio_path}: {e}")


from PIL import Image
display(Image.open("/kaggle/working/plasma/1139490/CSA36385_0._5.0.jpg"))




