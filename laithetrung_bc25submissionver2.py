!cp /kaggle/input/nvidia-dali-installation-package/nvidia-dali/* .
!pip install --no-index --find-links=. \
    nvidia_nvjpeg_cu12*.whl \
    nvidia_nvjpeg2k_cu12*.whl \
    nvidia_nvtiff_cu12*.whl \
    nvidia_nvimgcodec_cu12*.whl \
    packaging*.whl 
!cd /kaggle/input/nvidia-dali-installation-package/nvidia-dali && ls |grep nvidia_dali_nightly_cuda120 |xargs pip install  
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
import numpy as np
import joblib
import timm
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

# clear_output()
# print("Install and Import DONE")


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
    USE_AUDIO_AS_INPUT = True, # tiền xử lý lại từ đầu nếu set True, set False sẽ lấy đầu vào là ảnh.
    USE_SMOOTH_LABEL = True,
    # MEATA_DATA_PATH = Path('train_soundscape_data/meta_train_soundscape.csv'),
    SUBMISSION_SAMPLE_PATH = Path('/kaggle/input/birdclef-2025/sample_submission.csv'),
    DATA_PATH=Path("/kaggle/input/birdclef-2025/test_soundscapes"),
    OUTPUT_FOLDER =Path("evaluation"),
    MODEL_PATH = Path("/kaggle/input/modelbirdclef/best_epoch_rms_21_05.pth"),
    CLASS_NAME = joblib.load("/kaggle/input/modelbirdclef/label_encoder.pkl").tolist(),
    NUM_CLASSES = 206,
    COLOR_MAP ='inferno',
    SAMPLE_RATE=32000,
    WINDOW="hann",
    NFILTER_MEL=128,
    WINDOW_LENGTH= 1024,
    WINDOW_STEP= 512,
    FREQ_HIGH=14000,
    TARGET_DURATION_S = 5,
    TARGET_SAMPLES = 5*32000,
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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


from torchvision import transforms
# ============= Data transform to convert 2 tensor==============
import torch.nn.functional as F

class ResizeTensor:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, H, W] → resize → [1, 224, 224]
        x = F.interpolate(x.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        return x.squeeze(0)

class To3Channels:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, H, W] → [3, H, W]
        return x.expand(3, -1, -1)

class NormalizeTensor:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to: (x - mean) / std
        return (x - self.mean) / self.std

class TransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
# ========== TRANSFORMS ==========
tensor_transform = TransformCompose([
    ResizeTensor((224, 224)),  # Resize [1, 128, 312] → [1, 224, 224]
    To3Channels(),             # Convert [1, 224, 224] → [3, 224, 224]
    NormalizeTensor(0.5, 0.5)  # Normalize float32 tensor to [-1, 1]
])

# ============================
# 1. Hàm load và tiền xử lý audio
# ============================
def load_and_preprocess(audio_path, sr):
    """
    - Load file .ogg bằng librosa.
    # - Giảm nhiễu bằng noisereduce.
    # - Tăng âm lượng bằng pedalboard.
    """
    samples, sr = librosa.load(audio_path, sr=sr)
    # samples_nr = nr.reduce_noise(y=samples, sr=sr)
    # board = Pedalboard([Gain(gain_db=10)])
    # samples_proc = board(samples_nr, sr)
    return samples, sr

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

# -------------------------
# 3. Tạo Mel Spectrogram
# -------------------------

def compute_mel_spectrogram(audio_segment, cfg):
    audio_data = np.array(audio_segment, dtype=np.float32)

    @pipeline_def
    def mel_spectrogram_pipe(nfft, window_length, window_step, sample_rate, nfilter, freq_high, device="cpu"):
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
            freq_high=freq_high
        )
        mel_spectrogram_dB = fn.to_decibels(
            mel_spectrogram,
            device=device,
            multiplier=10.0,
            cutoff_db=-80
        )
        return mel_spectrogram_dB

    pipe = mel_spectrogram_pipe(
        device="cpu",
        batch_size=1,
        num_threads=1,
        nfft=cfg.WINDOW_LENGTH,
        window_length=cfg.WINDOW_LENGTH,
        window_step=cfg.WINDOW_STEP,
        sample_rate=cfg.SAMPLE_RATE,
        nfilter=cfg.NFILTER_MEL,
        freq_high=cfg.FREQ_HIGH,
    )

    pipe.build()
    outputs = pipe.run()
    mel_spectrogram_dali_db = np.array(outputs[0][0])  # No need for .as_cpu()

    return mel_spectrogram_dali_db
    
# ============================
 # 5. Smoothing Label
# ============================
 
def smooth_label(sub):
    cols = sub.columns[1:]
    groups = sub['row_id'].str.rsplit('_', n=1).str[0]
    groups = groups.values
    for group in np.unique(groups):
        sub_group = sub[group == groups]
        predictions = sub_group[cols].values
        new_predictions = predictions.copy()
        for i in range(1, predictions.shape[0]-1):
            new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
        new_predictions[0] = (predictions[0] * 0.9) + (predictions[1] * 0.1)
        new_predictions[-1] = (predictions[-1] * 0.9) + (predictions[-2] * 0.1)
        sub_group[cols] = new_predictions
        sub[group == groups] = sub_group
    return sub

#==============================
#6. Process 1 audio file
#==============================
def process_audio_file(audio_path, model, output_folder, cfg, overlap_percent):
    saved_data = [] # list of [row_id, saved image path] 
    samples_proc, sr = load_and_preprocess(audio_path, cfg.SAMPLE_RATE)
    # Tạo sliding windows, mỗi window TARGET_DURATION_S giây, bước 0.5 giây
    windows = get_sliding_windows(samples_proc, sr, cfg.TARGET_DURATION_S, overlap_percent = overlap_percent)
    
    # Lấy human voice intervals nếu tồn tại cho file này (dùng đường dẫn tương đối so với DATA_PATH)
    for start_sample, end_sample in windows:
        # Format for saving
        window_start_time = start_sample / sr
        base_name = os.path.splitext(Path(audio_path).name)[0]
        # output_filename = f"{base_name}_{window_start_time:.1f}_{(window_start_time+cfg.TARGET_DURATION_S):.1f}.jpg"
        row_id = base_name + f'_{int (window_start_time+cfg.TARGET_DURATION_S)}'

        chunk = samples_proc[start_sample:end_sample]
        # Preprocessing 
        segment = np.array(chunk, dtype=np.float32)
        spec = compute_mel_spectrogram(segment, cfg)
        spec = (spec + 80.0) / 80.0
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)
        x = tensor_transform(spec_tensor)     # [3, 224, 224]
        x = x.unsqueeze(0).to(cfg.DEVICE)  # [1, 3, 224, 224]
        
        # pil_img, saved_path = get_and_save_spectrogram_image(mel_spec_db, cfg, output_folder, output_filename,(640,480))
        # x = data_transforms['test'](pil_img).unsqueeze(0).to(cfg.DEVICE) # (1, C, H, W)
        # Model prediction
        with torch.no_grad():
            logits = model(x)  
            scores = torch.sigmoid(logits)  
            scores = scores.squeeze(0).cpu().numpy()  
        saved_data.append([row_id]+ list(scores))
    return saved_data   



# =====================Load model ===================
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=cfg.NUM_CLASSES)
model.load_state_dict(torch.load(str(cfg.MODEL_PATH), map_location=cfg.DEVICE))
model.to(cfg.DEVICE)
model.eval()



import time

# Set seed
np.random.seed(cfg.SEED)
# Prepare empty list for rows
rows = []
# sample submission:
submission_example = pd.read_csv(cfg.SUBMISSION_SAMPLE_PATH)
# Class labels from train audio
class_labels = sorted(cfg.CLASS_NAME)

if cfg.USE_AUDIO_AS_INPUT:
    # List of test soundscapes (only visible during submission)
    test_soundscape_path = cfg.DATA_PATH
    test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]
    
    # Open each soundscape and make predictions for 5-second segments
    # Use pandas df with 'row_id' plus class labels as columns

    for count,soundscape in enumerate(test_soundscapes):
        print("[",count+1, "/", len(test_soundscapes),"] processing file:",soundscape)
        rows += process_audio_file(audio_path= Path(soundscape),
                              model = model,
                              output_folder = cfg.OUTPUT_FOLDER,
                              cfg = cfg,
                              overlap_percent = 0.0)


# Build predictions DataFrame
predictions = pd.DataFrame(rows, columns=['row_id'] + class_labels)

# Now remap to submission columns
submission_cols = submission_example.columns[1:]

# Reindex predictions to match submission
submission = predictions[['row_id'] + list(submission_cols)]
# meta = predictions[['row_id','path']]
# Smoothing for final submission
if cfg.USE_SMOOTH_LABEL:
    submission = smooth_label(submission)
submission.to_csv('submission.csv', index=False)
# meta.to_csv('metadata_trainSoundscape.csv', index=False)



submission.head(5)





