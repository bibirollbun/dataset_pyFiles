!cp /kaggle/input/nvidia-dali-installation-package/nvidia-dali/* .
!pip install --no-index --find-links=. \
    nvidia_nvjpeg_cu12*.whl \
    nvidia_nvjpeg2k_cu12*.whl \
    nvidia_nvtiff_cu12*.whl \
    nvidia_nvimgcodec_cu12*.whl 
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
    USE_AUDIO_AS_INPUT = False, # tiền xử lý lại từ đầu nếu set True, set False sẽ lấy đầu vào là ảnh.
    USE_SMOOTH_LABEL = True,
    MEATA_DATA_PATH = Path('/kaggle/input/sounscape-bc25/meta_train_soundscape.csv'),
    SUBMISSION_SAMPLE_PATH = Path('/kaggle/input/birdclef-2025/sample_submission.csv'),
    DATA_PATH=Path("/kaggle/input/sounscape-bc25/train_soundscapes_data"),
    OUTPUT_FOLDER =Path("/kaggle/working/"),
    MODEL_PATH = Path("/kaggle/input/bc25-models-pt-files/latest_model.pth"),
    CLASS_NAME = np.load('/kaggle/input/bc25-models-pt-files/class_names.npy', allow_pickle=True).tolist(),
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


# ==============This cell is a copy of noice reduce lib ====================
import numpy as np
from joblib import Parallel, delayed
import tempfile
from tqdm.auto import tqdm
import numpy as np
from scipy.signal import filtfilt, fftconvolve, stft, istft
def sigmoid(x, shift, mult):
    """
    Using this sigmoid to discourage one network overpowering the other
    """
    return 1 / (1 + np.exp(-(x + shift) * mult))

def _smoothing_filter(n_grad_freq, n_grad_time):
    """Generates a filter to smooth the mask for the spectrogram

    Arguments:
        n_grad_freq {[type]} -- [how many frequency channels to smooth over with the mask.]
        n_grad_time {[type]} -- [how many time channels to smooth over with the mask.]
    """
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    return smoothing_filter


class SpectralGate:
    def __init__(
            self,
            y,
            sr,
            prop_decrease,
            chunk_size,
            padding,
            n_fft,
            win_length,
            hop_length,
            time_constant_s,
            freq_mask_smooth_hz,
            time_mask_smooth_ms,
            tmp_folder,
            use_tqdm,
            n_jobs,
    ):
        self.sr = sr
        # if this is a 1D single channel recording
        self.flat = False

        y = np.array(y)
        # reshape data to (#channels, #frames)
        if len(y.shape) == 1:
            self.y = np.expand_dims(y, 0)
            self.flat = True
        elif len(y.shape) > 2:
            raise ValueError("Waveform must be in shape (# frames, # channels)")
        else:
            self.y = y

        self._dtype = y.dtype
        # get the number of channels and frames in data
        self.n_channels, self.n_frames = self.y.shape
        self._chunk_size = chunk_size
        self.padding = padding
        self.n_jobs = n_jobs

        self.use_tqdm = use_tqdm
        # where to create a temp file for parallel
        # writing
        self._tmp_folder = tmp_folder

        ### Parameters for spectral gating
        self._n_fft = n_fft
        # set window and hop length for stft
        if win_length is None:
            self._win_length = self._n_fft
        else:
            self._win_length = win_length
        if hop_length is None:
            self._hop_length = self._win_length // 4
        else:
            self._hop_length = hop_length

        self._time_constant_s = time_constant_s

        self._prop_decrease = prop_decrease

        if (freq_mask_smooth_hz is None) & (time_mask_smooth_ms is None):
            self.smooth_mask = False
        else:
            self._generate_mask_smoothing_filter(
                freq_mask_smooth_hz, time_mask_smooth_ms
            )

    def _generate_mask_smoothing_filter(self, freq_mask_smooth_hz, time_mask_smooth_ms):
        if freq_mask_smooth_hz is None:
            n_grad_freq = 1
        else:
            # filter to smooth the mask
            n_grad_freq = int(freq_mask_smooth_hz / (self.sr / (self._n_fft / 2)))
            if n_grad_freq < 1:
                raise ValueError(
                    "freq_mask_smooth_hz needs to be at least {}Hz".format(
                        int((self.sr / (self._n_fft / 2)))
                    )
                )

        if time_mask_smooth_ms is None:
            n_grad_time = 1
        else:
            n_grad_time = int(
                time_mask_smooth_ms / ((self._hop_length / self.sr) * 1000)
            )
            if n_grad_time < 1:
                raise ValueError(
                    "time_mask_smooth_ms needs to be at least {}ms".format(
                        int((self._hop_length / self.sr) * 1000)
                    )
                )
        if (n_grad_time == 1) & (n_grad_freq == 1):
            self.smooth_mask = False
        else:
            self.smooth_mask = True
            self._smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)

    def _read_chunk(self, i1, i2):
        """read chunk and pad with zerros"""
        if i1 < 0:
            i1b = 0
        else:
            i1b = i1
        if i2 > self.n_frames:
            i2b = self.n_frames
        else:
            i2b = i2
        chunk = np.zeros((self.n_channels, i2 - i1))
        chunk[:, i1b - i1: i2b - i1] = self.y[:, i1b:i2b]
        return chunk

    def filter_chunk(self, start_frame, end_frame):
        """Pad and perform filtering"""
        i1 = start_frame - self.padding
        i2 = end_frame + self.padding
        padded_chunk = self._read_chunk(i1, i2)
        filtered_padded_chunk = self._do_filter(padded_chunk)
        return filtered_padded_chunk[:, start_frame - i1: end_frame - i1]

    def _get_filtered_chunk(self, ind):
        """Grabs a single chunk"""
        start0 = ind * self._chunk_size
        end0 = (ind + 1) * self._chunk_size
        return self.filter_chunk(start_frame=start0, end_frame=end0)

    def _do_filter(self, chunk):
        """Do the actual filtering"""
        raise NotImplementedError

    def _iterate_chunk(self, filtered_chunk, pos, end0, start0, ich):
        filtered_chunk0 = self._get_filtered_chunk(ich)
        filtered_chunk[:, pos: pos + end0 - start0] = filtered_chunk0[:, start0:end0]
        pos += end0 - start0

    def get_traces(self, start_frame=None, end_frame=None):
        """Grab filtered data iterating over chunks"""
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.n_frames

        if self._chunk_size is not None:
            if end_frame - start_frame > self._chunk_size:
                ich1 = int(start_frame / self._chunk_size)
                ich2 = int((end_frame - 1) / self._chunk_size)

                # write output to temp memmap for parallelization
                with tempfile.NamedTemporaryFile(prefix=self._tmp_folder) as fp:
                    # create temp file
                    filtered_chunk = np.memmap(
                        fp,
                        dtype=self._dtype,
                        shape=(self.n_channels, int(end_frame - start_frame)),
                        mode="w+",
                    )
                    pos_list = []
                    start_list = []
                    end_list = []
                    pos = 0
                    for ich in range(ich1, ich2 + 1):
                        if ich == ich1:
                            start0 = start_frame - ich * self._chunk_size
                        else:
                            start0 = 0
                        if ich == ich2:
                            end0 = end_frame - ich * self._chunk_size
                        else:
                            end0 = self._chunk_size
                        pos_list.append(pos)
                        start_list.append(start0)
                        end_list.append(end0)
                        pos += end0 - start0

                    Parallel(n_jobs=self.n_jobs)(
                        delayed(self._iterate_chunk)(
                            filtered_chunk, pos, end0, start0, ich
                        )
                        for pos, start0, end0, ich in zip(
                            tqdm(pos_list, disable=not (self.use_tqdm)),
                            start_list,
                            end_list,
                            range(ich1, ich2 + 1),
                        )
                    )
                    if self.flat:
                        return filtered_chunk.astype(self._dtype).flatten()
                    else:
                        return filtered_chunk.astype(self._dtype)

        filtered_chunk = self.filter_chunk(start_frame=0, end_frame=end_frame)
        if self.flat:
            return filtered_chunk.astype(self._dtype).flatten()
        else:
            return filtered_chunk.astype(self._dtype)


class SpectralGateNonStationary(SpectralGate):
    def __init__(
            self,
            y,
            sr,
            chunk_size,
            padding,
            n_fft,
            win_length,
            hop_length,
            time_constant_s,
            freq_mask_smooth_hz,
            time_mask_smooth_ms,
            thresh_n_mult_nonstationary,
            sigmoid_slope_nonstationary,
            tmp_folder,
            prop_decrease,
            use_tqdm,
            n_jobs,
    ):
        self._thresh_n_mult_nonstationary = thresh_n_mult_nonstationary
        self._sigmoid_slope_nonstationary = sigmoid_slope_nonstationary

        super().__init__(
            y=y,
            sr=sr,
            chunk_size=chunk_size,
            padding=padding,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            time_constant_s=time_constant_s,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
            tmp_folder=tmp_folder,
            prop_decrease=prop_decrease,
            use_tqdm=use_tqdm,
            n_jobs=n_jobs,
        )

    def spectral_gating_nonstationary(self, chunk):
        """non-stationary version of spectral gating"""
        denoised_channels = np.zeros(chunk.shape, chunk.dtype)
        for ci, channel in enumerate(chunk):
            _, _, sig_stft = stft(
                channel,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length,
                padded=False
            )
            # get abs of signal stft
            abs_sig_stft = np.abs(sig_stft)

            # get the smoothed mean of the signal
            sig_stft_smooth = get_time_smoothed_representation(
                abs_sig_stft,
                self.sr,
                self._hop_length,
                time_constant_s=self._time_constant_s,
            )

            # get the number of X above the mean the signal is
            sig_mult_above_thresh = (abs_sig_stft - sig_stft_smooth) / sig_stft_smooth
            # mask based on sigmoid
            sig_mask = sigmoid(
                sig_mult_above_thresh,
                -self._thresh_n_mult_nonstationary,
                self._sigmoid_slope_nonstationary,
            )

            if self.smooth_mask:
                # convolve the mask with a smoothing filter
                sig_mask = fftconvolve(sig_mask, self._smoothing_filter, mode="same")

            sig_mask = sig_mask * self._prop_decrease + np.ones(np.shape(sig_mask)) * (
                    1.0 - self._prop_decrease
            )

            # multiply signal with mask
            sig_stft_denoised = sig_stft * sig_mask

            # invert/recover the signal
            _, denoised_signal = istft(
                sig_stft_denoised,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length
            )
            denoised_channels[ci, : len(denoised_signal)] = denoised_signal
        return denoised_channels

    def _do_filter(self, chunk):
        """Do the actual filtering"""
        chunk_filtered = self.spectral_gating_nonstationary(chunk)

        return chunk_filtered


def get_time_smoothed_representation(
        spectral, samplerate, hop_length, time_constant_s=0.001
):
    t_frames = time_constant_s * samplerate / float(hop_length)
    # By default, this solves the equation for b:
    #   b**2  + (1 - b) / t_frames  - 2 = 0
    # which approximates the full-width half-max of the
    # squared frequency response of the IIR low-pass filt
    b = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)
    return filtfilt([b], [1, b - 1], spectral, axis=-1, padtype=None)


def reduce_noise(
        y,
        sr,
        stationary=False,
        y_noise=None,
        prop_decrease=1.0,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        thresh_n_mult_nonstationary=2,
        sigmoid_slope_nonstationary=10,
        n_std_thresh_stationary=1.5,
        tmp_folder=None,
        chunk_size=600000,
        padding=30000,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        clip_noise_stationary=True,
        use_tqdm=False,
        n_jobs=1,
        use_torch=False,
        device="cuda",
):
    sg = SpectralGateNonStationary(
        y=y,
        sr=sr,
        chunk_size=chunk_size,
        padding=padding,
        prop_decrease=prop_decrease,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        time_constant_s=time_constant_s,
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        time_mask_smooth_ms=time_mask_smooth_ms,
        thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
        sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
        tmp_folder=tmp_folder,
        use_tqdm=use_tqdm,
        n_jobs=n_jobs,
    )
    return sg.get_traces()


from torchvision import transforms
# ============= Data transform to convert 2 tensor==============
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
}
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
    samples_nr = reduce_noise(y=samples, sr=sr)
    #apply 10dB gain
    gain_db = 10
    gain_linear = 10**(gain_db / 20)
    samples_proc = samples_nr*gain_linear
    return samples_nr, sr

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
# 4. Tạo ảnh từ Mel Spectrogram
# ============================
from PIL import Image
import matplotlib.cm as cm

def get_spectrogram_image(mel_spec, cfg, output_size=(1280, 720)):
    # Normalize mel spectrogram to 0–1
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
    # List of colormaps
    cmap_name = cfg.COLOR_MAP
    cmap = cm.get_cmap(cmap_name)
    rgba_img = cmap(mel_spec)  # shape: (H, W, 4), values in [0, 1]
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb_img)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_resized = img.resize(output_size, Image.LANCZOS)
    return img_resized

def get_and_save_spectrogram_image(mel_spec, cfg, output_dir, output_filename, output_size=(1280, 720)):
    # List of colormaps
    cmap_name = cfg.COLOR_MAP
    out_path =  cfg.OUTPUT_FOLDER /cmap_name /output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / output_filename
    img_resized = get_spectrogram_image(mel_spec, cfg, output_size=output_size)
    # Save final image
    img_resized.save(output_file)
    return img_resized, output_file
    
def show_spectrogram(spec, title, sr, hop_length, y_axis="log", x_axis="time"):
    librosa.display.specshow(
        spec, sr=sr, y_axis=y_axis, x_axis=x_axis, hop_length=hop_length
    )
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

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
        output_filename = f"{base_name}_{window_start_time:.1f}_{(window_start_time+cfg.TARGET_DURATION_S):.1f}.jpg"
        row_id = base_name + f'_{int (window_start_time+cfg.TARGET_DURATION_S)}'

        chunk = samples_proc[start_sample:end_sample]
        # Preprocessing 
        segment = np.array(chunk, dtype=np.float32)
        mel_spec_db = compute_mel_spectrogram(segment, cfg)
        pil_img, saved_path = get_and_save_spectrogram_image(mel_spec_db, cfg, output_folder, output_filename,(640,480))
        x = data_transforms['test'](pil_img).unsqueeze(0) # (1, C, H, W)
        # Model prediction
        with torch.no_grad():
            logits = model(x)  
            scores = torch.sigmoid(logits)  
            scores = scores.squeeze(0).cpu().numpy()  
            
        saved_data.append([row_id,saved_path]+ list(scores))
    return saved_data   
#==============================
#6. Process 1 image file
#==============================
def process_image_file(row_id, file_path, model, cfg):
    pil_img = Image.open(file_path).convert('RGB')
    x = data_transforms['test'](pil_img).unsqueeze(0).to(cfg.DEVICE) # (1, C, H, W)
    # Model prediction
    with torch.no_grad():
        logits = model(x)  
        scores = torch.sigmoid(logits)  
        scores = scores.squeeze(0).cpu().numpy()  
    return [[row_id,file_path]+ list(scores)]


from typing import Dict, Optional
from transformers import (
    AutoConfig,
    ConvNextConfig,
    ConvNextForImageClassification,
    ConvNextModel,
)

#=============== Model Clas ========================

class ConvNextClassifier(nn.Module):
    def __init__(
        self,
        num_channels: int = 1,
        embedding_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.model = None
        self.architecture = None
        self._initialize_model()

    def _initialize_model(self) -> nn.Module:
        """Initializes the ConvNext model based on specified attributes.

        Returns:
            nn.Module: The initialized ConvNext model.
        """

        adjusted_state_dict = None

        model = ConvNextModel
        config = ConvNextConfig.from_pretrained("/kaggle/input/bc25-models-pt-files/convnext_base_facebook_224_22k")
        hidden_sizes = config.hidden_sizes
        hidden_sizes[-1] = self.embedding_size
        self.model = model.from_pretrained(
                "/kaggle/input/bc25-models-pt-files/convnext_base_facebook_224_22k",
                hidden_sizes=hidden_sizes,
                num_channels=self.num_channels,
                cache_dir=None,
                ignore_mismatched_sizes=True,
            )

        self.architecture = self.model.config.model_type

    def forward(self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        outputs = self.model(input_values)

        output = outputs.last_hidden_state

        return output
        
class CustomNet(nn.Module):
    def __init__(self, backbone, num_classes, add_on_layers_type="identity", use_prototype = False, num_prototypes = 5):
        super().__init__()
        self.backbone = backbone
        backbone_out = backbone.embedding_size
        if add_on_layers_type == "identity":
            self.add_on_layers = nn.Sequential(nn.Identity())
        elif add_on_layers_type == "upsample":
            self.add_on_layers = nn.Upsample(scale_factor=2, mode="bilinear")
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            self.add_on_layers,
            nn.Linear(backbone_out, num_classes)
        )
        
    def forward(self,x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits


# =====================Load model ===================
backbone = ConvNextClassifier(
    num_channels=1,
    embedding_size=1024,
)
model = CustomNet(backbone, cfg.NUM_CLASSES).to(cfg.DEVICE)
checkpoint = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE)

state_dict_full= checkpoint['model_state_dict']
prefix = '_orig_mod.'
model_dict = {}
for key, val in state_dict_full.items():
    new_key = key[len(prefix):]
    model_dict[new_key] = val
model.load_state_dict(model_dict, strict=False)
model = model.to(cfg.DEVICE)


# not only visualize, but also for pre-allocated
sig, rate = load_and_preprocess("/kaggle/input/birdclef-2025/train_soundscapes/H02_20230420_074000.ogg", 32000)
chunk = sig[32000*0:32000*5]
segment = np.array(chunk, dtype=np.float32)
mel_spec_db = compute_mel_spectrogram(segment, cfg)
pil_img = get_spectrogram_image(mel_spec_db,cfg,(640,480))
display(pil_img)


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
    test_soundscape_path = '/kaggle/input/birdclef-2025/train_soundscapes/'
    test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]
    
    # Open each soundscape and make predictions for 5-second segments
    # Use pandas df with 'row_id' plus class labels as columns

    for count,soundscape in enumerate(test_soundscapes):
        print("[",count+1, "/", len(test_soundscapes),"] processing file:",soundscape)
        rows += process_audio_file(audio_path= Path(soundscape),
                              model = model,
                              output_folder = cfg.OUTPUT_FOLDER,
                              cfg = cfg,
                              overlap_percent = 0.0,)

else: 
    metadata= pd.read_csv(cfg.MEATA_DATA_PATH)
    old_prefix = "/kaggle/working/train_soundscapes_data"
    new_prefix = str(cfg.DATA_PATH)
    metadata['path'] = metadata['path'].str.replace(old_prefix, new_prefix, regex=False)
    for idx, row in metadata.iterrows():
        print("[",idx+1, "/", len(metadata),"] processing file:",row['path'])
        rows += process_image_file(row_id = row['row_id'], 
                                   file_path = row['path'],
                                   model = model,
                                   cfg = cfg)
# Build predictions DataFrame
predictions = pd.DataFrame(rows, columns=['row_id', 'path'] + class_labels)

# Now remap to submission columns
submission_cols = submission_example.columns[1:]

# Reindex predictions to match submission
submission = predictions[['row_id'] + list(submission_cols)]
meta = predictions[['row_id','path']]
# Smoothing for final submission
if cfg.USE_SMOOTH_LABEL:
    submission = smooth_label(submission)
submission.to_csv('submission.csv', index=False)
meta.to_csv('metadata_trainSoundscape.csv', index=False)



submission.head(5)



meta.head(5)




