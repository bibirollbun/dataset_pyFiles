from pathlib import Path
import numpy as np 
import pandas as pd 

import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset
from torchaudio.transforms import Fade

from librosa.effects import trim
import matplotlib.pyplot as plt

from IPython.display import Audio


audio_root = Path('/kaggle/input/birdclef-2025/train_audio')
train_df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_spectrogram(stft, title="Spectrogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    _, axis = plt.subplots(1, 1)
    axis.imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)
    plt.tight_layout()

def extract_animal_segments(src, top_db=30):
    _,index = trim(src, top_db=top_db)
    return src[...,: index[0]], index[0]
    

N_FFT = 4096
N_HOP = 4
stft = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=N_HOP,
    power=None,
)


class HDemucsSegmentor:
    def __init__(self, device, segment, overlap):
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        self.model = bundle.get_model()
        self.model.to(device)
        self.sample_rate = bundle.sample_rate
        self.segment = segment
        self.overlap = overlap
        self.source = 3
        
    def forward(self, x, sample_rate):
        assert len(x.shape) == 2

        ref = x.mean(0)
        x = (x - ref.mean()) / ref.std()  # normalization
        
        if x.shape[0] == 1:
            x = x.repeat(2,1)
            
        x = x.unsqueeze(0)  # Add batch dimension
        batch, channels, length = x.shape

        chunk_len = int(self.sample_rate * self.segment * (1 + self.overlap))
        start = 0
        overlap_frames = int(self.overlap * self.sample_rate)
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")
        final = torch.zeros(batch, channels, length, device=x.device)  # Use x.device

        while start < length:
            end = min(start + chunk_len, length)  # Ensure end does not exceed length
            chunk = x[:, :, start:end]

            with torch.no_grad():
                out = self.model.forward(chunk)

            out = fade(out)  # Apply fade
            out = out[:, self.source, :, :]  # Select the desired source

            out_len = out.shape[-1]  # Get output length
            final_len = final[:, :, start:end].shape[-1]  # Get final segment length

            if out_len != final_len:
                out = out[:, :, :final_len]  # Trim to match

            final[:, :, start:end] += out  # Accumulate

            if start == 0:
                fade.fade_in_len = overlap_frames  # Adjust fade-in after first chunk

            start += chunk_len - overlap_frames  # Move by segment minus overlap

        return final[0]
segmentor = HDemucsSegmentor(device, 10., .1)


start_segment = 0
end_segment = 500000
waveform, sample_rate = torchaudio.load(audio_root.joinpath(train_df.iloc[1].filename))
plot_spectrogram(stft(waveform[...,start_segment:end_segment])[0], "Raw Audio Spectrogram")
Audio(waveform[...,start_segment:end_segment].cpu(), rate=sample_rate)


roi_wav,_ = extract_animal_segments(waveform[...,start_segment:end_segment])
plot_spectrogram(stft(roi_wav)[0], "Raw Audio Spectrogram")
Audio(roi_wav, rate=sample_rate)


vocal_source = segmentor.forward(waveform.to(device), sample_rate)
plot_spectrogram(stft(vocal_source[:,start_segment:end_segment].cpu())[0], f"Spectrogram - Vocal")
Audio(vocal_source[0,start_segment:end_segment].cpu(), rate=sample_rate)


_,start_at = extract_animal_segments(vocal_source[0,start_segment:end_segment].cpu().unsqueeze(0), 50)
roi_wav = waveform[...,start_segment:end_segment]
plot_spectrogram(stft(roi_wav[:,:start_at])[0], "Proccessed Audio Spectrogram")
Audio(roi_wav[:,:start_at], rate=sample_rate)

