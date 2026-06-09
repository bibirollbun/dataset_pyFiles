import numpy as np
import pandas as pd
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from joblib import Parallel, delayed


df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
input_dir = '/kaggle/input/birdclef-2025/train_audio/'
output_dir = 'train_audio_specs/'
filenames = list(df.filename)


to_spec = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=32000,
        n_mels=128,
        n_fft=1920,
        hop_length=640,
        center=False,
        power=2,
    ),
    torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=80.0,
    )
)

def show_spec(spec):
    plt.imshow(spec)
    plt.colorbar()
    plt.show()

size = to_spec(torch.zeros(32000*5)).shape

for f in filenames[:1]:
    f = input_dir + f
    audio = torchaudio.load(f)[0][0]
    spec = to_spec(audio)[:, :size[1]]
    show_spec(spec)

print(size)


class Quantizer:
    def __init__(self, num_bits):
        self.range = 2**num_bits
        self.max = 2**(num_bits - 1) - 1
        self.min = -2**(num_bits - 1)
        if num_bits <= 8:
            self.dtype = torch.int8
        elif num_bits <= 16:
            self.dtype = torch.int16
        elif num_bits <= 32:
            self.dtype = torch.int32

    def quantize(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:  # Edge case: all values are the same
            return torch.full_like(tensor, 0, dtype=self.dtype), min_val, max_val
        scale = self.range / (max_val - min_val)
        quantized_tensor = torch.round((tensor - min_val) * scale + self.min).clamp(self.min, self.max).to(self.dtype)
        return quantized_tensor, min_val, max_val

    def dequantize(self, quantized_tensor, min_val, max_val):
        if min_val == max_val:
            return torch.full_like(quantized_tensor, min_val, dtype=torch.float32)
        scale = (max_val - min_val) / self.range
        return (quantized_tensor.to(torch.float32) - self.min) * scale + min_val


q = Quantizer(num_bits=16)

def process(f):
    output_path = output_dir + os.path.splitext(f)[0] + '.pt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio = torchaudio.load(input_dir + f)[0][0]
    spec, min_val, max_val = q.quantize(to_spec(audio))
    torch.save(spec, output_path)
    return {'filename': f, 'min_value': min_val.item(), 'max_value': max_val.item()}

quantize_params = Parallel(n_jobs=-1)(
    delayed(process)(f)
    for f in tqdm(filenames, desc="Getting specs & running inference")
)
quantize_df = pd.DataFrame(quantize_params)
quantize_df.to_parquet('quantize_params.parquet')
quantize_df.head()


for f, min_value, max_value in quantize_df[:1].iloc:
    output_path = output_dir + os.path.splitext(f)[0] + '.pt'
    spec = torch.load(output_path, weights_only=True)[:, :size[1]]
    spec = q.dequantize(spec, min_value, max_value)
    show_spec(spec)

