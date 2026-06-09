import numpy as np
import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt


class Quantizer:
    def __init__(self, num_bits):
        self.max = 2**num_bits - 1
        self.mid = 2**(num_bits - 1)
        if num_bits <= 8:
            self.dtype = torch.uint8
        elif num_bits <= 16:
            self.dtype = torch.uint16
        elif num_bits <= 32:
            self.dtype = torch.uint32

    def quantize(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:  # Edge case: all values are the same
            return torch.full_like(tensor, self.mid, dtype=torch.uint16), min_val, max_val
        scale = self.max / (max_val - min_val)
        quantized_tensor = torch.round((tensor - min_val) * scale).clamp(0, self.max).to(self.dtype)
        return quantized_tensor, min_val, max_val

    def dequantize(self, quantized_tensor, min_val, max_val):
        if min_val == max_val:
            return torch.full_like(quantized_tensor, min_val, dtype=torch.float32)
        scale = (max_val - min_val) / self.max
        return quantized_tensor.to(torch.float32) * scale + min_val

q = Quantizer(num_bits=16)


df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv').merge(
    pd.read_parquet('/kaggle/input/birdclef-2025-16-bit-melspecs/quantize_params.parquet'),
    on="filename"
)
df.head()


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


filename = df.filename[0]
original_audio = torchaudio.load('/kaggle/input/birdclef-2025/train_audio/' + filename)[0][0]
original_spec = to_spec(original_audio)

plt.imshow(original_spec[:, :248])
plt.colorbar();


spec_filename = filename.split('.')[0] + '.npy'
quantized_spec = np.load('/kaggle/input/birdclef-2025-16-bit-melspecs/train_audio_specs/' + spec_filename)
dequantized_spec = q.dequantize(torch.tensor(quantized_spec), df.min_value[0], df.max_value[0])

plt.imshow(dequantized_spec[:, :248])
plt.colorbar();


pd.Series((original_spec - dequantized_spec).abs().flatten()).describe()




