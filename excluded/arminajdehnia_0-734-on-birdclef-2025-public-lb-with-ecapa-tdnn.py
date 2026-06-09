# This package is responsible to normalize and put some variation on our Audio files.
# If you Turn The Internet On, This Block doesn`t necessary to be run and instead of this you can run following command
# !pip install audiomentations
# !pip install /kaggle/input/birdclef-packages/other/default/3/numpy_minmax-0.3.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# !pip install /kaggle/input/birdclef-packages/other/default/3/python_stretch-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# !pip install /kaggle/input/birdclef-packages/other/default/3/numpy_rms-0.4.2-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# !pip install /kaggle/input/birdclef-packages/other/default/4/scipy-1.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# !pip install /kaggle/input/birdclef-packages/other/default/3/audiomentations-0.39.0-py3-none-any.whl


import math
import os
from typing import List
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm

import numpy as np

import librosa
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# import audiomentations as A

import ecapa_tdnn as ecapa


DEVICE = torch.device('cpu')
states = torch.load('/kaggle/input/birdclef-ecapa-tdnn/pytorch/default/2/ecapa-tdnn-5sec-best-v0.0.1.pt', map_location=DEVICE)
meta = states['meta']

# Audio parameters
FS = 32000  
WINDOW_SIZE = 5  

class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
label2id = meta['label2id']
id2label = {v:k for k,v in label2id.items()}


class CLEFDataset(Dataset):
    def __init__(self, audio_list:List[Path]):
        self.audio_list = audio_list
        # self.transform = transform
        # self.ext = ext
        # self.sr = sample_rate

    def __len__(self)->int:
        return len(self.audio_list)

    def __getitem__(self, idx):
        wav, sr = librosa.load(self.audio_list[idx], sr=FS)
        wav = wav.reshape(-1, FS*5)
        # wav = wav.numpy().squeeze()

        # if self.transform is not None:
        #     wav = self.transform(wav, self.sr)

        # wav = torch.from_numpy(wav).unsqueeze(0)
        
        # if self.ext is not None:
        #     wav = self.ext(wav)
        return wav, sr, Path(self.audio_list[idx]).stem


# tests_audio = list(Path('/kaggle/input/birdclef-2025/test_soundscapes').glob('*.ogg'))
# tests_audio = list(Path('/kaggle/input/birdclef-2025/train_soundscapes').glob('*.ogg'))[:500]
# tests_path = '/kaggle/input/birdclef-2025/train_soundscapes'
tests_path = '/kaggle/input/birdclef-2025/test_soundscapes'
tests_audio = [os.path.join(tests_path, afile) for afile in sorted(os.listdir(tests_path)) if afile.endswith('.ogg')]

if len(tests_audio) == 0:
    tests_path = '/kaggle/input/birdclef-2025/train_soundscapes'
    tests_audio = [os.path.join(tests_path, afile) for afile in sorted(os.listdir(tests_path)) if afile.endswith('.ogg')][:3]

# test_transforms = A.Compose(
#     [
#         A.AdjustDuration(duration_samples=int(meta['wav_len'] * meta['sr']),p=1.),
#     ]
# )


class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=self.sample_rate, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear

class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x

class CosineClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_blocks=0,
        inter_dim=512,
        out_neurons=1000,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for index in range(num_blocks):
            self.blocks.append(
                DenseLayer(input_dim, inter_dim, config_str='batchnorm')
            )
            input_dim = inter_dim

        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [B, dim]
        for layer in self.blocks:
            x = layer(x)

        # normalized
        x = F.linear(F.normalize(x), F.normalize(self.weight))
        return x
        
class BirdCLEFClassifier(nn.Module):
    def __init__(self, backbone, n_classes:int, input_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = CosineClassifier(input_dim=input_dim, out_neurons=n_classes)


    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


fbank = FBank(n_mels=meta['fbank_features'], mean_nor=True, sample_rate=meta['sr'])

# Define Feature Extractor
extractor = ecapa.ECAPA_TDNN(input_size=meta['fbank_features'], lin_neurons=meta['lin_out'], channels=meta['channels'])
extractor.fc = ecapa.Conv1d(in_channels=meta['channels'][-1] * 2, out_channels=meta['lin_out'], kernel_size=1)

model = BirdCLEFClassifier(extractor, len(meta['label2id']), meta['lin_out'])
model.load_state_dict(states['state_dict'])

model.to(DEVICE)


loader = DataLoader(CLEFDataset(tests_audio), batch_size=1, num_workers=2, shuffle=False, prefetch_factor=2)
class_labels_models = [id2label[i] for i in range(len(id2label))]
model.eval()

row_ids = []
matrix = []
with torch.inference_mode():    
    bar = tqdm(enumerate(loader), total=len(loader))
    # Read Audio Paths
    for idx, (audio, sr, soundscape) in bar:
        audio = audio[0]
        sr = sr.item()
        soundscape = soundscape[0]
        feats = []
        for feat in audio:
            feat = fbank(feat).unsqueeze(0)
            feats.append(feat)
        audio = torch.concatenate(feats, dim=0)
        outputs = model(audio)
        logits = F.softmax(outputs, dim=-1).detach().numpy()
        soundscape = os.path.basename(soundscape).split('.')[0]
        row_id = [f"{soundscape}_{(i+1)*5}" for i in range(0, logits.shape[0])]

        row_ids += row_id
        matrix.append(logits)
matrix = np.concatenate(matrix)
matrix = np.concatenate([np.array(row_ids).reshape(-1, 1), matrix], axis=1)
sample_sub = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
sub_csv = pd.DataFrame(matrix, columns=["row_id", *class_labels])
sub_csv.to_csv('submission.csv', index=False)


sub_csv.head()

