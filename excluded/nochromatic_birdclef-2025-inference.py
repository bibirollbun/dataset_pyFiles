from pathlib import Path
from typing import Callable

import pandas as pd
import timm
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as tt


from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Config:
    num_seconds: int = 5
    sample_rate: int = 32000

    n_mels: int = 128
    n_fft: int = 1024
    f_min: float = 0
    f_max: float = 16000
    mel_scale: str = "slaney"

    in_chans = 1

    device: str = "cpu"
    model: str = "tf_efficientnet_b0"


class Compose:
    # https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

class Scale:
    def __init__(self, min: float = 0, max: float = 1):
        self.min = min
        self.max = max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        return self.min + (self.max - self.min) * (x - x.min()) / (x.max() - x.min() + eps)

def create_melspec(config: Config):
    return Compose(
        [
            tt.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                f_min=config.f_min,
                f_max=config.f_max,
                n_mels=config.n_mels,
                mel_scale=config.mel_scale,
            ),
            tt.AmplitudeToDB(stype="power", top_db=80),
            Scale(),
        ]
    )


class InferenceModel:
    def __init__(self, model: torch.nn.Module, path: Path, device: str):
        self.model = model
        self.model.load_state_dict(
            torch.load(path, weights_only=True, map_location=torch.device(device))
        )
        self.model.eval()

    @torch.inference_mode()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(X))


class BirdSoundCNN(nn.Module):
    def __init__(self, num_classes: int, in_chans: int):
        super().__init__()

        self.encoder = nn.Sequential(
            self.conv_block(in_chans, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128 * 14 * 37, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def conv_block(in_chans: int, out_chans: int, kernel_size: int = 3) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.decoder(x)


class PreTrainedModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.model(x)


def create_model(config: Config, num_classes: int, **kwargs) -> nn.Module:
    if config.model == "cnn":
        return BirdSoundCNN(
            num_classes=num_classes,
            in_chans=config.in_chans,
            **kwargs,
        )

    return PreTrainedModel(
        config.model,
        num_classes=num_classes,
        in_chans=config.in_chans,
        **kwargs,
    )


import csv
import os

if __name__ == "__main__":
    config = Config()

    root = Path("../input/birdclef-2025")
    labels = pd.read_csv(root / "sample_submission.csv", index_col=0, nrows=0).columns

    paths = [
        Path("/kaggle/input/ef/pytorch/with_noise/3/PreTrainedModel_9_20250726_0916.pt"),
        Path("/kaggle/input/ef/pytorch/default/3/PreTrainedModel_7_20250726_0340.pt"),
    ]
    models = [
        InferenceModel(create_model(config, len(labels)), path, config.device)
        for path in paths
    ]

    with open("submission.csv", "w", newline="") as f:
        writer = csv.writer(f, lineterminator=os.linesep)
        writer.writerow(["row_id", *labels])
    
        segments = 12

        melspec = create_melspec(config)
        for file in (root / "test_soundscapes").glob("*.ogg"):
            row_ids = [file.stem + f"_{config.num_seconds * (i + 1)}" for i in range(segments)]

            num_frames = config.sample_rate * config.num_seconds
            waveform, sample_rate = torchaudio.load(file, num_frames=(num_frames * segments))
            waveform = waveform.reshape(-1, 1, num_frames)
    
            print(f"Predicting file: {file}")
            spec = melspec(waveform)
            predictions = torch.stack([model.predict(spec) for model in models]).mean(0).tolist()
            #predictions = model.predict(melspec(waveform)).tolist()
    
            writer.writerows((row_id, *prediction) for row_id, prediction in zip(row_ids, predictions))


df = pd.read_csv("submission.csv")
df.head()

