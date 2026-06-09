from enum import StrEnum
from pathlib import Path
from typing import Callable
import random

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as tt


torch.backends.cudnn.benchmark = True


from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Config:
    num_seconds: int = 5
    sample_rate: int = 32000

    batch_size: int = 64
    random_state: int = 42
    shuffle: bool = True
    val_size: float = 0.25

    picker_method: str = "random"
    human_voice_segments_file: str = "../input/birdclef-human-voice/hvs.csv"

    n_mels: int = 128
    n_fft: int = 1024
    f_min: float = 0
    f_max: float = 16000
    mel_scale: str = "slaney"

    in_chans = 1

    iid_masks: bool = True
    n_freq: int = n_fft // 2 + 1

    mask_prob: float = 0.5
    freq_mask_param: int = n_mels // 3
    time_mask_param: int = 100

    max_noise_factor: float = 0.75

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    loss: str = "focal"
    model: str = "tf_efficientnet_b0"
    optimizer: str = "adam"

    early_stopper_patience: int = 5
    early_stopper_min_delta: float = 0


class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        filenames: pd.Series,
        targets: torch.Tensor,
        picker: Callable,
        transformer: Callable,
    ):
        self.filenames = filenames
        self.targets = targets

        self.picker = picker
        self.transformer = transformer

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, sample_rate = self.picker(self.filenames.iloc[i])
        sample = self.transformer(waveform)
        return sample, self.targets[i]


def read_segments(segment_file: str | Path) -> dict[str, list[dict]]:
    return (
        pd.read_csv(segment_file)
        .groupby("filename")
        .apply(lambda x: x[["start", "end"]].to_dict(orient="records"), include_groups=False)
        .to_dict()
    )


def create_mask(segments: list[dict], size: int) -> np.ndarray:
    mask = np.ones(size, dtype=bool)
    for segment in segments:
        mask[segment["start"] : segment["end"]] = False
    return mask


class PickerMethod(StrEnum):
    FIRST = "first"
    RANDOM = "random"
    RMS = "rms"


class Picker:
    def __init__(
        self,
        method: PickerMethod,
        num_seconds: int,
        sample_rate: int,
        human_voice_segments: dict[str, list[dict]] = {},
    ):
        self.method = method
        self.num_frames = num_seconds * sample_rate
        self.sample_rate = sample_rate
        self.human_voice_segments = human_voice_segments

    def frame_offset(self, total_num_frames: int) -> int:
        if self.method == PickerMethod.FIRST:
            return 0

        if self.method == PickerMethod.RANDOM:
            return random.randint(0, max(0, total_num_frames - self.num_frames))

        raise NotImplementedError(PickerMethod.RMS)

    def __call__(self, filename: Path) -> tuple[torch.Tensor, int]:
        if filename not in self.human_voice_segments:
            metadata = torchaudio.info(filename)
            if metadata.sample_rate != self.sample_rate:
                raise NotImplementedError("Resampling")

            offset = self.frame_offset(metadata.num_frames)
            waveform, sample_rate = torchaudio.load(
                filename,
                frame_offset=offset,
                num_frames=self.num_frames,
            )
        else:
            waveform, sample_rate = torchaudio.load(filename)
            if sample_rate != self.sample_rate:
                raise NotImplementedError("Resampling")

            total_num_frames = waveform.shape[1]
            non_voice_mask = create_mask(self.human_voice_segments[str(filename)], total_num_frames)
            waveform = waveform[non_voice_mask]

            offset = self.frame_offset(total_num_frames)
            waveform = waveform[:, offset : 1 + min(total_num_frames, offset + self.num_frames)]

        if waveform.shape[1] <= self.num_frames:
            times = (self.num_frames - 1) // waveform.shape[1] + 1
            waveform = waveform.repeat(1, times)[:, : self.num_frames]

        assert waveform.shape[1] == self.num_frames
        return waveform, sample_rate

    @classmethod
    def from_config(cls, config: Config) -> "Picker":
        return Picker(
            method=PickerMethod(config.picker_method),
            num_seconds=config.num_seconds,
            sample_rate=config.sample_rate,
            human_voice_segments=read_segments(config.human_voice_segments_file),
        )


class Compose:
    # https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


# class OneOf:
#     def __init__(self, transforms: list[Callable]):
#         self.transforms = transforms
#
#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         transform = random.choice(self.transforms)
#         return transform(x)


class RandomTransform:
    def __init__(self, transform: Callable, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x) if random.random() < self.p else x


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


def create_specaug(config: Config):
    return Compose(
        [
            RandomTransform(create_timemask(config), p=config.mask_prob),
            RandomTransform(create_freqmask(config), p=config.mask_prob),
        ]
    )


def create_timemask(config: Config):
    return tt.TimeMasking(time_mask_param=config.time_mask_param, iid_masks=config.iid_masks)


def create_freqmask(config: Config):
    return tt.FrequencyMasking(freq_mask_param=config.freq_mask_param, iid_masks=config.iid_masks)


class NoiseAugment:
    def __init__(self, config: Config):
        melspec = create_melspec(config)

        self.max_factor = config.max_noise_factor

        def spec_signal_mask(specgram: torch.Tensor) -> torch.Tensor:
            mean, std = torch.mean(specgram, dim=1), torch.std(specgram, dim=1)
            return specgram >= (mean + 1 * std)

        def suppress_signal(waveform: torch.Tensor) -> torch.Tensor:
            specgram = melspec(waveform)
            mask = spec_signal_mask(specgram)
            specgram[mask] = 0
            mean = torch.mean(specgram, dim=1)
            specgram[mask] = (torch.ones_like(mask) * mean)[mask]
            return specgram

        def noise(file: Path) -> torch.Tensor:
            offset = random.randint(0, (60 - config.num_seconds) * config.sample_rate)
            waveform, sample_rate = torchaudio.load(
                file,
                frame_offset=offset,
                num_frames=config.num_seconds * config.sample_rate,
            )

            return suppress_signal(waveform)

        files = Path("../input/birdclef-2025/train_soundscapes").glob("*.ogg")
        self.noise = [noise(file) for file in random.sample(list(files), 100)]

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        return spec + random.uniform(0, self.max_factor) * random.choice(self.noise)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, device: str, alpha: torch.Tensor, gamma: float = 2, reduction: str = "mean"):
        super().__init__()

        self.alpha = alpha.to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = torch.exp(-bce)

        # https://stackoverflow.com/a/78781507
        focal_loss = self.alpha[target.long()] * ((1 - p_t) ** self.gamma) * bce

        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "mean":
            return focal_loss.mean()

        return focal_loss


def create_loss(config: Config, targets: torch.Tensor, **kwargs) -> nn.Module:
    if config.loss != "focal":
        return nn.BCEWithLogitsLoss(**kwargs)

    alpha = targets.sum() / targets.sum(0)
    alpha = 100 * (alpha / alpha.sum())
    return FocalBCEWithLogitsLoss(device=config.device, alpha=alpha, **kwargs)


from datetime import datetime



class EarlyStopperAction(StrEnum):
    CONTINUE = "continue"
    SAVE = "save"
    STOP = "stop"


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_vloss = float("inf")

    def __call__(self, vloss: float) -> EarlyStopperAction:
        if vloss >= self.best_vloss:
            self.counter += 1
            if self.counter >= self.patience:
                return EarlyStopperAction.STOP

        elif vloss < self.best_vloss - self.min_delta:
            self.best_vloss = vloss
            self.counter = 0
            return EarlyStopperAction.SAVE

        return EarlyStopperAction.CONTINUE


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss: nn.Module,
        device: str | None = None,
        early_stopper: Callable | None = None,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ):
        self.epoch = 0

        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.early_stopper = early_stopper
        self.device = device
        self.scaler = torch.GradScaler(device)

    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self.model(X.to(self.device))
        return self.loss(y_pred, y.to(self.device))

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        running_loss = 0.0

        for i, (X, y) in enumerate(dataloader):
            self.optimizer.zero_grad()

            with torch.autocast(self.device):
                y_pred = self.model(X.to(self.device))
                loss = self.loss(y_pred, y.to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            running_loss += loss
            avg_loss = running_loss / (i + 1)
            if i % 50 == 49:
                print(f"  batch {i + 1:4}: loss: {avg_loss.item():.6f}")

        return running_loss.item() / (i + 1)

    def train(self, epochs: int, dataloader: DataLoader, val_dataloader: DataLoader | None = None):
        for epoch in range(epochs):
            self.epoch += 1

            self.model.train()
            tloss = self.train_one_epoch(dataloader)

            vloss = self.validate(val_dataloader) if val_dataloader else 0
            print(f"Epoch {self.epoch:2}: train loss: {tloss:.6f}  val loss: {vloss:.6f}")

            if val_dataloader and self.early_stopper:
                action = self.early_stopper(vloss)
                if action == EarlyStopperAction.STOP:
                    break
                if action == EarlyStopperAction.SAVE:
                    self.save()

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for i, (X, y) in enumerate(dataloader):
                total_loss += self.compute_loss(X, y)
            return total_loss.item() / (i + 1)

    def save(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        path = f"{self.model.__class__.__name__}_{self.epoch}_{timestamp}.pt"
        torch.save(self.model.state_dict(), path)


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


import timm


class PreTrainedModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.model(x)



import math
import sys
import time

from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot


def create_optimizer(config: Config, model: nn.Module, **kwargs) -> optim.Optimizer:
    return optim.Adam(model.parameters(), **kwargs)


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


def one_hot_encode(primary_label: pd.Series, labels: list[str]) -> torch.Tensor:
    label_to_int = {label: i for i, label in enumerate(labels)}
    targets = primary_label.map(label_to_int).to_numpy()
    return one_hot(torch.LongTensor(targets), num_classes=len(labels)).float()


def duplicate_undersampled_classes(df: pd.DataFrame, min_samples: int = 4):
    samples_per_label = df.primary_label.value_counts()
    undersampled_labels = samples_per_label[samples_per_label < min_samples].index

    def duplicate_samples(label: str):
        samples = df[df.primary_label == label]
        return samples.sample(
            min_samples - len(samples),
            replace=True,
            random_state=config.random_state,
        )

    return pd.concat([df, *[duplicate_samples(label) for label in undersampled_labels]])


if __name__ == "__main__":
    config = Config()
    print(config)

    root = Path("../input/birdclef-2025")
    df = pd.read_csv(root / "train.csv")
    df = duplicate_undersampled_classes(df, min_samples=math.ceil(1 / config.val_size))

    filenames = root / "train_audio" / df.filename

    label_names = pd.read_csv(root / "sample_submission.csv", index_col=0, nrows=0).columns
    num_classes = len(label_names)

    targets = one_hot_encode(df.primary_label, label_names)

    train_filenames, val_filenames, train_targets, val_targets = train_test_split(
        filenames,
        targets,
        random_state=config.random_state,
        stratify=df.primary_label,
    )

    picker  = Picker.from_config(config)
    melspec = create_melspec(config)
    specaug = create_specaug(config)
    transformer = Compose([melspec, specaug])

    train_dataset = BirdCLEFDataset(train_filenames, train_targets, picker, transformer)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, pin_memory=True, num_workers=2)

    val_dataset = BirdCLEFDataset(val_filenames, val_targets, picker, melspec)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=config.shuffle, pin_memory=True, num_workers=2)

    model = create_model(config, num_classes)
    print(model)

    optimizer = create_optimizer(config, model)
    loss_fn = create_loss(config, train_targets)

    early_stopper = EarlyStopper(
        patience=config.early_stopper_patience,
        min_delta=config.early_stopper_min_delta,
    )
    trainer = Trainer(model, optimizer, loss_fn, device=config.device, early_stopper=early_stopper)

    start = time.time()
    trainer.train(config.epochs, train_dataloader, val_dataloader)
    train_duration = (time.time() - start) / 60
    print(f"Training duration: {train_duration:.6f}")

