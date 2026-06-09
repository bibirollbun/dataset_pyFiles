import cv2
import re
import torch
import torch.nn as nn
import time

import numpy as np
import librosa as lb
import soundfile as sf
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

import sys
sys.path.append('../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest')
from resnest.torch import resnest50


NUM_CLASSES = 397
SAMPLE_RATE = 32_000
AUDIO_DURATION = 5
PREDICTION_THRESHOLD = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", DEVICE)

TEST_AUDIO_PATH = Path("../input/birdclef-2021/test_soundscapes")
SUBMISSION_SAMPLE_PATH = "../input/birdclef-2021/sample_submission.csv"
TRAIN_LABELS_PATH = None

if not list(TEST_AUDIO_PATH.glob("*.ogg")):
    TEST_AUDIO_PATH = Path("../input/birdclef-2021/train_soundscapes")
    SUBMISSION_SAMPLE_PATH = None
    TRAIN_LABELS_PATH = Path("../input/birdclef-2021/train_soundscape_labels.csv")


class MelSpectrogramGenerator:
    def __init__(self, sample_rate, n_mels, f_min, f_max, **kwargs):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        kwargs["n_fft"] = kwargs.get("n_fft", self.sample_rate // 10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sample_rate // (10 * 4))
        self.kwargs = kwargs

    def __call__(self, audio):
        mel_spec = lb.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate, 
            n_mels=self.n_mels, 
            fmin=self.f_min, 
            fmax=self.f_max, 
            **self.kwargs
        )
        mel_spec_db = lb.power_to_db(mel_spec).astype(np.float32)
        return mel_spec_db


def convert_to_grayscale(spectrogram, epsilon=1e-6, mean_val=None, std_val=None):
    mean_val = mean_val or spectrogram.mean()
    std_val = std_val or spectrogram.std()
    normalized = (spectrogram - mean_val) / (std_val + epsilon)
    
    min_val, max_val = normalized.min(), normalized.max()

    if (max_val - min_val) > epsilon:
        clipped = np.clip(normalized, min_val, max_val)
        scaled = 255 * (clipped - min_val) / (max_val - min_val)
        uint8_array = scaled.astype(np.uint8)
    else:
        uint8_array = np.zeros_like(spectrogram, dtype=np.uint8)

    return uint8_array

def adjust_audio_length(audio, target_length):
    if len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.concatenate([audio, np.zeros(padding)])
    elif len(audio) > target_length:
        audio = audio[:target_length]
    return audio



class BirdAudioDataset(Dataset):
    def __init__(self, metadata, sr=SAMPLE_RATE, n_mels=128, f_min=0, f_max=None, 
                 duration=AUDIO_DURATION, step_size=None, resample_type="kaiser_fast", 
                 do_resample=True):
        
        self.metadata = metadata
        self.sample_rate = sr
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or self.sample_rate // 2
        self.duration = duration
        self.audio_length = self.duration * self.sample_rate
        self.step_size = step_size or self.audio_length
        self.resample_type = resample_type
        self.do_resample = do_resample

        self.mel_generator = MelSpectrogramGenerator(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        )

    def __len__(self):
        return len(self.metadata)
    
    @staticmethod
    def normalize_image(image_array):
        normalized = image_array.astype("float32", copy=False) / 255.0
        stacked = np.stack([normalized, normalized, normalized])
        return stacked
    
    def convert_audio_to_image(self, audio_segment):
        mel_spectrogram = self.mel_generator(audio_segment)
        image = convert_to_grayscale(mel_spectrogram)
        normalized_image = self.normalize_image(image)
        return normalized_image

    def load_and_process_audio(self, file_path):
        audio_data, original_sr = sf.read(file_path, dtype="float32")

        if self.do_resample and original_sr != self.sample_rate:
            audio_data = lb.resample(audio_data, original_sr, self.sample_rate, 
                                   res_type=self.resample_type)
        
        audio_segments = []
        for i in range(self.audio_length, len(audio_data) + self.step_size, self.step_size):
            start_idx = max(0, i - self.audio_length)
            end_idx = start_idx + self.audio_length
            segment = audio_data[start_idx:end_idx]
            audio_segments.append(segment)
            
        if len(audio_segments[-1]) < self.audio_length:
            audio_segments = audio_segments[:-1]
            
        spectrogram_images = [self.convert_audio_to_image(segment) for segment in audio_segments]
        stacked_images = np.stack(spectrogram_images)
        
        return stacked_images
    
    def __getitem__(self, index):
        return self.load_and_process_audio(self.metadata.loc[index, "filepath"])


audio_files_df = pd.DataFrame(
    [(path.stem, *path.stem.split("_"), path) for path in Path(TEST_AUDIO_PATH).glob("*.ogg")],
    columns=["filename", "id", "site", "date", "filepath"]
)
print(f"Загружено файлов: {audio_files_df.shape[0]}")
print(audio_files_df.head())

train_metadata = pd.read_csv("../input/birdclef-2021/train_metadata.csv")
LABEL_MAPPING = {label: idx for idx, label in enumerate(sorted(train_metadata["primary_label"].unique()))}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


test_dataset = BirdAudioDataset(metadata=audio_files_df)
print(f"Размер тестового датасета: {len(test_dataset)}")
print(f"Форма первого элемента: {test_dataset[0].shape}")


def initialize_model(model_checkpoint_path, num_classes=NUM_CLASSES):
    model = resnest50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    cpu_device = torch.device("cpu")
    checkpoint = torch.load(model_checkpoint_path, map_location=cpu_device)
    
    updated_state_dict = {}
    for key in list(checkpoint.keys()):
        updated_state_dict[key.replace("model.", "")] = checkpoint[key]
    
    model.load_state_dict(updated_state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


model_paths = [
    Path("../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth"),
]

trained_models = [
    initialize_model(model_path.as_posix()) for model_path in model_paths
]


@torch.no_grad()
def apply_threshold_to_predictions(predictions, threshold=None):
    threshold = threshold or PREDICTION_THRESHOLD
    sorted_indices = (-predictions).argsort(1)
    num_positive = (predictions > threshold).sum(1)
    
    thresholded_predictions = []
    for indices, count in zip(sorted_indices, num_positive):
        thresholded_predictions.append(indices[:count].cpu().numpy().tolist())
    
    return thresholded_predictions

def convert_to_bird_names(prediction_indices):
    bird_name_list = []
    for pred in prediction_indices:
        if not pred:
            bird_name_list.append("nocall")
        else:
            bird_names = [REVERSE_LABEL_MAPPING[bird_id] for bird_id in pred]
            bird_name_list.append(" ".join(bird_names))
    return bird_name_list


def generate_predictions(models, dataset, return_names=True):
    all_predictions = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            batch_data = torch.from_numpy(dataset[idx]).to(DEVICE)
            
            ensemble_prediction = 0.0
            for model in models:
                output = model(batch_data)
                probabilities = torch.sigmoid(output)
                ensemble_prediction += probabilities

            ensemble_prediction /= len(models)
            
            if return_names:
                thresholded = apply_threshold_to_predictions(ensemble_prediction)
                final_predictions = convert_to_bird_names(thresholded)
            else:
                final_predictions = ensemble_prediction

            all_predictions.append(final_predictions)
    
    return all_predictions


raw_predictions = generate_predictions(trained_models, test_dataset, return_names=False)
print(f"Получено предсказаний: {len(raw_predictions)}")

final_predictions = [
    convert_to_bird_names(apply_threshold_to_predictions(pred, threshold=PREDICTION_THRESHOLD))
    for pred in raw_predictions
]


def create_submission_dataframe(metadata, predictions):
    submission_dict = {
        "row_id": [],
        "birds": [],
    }
    
    for row, pred_list in zip(metadata.itertuples(False), predictions):
        row_ids = [f"{row.id}_{row.site}_{5*i}" for i in range(1, len(pred_list)+1)]
        submission_dict["birds"].extend(pred_list)
        submission_dict["row_id"].extend(row_ids)
    
    submission_df = pd.DataFrame(submission_dict)
    
    if SUBMISSION_SAMPLE_PATH:
        sample_submission = pd.read_csv(SUBMISSION_SAMPLE_PATH, usecols=["row_id"])
        submission_df = sample_submission.merge(submission_df, on="row_id", how="left")
        submission_df["birds"] = submission_df["birds"].fillna("nocall")
    
    return submission_df

submission_df = create_submission_dataframe(audio_files_df, final_predictions)

submission_df.to_csv("submission.csv", index=False)

