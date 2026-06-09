from datasets import load_dataset

import librosa
import matplotlib.pyplot as plt

import torch
from torchaudio import transforms

import IPython.display as ipd


import sys

sys.path.append("/mnt/home/rheinrich/deep_bird_detect/datapipeline/Bird2Vec")

from birdset.augmentations import AudioAugmentor


esc = load_dataset("ashraq/esc50", split="train")


def normalize_audio(audio_tensor, mean, std):
    return (audio_tensor - mean) / std


def preprocess(
    waveform,
    use_spectrogram,
    spectrogram_augmentations=None,
    waveform_augmentations=None,
    n_fft=1024,
    hop_length=512,
    n_mels=None,
):
    audio_augmentor = AudioAugmentor(
        sample_rate=waveform["sampling_rate"],
        use_spectrogram=use_spectrogram,
        spectrogram_augmentations=spectrogram_augmentations,
        waveform_augmentations=waveform_augmentations,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        db_scale=True,
    )
    audio_augmented = audio_augmentor.combined_augmentations(waveform["array"])

    # audio_augmented = normalize_audio(audio_augmented)
    return audio_augmented


# path to the aufio files which are used for creating background noise
speech_command_path = (
    "/mnt/home/rheinrich/deep_bird_detect/datapipeline/speech_command_dataset"
)

# dictionary which defines the waveform data augmentations to be used
waveform_augmentations_dict = {
    "colored_noise": {
        "prob": 0.5,
        "min_snr_in_db": 3.0,
        "max_snr_in_db": 30.0,
        "min_f_decay": -2.0,
        "max_f_decay": 2.0,
    },
    "background_noise": {
        "background_paths": speech_command_path,
        "prob": 0.5,
        "min_snr_in_db": 3.0,
        "max_snr_in_db": 30.0,
    },
    "pitch_shift": {
        "prob": 0.5,
        "min_transpose_semitones": -4.0,
        "max_transpose_semitones": 4.0,
    },
    "time_mask": {"prob": 0.5, "min_band_part": 0.0, "max_band_part": 0.5},
    "time_stretch": {"prob": 0.5, "min_rate": 0.8, "max_rate": 1.25},
}


def train_transforms_waveform(examples):
    examples["audio_array_preprocessed"] = [
        preprocess(
            waveform=audio,
            waveform_augmentations=waveform_augmentations_dict,
            use_spectrogram=False,
        )
        for audio in examples["audio"]
    ]
    return examples


esc.set_transform(train_transforms_waveform)


waveform = esc[101]["audio"]["array"]
sr = esc[101]["audio"]["sampling_rate"]
ipd.Audio(data=waveform, rate=sr)


waveform_augmented = esc[101]["audio_array_preprocessed"]
sr = esc[101]["audio"]["sampling_rate"]
ipd.Audio(data=waveform_augmented, rate=sr)


esc = load_dataset("ashraq/esc50", split="train")


# path to the aufio files which are used for creating background noise
speech_command_path = (
    "/mnt/home/rheinrich/deep_bird_detect/datapipeline/speech_command_dataset"
)

# dictionary which defines the waveform data augmentations to be used
waveform_augmentations_dict = {
    "colored_noise": {
        "prob": 0.5,
        "min_snr_in_db": 3.0,
        "max_snr_in_db": 30.0,
        "min_f_decay": -2.0,
        "max_f_decay": 2.0,
    },
    "background_noise": {
        "background_paths": speech_command_path,
        "prob": 0.5,
        "min_snr_in_db": 3.0,
        "max_snr_in_db": 30.0,
    },
    "pitch_shift": {
        "prob": 0.5,
        "min_transpose_semitones": -4.0,
        "max_transpose_semitones": 4.0,
    },
}


# dictionary which defines the spectrogram data augmentations to be used
spectrogram_augmentations_dict = {
    "time_masking": {"time_mask_param": 100, "prob": 0.5},
    "frequency_masking": {"freq_mask_param": 100, "prob": 0.5},
    "time_stretch": {
        "prob": 0.33,
        "min_rate": 0.8,
        "max_rate": 1.25,
    },
}


def train_transforms_with_spec(examples):
    examples["audio_array_preprocessed"] = [
        preprocess(
            waveform=audio,
            use_spectrogram=True,
            spectrogram_augmentations=spectrogram_augmentations_dict,
            waveform_augmentations=waveform_augmentations_dict,
        )
        for audio in examples["audio"]
    ]
    return examples


esc.set_transform(train_transforms_with_spec)


spectrogram = esc[101]["audio_array_preprocessed"]


plt.imshow(spectrogram.squeeze().numpy())


transform_to_waveform = transforms.GriffinLim(n_fft=1024, hop_length=512)


spectrogram_unscaled = spectrogram.numpy()
spectrogram_unscaled = librosa.db_to_power(spectrogram_unscaled)
spectrogram_unscaled = torch.from_numpy(spectrogram_unscaled)
waveform_from_spectrogram = transform_to_waveform(spectrogram_unscaled)


sr = esc[101]["audio"]["sampling_rate"]
ipd.Audio(data=waveform_from_spectrogram, rate=sr)


from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def collate_batch(batch):
    input_features = [x["audio_array_preprocessed"] for x in batch]
    input_features = torch.cat(input_features, 0)
    targets = [x["target"] for x in batch]
    targets = torch.Tensor(targets)
    return input_features, targets


dataloader = DataLoader(esc, batch_size=5, shuffle=True, collate_fn=collate_batch)


for i in dataloader:
    print(i)
    break




