import numpy as np
import pandas as pd
import os
from pathlib import Path

input_path = Path('/kaggle/input')
birdclef_path = input_path / 'birdclef-2025'
meta_path = input_path / 'birdclef-2025-train-meta-extra-updated-1' / 'train_meta_extra_updated_2.csv'
taxonomy_path = birdclef_path / 'taxonomy.csv'
train_path = birdclef_path / 'train_audio'
train_soundscapes_path = birdclef_path / 'train_soundscapes'
test_soundscapes_path = birdclef_path / 'test_soundscapes'
checkpoint_path = input_path / 'birdclef_2025_efficientnet_b0' / 'pytorch' / 'default' / '1' / 'efficientnet_b0_0000.ckpt'


import random

config = {
    'experiment_name': 'efficientnet_0000',
    'seed': 42,
    'spectrogram_input': True,
    'spectrogram_params': {
        'n_mels': 256,
        'fmin': 20,
        'fmax': 15_000
    },
    'validation_size': 0.2,
    'batch_size': 128,
    'cw_len': 5000,
    'pretrained_model': True,
    'early_stopping_patience': 3,
    'N_batches': 1.0,
    'N_epochs': 100,
    'sample': False,
    'sample_fraction': 1.0,
    'oversample': True,
    'oversample_factor': 1.0,
    'model': 'EfficientNet'
}

seed = config['seed']
random.seed(seed)
rng = np.random.default_rng(seed=seed)


import torch
import torch.nn as nn
import timm

class EfficientNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool):
        super().__init__()

        self.model = timm.create_model(
            'efficientnet_b0',
            in_chans=1,
            num_classes=0,
            pretrained=False
        )
        pool_out_shape = 1280
        self.classifier = nn.Linear(pool_out_shape, num_classes)

    def forward(self, x):
        features = self.model(x)
        output = self.classifier(features)
        return output


meta_df = pd.read_csv(meta_path)
all_species = sorted(meta_df['primary_label'].unique().tolist())


match config['model']:
    case 'EfficientNet':
        model = EfficientNet(len(all_species), config['pretrained_model'])
    case _:
        raise ValueError(f"Unknown model: {config['model']}")

checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))['state_dict']
state_dict = {k.partition('model.')[2]: v for k,v in checkpoint.items()}
model.load_state_dict(state_dict)


if len(os.listdir(test_soundscapes_path)) == 1:
    debug = True
    soundscapes_path = train_soundscapes_path
else:
    soundscapes_path = test_soundscapes_path

soundscape_files = [afile for afile in sorted(os.listdir(soundscapes_path)) if afile.endswith('.ogg')]
if debug:
    soundscape_files = soundscape_files[:700]


from tqdm import tqdm
import librosa
import cv2
from itertools import islice
import torch.nn.functional as F

def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

model.eval()
class_labels = all_species
predict_batch_size = 16

fs = 32_000

def get_mel_spec(audio: np.ndarray, spectrogram_params: dict) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=fs,
        power=2.0,
        **spectrogram_params
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + np.finfo(np.float32).eps)
    return np.expand_dims(cv2.resize(mel_spec, (256, 256), interpolation=cv2.INTER_LINEAR), 0)

def get_soundscape(soundscape):
    # Load audio
    sig, rate = librosa.load(path=soundscapes_path / soundscape, sr=None)

    # Split into 5-second chunks
    chunks = []
    for i in range(0, len(sig), rate*5):
        chunk = sig[i:i+rate*5]
        chunks.append(get_mel_spec(chunk, config['spectrogram_params']))

    row_id_prefix = os.path.basename(soundscape).split('.')[0]
    row_ids = [row_id_prefix + f'_{i * 5 + 5}' for i in range(len(chunks))]
    return chunks, row_ids

# Open each soundscape and make predictions for 5-second segments
# Use pandas df with 'row_id' plus class labels as columns
predictions = pd.DataFrame(columns=['row_id'] + class_labels)
for soundscapes in tqdm(batched(soundscape_files, predict_batch_size)):
    chunks = []
    all_row_ids = []
    for soundscape in soundscapes:
        chunk, row_ids = get_soundscape(soundscape)
        chunks.extend(chunk)
        all_row_ids.extend(row_ids)

    all_row_ids = pd.Series(all_row_ids)
    chunks = torch.tensor(np.array(chunks))
    with torch.no_grad():
        scores = pd.DataFrame(F.sigmoid(model(chunks)))
    
    new_rows = pd.concat([all_row_ids, scores], axis=1)
    new_rows.columns = ['row_id'] + class_labels

    predictions = pd.concat([predictions, new_rows], axis=0, ignore_index=True)
    
# Save prediction as csv
predictions.to_csv('submission.csv', index=False)
predictions.head()




