import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

import tensorflow_hub as hub
import tensorflow as tf

import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler


df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
AUDIO_PATH = Path('/kaggle/input/birdclef-2025/train_audio')
SOUNDSCAPE_PATH = Path('/kaggle/input/birdclef-2025/train_soundscapes')
model_path = 'https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/8'
model = hub.load(model_path)
model_labels_df = pd.read_csv(hub.resolve(model_path) + "/assets/label.csv")

SAMPLE_RATE = 32000
WINDOW = 5*SAMPLE_RATE


index_to_label = sorted(df.primary_label.unique())
label_to_index = {v: k for k, v in enumerate(index_to_label)}
model_labels = {v: k for k, v in enumerate(model_labels_df.ebird2021)}
model_bc_indexes = [model_labels[label] if label in model_labels else -1 for label in index_to_label]

# filter out birds that the model doesn't predict
missing_birds = set(np.array(index_to_label)[np.array(model_bc_indexes) == -1])
len(missing_birds)


# use a torch dataloader to decode audio in parallel on CPU while GPU is running
class SoundScapeDataset(Dataset):
    def __init__(self):
        self.path = SOUNDSCAPE_PATH
        self.files = [f for f in os.listdir(self.path) if f.endswith('.ogg')]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        try:
            file_path = os.path.join(self.path, self.files[i])
            audio, sr = torchaudio.load(file_path)
            return audio[0], self.files[i]
        except Exception as e:
            print(f"Error loading {self.files[i]}: {str(e)}")
            return torch.zeros(1, 16000), "error_file"  # 返回空数据
dataloader = DataLoader(
    SoundScapeDataset(),
    batch_size=1,
    # sampler=SequentialSampler(range(10)),
    num_workers=os.cpu_count()
)

# embeddings are formated like {"filename": np.array(nx1280)} 
# (where n = the number of non overlapping 5 sec chunks in the audio)
all_embeddings = {}

# predictiones formated like {"filename": np.array(nx264)} 
all_predictions = {}

with tf.device('/gpu:0'):
    for audio, filename in tqdm(dataloader):
        audio = audio[0]
        filename = filename[0]
        file_embeddings = []
        file_predictions = []
        for i in range(0, len(audio), WINDOW):
            clip = audio[i:i+WINDOW]
            if len(clip) < WINDOW:
                clip = np.concatenate([clip, np.zeros(WINDOW - len(clip))])
            result = model.infer_tf(clip[np.newaxis, :])
            file_embeddings.append(result['embedding'].numpy())
            prediction = np.concatenate([result['label'].numpy(), -100], axis=None) # add -100 logit for unpredicted birds
            file_predictions.append(prediction[model_bc_indexes])
        all_embeddings[filename] = np.stack(file_embeddings)
        all_predictions[filename] = np.stack(file_predictions)

torch.save(all_embeddings, 'train_soundscape_embeddings.pt')
torch.save(all_predictions, 'train_soundscape_predictions.pt')


# use a torch dataloader to decode audio in parallel on CPU while GPU is running
class AudioDataset(Dataset):
    def __len__(self):
        return len(df)
    def __getitem__(self, i):
        filename = df.filename[i]
        audio = torchaudio.load(AUDIO_PATH / filename)[0].numpy()[0]
        return audio, filename
dataloader = DataLoader(
    AudioDataset(),
    batch_size=1,
    # sampler=SequentialSampler(range(10)),
    num_workers=os.cpu_count()
)

# embeddings are formated like {"filename": np.array(nx1280)} 
# (where n = the number of non overlapping 5 sec chunks in the audio)
all_embeddings = {}

# predictiones formated like {"filename": np.array(nx264)} 
all_predictions = {}

with tf.device('/gpu:0'):
    for audio, filename in tqdm(dataloader):
        audio = audio[0]
        filename = filename[0]
        file_embeddings = []
        file_predictions = []
        for i in range(0, len(audio), WINDOW):
            clip = audio[i:i+WINDOW]
            if len(clip) < WINDOW:
                clip = np.concatenate([clip, np.zeros(WINDOW - len(clip))])
            result = model.infer_tf(clip[np.newaxis, :])
            file_embeddings.append(result['embedding'].numpy())
            prediction = np.concatenate([result['label'].numpy(), -100], axis=None) # add -100 logit for unpredicted birds
            file_predictions.append(prediction[model_bc_indexes])
        all_embeddings[filename] = np.stack(file_embeddings)
        all_predictions[filename] = np.stack(file_predictions)

torch.save(all_embeddings, 'train_audio_embeddings.pt')
torch.save(all_predictions, 'train_audio_predictions.pt')

