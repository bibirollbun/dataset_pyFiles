import pandas as pd
import pandas.api.types

import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype  for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return kaggle_metric_utilities.safe_call_score(sklearn.metrics.roc_auc_score, solution[scored_columns].values, submission[scored_columns].values, average='macro')


import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
import numpy as np
import seaborn as sns

annotation_path = '/kaggle/input/birdclef-2025/train.csv'
train_df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
train_soundscape_dir = '/kaggle/input/birdclef-2025/train_soundscapes/'
train_audio_dir = '/kaggle/input/birdclef-2025/train_audio/'
test_soundscape_dir = '/kaggle/input/birdclef-2025/test_soundscapes/'


train_df.head(5)


print("Number of unique labels is: ", end="")
print(len(train_df['primary_label'].unique()))


train_df.info()


count_unique_by_rating = train_df.groupby('rating')['primary_label'].nunique()
plt.title("Count of unique primary_labels by rating")
plt.bar(count_unique_by_rating.index, count_unique_by_rating.values, width=0.2)
plt.xlabel('Rating')
plt.ylabel('Count of unique primary_label')
plt.show()
count_unique_by_rating


pd.set_option('display.max_colwidth', None)
pivot_table = pd.crosstab(train_df['common_name'], train_df['rating'])
pivot_table


train_df.groupby('common_name').size().sort_values(ascending=True)


import os
import pandas as pd


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import Spectrogram
from torchaudio import load
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm


def WindowingMelSpec(mel_spec, mel_spec_freq, win_length:int=100, hop_length:int=50, n_windows_limit:int=600):
    """    
        Windowing spectrogram with 50% overlap.
        Window length = sample_rate * seconds
        Overlap 50%
        Max number of window should be 500 windows which should cover the given setting for 5 minutes audio
    
        Default setting: 10ms = 1 mel_spec frame = 10ms * 32_000 samples so if we want 1 second window frame with 0.5 overlap
        we should do (1000ms / 10ms) = 100 frames with Hop = (100 * 0.5) = 50
        5 min = 300_000 ms so max_windows = 300_000ms / hop_duration = 300_000 / (50 * 10) = 600 windows
    """
    # Input tensor mel_spec = (mel_spec_freq, num_of_frames)
    # Output tensor
    windows = []
    num_frames = mel_spec.shape[-1]
    mel_spec = mel_spec.squeeze()

    for start in range(0, num_frames - win_length + 1, hop_length):
        end = start + win_length
        window = mel_spec[:, start:end]
        windows.append(window)

    n_windows = min(len(windows), n_windows_limit)

    sample_windows = torch.zeros((n_windows_limit, mel_spec_freq, win_length))
    for i in range(n_windows):
        sample_windows[i] = windows[i]

    return sample_windows, n_windows

class WindowingModule(nn.Module):
    def __init__(self, mel_spec_freq=128, win_length=100, hop_length=50, n_windows_limit=600):
        super().__init__()
        self.mel_spec_freq = mel_spec_freq
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_windows_limit = n_windows_limit

    def forward(self, mel_spec):
        windows, n_windows = WindowingMelSpec(
            mel_spec,
            mel_spec_freq=self.mel_spec_freq,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_windows_limit=self.n_windows_limit
        )
        return windows, n_windows



SAMPLE_RATE = 32_000
MEL_SPEC_WINDOW = SAMPLE_RATE * 0.025
MEL_SPEC_HOP = SAMPLE_RATE * 0.01
N_MELS = 128
NFFT = 1024


    
mel_transform = nn.Sequential(
    T.Preemphasis(),
    T.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft=NFFT,
        win_length=int(MEL_SPEC_WINDOW),
        hop_length=int(MEL_SPEC_HOP),
        n_mels=N_MELS
    ),
    T.AmplitudeToDB(stype='power'),
    WindowingModule(mel_spec_freq=N_MELS)
)


class SoundscapeData(Dataset):
    def __init__(self, audio_dir, transform=None, transform_target=None):
        self.audio_dir = audio_dir
        self.audio_paths = os.listdir(audio_dir)
        self.transform = transform

    def __getitem__(self, idx):
        """
            Return windowed mel spectrograms and the number of valid windows
        """
        audio, rate = torchaudio.load(self.audio_dir + self.audio_paths[idx])

        if self.transform:
            audio, n_windows = self.transform(audio)
            return audio, n_windows

        return audio
        
    def __len__(self):
        return len(self.audio_dir)


import os
import torch
from torch.utils.data import Dataset

class LabeledData(Dataset):
    def __init__(self, annotation_df, audio_dir, transform=None, transform_target=None):
        self.annotation_df = annotation_df
        self.transform = transform
        self.audio_dir = audio_dir

        self.unique_labels = sorted(annotation_df['primary_label'].unique())
        self.label_to_id = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.transform_target = transform_target

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.annotation_df.iloc[idx]['filename'])
        audio, rate = torchaudio.load(audio_path)

        label_str = self.annotation_df.iloc[idx]['primary_label']
        label_id = self.label_to_id[label_str]

        if self.transform:
            audio, n_windows = self.transform(audio)

        if self.transform_target:
            label_id = self.transform_target(label_id)

        return audio, n_windows, label_id

    def __len__(self):
        return len(self.annotation_df)



mel_soundscape_dataset = SoundscapeData(train_soundscape_dir, transform=mel_transform)
mel_train_dataset = LabeledData(train_df, train_audio_dir, transform=mel_transform)


print(mel_train_dataset[100][0].shape)
print(mel_soundscape_dataset[100][0].shape)


def collate_fn(batch):
    mel_specs = []
    labels = []
    for mel_spec, n_windows, label in batch:
        mel_specs.append(mel_spec[:])
        labels.append(label)

    train = torch.stack(mel_specs)
    return (train, labels)


from torch.utils.data import WeightedRandomSampler

labels = train_df['primary_label'].values

unique_labels = np.unique(labels)
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

int_labels = np.array([label_to_id[label] for label in labels])

class_counts = np.bincount(int_labels)
class_weights = 1. / class_counts
sample_weights = class_weights[int_labels]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


mel_train_loader = DataLoader(mel_train_dataset, batch_size=50, collate_fn=collate_fn, sampler=sampler)


X, Y = next(iter(mel_train_loader))


encoder = AudioEncoder()
projection_head = ProjectionHead(in_dim=512)
model = SimCLR(encoder, projection_head)
model.load_state_dict(torch.load('/kaggle/working/simclr_epoch_10.pt'))

class AudioClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.num_classes = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)


train_losses = []
epochs = 100

for epoch in epochs:
    model.train()
    
    for step, data, label in enumerate(tqdm(mel_train_loader, desc=f"Epoch {epoch + 1} batch size: {augmented_soundscape_pairs_loader.batch_size}")):
        


len(mel_train_dataset)


# Training setupa
loss = []
epochs = 50


def collate_soundscape(batch):
    mel_windows_list, n_windows_list = zip(*batch)

    batch_tensor = torch.stack(mel_windows_list)
    n_windows_tensor = torch.tensor(n_windows_list)


    return batch_tensor, n_windows_tensor


from torchvision import models

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet18()
        # Replace first layer to accept 1-channel audio (mel spectrogram)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Exclude FC layer
        self.out_dim = base_model.fc.in_features  # 512

    def forward(self, x):  # x: (batch, 1, 128, 100)
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # (batch, 512)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h):
        return self.proj(h)

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z


import random

class LightAugmentAudioModule(nn.Module):
    def __init__(self, augment_prob=1):
        super().__init__()
        self.augment_prob = augment_prob
        
    def forward(self, audio):
        if not self.training or random.random() > self.augment_prob:
            return audio
            
        original_shape = audio.shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Only fast augmentations
        # Add noise
        if random.random() < 0.4:
            noise = torch.randn_like(audio) * random.uniform(0.001, 0.005)
            audio = audio + noise
        
        # Volume scaling
        if random.random() < 0.5:
            audio = audio * random.uniform(0.8, 1.2)
        
        # Time masking
        if random.random() < 0.3:
            seq_len = audio.shape[-1]
            mask_length = int(seq_len * random.uniform(0.01, 0.1))
            if mask_length > 0:
                start_idx = random.randint(0, max(1, seq_len - mask_length))
                audio[..., start_idx:start_idx + mask_length] = 0
        
        # Polarity flip
        if random.random() < 0.1:
            audio = -audio
        
        audio = torch.clamp(audio, -1.0, 1.0)
        
        if len(original_shape) == 1:
            audio = audio.squeeze(0)
            
        return audio


class AugmentedSoundscapePairData(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.audio_paths = os.listdir(audio_dir)
        self.mel_transform = mel_transform
        self.transform = transform

    def __len__(self):
        return len(self.audio_paths)
        
    def __getitem__(self, idx):
        raw_audio, rate = torchaudio.load(self.audio_dir + self.audio_paths[idx])

        augmented_mel_spec1, n_windows1 = self.transform(raw_audio)
        augmented_mel_spec2, n_windows2 = self.transform(raw_audio)
        
        return augmented_mel_spec1, augmented_mel_spec2, n_windows1


augmented_mel_transform = nn.Sequential(
    LightAugmentAudioModule(),
    mel_transform
)


augmented_soundscape_pairs_data = AugmentedSoundscapePairData(train_soundscape_dir, transform=augmented_mel_transform)

# batch_size * tuples([max_windows, mel_freq, win_size], [max_windows, mel_freq, win_size], valid_windows)


def simclr_collate_fn(batch):
    view1, view2 = [], []

    for mel1, mel2, valid in batch:
        view1.append(mel1[:valid])
        view2.append(mel2[:valid])

    view1 = torch.cat(view1, dim=0).unsqueeze(1)  # (N, 1, 128, 100)
    view2 = torch.cat(view2, dim=0).unsqueeze(1)
    return view1.float(), view2.float()            # be sure they’re float32



augmented_soundscape_pairs_loader = DataLoader(augmented_soundscape_pairs_data, batch_size=1, shuffle=True, collate_fn=simclr_collate_fn)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2N, dim]

        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2N, 2N]

        labels = torch.arange(N, device=z.device)
        labels = torch.cat([labels + N, labels], dim=0)

        # mask out self-similarities
        mask = torch.eye(2 * N, device=z.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        positives = similarity_matrix[torch.arange(2 * N), labels]
        loss = -positives / self.temperature + torch.logsumexp(similarity_matrix / self.temperature, dim=1)
        return loss.mean()


import os
import torch
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = AudioEncoder().to(device)
projection_head = ProjectionHead(in_dim=encoder.out_dim).to(device)
model = SimCLR(encoder, projection_head).to(device)

criterion = NTXentLoss(temperature=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

checkpoint_path = "/kaggle/working/simclr_epoch_3.pt"
current_epoch = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        current_epoch = checkpoint['epoch']
        print("Loaded model weights from checkpoint.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights only (raw state_dict).")
else:
    print("No checkpoint found, starting from scratch.")

print(f"Current epoch: {current_epoch}")


from datetime import datetime

epochs = 20
train_losses = []

for epoch in range(current_epoch, current_epoch + epochs):
    model.train()
    total_loss = 0
    cum_loss = 0

    for step, (view1, view2) in enumerate(tqdm(augmented_soundscape_pairs_loader, desc=f"Epoch {epoch + 1}")):
        view1, view2 = view1.to(device), view2.to(device)

        _, z1 = model(view1)
        _, z2 = model(view2)

        loss = criterion(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        cum_loss += loss.item()
        if step % 1000 == 0:
            show_loss = None
            if step == 0:
                show_loss = cum_loss / 1
            else:
                show_loss = cum_loss / step
                
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1} Step {step} Cumulative Average Loss: {show_loss:.4f}")

    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"/kaggle/working/simclr_epoch_{epoch+1}.pt")
    print(f"Saved checkpoint at epoch {epoch+1}")
        
    avg_loss = total_loss / len(augmented_soundscape_pairs_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "/kaggle/working/simclr_audio.pth")


plt.plot(range(1, 4 + 1), train_losses, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

