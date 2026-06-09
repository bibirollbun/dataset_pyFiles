import os
import random
import soundfile as sf
import librosa
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


class BirdDataset(Dataset):
    def __init__(self, metadata, audio_dir, sr=32000, duration=5, n_mels=128):
        self.metadata = pd.read_csv("/kaggle/input/metadata-clean/metadata_clean.csv")
        self.audio_dir = audio_dir
        self.sr = sr
        self.duration = duration
        self.n_samples = sr * duration
        self.n_mels = n_mels
        self.label_map = {
            label: i for i, label in enumerate(metadata["species"].unique())
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        max_attempts = 10  # Prevent infinite loops

        for attempt in range(max_attempts):
            row = self.metadata.iloc[idx]
            path = os.path.join(self.audio_dir, row["filename"])
            label = self.label_map[row["species"]]

            if not os.path.exists(path):
                print(f"ЁЯЪл Missing file: {path}")
                idx = random.randint(0, len(self.metadata) - 1)
                continue

            try:
                y, _ = librosa.load(path, sr=self.sr, duration=self.duration)
                break  # success
            except Exception as e:
                print(f"тЪая╕П Error loading file: {path}\n{e}")
                idx = random.randint(0, len(self.metadata) - 1)

        else:
            raise RuntimeError("Too many corrupted or missing files in a row.")

        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)))

        mel = librosa.feature.melspectrogram(y=y, sr=self.sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel_db).unsqueeze(0)
        return mel_tensor, label

# CNN Model
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# === Load clean metadata ===
metadata = pd.read_csv("/kaggle/input/metadata-clean/metadata_clean.csv")
dataset = BirdDataset(metadata, audio_dir="/kaggle/input/birdclef-2025/train_audio")
label_map = dataset.label_map
inv_label_map = {v: k for k, v in label_map.items()}

# === Load model ===
model = CNNClassifier(num_classes=len(label_map))
model.load_state_dict(torch.load("/kaggle/input/birdclef-l-cn240/pytorch/default/1/birdclef_cnn.pth", weights_only=True))
model.eval()

# === Make predictions ===
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, (inputs, label) in enumerate(loader):
    with torch.no_grad():
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs, 1)
        true_label = inv_label_map[label.item()]
        pred_label = inv_label_map[predicted.item()]
        print(f"Sample {i}: True = {true_label}, Predicted = {pred_label}")

    if i == 9:  # just show 10 predictions
        break



import os
import librosa
import numpy as np
import pandas as pd

# Set seed
np.random.seed(42)

# Class labels from train audio
class_labels = sorted(os.listdir("/kaggle/input/birdclef-2025/train_audio/"))

# List of test soundscapes (only visible during submission)
test_soundscape_path = "/kaggle/input/birdclef-2025/test_soundscapes/"
test_soundscapes = [
    os.path.join(test_soundscape_path, afile)
    for afile in sorted(os.listdir(test_soundscape_path))
    if afile.endswith(".ogg")
]

# Open each soundscape and make predictions for 5-second segments
# Use pandas df with 'row_id' plus class labels as columns
predictions = pd.DataFrame(columns=["row_id"] + class_labels)
for soundscape in test_soundscapes:

    # Load audio
    sig, rate = librosa.load(path=soundscape, sr=None)

    # Split into 5-second chunks
    chunks = []
    for i in range(0, len(sig), rate * 5):
        chunk = sig[i : i + rate * 5]
        chunks.append(chunk)

    # Make predictions for each chunk
    for i, chunk in enumerate(chunks):

        # Get row id  (soundscape id + end time of 5s chunk)
        row_id = os.path.basename(soundscape).split(".")[0] + f"_{i * 5 + 5}"

        # Make prediction (let's use random scores for now)
        # scores = model.predict...
        scores = np.random.rand(len(class_labels))

        # Append to predictions as new row
        new_row = pd.DataFrame(
            [[row_id] + list(scores)], columns=["row_id"] + class_labels
        )
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)

# Save prediction as csv
predictions.to_csv("submission.csv", index=False)
predictions.head()


