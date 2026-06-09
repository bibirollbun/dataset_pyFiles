# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import joblib

# ==== Constants ====
SAMPLE_RATE = 32000
AUDIO_LEN = 5 * SAMPLE_RATE
BATCH_SIZE = 32
EMBED_DIM = 2048
NUM_CLASSES = 206
EPOCHS = 5
TRAIN_CSV = '/kaggle/input/birdclef-2025/train.csv'
AUDIO_DIR = '/kaggle/input/birdclef-2025/train_audio'

# ==== Load Metadata ====
df = pd.read_csv(TRAIN_CSV)

# Use flat structure: train_audio/CSA36385.ogg etc.
df['filepath'] = df['filename'].apply(lambda x: os.path.join(AUDIO_DIR, x))

# ==== Multi-label binarizer for primary labels ====
mlb = MultiLabelBinarizer()
mlb.fit([[label] for label in df['primary_label'].unique()])
joblib.dump(mlb, 'label_binarizer.pkl')  # Save for inference

# ==== Load pretrained CNN14 from torch.hub ====
panns_model = torch.hub.load('qiuqiangkong/panns_transfer_cnn14', 'cnn14', pretrained=True)
panns_model.eval()
for param in panns_model.parameters():
    param.requires_grad = False

# ==== Dataset ====
class BirdSoundDataset(Dataset):
    def __init__(self, dataframe, label_binarizer):
        self.df = dataframe
        self.label_binarizer = label_binarizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        label = row['primary_label']

        try:
            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
        except:
            audio = np.zeros(AUDIO_LEN)

        if len(audio) > AUDIO_LEN:
            audio = audio[:AUDIO_LEN]
        elif len(audio) < AUDIO_LEN:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))

        waveform = torch.tensor(audio).unsqueeze(0).float()

        with torch.no_grad():
            embedding = panns_model(waveform)['embedding'].squeeze()

        y = self.label_binarizer.transform([[label]])[0]
        return embedding, torch.tensor(y, dtype=torch.float32)

# ==== DataLoader ====
dataset = BirdSoundDataset(df, mlb)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==== Classifier Head ====
class SpeciesClassifier(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, out_dim=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

model = SpeciesClassifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==== Training Loop ====
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in tqdm(train_loader):
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# ==== Save Model ====
torch.save(model.state_dict(), 'species_classifier.pth')
print("✅ Model and label binarizer saved.")



import os
import librosa
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

# Constants
SAMPLE_RATE = 32000
AUDIO_LEN = 5 * SAMPLE_RATE  # 5 seconds
TEST_DIR = '/kaggle/input/birdclef-2025/test_soundscapes'
SAMPLE_SUB_PATH = '/kaggle/input/birdclef-2025/sample_submission.csv'

# Load sample submission to get structure
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
species_ids = sample_sub.columns.tolist()[1:]  # skip 'row_id'

# Load label binarizer and trained classifier
mlb = joblib.load('label_binarizer.pkl')

class SpeciesClassifier(nn.Module):
    def __init__(self, in_dim=2048, out_dim=206):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, out_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

classifier = SpeciesClassifier()
classifier.load_state_dict(torch.load('species_classifier.pth', map_location='cpu'))
classifier.eval()

# Load pretrained PANNs CNN14
panns_model = torch.hub.load('qiuqiangkong/panns_transfer_cnn14', 'cnn14', pretrained=True)
panns_model.eval()
for param in panns_model.parameters():
    param.requires_grad = False

# Generate submission
row_ids, predictions = [], []

for filename in sorted(os.listdir(TEST_DIR)):
    if not filename.endswith('.ogg'):
        continue

    filepath = os.path.join(TEST_DIR, filename)
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=audio, sr=SAMPLE_RATE)

    for start in range(0, int(duration), 5):
        end = start + 5
        if end > duration:
            break

        segment = audio[start*SAMPLE_RATE:end*SAMPLE_RATE]
        if len(segment) < AUDIO_LEN:
            segment = np.pad(segment, (0, AUDIO_LEN - len(segment)))

        waveform = torch.tensor(segment).unsqueeze(0).float()

        with torch.no_grad():
            embedding = panns_model(waveform)['embedding']
            probs = classifier(embedding).squeeze().numpy()

        row_id = f"{filename.replace('.ogg', '')}_{end}"
        row_ids.append(row_id)
        predictions.append(probs)

# Build submission DataFrame
submission_df = pd.DataFrame(predictions, columns=species_ids)
submission_df.insert(0, 'row_id', row_ids)

# Ensure it matches sample_submission structure
submission_df = submission_df[sample_sub.columns]

# Save
submission_df.to_csv('submission.csv', index=False)
print("✅ submission.csv created.")


















