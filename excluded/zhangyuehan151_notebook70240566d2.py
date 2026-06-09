import pandas as pd

# Load metadata
df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")

# Sample 3% for debugging
df_small = df.sample(frac=0.03, random_state=42).reset_index(drop=True)



import torch
from torch.utils.data import Dataset
import torchaudio
import torchvision.transforms as T
import numpy as np
import os

class BirdclefDataset(Dataset):
    def __init__(self, df, audio_dir, duration=5.0, sr=32000, transform=None):
        self.df = df
        self.audio_dir = audio_dir
        self.duration = duration
        self.sr = sr
        self.transform = transform
        self.samples = int(duration * sr)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=50,
            f_max=16000
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        label = row["primary_label"]

        filepath = os.path.join(self.audio_dir, filename)

        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[1] < self.samples:
            # pad if too short
            pad_len = self.samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :self.samples]

        mel = self.mel_transform(waveform).squeeze(0)  # (n_mels, time)

        # Normalize and convert to image format
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = torch.stack([mel, mel, mel], dim=0)  # (3, H, W)

        if self.transform:
            mel = self.transform(mel)

        return mel, label



transform = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])



from torch.utils.data import DataLoader

train_ds = BirdclefDataset(df_small, audio_dir='/kaggle/input/birdclef-2025/train_audio', transform=transform)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)




for x, y in train_dl:
    print(x.shape)  # should be [B, 3, 224, 224]
    print(y)
    break



import torchvision.models as models
import torch.nn as nn

NUM_CLASSES = df["primary_label"].nunique()  # should be 206 for full dataset


model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)



import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()  # multilabel setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



species_list = sorted(df['primary_label'].unique().tolist())
label_dict = {species: idx for idx, species in enumerate(species_list)}

def encode_labels(species_names, num_classes=206):
    # Create a one-hot encoded tensor for the labels
    label_tensor = torch.zeros(len(species_names), num_classes)
    for i, species in enumerate(species_names):
        label_index = label_dict.get(species, -1)  # Get the label index from the dictionary
        if label_index >= 0:  # If a valid species name
            label_tensor[i, label_index] = 1  # Set the index corresponding to the species
    return label_tensor




for batch in train_dl:
    mel = batch[0]  # mel spectrograms
    label_strs = batch[1]  # species names
    
    # Convert species names to indices
    label_tensor = encode_labels(label_strs).to(device).float()


    mel = mel.to(device)
    label_tensor = label_tensor.to(device).float()

    optimizer.zero_grad()
    output = model(mel)
    loss = criterion(output, label_tensor)
    loss.backward()
    optimizer.step()



print(batch[1])


for epoch in range(2):  # just a couple of epochs for testing
    model.train()
    running_loss = 0.0
    for mel, label_strs in train_dl:
        mel = mel.to(device)
        label_tensor = encode_labels(label_strs).to(device).float()

        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label_tensor)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_dl):.4f}")



for epoch in range(2):  # just a couple of epochs for testing
    model.train()
    running_loss = 0.0
    for mel, label_strs in train_dl:
        mel = mel.to(device)
        label_tensor = encode_labels(label_strs).to(device).float()

        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label_tensor)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_dl):.4f}")



torch.save(model.state_dict(), "birdclef_test_model.pth")



from IPython.display import FileLink

# Just show a clickable link
FileLink(r'./birdclef_test_model.pth')



import os
import torch
import librosa
import numpy as np
import pandas as pd
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Set seed
np.random.seed(42)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model (same architecture as used for training)
model = models.efficientnet_b0(pretrained=False)  # Do not load pre-trained weights
NUM_CLASSES = 206  # Update to your actual number of classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

# Load model weights
model.load_state_dict(torch.load("birdclef_test_model.pth"))
model = model.to(device)
model.eval()

# Transform to match model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels from train audio (ensure these match the labels used during training)
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))

test_soundscape_path = '/kaggle/input/birdclef-2025/train_soundscapes/'
test_soundscapes_all = [os.path.join(test_soundscape_path, afile) 
                        for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]

# Use only the first 3% for testing/debugging
num_files = max(1, int(0.01 * len(test_soundscapes_all)))  # at least 1 file
test_soundscapes = test_soundscapes_all[:num_files]

# DataFrame for predictions
predictions = pd.DataFrame(columns=['row_id'] + class_labels)

# Helper function to process the 5-second chunks
def process_audio_chunk(chunk, sr=32000):
    # Convert chunk to Mel Spectrogram (same process as during training)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=50,
        f_max=16000
    ).to(device)
    waveform = torch.tensor(chunk).unsqueeze(0).float().to(device)
    mel = mel_transform(waveform).squeeze(0)  # (n_mels, time)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)  # Normalize
    mel = torch.stack([mel, mel, mel], dim=0)  # Make 3-channel image (H, W, C)
    mel = transform(mel)  # Apply resize and normalization
    return mel.unsqueeze(0)  # Add batch dimension

# Loop through each test soundscape
for soundscape in test_soundscapes:
    # Load audio file
    sig, rate = librosa.load(path=soundscape, sr=32000)

    # Split into 5-second chunks
    chunks = [sig[i:i + rate * 5] for i in range(0, len(sig), rate * 5)]

    # Predict for each chunk
    for i, chunk in enumerate(chunks):
        # Create row_id based on soundscape and chunk number
        row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'

        # Process the chunk (convert to Mel spectrogram)
        mel = process_audio_chunk(chunk)

        # Make prediction (output from model)
        with torch.no_grad():
            output = model(mel)
            probs = torch.sigmoid(output)  # Apply sigmoid to get probabilities

        # Apply threshold to get the labels (0.5 is commonly used)
        threshold = 0
        pred_labels = [class_labels[i] for i, p in enumerate(probs[0]) if p >= threshold]
        
        # Create a new row for the prediction
        new_row = pd.DataFrame([[row_id] + [1 if label in pred_labels else 0 for label in class_labels]],
                               columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)

# Save the predictions to CSV
predictions.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')



# 1. Check output logits range
with torch.no_grad():
    output = model(mel)
print("Logits:", output[0][:10])  # first 10 classes

# 2. Check model weights
print("Weight avg:", next(model.parameters()).abs().mean().item())


