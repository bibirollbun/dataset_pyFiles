import pandas as pd
import numpy as np


import pandas as pd

# Train Õ¿Õ¾ÕµÕ¡Õ¬Õ¶Õ¥Ö€Õ¨
df_train = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
print("ğŸ“Š train.csv shape:", df_train.shape)
display(df_train.head())

# Taxonomy Õ¿Õ¾ÕµÕ¡Õ¬Õ¶Õ¥Ö€Õ¨
df_taxonomy = pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")
print("\nğŸŒ¿ taxonomy.csv shape:", df_taxonomy.shape)
display(df_taxonomy.head())

# Õ†Õ¥Ö€Õ¯Õ¡ÕµÕ¡Ö�Õ´Õ¡Õ¶ Õ¶Õ´Õ¸Ö‚Õ·
df_sample = pd.read_csv("/kaggle/input/birdclef-2025/sample_submission.csv")
print("\nğŸ“¤ sample_submission.csv shape:", df_sample.shape)
display(df_sample.head())



pip install librosa matplotlib soundfile



import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def show_mel_spectrogram(file_path, sr=32000):
    audio, _ = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

# Õ•Ö€Õ«Õ¶Õ¡Õ¯ 1-Õ«Õ¶ train Ö†Õ¡ÕµÕ¬Õ¨ Õ¾Õ¥Ö€Ö�Õ¶Õ¥Õ¬
example_path = os.path.join("/kaggle/input/birdclef-2025/train_audio", "/kaggle/input/birdclef-2025/train_audio/1139490", "/kaggle/input/birdclef-2025/train_audio/1139490/CSA36385.ogg")
show_mel_spectrogram(example_path)



import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def extract_mel_spectrograms(file_path, sr=32000, duration=5, hop_length=512, n_mels=128):
    audio, _ = librosa.load(file_path, sr=sr)
    total_secs = int(len(audio) / sr)
    segments = []

    for start in range(0, total_secs, duration):
        end = start + duration
        if end > total_secs:
            break
        start_sample = start * sr
        end_sample = end * sr
        segment = audio[start_sample:end_sample]

        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        segments.append(mel_db)

    return segments

# Õ•Ö€Õ«Õ¶Õ¡Õ¯Õ� ÖƒÕ¸Ö€Õ±Õ¥Õ¶Ö„ 1 Ö†Õ¡ÕµÕ¬Õ« Õ¾Ö€Õ¡
example_path = os.path.join("/kaggle/input/birdclef-2025/train_audio", "1139490", "CSA36385.ogg")
mel_segments = extract_mel_spectrograms(example_path)

print(f"ğŸ“ˆ ÕŠÕ¡Õ¿Ö€Õ¡Õ½Õ¿Õ¾Õ¡Õ® Õ§ {len(mel_segments)} Õ¯Õ¿Õ¸Ö€")
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_segments[0], sr=32000, x_axis='time', y_axis='mel')
plt.colorbar()
plt.title("Mel Spectrogram - Segment 1")
plt.tight_layout()
plt.show()



from glob import glob

def build_mel_dataset(train_df, base_audio_path, max_files=50):  # max_files=50 ÕªÕ¡Õ´Õ¡Õ¶Õ¡Õ¯Õ¡Õ¾Õ¸Ö€Õ¡ÕºÕ¥Õ½
    dataset = []

    for i, row in tqdm(train_df.iterrows(), total=min(len(train_df), max_files)):
        label = row['primary_label']
        rel_path = row['filename']  # e.g., 1139490/CSA36385.ogg
        full_path = os.path.join(base_audio_path, rel_path)

        if not os.path.exists(full_path):
            continue

        try:
            segments = extract_mel_spectrograms(full_path)
            for mel in segments:
                dataset.append({
                    "mel": mel,
                    "label": label
                })
        except Exception as e:
            print(f"â�Œ Error with {full_path}: {e}")
            continue

        if i + 1 >= max_files:  # early stop
            break

    return dataset

# Õ•Õ£Õ¿Õ¡Õ£Õ¸Ö€Õ®Õ¸Ö‚Õ´ Õ¥Õ¶Ö„
mel_dataset = build_mel_dataset(df_train, base_audio_path="/kaggle/input/birdclef-2025/train_audio", max_files=30)

print(f"\nâœ… Õ�Õ¿Õ¡Ö�Õ¾Õ¥Ö� {len(mel_dataset)} Õ´Õ¥Õ¬ Õ½ÕºÕ¥Õ¯Õ¿Ö€Õ¸Õ£Ö€Õ¡Õ´ Õ¯Õ¿Õ¸Ö€ Õ¨Õ¶Õ¤Õ°Õ¡Õ¶Õ¸Ö‚Ö€ {len(set([x['label'] for x in mel_dataset]))} Õ¿Õ¡Ö€Õ¢Õ¥Ö€ Õ©Õ¥Õ£Õ¥Ö€Õ¸Õ¾")



from sklearn.preprocessing import LabelEncoder
import torch

# Õ�Õ¿Õ¥Õ²Õ®Õ¸Ö‚Õ´ Õ¥Õ¶Ö„ label encoder
label_encoder = LabelEncoder()
all_labels = [x['label'] for x in mel_dataset]
label_encoder.fit(all_labels)
num_classes = len(label_encoder.classes_)

print("ğŸ“š Ô´Õ¡Õ½Õ¥Ö€Õ« Ö„Õ¡Õ¶Õ¡Õ¯:", num_classes)



from torch.utils.data import Dataset

class BirdClefDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel = self.data[idx]['mel']
        label = self.data[idx]['label']

        # Normalize spectrogram
        mel = (mel - mel.min()) / (mel.max() - mel.min())

        # To tensor, add channel dim
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # shape: [1, H, W]

        # Label as index (for now)
        label_index = self.encoder.transform([label])[0]
        label_tensor = torch.tensor(label_index, dtype=torch.long)

        return mel_tensor, label_tensor



dataset = BirdClefDataset(mel_dataset, label_encoder)
x, y = dataset[0]
print("ğŸ“� Input shape:", x.shape)
print("ğŸ�·ï¸� Label index:", y.item(), "=", label_encoder.inverse_transform([y.item()])[0])



import torch.nn as nn
import torch.nn.functional as F

class BirdCLEFCNN(nn.Module):
    def __init__(self, num_classes):
        super(BirdCLEFCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 32 * 78, 128)  # depends on input shape
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, H/2, W/2]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, H/4, W/4]
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x



from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdCLEFCNN(num_classes=num_classes).to(device)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train 1 epoch
model.train()
for epoch in range(1):
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"ğŸ“š Epoch {epoch+1} â€” Loss: {running_loss:.4f} â€” Accuracy: {acc:.4f}")



mel_dataset = build_mel_dataset(df_train, base_audio_path="/kaggle/input/birdclef-2025/train_audio", max_files=1000)


from collections import Counter

label_counts = Counter([x['label'] for x in mel_dataset])
valid_labels = set([label for label, count in label_counts.items() if count >= 2])
filtered_dataset = [x for x in mel_dataset if x['label'] in valid_labels]

print(f"ğŸ“¦ Õ†Õ¡Õ­Ö„Õ¡Õ¶: {len(mel_dataset)} | Õ€Õ¥Õ¿Õ¸ Ö†Õ«Õ¬Õ¿Ö€Õ¸Ö‚Õ´Õ«Ö�: {len(filtered_dataset)}")



from sklearn.preprocessing import LabelEncoder

filtered_labels = [x['label'] for x in filtered_dataset]
label_encoder = LabelEncoder()
label_encoder.fit(filtered_labels)

num_classes = len(label_encoder.classes_)
print("ğŸ“š Õ�Õ¥Ö€Õ»Õ¶Õ¡Õ¯Õ¡Õ¶ Õ¤Õ¡Õ½Õ¥Ö€Õ« Ö„Õ¡Õ¶Õ¡Õ¯:", num_classes)



from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(
    filtered_dataset,
    test_size=0.2,
    stratify=filtered_labels,
    random_state=42
)

print(f"âœ… Train segments: {len(train_data)}, Validation segments: {len(val_data)}")



train_dataset = BirdClefDataset(train_data, label_encoder)
val_dataset = BirdClefDataset(val_data, label_encoder)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



model = BirdCLEFCNN(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    train_preds, train_labels = [], []
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    train_acc = accuracy_score(train_labels, train_preds)

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)

    print(f"ğŸ“š Epoch {epoch+1}: Train Loss={train_loss:.3f}, Acc={train_acc:.4f} | Val Loss={val_loss:.3f}, Acc={val_acc:.4f}")



import torch.nn.functional as F

class BirdClefEffDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel = self.data[idx]['mel']
        label = self.data[idx]['label']

        mel = (mel - mel.min()) / (mel.max() - mel.min())  # normalize
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Resize to 224Ã—224
        mel_tensor = F.interpolate(mel_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        mel_tensor = mel_tensor.squeeze(0)  # back to [1, 224, 224]

        label_index = self.encoder.transform([label])[0]
        label_tensor = torch.tensor(label_index, dtype=torch.long)

        return mel_tensor, label_tensor



import random

class BirdClefEffAugmentedDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel = self.data[idx]['mel']
        label = self.data[idx]['label']

        mel = (mel - mel.min()) / (mel.max() - mel.min())
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # SpecAugment-style masking
        if random.random() < 0.8:
            time_mask = random.randint(10, 40)
            freq_mask = random.randint(5, 20)
            time_start = random.randint(0, mel_tensor.shape[2] - time_mask)
            freq_start = random.randint(0, mel_tensor.shape[1] - freq_mask)
            mel_tensor[0, freq_start:freq_start+freq_mask, :] = 0
            mel_tensor[0, :, time_start:time_start+time_mask] = 0

        # Resize to 224x224
        mel_tensor = F.interpolate(mel_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        mel_tensor = mel_tensor.squeeze(0)

        label_tensor = torch.tensor(self.encoder.transform([label])[0], dtype=torch.long)
        return mel_tensor, label_tensor



import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



criterion = LabelSmoothingCrossEntropy(smoothing=0.1)


from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=5)



scheduler.step()



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()



import torchaudio
import librosa
import os
import numpy as np

def predict_with_tta(model, filepath, label_encoder, device, tta_segments=[0, 5, 10]):
    y, sr = librosa.load(filepath, sr=32000)
    model.eval()
    all_outputs = []

    for start in tta_segments:
        start_sample = start * sr
        end_sample = start_sample + 5 * sr
        if end_sample > len(y):
            break
        segment = y[start_sample:end_sample]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
        tensor = tensor.to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            all_outputs.append(probs.cpu().numpy())

    # Õ„Õ«Õ»Õ«Õ¶Õ¡Ö�Õ¶Õ¥Õ¬ Õ¢Õ¸Õ¬Õ¸Ö€ segment-Õ¶Õ¥Ö€Õ« Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Ö„Õ¶Õ¥Ö€Õ¨
    mean_probs = np.mean(all_outputs, axis=0)
    return mean_probs.flatten()



import pandas as pd

def generate_submission(model, label_encoder, device, test_folder='/kaggle/input/birdclef-2025/test_soundscapes', sample_csv='/kaggle/input/birdclef-2025/sample_submission.csv', output_csv='/kaggle/working/submission.csv'):
    sample_df = pd.read_csv(sample_csv)
    row_ids = sample_df['row_id'].values
    label_columns = sample_df.columns[1:]

    final_preds = []

    for row_id in row_ids:
        filename = row_id.split("_")[1]
        full_path = os.path.join(test_folder, f"soundscape_{filename}.ogg")
        if not os.path.exists(full_path):
            print("â�Œ File not found:", full_path)
            final_preds.append([0.004] * len(label_columns))  # fallback
            continue

        probs = predict_with_tta(model, full_path, label_encoder, device)
        final_preds.append(probs)

    submission_df = pd.DataFrame(final_preds, columns=label_columns)
    submission_df.insert(0, "row_id", row_ids)
    submission_df.to_csv(output_csv, index=False)
    print(f"âœ… submission.csv saved to {output_csv}")



def pseudo_label_dataset(model, folder, label_encoder, device, confidence_threshold=0.9):
    pseudo_data = []

    for fname in os.listdir(folder):
        if not fname.endswith(".ogg"):
            continue
        path = os.path.join(folder, fname)
        probs = predict_with_tta(model, path, label_encoder, device)

        top_prob = np.max(probs)
        top_idx = np.argmax(probs)

        if top_prob >= confidence_threshold:
            label_id = label_encoder.inverse_transform([top_idx])[0]
            y, sr = librosa.load(path, sr=32000)
            total_secs = int(len(y) / sr)

            for start in range(0, total_secs, 5):
                end = start + 5
                if end > total_secs:
                    break
                start_sample = start * sr
                end_sample = end * sr
                segment = y[start_sample:end_sample]

                mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

                pseudo_data.append({"mel": mel_db, "label": label_id})

    print(f"âœ… Õ�Õ¿Õ¡Ö�Õ¾Õ¥Ö� {len(pseudo_data)} Õ¾Õ½Õ¿Õ¡Õ°Õ¾Õ¡Õ® pseudo-label Õ¶Õ´Õ¸Ö‚Õ·")
    return pseudo_data



pseudo_data = pseudo_label_dataset(
    model=model,
    folder="/kaggle/input/birdclef-2025/test_soundscapes",
    label_encoder=label_encoder,
    device=device,
    confidence_threshold=0.9
)



pseudo_data = pseudo_label_dataset(
    model=model,
    folder="/kaggle/input/birdclef-2025/test_soundscapes",
    label_encoder=label_encoder,
    device=device,
    confidence_threshold=0.7  # ğŸ‘ˆ Õ¶Õ¡Õ­Õ¯Õ«Õ¶ 0.9-Õ« ÖƒÕ¸Õ­Õ¡Ö€Õ¥Õ¶
)



import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        focal_weight = (1 - probs) ** self.gamma
        loss = -targets_onehot * focal_weight * log_probs
        return loss.sum(dim=1).mean()



criterion = FocalLoss(gamma=2.0)



class BirdClefEffAugmentedDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel = self.data[idx]['mel']
        label = self.data[idx]['label']

        mel = (mel - mel.min()) / (mel.max() - mel.min())
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        # SpecAugment
        if random.random() < 0.8:
            time_mask = random.randint(10, 40)
            freq_mask = random.randint(5, 20)
            time_start = random.randint(0, mel_tensor.shape[2] - time_mask)
            freq_start = random.randint(0, mel_tensor.shape[1] - freq_mask)
            mel_tensor[0, freq_start:freq_start+freq_mask, :] = 0
            mel_tensor[0, :, time_start:time_start+time_mask] = 0

        mel_tensor = F.interpolate(mel_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        mel_tensor = mel_tensor.squeeze(0)
        label_tensor = torch.tensor(self.encoder.transform([label])[0], dtype=torch.long)

        return mel_tensor, label_tensor



import torch
from sklearn.metrics import accuracy_score

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Low LR for fine-tuning
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = FocalLoss(gamma=2.0)

for epoch in range(10):
    model.train()
    total_loss = 0.0
    preds, labels_all = [], []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    acc = accuracy_score(labels_all, preds)
    scheduler.step()
    print(f"ğŸ“š Final Epoch {epoch+1}: Loss={total_loss:.2f} | Acc={acc:.4f}")



torch.save(model.state_dict(), "best_model_final.pth")


# generate_submission(
#     model=model,
#     label_encoder=label_encoder,
#     device=device,
#     test_folder='/kaggle/input/birdclef-2025/test_soundscapes',
#     sample_csv='/kaggle/input/birdclef-2025/sample_submission.csv',
#     output_csv= '/kaggle/working/submission.csv'
# )



from pathlib import Path

def generate_submission(model, label_encoder, device, test_folder='/kaggle/input/birdclef-2025/test_soundscapes', sample_csv='/kaggle/input/birdclef-2025/sample_submission.csv', output_csv='/kaggle/working/submission.csv'):
    sample_df = pd.read_csv(sample_csv)
    row_ids = sample_df['row_id'].values
    label_columns = sample_df.columns[1:]

    final_preds = []

    for row_id in row_ids:
        soundscape_id = "_".join(row_id.split("_")[:2])
        full_path = os.path.join(test_folder, f"{soundscape_id}.ogg")

        if not os.path.exists(full_path):
            print("â�Œ File not found:", full_path)
            final_preds.append([0.004] * len(label_columns))  # fallback
            continue

        try:
            probs = predict_with_tta(model, full_path, label_encoder, device)
            final_preds.append(probs)
        except Exception as e:
            print(f"â�Œ Error processing {full_path}: {e}")
            final_preds.append([0.004] * len(label_columns))

    submission_df = pd.DataFrame(final_preds, columns=label_columns)
    submission_df.insert(0, "row_id", row_ids)
    submission_df.to_csv(output_csv, index=False)
    print(f"âœ… submission.csv saved to {output_csv}")


