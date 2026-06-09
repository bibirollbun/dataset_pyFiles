import os
INPUT_DIR = '/kaggle/input/birdclef-2025/'  # Kaggle's input directory
OUTPUT_DIR = '/kaggle/working'             # Kaggle's output directory

# Verify files in input directory
for dirname, _, filenames in os.walk(INPUT_DIR):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# --- Imports ---
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Load Competition Data ---
try:
    train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")
    sample_submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")
except:
    # Fallback for local testing
    train_df = pd.read_csv("train.csv")
    sample_submission = pd.read_csv("sample_submission.csv")

# --- Create Label Map ---
all_labels = sorted(train_df["primary_label"].unique())
label_map = {label: i for i, label in enumerate(all_labels)}
num_classes = len(label_map)

# --- Feature Extraction ---
def extract_melspectrogram(file_path, sr=32000, n_mels=64, duration=5, hop_length=512):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        if len(y) < duration * sr:
            y = np.pad(y, (0, duration * sr - len(y)))
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return mel_db
    except:
        return np.zeros((n_mels, int(sr * duration / hop_length) + 1))

# --- Dataset Class ---
class BirdDataset(Dataset):
    def __init__(self, df, audio_dir, label_map, duration=5):
        self.df = df
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.duration = duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.label_map[row["primary_label"]]
        file_path = os.path.join(self.audio_dir, row["filename"])
        mel = extract_melspectrogram(file_path, duration=self.duration)
        mel_tensor = torch.tensor(mel).unsqueeze(0).float()
        return mel_tensor, label

# --- Model Definition ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class CNN14(nn.Module):
    def __init__(self, num_classes):
        super(CNN14, self).__init__()
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        
        self.pool = nn.AvgPool2d(2)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(self.conv_block1(x))
        x = self.pool(self.conv_block2(x))
        x = self.pool(self.conv_block3(x))
        x = self.pool(self.conv_block4(x))
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.fc(x))

# --- Prediction Function ---
def predict(model, test_dir, sample_submission):
    model.eval()
    device = next(model.parameters()).device
    results = []
    
    for row_id in tqdm(sample_submission["row_id"], desc="Predicting"):
        file_name = row_id.split("_")[0] + "_" + row_id.split("_")[1] + ".ogg"
        path = os.path.join(test_dir, file_name)
        
        mel = extract_melspectrogram(path)
        input_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            pred = model(input_tensor).cpu().numpy().flatten()
        
        results.append(pred)
    
    return np.stack(results)

# --- Main Execution ---
def main():
    # Prepare data
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    train_dataset = BirdDataset(train_data, f"{INPUT_DIR}/train_audio", label_map)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN14(num_classes).to(device)
    
    # Train (simplified for Kaggle - use pre-trained in final submission)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    print("Starting training...")
    for epoch in range(3):  # Reduced epochs for Kaggle demo
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), F.one_hot(y, num_classes).float().to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
    
    # Generate predictions
    print("Generating predictions...")
    test_predictions = predict(model, f"{INPUT_DIR}/test_soundscapes", sample_submission)
    
    # Create submission
    submission_df = sample_submission.copy()
    submission_df.iloc[:, 1:] = test_predictions
    submission_df.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
    print("Submission saved to /kaggle/working/submission.csv")

if __name__ == "__main__":
    main()




