import os                                                                                                                                                                                                                                                                                                

# Define the output directory
OUTPUT_DIR = '/kaggle/working/'

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


import os
import librosa
import numpy as np

# Choose a single folder to process
folder_name = "bkmtou1"  # Change this to the folder you want to process

folder_path = f'/kaggle/input/birdclef-2025/train_audio/{folder_name}/'
audio_files = [f for f in os.listdir(folder_path) if f.endswith('.ogg')]

# Process each audio file in the folder
for audio_file in audio_files:
    audio_path = os.path.join(folder_path, audio_file)
    audio, sr = librosa.load(audio_path, sr=32000)  # Use 32 kHz sampling rate as required
    # Add your audio processing logic here
    print(f"Processed: {audio_file}")


import gc

# After processing each file or batch
gc.collect()


import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# Define constants
TRAIN_AUDIO_PATH = "/kaggle/input/birdclef-2025/train_audio"
OUTPUT_DIR = "/kaggle/working/"
TARGET_FOLDER = "blbwre1"  # you can change this to any subfolder
MAX_FILES = 10
SAMPLE_RATE = 16000
DURATION_SEC = 5
AUDIO_LENGTH = SAMPLE_RATE * DURATION_SEC

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mel spectrogram transform
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=128,
)

# Utility to load and trim/pad audio
def load_trim_pad(path):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform[:, :AUDIO_LENGTH]
    pad_len = AUDIO_LENGTH - waveform.shape[1]
    if pad_len > 0:
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    return waveform

# Get a small sample of files
folder_path = os.path.join(TRAIN_AUDIO_PATH, TARGET_FOLDER)
all_files = [f for f in os.listdir(folder_path) if f.endswith(".ogg")]
sample_files = random.sample(all_files, min(MAX_FILES, len(all_files)))

# Process and save mel spectrograms
for filename in sample_files:
    filepath = os.path.join(folder_path, filename)
    waveform = load_trim_pad(filepath)
    mel_spec = mel_transform(waveform)

    # Optional: Save spectrograms for quick visualization or caching
    save_path = os.path.join(OUTPUT_DIR, filename.replace(".ogg", ".pt"))
    torch.save(mel_spec, save_path)

    print(f"Processed {filename} → {save_path}")


import torchaudio.functional as F

def apply_random_augmentations(waveform, sample_rate):
    if random.random() < 0.5:
        rate = random.uniform(0.8, 1.2)  # time stretch (0.8x to 1.2x)
        waveform = F.phase_vocoder(
            T.Spectrogram()(waveform), rate, torch.zeros(1)  # just placeholder phase
        ) if waveform.size(1) > 1 else waveform  # avoid if 1-frame
    if random.random() < 0.5:
        n_steps = random.uniform(-2, 2)  # pitch shift in semitones
        waveform = F.pitch_shift(waveform, sample_rate, n_steps)
    if random.random() < 0.5:
        gain_db = random.uniform(-6, 6)  # volume adjustment
        waveform = waveform * (10 ** (gain_db / 20))
    return waveform


waveform = load_trim_pad(filepath)
waveform = apply_random_augmentations(waveform, SAMPLE_RATE)


import matplotlib.pyplot as plt

def plot_augmented_spectrogram(waveform, sample_rate):
    # Original mel spectrogram
    mel_original = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128)(waveform)

    # Apply augmentations
    augmented_waveform = apply_random_augmentations(waveform.clone(), sample_rate)
    mel_augmented = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128)(augmented_waveform)

    # Convert to log scale for better visibility
    mel_original_db = torchaudio.transforms.AmplitudeToDB()(mel_original)
    mel_augmented_db = torchaudio.transforms.AmplitudeToDB()(mel_augmented)

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].imshow(mel_original_db.squeeze().numpy(), origin="lower", aspect="auto", cmap="viridis")
    axs[0].set_title("Original Mel Spectrogram")

    axs[1].imshow(mel_augmented_db.squeeze().numpy(), origin="lower", aspect="auto", cmap="magma")
    axs[1].set_title("Augmented Mel Spectrogram")

    for ax in axs:
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel bins")
    plt.tight_layout()
    plt.show()


import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to the audio file
audio_path = '/kaggle/input/birdclef-2025/train_audio/bkmtou1/XC383275.ogg'

# Load the audio file
y, sr = librosa.load(audio_path, sr=32000)

# Extract features (e.g., Mel-frequency cepstral coefficients)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

# Print the shape of the MFCC features
print(f"MFCC shape: {mfcc.shape}")

# Save the MFCC features to a NumPy file in the output directory
output_path = os.path.join(OUTPUT_DIR, 'mfcc_features.npy')
np.save(output_path, mfcc)
print(f"MFCC features saved to {output_path}")

# Optionally, you can visualize the MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()


#  Fixed Model Definition
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Ensure you have the model class defined earlier
class BirdCLEFCRNN(nn.Module):
    def __init__(self, num_classes=30):
        super(BirdCLEFCRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 16, H/2, W/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, H/4, W/4)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((64, 32))  # Force output to (64, 32)
        )
        self.gru = nn.GRU(input_size=64 * 64, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)  # (B, 64, 64, 32)
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, W=32, C=64, H=64)
        x = x.view(B, W, C * H)  # (B, 32, 4096)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])  # Last time step
        return x

# Instantiate the model
model = BirdCLEFCRNN(num_classes=30).to(device)

# Check if model is correctly defined
print(model)


import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Dummy parameters
batch_size = 8
num_classes = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
model = BirdCLEFCRNN(num_classes=num_classes).to(device)

# Dummy data
dummy_x = torch.randn(batch_size, 1, 128, 256).to(device)
dummy_y = torch.randint(0, num_classes, (batch_size,)).to(device)

# DataLoader
dataset = TensorDataset(dummy_x, dummy_y)
loader = DataLoader(dataset, batch_size=batch_size)

# Optimizer + Loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Dummy training loop (1 epoch)
model.train()
for batch_x, batch_y in loader:
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    print(f"Dummy loss: {loss.item():.4f}")


import torch
import torch.nn as nn

# Ensure you have the model class defined earlier
class BirdCLEFCRNN(nn.Module):
    def __init__(self, num_classes=30):
        super(BirdCLEFCRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((64, 32)),
        )
        self.gru = nn.GRU(32, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flattening the tensor for GRU
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])  # Get the output from the last GRU step
        return x

# Instantiate the model
model = BirdCLEFCRNN(num_classes=30).to(device)

# Check if model is correctly defined
print(model)


import torch
from torch.utils.data import Dataset
import os

class BirdCLEFMelDataset(Dataset):
    def __init__(self, data_dir, label_map, transform=None):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".pt")]
        self.label_map = label_map  # Dict[str -> int] or list index for multi-label
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        tensor = torch.load(os.path.join(self.data_dir, fname))  # shape: [1, H, W]
        
        # Dummy label: Replace with actual label extraction logic if needed
        species_id = fname.split(".")[0].split("_")[0]  # e.g., XC123456 → XC123456
        label = self.label_map.get(species_id, 0)  # default to 0 or multi-hot
        
        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)


# Dummy label map for testing (assuming 30 classes)
import random
label_map = {fname.split(".")[0]: random.randint(0, 29) for fname in os.listdir("/kaggle/working") if fname.endswith(".pt")}


from torch.utils.data import DataLoader

train_dataset = BirdCLEFMelDataset("/kaggle/working", label_map)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)


for batch in train_loader:
    x, y = batch
    print("Input:", x.shape)  # Expected: [B, 1, H, W]
    print("Labels:", y.shape)  # Expected: [B] or [B, num_classes]
    break


%%time
import torchaudio
import torchaudio.transforms as T

def preprocess_audio_file(file_path, sample_rate=32000, n_mels=128, mel_len=313):
    waveform, sr = torchaudio.load(file_path)

    # Resample if needed
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Mel spectrogram
    mel_spec = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels
    )(waveform)

    # Convert to log scale (dB)
    log_mel = T.AmplitudeToDB()(mel_spec)

    # Ensure fixed length (padding or truncating to fixed time frames)
    if log_mel.shape[-1] < mel_len:
        pad_amount = mel_len - log_mel.shape[-1]
        log_mel = torch.nn.functional.pad(log_mel, (0, pad_amount))
    else:
        log_mel = log_mel[:, :, :mel_len]

    return log_mel  # shape: [1, n_mels, mel_len]


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Sample data for ground truth and predictions
ground_truth_data = {
    'recording_id': ['0001', '0002', '0003', '0004', '0005'],
    'x': [10.5, 15.4, 12.3, 18.2, 14.1],
    'y': [20.3, 25.6, 22.1, 28.4, 24.3],
    'z': [30.7, 35.8, 32.9, 38.6, 34.5]
}

predictions_data = {
    'recording_id': ['0001', '0002', '0003', '0004', '0005'],
    'x': [10.6, 15.5, 12.4, 18.3, 14.2],
    'y': [20.4, 25.7, 22.2, 28.5, 24.4],
    'z': [30.8, 35.9, 33.0, 38.7, 34.6]
}

# Convert dictionaries to DataFrames
ground_truth = pd.DataFrame(ground_truth_data)
predictions = pd.DataFrame(predictions_data)

# Ensure both DataFrames are aligned by recording_id
ground_truth = ground_truth.set_index('recording_id')
predictions = predictions.set_index('recording_id')

# Function to calculate Euclidean distance
def euclidean_distance(row):
    y_true = ground_truth.loc[row.name]
    y_pred = row
    d = np.sqrt((y_true['x'] - y_pred['x'])**2 + 
                (y_true['y'] - y_pred['y'])**2 + 
                (y_true['z'] - y_pred['z'])**2)
    return d

# Calculate Euclidean distance for each prediction
predictions['distance'] = predictions.apply(euclidean_distance, axis=1)

# Determine True Positives (TP) and False Negatives (FN)
threshold = 10.0  # Angstroms
predictions['is_TP'] = predictions['distance'] <= threshold

# Calculate TP, FP, and FN
TP = predictions['is_TP'].sum()
FP = predictions.shape[0] - TP  # False Positives are all predictions that are not True Positives
FN = ground_truth.shape[0] - TP  # False Negatives are all ground truth that are not True Positives

# Calculate Precision and Recall
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1 Score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optionally, you can save the results to a CSV file
results = pd.DataFrame({
    'TP': [TP],
    'FP': [FP],
    'FN': [FN],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1]
})
results.to_csv('evaluation_results.csv', index=False)


import matplotlib.pyplot as plt

# Plot the distribution of distances
plt.hist(predictions['distance'], bins=10, alpha=0.7, color='blue')
plt.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2)
plt.title('Distribution of Euclidean Distances')
plt.xlabel('Distance (Angstroms)')
plt.ylabel('Frequency')
plt.show()

# Plot true positives and false negatives
plt.figure(figsize=(10, 8))
plt.scatter(ground_truth['x'], ground_truth['y'], label='Ground Truth', color='green')
plt.scatter(predictions.loc[predictions['is_TP'], 'x'], predictions.loc[predictions['is_TP'], 'y'], label='True Positives', color='blue')
plt.scatter(predictions.loc[~predictions['is_TP'], 'x'], predictions.loc[~predictions['is_TP'], 'y'], label='False Positives', color='red')
plt.title('True Positives and False Positives')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []

for train_index, val_index in kfold.split(ground_truth):
    train_gt = ground_truth.iloc[train_index]
    val_gt = ground_truth.iloc[val_index]
    train_pred = predictions.iloc[train_index]
    val_pred = predictions.iloc[val_index]

    # Calculate distances for validation set
    val_pred['distance'] = val_pred.apply(euclidean_distance, axis=1)
    val_pred['is_TP'] = val_pred['distance'] <= threshold

    TP = val_pred['is_TP'].sum()
    FP = val_pred.shape[0] - TP
    FN = val_gt.shape[0] - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    f1_scores.append(f1)

print(f"Average F1 Score: {np.mean(f1_scores):.4f}")

