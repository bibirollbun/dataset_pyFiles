# Import pandas for data handling
import pandas as pd

# Define the path to the metadata file
metadata_path = '/kaggle/input/birdclef-2025/train.csv'

# Load the metadata into a DataFrame
metadata = pd.read_csv(metadata_path)

# Display the first few rows to inspect the structure
print("First few rows of the metadata:")
print(metadata.head())

# Check for missing values in each column
print("\nMissing values in each column:")
print(metadata.isnull().sum())

# Display the unique bird species (assuming 'primary_label' is the species column)
print("\nUnique bird species:")
print(metadata['primary_label'].unique())


import pandas as pd
import librosa

# Load metadata
metadata_path = '/kaggle/input/birdclef-2025/train.csv'
metadata = pd.read_csv(metadata_path)

# Example audio file path
example_file_path = '/kaggle/input/birdclef-2025/train_audio/' + metadata['filename'].iloc[0]

# Preprocess audio function (flexible for any file path)
def preprocess_audio(file_path, sample_rate=32000, duration=5):
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    return audio, sr

# Test the function
audio, sr = preprocess_audio(example_file_path)
print(f"Loaded audio with sample rate {sr} and length {len(audio)}")


# Import libraries for audio processing
import librosa
import numpy as np

def preprocess_audio(file_path, target_sr=22050, duration=5):
    """
    Load and preprocess an audio file.
    
    Parameters:
    - file_path: Path to the audio file (str).
    - target_sr: Target sample rate in Hz (int, default: 22050).
    - duration: Target duration in seconds (int, default: 5).
    
    Returns:
    - audio: Preprocessed audio array (numpy array).
    """
    try:
        # Load audio file with the target sample rate
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Calculate target length in samples
        target_length = int(target_sr * duration)
        
        # Trim or pad audio to the target duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Normalize audio to range [-1, 1]
        audio = librosa.util.normalize(audio)
        
        # Optional: Basic noise reduction with pre-emphasis
        audio = librosa.effects.preemphasis(audio)
        
        return audio
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Test the function with an example file
# Replace 'species1/audio1.wav' with an actual filename from your metadata
example_file_path = '/kaggle/input/birdclef-2025/train_audio/' + metadata['filename'].iloc[0]
preprocessed_audio = preprocess_audio(example_file_path)
if preprocessed_audio is not None:
    print(f"Shape of preprocessed audio: {preprocessed_audio.shape}")


# Import necessary libraries
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Reuse the preprocess_audio function from Step 1
def preprocess_audio(file_path, target_sr=22050, duration=5):
    """
    Load and preprocess an audio file.
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        target_length = int(target_sr * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        audio = librosa.util.normalize(audio)
        audio = librosa.effects.preemphasis(audio)
        return audio
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to extract Mel spectrogram features
def extract_mel_spectrogram(audio, sr=22050, n_mels=128, hop_length=512, n_fft=2048):
    """
    Convert audio to Mel spectrogram.
    
    Parameters:
    - audio: Preprocessed audio array (numpy array).
    - sr: Sample rate (int, default: 22050).
    - n_mels: Number of Mel bands (int, default: 128).
    - hop_length: Number of samples between successive frames (int, default: 512).
    - n_fft: Length of FFT window (int, default: 2048).
    
    Returns:
    - mel_spec_db: Mel spectrogram in decibels (numpy array).
    """
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    
    # Convert to decibels (log scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

# Function to resize spectrogram for EfficientNet B0 input (224x224)
def prepare_for_efficientnet(mel_spec_db, target_size=(224, 224)):
    """
    Resize Mel spectrogram to match EfficientNet B0 input size.
    
    Parameters:
    - mel_spec_db: Mel spectrogram in decibels (numpy array).
    - target_size: Desired output size (tuple, default: (224, 224)).
    
    Returns:
    - resized_spec: Resized spectrogram (numpy array).
    """
    # Normalize spectrogram values to [0, 1]
    scaler = StandardScaler()
    mel_spec_normalized = scaler.fit_transform(mel_spec_db)
    
    # Resize to target size (EfficientNet B0 expects 224x224x3, we'll replicate channels later)
    from scipy.ndimage import zoom
    height, width = mel_spec_normalized.shape
    zoom_factors = (target_size[0] / height, target_size[1] / width)
    resized_spec = zoom(mel_spec_normalized, zoom_factors, order=1)
    
    # Ensure the shape matches target_size
    resized_spec = resized_spec[:target_size[0], :target_size[1]]
    
    return resized_spec

# Test the feature extraction pipeline
# Load metadata to get a sample file
metadata_path = '/kaggle/input/birdclef-2025/train.csv'
metadata = pd.read_csv(metadata_path)
example_file_path = '/kaggle/input/birdclef-2025/train_audio/' + metadata['filename'].iloc[0]

# Step 1: Preprocess audio
audio = preprocess_audio(example_file_path)
if audio is not None:
    print(f"Preprocessed audio shape: {audio.shape}")
    
    # Step 2: Extract Mel spectrogram
    mel_spec_db = extract_mel_spectrogram(audio)
    print(f"Mel spectrogram shape: {mel_spec_db.shape}")
    
    # Step 2: Prepare for EfficientNet
    resized_spec = prepare_for_efficientnet(mel_spec_db)
    print(f"Resized spectrogram shape: {resized_spec.shape}")
    
    # Optional: Visualize the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=22050, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()


# Install transformers if not already available
!pip install transformers

# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from transformers import EfficientNetModel
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom
import os

# Reuse Step 1 & 2 functions
def preprocess_audio(file_path, target_sr=22050, duration=5):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        target_length = int(target_sr * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        audio = librosa.util.normalize(audio)
        audio = librosa.effects.preemphasis(audio)
        return audio
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_mel_spectrogram(audio, sr=22050, n_mels=128, hop_length=512, n_fft=2048):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def prepare_for_efficientnet(mel_spec_db, target_size=(224, 224)):
    scaler = StandardScaler()
    mel_spec_normalized = scaler.fit_transform(mel_spec_db)
    height, width = mel_spec_normalized.shape
    zoom_factors = (target_size[0] / height, target_size[1] / width)
    resized_spec = zoom(mel_spec_normalized, zoom_factors, order=1)
    resized_spec = resized_spec[:target_size[0], :target_size[1]]
    return resized_spec

# Custom Dataset with lightweight initialization
class BirdCLEFDataset(Dataset):
    def __init__(self, metadata, audio_dir, label_map):
        self.audio_dir = audio_dir
        self.label_map = label_map
        valid_metadata = []
        for idx, row in metadata.iterrows():
            file_path = f"{audio_dir}/{row['filename']}"
            if os.path.exists(file_path):
                valid_metadata.append(row)
            else:
                print(f"Skipping {file_path}: File not found")
        self.metadata = pd.DataFrame(valid_metadata)
        print(f"Dataset size after filtering: {len(self.metadata)} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        file_path = f"{self.audio_dir}/{self.metadata['filename'].iloc[idx]}"
        audio = preprocess_audio(file_path)
        if audio is None:
            dummy_img = torch.zeros(3, 224, 224, dtype=torch.float32)
            dummy_label = self.label_map[self.metadata['primary_label'].iloc[idx]]
            return dummy_img, dummy_label
        
        mel_spec_db = extract_mel_spectrogram(audio)
        resized_spec = prepare_for_efficientnet(mel_spec_db)
        img = np.stack([resized_spec] * 3, axis=-1)  # Shape: (224, 224, 3)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Shape: (3, 224, 224)
        
        label = self.label_map[self.metadata['primary_label'].iloc[idx]]
        return img, label

# Load and prepare data
metadata_path = '/kaggle/input/birdclef-2025/train.csv'
metadata = pd.read_csv(metadata_path)
train_metadata, val_metadata = train_test_split(
    metadata,
    test_size=0.2,
    stratify=metadata['primary_label'],
    random_state=42
)

# Create label mapping
unique_labels = metadata['primary_label'].unique()
label_map = {label: idx for idx, label in enumerate(unique_labels)}
num_classes = len(unique_labels)
print(f"Number of classes: {num_classes}")

audio_dir = '/kaggle/input/birdclef-2025/train_audio'
train_dataset = BirdCLEFDataset(train_metadata, audio_dir, label_map)
val_dataset = BirdCLEFDataset(val_metadata, audio_dir, label_map)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Load EfficientNet B0 from Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = EfficientNetModel.from_pretrained('google/efficientnet-b0')

# Custom EfficientNet
class CustomEfficientNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1280, num_classes)  # pooler_output is (batch_size, 1280)
    
    def forward(self, x):
        # Input: (batch_size, 3, 224, 224)
        outputs = self.base_model(x)
        x = outputs.pooler_output  # Shape: (batch_size, 1280)
        x = self.fc(x)  # Shape: (batch_size, num_classes)
        return x

model = CustomEfficientNet(base_model, num_classes)
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")


# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# Save the trained model
output_path = '/kaggle/working/output/trained_efficientnet_b0.pth'
torch.save(model.state_dict(), output_path)
print(f"Model saved to {output_path}")

