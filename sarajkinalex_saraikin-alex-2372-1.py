import kagglehub

audio_dataset_path = kagglehub.competition_download("freesound-audio-tagging")
efficientnet_weights_path = kagglehub.model_download(
    "tensorflow/efficientnet/TensorFlow2/b0-classification/1"
)

print("Ğ˜Ñ�Ñ‚Ğ¾Ñ‡Ğ½Ğ¸ĞºĞ¸ ÑƒÑ�Ğ¿ĞµÑˆĞ½Ğ¾ Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶ĞµĞ½Ñ‹.")


import numpy as np
from torchvision import models
import pandas as pd
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from torch import nn
import torch
import librosa.display

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼Ğ¾Ğµ ÑƒÑ�Ñ‚Ñ€Ğ¾Ğ¹Ñ�Ñ‚Ğ²Ğ¾: {device}")


# Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµĞ¼ Ğ¼ĞµÑ‚Ğ°Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ´Ğ»Ñ� Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ³Ğ¾ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğ°
df = pd.read_csv("../input/freesound-audio-tagging/train.csv")

# Ğ˜Ğ·Ğ²Ğ»ĞµĞºĞ°ĞµĞ¼ ÑƒĞ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¼ĞµÑ‚ĞºĞ¸ Ğ¸ Ñ�Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ�Ğ¾Ğ¾Ñ‚Ğ²ĞµÑ‚Ñ�Ñ‚Ğ²Ğ¸Ñ�
class_labels = sorted(df["label"].dropna().unique())
label_mapping = {label: i for i, label in enumerate(class_labels)}
print(f"Ğ’Ñ�ĞµĞ³Ğ¾ Ñ€Ğ°Ğ·Ğ»Ğ¸Ñ‡Ğ½Ñ‹Ñ… ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹: {len(class_labels)}")

# Ğ�Ğ¿Ñ€ĞµĞ´ĞµĞ»Ñ�ĞµĞ¼ Ğ¿ÑƒÑ‚Ğ¸ Ğº Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ñ�Ğ¼ Ñ� Ğ°ÑƒĞ´Ğ¸Ğ¾Ñ„Ğ°Ğ¹Ğ»Ğ°Ğ¼Ğ¸
train_audio_dir = "../input/freesound-audio-tagging/audio_train/"
test_audio_dir = "../input/freesound-audio-tagging/audio_test/"

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¾Ğ±Ñ€Ğ°Ğ·Ñ†Ğ¾Ğ²
total_samples = len(df)
print(f"Ğ�Ğ±Ñ‰ĞµĞµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰Ğ¸Ñ… Ğ¾Ğ±Ñ€Ğ°Ğ·Ñ†Ğ¾Ğ²: {total_samples}")


class AudioVisualDataset(Dataset):
    def __init__(self, metadata, mode='train'):
        self.metadata = metadata
        self.mode = mode
    
    def __len__(self):
        return self.metadata.shape[0]
    
    def __getitem__(self, index):
        sample = self.metadata.iloc[index]
        audio_file = sample["fname"]
        
        if self.mode == 'test':
            full_path = test_audio_dir + audio_file
        else:
            full_path = train_audio_dir + audio_file
        
        try:
            # Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ¸ Ğ¿Ñ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ°ÑƒĞ´Ğ¸Ğ¾
            audio_signal, _ = librosa.load(full_path, sr=None)
            
            # Ğ˜Ğ·Ğ²Ğ»ĞµÑ‡ĞµĞ½Ğ¸Ğµ Ğ¼ĞµĞ»-Ñ�Ğ¿ĞµĞºÑ‚Ñ€Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñ‹
            mel_spec = librosa.feature.melspectrogram(y=audio_signal)
            mel_spec_db = librosa.power_to_db(mel_spec, top_db=None)
            
            # ĞŸĞ¾Ğ´Ğ³Ğ¾Ñ‚Ğ¾Ğ²ĞºĞ° Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ� Ğ´Ğ»Ñ� CNN
            processed_image = cv2.resize(mel_spec_db, (128, 128))
        except Exception as e:
            # Ğ ĞµĞ·ĞµÑ€Ğ²Ğ½Ñ‹Ğ¹ Ğ²Ğ°Ñ€Ğ¸Ğ°Ğ½Ñ‚ Ğ¿Ñ€Ğ¸ Ğ¾ÑˆĞ¸Ğ±ĞºĞµ Ñ‡Ñ‚ĞµĞ½Ğ¸Ñ� Ñ„Ğ°Ğ¹Ğ»Ğ°
            processed_image = np.random.randn(128, 128).clip(-1, 1)
        
        # ĞŸÑ€Ğ¸Ğ²ĞµĞ´ĞµĞ½Ğ¸Ğµ Ğº Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ñƒ [channels, height, width]
        img_tensor = torch.FloatTensor(np.stack([processed_image] * 3, axis=0))
        
        if self.mode == 'test':
            return img_tensor
        
        # ĞŸĞ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ğ¾Ğ³Ğ¾ Ğ¸Ğ´ĞµĞ½Ñ‚Ğ¸Ñ„Ğ¸ĞºĞ°Ñ‚Ğ¾Ñ€Ğ° ĞºĞ»Ğ°Ñ�Ñ�Ğ° Ğ´Ğ»Ñ� Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ³Ğ¾ Ñ€ĞµĞ¶Ğ¸Ğ¼Ğ°
        class_id = label_mapping[sample["label"]]
        return img_tensor, class_id


RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 5

# Ğ Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ½Ğ° Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½ÑƒÑ� Ğ¸ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½ÑƒÑ� Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸
train_part, valid_part = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['label'], 
    random_state=RANDOM_SEED
)

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ¾Ğ² Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğ¾Ğ²
training_dataset = AudioVisualDataset(train_part, mode='train')
validation_dataset = AudioVisualDataset(valid_part, mode='train')

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ğ·Ğ°Ğ³Ñ€ÑƒĞ·Ñ‡Ğ¸ĞºĞ¾Ğ² Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
training_loader = DataLoader(
    training_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

validation_loader = DataLoader(
    validation_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

# Ğ’Ñ‹Ğ²Ğ¾Ğ´ Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ğ¸ Ğ¾ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ğ°Ñ… Ğ²Ñ‹Ğ±Ğ¾Ñ€Ğ¾Ğº
print(f"Ğ Ğ°Ğ·Ğ¼ĞµÑ€ Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸: {len(train_part):,}")
print(f"Ğ Ğ°Ğ·Ğ¼ĞµÑ€ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸: {len(valid_part):,}")
print(f"ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ±Ğ°Ñ‚Ñ‡ĞµĞ¹ Ğ² Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ğ¾Ğ¼ Ğ·Ğ°Ğ³Ñ€ÑƒĞ·Ñ‡Ğ¸ĞºĞµ: {len(training_loader)}")
print(f"ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ±Ğ°Ñ‚Ñ‡ĞµĞ¹ Ğ² Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ¼ Ğ·Ğ°Ğ³Ñ€ÑƒĞ·Ñ‡Ğ¸ĞºĞµ: {len(validation_loader)}")


# Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼ Ğ¿Ñ€ĞµĞ´Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ½ÑƒÑ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ EfficientNet
base_model = models.efficientnet_b0(weights='DEFAULT')

# Ğ—Ğ°Ğ¼ĞµĞ½Ñ�ĞµĞ¼ Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ğ¹ ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ñ‹Ğ¹ Ñ�Ğ»Ğ¾Ğ¹ Ğ½Ğ° Ğ½Ğ°Ñˆ
# Ñ� ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾Ğ¼ Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ¾Ğ², Ñ€Ğ°Ğ²Ğ½Ñ‹Ğ¼ Ñ‡Ğ¸Ñ�Ğ»Ñƒ ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ğ² Ğ½Ğ°ÑˆĞµĞ¼ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğµ
in_features = base_model.classifier[1].in_features
base_model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, len(class_labels))
)

# ĞŸĞµÑ€ĞµĞ½Ğ¾Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ½Ğ° Ğ´Ğ¾Ñ�Ñ‚ÑƒĞ¿Ğ½Ğ¾Ğµ ÑƒÑ�Ñ‚Ñ€Ğ¾Ğ¹Ñ�Ñ‚Ğ²Ğ¾ (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = base_model.to(device)

# Ğ’Ñ‹Ğ²Ğ¾Ğ´Ğ¸Ğ¼ Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ Ñ�Ñ‚Ñ€ÑƒĞºÑ‚ÑƒÑ€Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
print(f"ĞœĞ¾Ğ´ĞµĞ»ÑŒ Ğ¿ĞµÑ€ĞµĞ½ĞµÑ�ĞµĞ½Ğ° Ğ½Ğ° ÑƒÑ�Ñ‚Ñ€Ğ¾Ğ¹Ñ�Ñ‚Ğ²Ğ¾: {device}")
print(f"ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ½Ñ‹Ñ… ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ²: {len(class_labels)}")
print(f"Ğ’Ñ…Ğ¾Ğ´Ğ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ğ² Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½ĞµĞ¼ Ñ�Ğ»Ğ¾Ğµ: {in_features}")

# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¾Ñ‚Ğ´ĞµĞ»ÑŒĞ½ÑƒÑ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½ÑƒÑ� Ğ´Ğ»Ñ� Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�
audio_classifier = base_model


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(audio_classifier.parameters(), lr=1e-3)

training_history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

for epoch_idx in range(NUM_EPOCHS):
    # Ğ ĞµĞ¶Ğ¸Ğ¼ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�
    audio_classifier.train()
    running_train_loss = 0.0
    correct_train_preds = 0
    
    # Ğ¢Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ğ¹ Ñ†Ğ¸ĞºĞ»
    for inputs, targets in training_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Ğ�Ğ±Ğ½ÑƒĞ»Ñ�ĞµĞ¼ Ğ³Ñ€Ğ°Ğ´Ğ¸ĞµĞ½Ñ‚Ñ‹
        optimizer.zero_grad(set_to_none=True)
        
        # ĞŸÑ€Ñ�Ğ¼Ğ¾Ğ¹ Ğ¿Ñ€Ğ¾Ñ…Ğ¾Ğ´
        outputs = audio_classifier(inputs)
        batch_loss = criterion(outputs, targets)
        
        # Ğ�Ğ±Ñ€Ğ°Ñ‚Ğ½Ñ‹Ğ¹ Ğ¿Ñ€Ğ¾Ñ…Ğ¾Ğ´ Ğ¸ Ğ¾Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
        batch_loss.backward()
        optimizer.step()
        
        # Ğ�ĞºĞºÑƒĞ¼ÑƒĞ»Ğ¸Ñ€ÑƒĞµĞ¼ Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºÑƒ
        running_train_loss += batch_loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_train_preds += (predicted == targets).sum().item()
    
    # Ğ ĞµĞ¶Ğ¸Ğ¼ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸
    audio_classifier.eval()
    running_val_loss = 0.0
    correct_val_preds = 0
    
    with torch.inference_mode():
        for inputs, targets in validation_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = audio_classifier(inputs)
            batch_loss = criterion(outputs, targets)
            
            running_val_loss += batch_loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_val_preds += (predicted == targets).sum().item()
    
    # Ğ Ğ°Ñ�Ñ‡ĞµÑ‚ Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº
    avg_train_loss = running_train_loss / len(train_part)
    avg_val_loss = running_val_loss / len(valid_part)
    train_accuracy = 100.0 * correct_train_preds / len(train_part)
    val_accuracy = 100.0 * correct_val_preds / len(valid_part)
    
    # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ğµ Ğ¸Ñ�Ñ‚Ğ¾Ñ€Ğ¸Ğ¸
    training_history['train_loss'].append(avg_train_loss)
    training_history['val_loss'].append(avg_val_loss)
    training_history['train_acc'].append(train_accuracy)
    training_history['val_acc'].append(val_accuracy)
    
    # Ğ’Ñ‹Ğ²Ğ¾Ğ´ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¾Ğ² Ñ�Ğ¿Ğ¾Ñ…Ğ¸
    print(f"Ğ­Ğ¿Ğ¾Ñ…Ğ° [{epoch_idx+1}/{NUM_EPOCHS}]")
    print(f"  Ğ¢Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²ĞºĞ°:   ĞŸĞ¾Ñ‚ĞµÑ€Ñ� = {avg_train_loss:.4f}, Ğ¢Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ = {train_accuracy:.2f}%")
    print(f"  Ğ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ�:    ĞŸĞ¾Ñ‚ĞµÑ€Ñ� = {avg_val_loss:.4f}, Ğ¢Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ = {val_accuracy:.2f}%")
    
    # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ñ€Ğ°Ğ·Ğ´ĞµĞ»Ğ¸Ñ‚ĞµĞ»ÑŒ Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ�Ğ¿Ğ¾Ñ…Ğ°Ğ¼Ğ¸ Ğ´Ğ»Ñ� Ñ‡Ğ¸Ñ‚Ğ°ĞµĞ¼Ğ¾Ñ�Ñ‚Ğ¸
    if epoch_idx < NUM_EPOCHS - 1:
        print("-" * 50)

print("\nĞ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½Ğ¾!")


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np

# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ° Ñ�Ñ‚Ğ¸Ğ»Ñ� Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¾Ğ²
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 12

# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ„Ğ¸Ğ³ÑƒÑ€Ñƒ Ñ� Ğ´Ğ²ÑƒĞ¼Ñ� Ğ¿Ğ¾Ğ´Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ°Ğ¼Ğ¸
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ğ“Ñ€Ğ°Ñ„Ğ¸Ğº 1: Ğ¤ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ¿Ğ¾Ñ‚ĞµÑ€ÑŒ (Loss)
axes[0].plot(training_history['train_loss'], 
            label='Ğ¢Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ğ°Ñ� Ğ¿Ğ¾Ñ‚ĞµÑ€Ñ�', 
            linewidth=2.5, 
            marker='o',
            markersize=8,
            alpha=0.8)

axes[0].plot(training_history['val_loss'], 
            label='Ğ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ°Ñ� Ğ¿Ğ¾Ñ‚ĞµÑ€Ñ�', 
            linewidth=2.5, 
            marker='s',
            markersize=8,
            alpha=0.8)

axes[0].set_title('Ğ”Ğ¸Ğ½Ğ°Ğ¼Ğ¸ĞºĞ° Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ğ¸ Ğ¿Ğ¾Ñ‚ĞµÑ€ÑŒ', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Ğ­Ğ¿Ğ¾Ñ…Ğ°', fontsize=12)
axes[0].set_ylabel('ĞŸĞ¾Ñ‚ĞµÑ€Ñ�', fontsize=12)
axes[0].legend(loc='upper right', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

# Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ°Ğ½Ğ½Ğ¾Ñ‚Ğ°Ñ†Ğ¸Ñ� Ñ� Ñ„Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¼Ğ¸ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ�Ğ¼Ğ¸
final_train_loss = training_history['train_loss'][-1]
final_val_loss = training_history['val_loss'][-1]
axes[0].annotate(f'Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ğ°Ñ�:\nĞ¢Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²ĞºĞ°: {final_train_loss:.4f}\nĞ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ�: {final_val_loss:.4f}', 
                 xy=(0.98, 0.02), 
                 xycoords='axes fraction',
                 ha='right', 
                 va='bottom',
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Ğ“Ñ€Ğ°Ñ„Ğ¸Ğº 2: Ğ¢Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ (Accuracy)
axes[1].plot(training_history['train_acc'], 
            label='Ğ¢Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ğ°Ñ� Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ', 
            linewidth=2.5, 
            marker='o',
            markersize=8,
            alpha=0.8,
            color='green')

axes[1].plot(training_history['val_acc'], 
            label='Ğ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ°Ñ� Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ', 
            linewidth=2.5, 
            marker='s',
            markersize=8,
            alpha=0.8,
            color='orange')

axes[1].set_title('Ğ”Ğ¸Ğ½Ğ°Ğ¼Ğ¸ĞºĞ° Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚Ğ¸ ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Ğ­Ğ¿Ğ¾Ñ…Ğ°', fontsize=12)
axes[1].set_ylabel('Ğ¢Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ (%)', fontsize=12)
axes[1].legend(loc='lower right', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

# Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ°Ğ½Ğ½Ğ¾Ñ‚Ğ°Ñ†Ğ¸Ñ� Ñ� Ñ„Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¼Ğ¸ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ�Ğ¼Ğ¸
final_train_acc = training_history['train_acc'][-1]
final_val_acc = training_history['val_acc'][-1]
axes[1].annotate(f'Ğ¤Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ğ°Ñ�:\nĞ¢Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²ĞºĞ°: {final_train_acc:.2f}%\nĞ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ�: {final_val_acc:.2f}%', 
                 xy=(0.98, 0.98), 
                 xycoords='axes fraction',
                 ha='right', 
                 va='top',
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Ğ“Ñ€Ğ°Ñ„Ğ¸Ğº 3: Ğ Ğ°Ğ·Ğ½Ğ¸Ñ†Ğ° Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ğ¾Ğ¹ Ğ¸ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ¹ Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒÑ�
epochs = range(1, len(training_history['train_acc']) + 1)
accuracy_gap = np.array(training_history['train_acc']) - np.array(training_history['val_acc'])

bars = axes[2].bar(epochs, accuracy_gap, 
                  color=['red' if gap > 10 else 'orange' if gap > 5 else 'green' for gap in accuracy_gap],
                  alpha=0.7,
                  edgecolor='black',
                  linewidth=1)

axes[2].set_title('Ğ Ğ°Ğ·Ñ€Ñ‹Ğ² Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ğ¾Ğ¹ Ğ¸ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ¹ Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒÑ�', 
                 fontsize=14, fontweight='bold', pad=15)
axes[2].set_xlabel('Ğ­Ğ¿Ğ¾Ñ…Ğ°', fontsize=12)
axes[2].set_ylabel('Ğ Ğ°Ğ·Ğ½Ğ¸Ñ†Ğ° (%)', fontsize=12)
axes[2].axhline(y=0, color='black', linewidth=0.8, linestyle='-')
axes[2].axhline(y=5, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='ĞŸĞ¾Ñ€Ğ¾Ğ³ 5%')
axes[2].axhline(y=10, color='red', linewidth=1, linestyle='--', alpha=0.5, label='ĞŸĞ¾Ñ€Ğ¾Ğ³ 10%')
axes[2].legend(loc='upper right', fontsize=10)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

# Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ½Ğ° Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ñ‹
for bar, gap in zip(bars, accuracy_gap):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., 
                height + (0.1 if height >= 0 else -0.8),
                f'{gap:.1f}%', 
                ha='center', 
                va='bottom' if height >= 0 else 'top',
                fontsize=9,
                fontweight='bold')

# Ğ”Ğ¾Ğ¿Ğ¾Ğ»Ğ½Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ°Ñ� Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¿Ğ¾Ğ´ Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ°Ğ¼Ğ¸
plt.suptitle(f'Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ñ‹ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ·Ğ° {NUM_EPOCHS} Ñ�Ğ¿Ğ¾Ñ…', 
             fontsize=16, 
             fontweight='bold', 
             y=1.02)

# ĞšĞ¾Ğ¼Ğ¿Ğ°ĞºÑ‚Ğ½Ğ¾Ğµ Ñ€Ğ°Ñ�Ğ¿Ğ¾Ğ»Ğ¾Ğ¶ĞµĞ½Ğ¸Ğµ
plt.tight_layout()

plt.show()


# Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
test_meta = pd.read_csv("../input/freesound-audio-tagging/sample_submission.csv")
test_set = AudioVisualDataset(test_meta, mode='test')
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Ğ˜Ğ½Ñ„ĞµÑ€ĞµĞ½Ñ�
audio_classifier.eval()
preds_list = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = audio_classifier(batch)
        preds_list.append(outputs.cpu())

# Ğ¡Ğ±Ğ¾Ñ€ĞºĞ° Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¹
all_preds = torch.cat(preds_list, dim=0)
probs = torch.softmax(all_preds, dim=1).numpy()

print(f"âœ… Ğ˜Ğ½Ñ„ĞµÑ€ĞµĞ½Ñ� Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½. ĞŸĞ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ¾ {len(probs)} Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¹")


# Ğ¤Ğ¾Ñ€Ğ¼Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ submission
submission = pd.DataFrame(probs, columns=class_labels)
submission.insert(0, 'fname', test_meta['fname'].values)
submission.to_csv('my_submission.csv', index=False)
print("ğŸ’¾ Submission Ñ„Ğ°Ğ¹Ğ» Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½")

