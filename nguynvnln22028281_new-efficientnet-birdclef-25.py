# Import necessary libraries
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm # Or standard tqdm if not in notebook
import time
import copy # To save best model state
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchaudio.transforms as T # For SpecAugment


# --- 1. Configuration & Load Data ---
# Adjust these paths based on your Kaggle environment
# Create dummy paths and data if not on Kaggle for local testing
IS_KAGGLE = os.path.exists("/kaggle/input")
if IS_KAGGLE:
    CSV_PATH = "/kaggle/input/birdclef-2025/train.csv"
    AUDIO_BASE_PATH = "/kaggle/input/birdclef-2025/train_audio/"
else:
    print("Kaggle paths not found. Creating dummy data for local testing.")
    if not os.path.exists("./kaggle/input/birdclef-2025/"):
        os.makedirs("./kaggle/input/birdclef-2025/", exist_ok=True)
    if not os.path.exists("./kaggle/input/birdclef-2025/train_audio/"):
        os.makedirs("./kaggle/input/birdclef-2025/train_audio/", exist_ok=True)

    CSV_PATH = "./kaggle/input/birdclef-2025/train.csv"
    AUDIO_BASE_PATH = "./kaggle/input/birdclef-2025/train_audio/"

    # Create dummy CSV
    dummy_filenames = [f'bird_{i}.ogg' for i in range(20)] # Increased dummy files
    dummy_data = {'filename': dummy_filenames,
                  'primary_label': [f'species_{i%5}' for i in range(20)], # 5 dummy species
                  'secondary_labels': [[] for _ in range(20)]}
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(CSV_PATH, index=False)

    # Create dummy audio files
    try:
        import soundfile as sf
        for fname in dummy_filenames:
            dummy_audio = np.random.uniform(-0.5, 0.5, 32000 * 5) # 5 seconds of random noise
            sf.write(os.path.join(AUDIO_BASE_PATH, fname), dummy_audio, 32000)
        print(f"Dummy data created at {CSV_PATH} and {AUDIO_BASE_PATH}")
    except ImportError:
        print("Please install 'soundfile' to create dummy audio files: pip install soundfile")
        exit()


# Audio Processing Parameters
SAMPLE_RATE = 32000
DURATION = 5  # seconds
N_MELS = 128
FMIN = 20
FMAX = 14000 # Birds typically don't vocalize much higher than this
HOP_LENGTH = int(SAMPLE_RATE * 0.01)  # 10ms hop: 32000 * 0.01 = 320
N_FFT = int(SAMPLE_RATE * 0.025)     # 25ms window: 32000 * 0.025 = 800, round to 1024 for FFT efficiency
N_FFT = 1024 # Or 2048 as you had, both are fine. Let's try 1024.

# Augmentation Parameters
APPLY_AUGMENTATION_PROB = 0.6 # Probability of applying augmentations to a training sample
NOISE_LEVEL = 0.005
TIME_SHIFT_FRACTION = 0.2 # Max fraction of duration to shift
PITCH_SHIFT_STEPS = 2 # Max semitones for pitch shift

# Model & Training Parameters
MODEL_NAME = "efficientnet_b3" # e.g., "efficientnet_b0", "efficientnet_b2"
BATCH_SIZE = 32
EPOCHS = 25 # Pre-trained models can benefit from more epochs with good augmentation
LEARNING_RATE = 3e-4 # Often a good starting LR for AdamW with pre-trained models
WEIGHT_DECAY = 0.01 # For AdamW
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 2 # os.cpu_count() // 2 if you have many cores
LABEL_SMOOTHING = 0.1 # Set to 0 to disable
PRELOAD_DATA_IN_RAM = True # Set to True if you have enough RAM and want faster epochs after initial load

# --- Set Seed for Reproducibility ---
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # 영향 줄 수 있는 설정들
    # torch.backends.cudnn.deterministic = True # Can slow down training
    # torch.backends.cudnn.benchmark = False    # Can slow down training if input sizes vary

seed_everything(RANDOM_SEED)

# --- Determine Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load the training metadata ---
try:
    data = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded {CSV_PATH}")
    print(f"Data shape: {data.shape}")
except FileNotFoundError:
    print(f"Error: Could not find {CSV_PATH}")
    exit()

# Construct full audio file paths
data['full_path'] = data['filename'].apply(lambda x: os.path.join(AUDIO_BASE_PATH, x))



# --- 2. Label Encoding ---
encoder = LabelEncoder()
data['label_encoded'] = encoder.fit_transform(data['primary_label'])
NUM_CLASSES = len(encoder.classes_)
print(f"\nNumber of unique classes: {NUM_CLASSES}")
# label_to_int = {label: i for i, label in enumerate(encoder.classes_)}
# int_to_label = {i: label for i, label in enumerate(encoder.classes_)}


# --- 3. Train/Validation Split ---
train_df, val_df = train_test_split(
    data,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    stratify=data['label_encoded'] # Crucial for imbalanced datasets
)
print(f"\nTraining set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Calculate expected spectrogram shape (for reference and padding)
# For librosa.stft, frame length is N_FFT. Number of frames is roughly len(y) / hop_length.
# For melspectrogram, the time dimension is ceil(samples / hop_length)
# After padding/truncating audio to DURATION * SAMPLE_RATE:
TARGET_LENGTH_SAMPLES = int(SAMPLE_RATE * DURATION)
N_FRAMES = int(np.ceil(TARGET_LENGTH_SAMPLES / HOP_LENGTH)) + 1 # Librosa can add a frame due to centering
print(f"Expected Spectrogram Shape (H, W) after processing: ({N_MELS}, {N_FRAMES})")


# --- 4. Audio Preprocessing & Augmentation Function ---
def load_and_preprocess_audio(
    file_path, target_sr=SAMPLE_RATE, duration_samples=TARGET_LENGTH_SAMPLES,
    n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT,
    fmin=FMIN, fmax=FMAX, is_train=False):
    """Loads, augments (if train), pads/truncates, and computes Mel spectrogram."""
    try:
        y, sr = librosa.load(file_path, sr=None) # Load native SR
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # --- Augmentations (on raw audio, only for training) ---
        if is_train and random.random() < APPLY_AUGMENTATION_PROB:
            # 1. Time Shifting
            if random.random() < 0.5:
                max_shift = int(len(y) * TIME_SHIFT_FRACTION)
                shift = random.randint(-max_shift, max_shift)
                y = np.roll(y, shift)

            # 2. Adding Noise
            if random.random() < 0.5:
                noise = np.random.randn(len(y)) * NOISE_LEVEL
                y = y + noise

            # 3. Pitch Shifting (can be slow, apply with lower probability if needed)
            # if random.random() < 0.3:
            #     n_steps = random.uniform(-PITCH_SHIFT_STEPS, PITCH_SHIFT_STEPS)
            #     y = librosa.effects.pitch_shift(y, sr=target_sr, n_steps=n_steps)

        # Pad or truncate audio to target_length_samples
        if len(y) < duration_samples:
            padding = duration_samples - len(y)
            offset = random.randint(0, padding) if is_train else padding // 2 # Pad randomly for train
            y = np.pad(y, (offset, padding - offset), mode='constant')
        elif len(y) > duration_samples:
            start = random.randint(0, len(y) - duration_samples) if is_train else 0 # Random crop for train
            y = y[start : start + duration_samples]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=target_sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax, window='hann' # Hann window is common
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1] (or standardize if preferred)
        min_val = np.min(mel_spec_db)
        max_val = np.max(mel_spec_db)
        if max_val > min_val:
            mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val)
        else: # Handle silent clips
            mel_spec_db = np.zeros_like(mel_spec_db)

        # Ensure consistent shape (especially time dimension) due to potential minor librosa variations
        if mel_spec_db.shape[1] < N_FRAMES:
            pad_width = N_FRAMES - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad_width)), mode='constant', constant_values=0.)
        elif mel_spec_db.shape[1] > N_FRAMES:
            mel_spec_db = mel_spec_db[:, :N_FRAMES]


        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((n_mels, N_FRAMES)) # Return zeros on error, with expected shape


# --- 5. Create PyTorch Dataset ---
class BirdSoundDataset(Dataset):
    def __init__(self, dataframe, is_train=False, preload_in_ram=PRELOAD_DATA_IN_RAM,
                 apply_spec_augment=True):
        self.dataframe = dataframe
        self.is_train = is_train
        self.preload_in_ram = preload_in_ram
        self.apply_spec_augment = apply_spec_augment

        if self.is_train and self.apply_spec_augment:
            # SpecAugment: Applied on the Mel spectrogram tensor
            self.spec_augmenter =torch.nn.Sequential(
                    T.FrequencyMasking(freq_mask_param=N_MELS // 8),
                    T.TimeMasking(time_mask_param=int(N_FRAMES * 0.1))
                )
            # self.spec_augmenter = nn.Sequential( # Alternative definition
            #     T.FrequencyMasking(freq_mask_param=N_MELS // 8), # Mask up to 1/8th of mel bands
            #     T.TimeMasking(time_mask_param=int(N_FRAMES * 0.1)) # Mask up to 10% of time steps
            # )
        else:
            self.spec_augmenter = None

        if self.preload_in_ram:
            self.spectrograms = []
            self.labels = []
            print(f"Preloading {len(dataframe)} audio files into RAM for {'training' if is_train else 'validation'}...")
            for idx in tqdm(range(len(dataframe))):
                row = self.dataframe.iloc[idx]
                file_path = row['full_path']
                label = row['label_encoded']
                # For preloading, we don't apply instance-specific augmentations like SpecAugment here,
                # as they should be random for each epoch. Audio-level augs are applied during loading.
                spectrogram = load_and_preprocess_audio(file_path, is_train=self.is_train) # is_train for audio augs
                spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
                spectrogram_tensor_3channel = spectrogram_tensor.repeat(3, 1, 1)
                self.spectrograms.append(spectrogram_tensor_3channel)
                self.labels.append(torch.tensor(label, dtype=torch.long))
            print(f"All {len(self.spectrograms)} spectrograms loaded into memory!")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.preload_in_ram:
            spectrogram_3channel = self.spectrograms[idx]
            label = self.labels[idx]
        else:
            row = self.dataframe.iloc[idx]
            file_path = row['full_path']
            label_val = row['label_encoded']

            # Load and process audio (includes audio-level augmentations if is_train)
            spectrogram = load_and_preprocess_audio(file_path, is_train=self.is_train)
            spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0) # (1, H, W)
            spectrogram_3channel = spectrogram_tensor.repeat(3, 1, 1) # (3, H, W)
            label = torch.tensor(label_val, dtype=torch.long)

        # Apply SpecAugment (if training and enabled)
        if self.is_train and self.spec_augmenter and random.random() < APPLY_AUGMENTATION_PROB:
            # Ensure spec_augmenter is on the same device if it contains parameters,
            # or apply before moving tensor to GPU in training loop.
            # For torchaudio.transforms, they are typically stateless or handle device internally.
            try:
                spectrogram_3channel = self.spec_augmenter(spectrogram_3channel)
            except Exception as e: # Catch potential errors with SpecAugment on edge cases
                # print(f"SpecAugment error for sample {idx}, shape {spectrogram_3channel.shape}: {e}")
                pass # Skip SpecAugment for this sample if it errors

        return spectrogram_3channel, label

#train_data = BirdSoundDataset(train_df, is_train=True, preload_in_ram=PRELOAD_DATA_IN_RAM)
#val_data = BirdSoundDataset(val_df, is_train=False, preload_in_ram=PRELOAD_DATA_IN_RAM, apply_spec_augment=False)




train_data = torch.load('/kaggle/input/train-databridclef/train_dataset.pt')
val_data = torch.load("/kaggle/input/notebook863a8ddda3/val_dataset.pt")

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True # drop_last can be useful
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)


# --- 6. Build Custom Model with EfficientNet Backbone ---
print(f"\nBuilding custom model with {MODEL_NAME} backbone...")

# Define a custom model class that uses EfficientNet as a backbone
class BirdClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=True):
        super(BirdClassifier, self).__init__()
        
        # Get the EfficientNet backbone
        if backbone_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            backbone_features = backbone.features
            num_ftrs = 1536  # For EfficientNet B3
        elif backbone_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b2(weights=weights)
            backbone_features = backbone.features
            num_ftrs = 1408  # For EfficientNet B2
        else:
            raise ValueError(f"Backbone model {backbone_name} not supported yet")
        
        # Extract the feature extractor (everything except the classifier)
        self.backbone = backbone_features
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Custom classifier with an additional hidden layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 512),  # Add a hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Softmax activation for final layer (not included in training since CrossEntropyLoss has it)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Extract features using the backbone
        x = self.backbone(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Classification head
        x = self.classifier(x)
        
        # Note: We don't apply softmax during training since CrossEntropyLoss includes it
        # We only apply it during inference or when raw probabilities are needed
        return x
    
    def predict_proba(self, x):
        # For getting probabilities during inference
        logits = self.forward(x)
        return self.softmax(logits)

# Initialize the model
model = BirdClassifier(MODEL_NAME, NUM_CLASSES, pretrained=True)
model = model.to(device)

# Fine-tuning strategy
# Option 1: Unfreeze all layers for end-to-end fine-tuning
for param in model.parameters():
    param.requires_grad = True
print("Fine-tuning: Training all layers.")

# Option 2: Freeze backbone and train only classifier for the first few epochs
# Comment this section if you want to train all layers from the beginning
# for name, param in model.backbone.named_parameters():
#     param.requires_grad = False
# print("Fine-tuning: Only training the classifier initially.")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total model parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")



# --- 7. Define Loss Function and Optimizer ---

# Label Smoothing Cross Entropy
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

if LABEL_SMOOTHING > 0.0 and NUM_CLASSES > 1: # Label smoothing only makes sense for >1 classes
    criterion = LabelSmoothingCrossEntropy(num_classes=NUM_CLASSES, smoothing=LABEL_SMOOTHING).to(device)
    print(f"Using Label Smoothing Cross Entropy with smoothing={LABEL_SMOOTHING}")
else:
    criterion = nn.CrossEntropyLoss().to(device)
    print("Using standard CrossEntropyLoss.")

# Optimizer: AdamW
# For differential learning rates:
param_groups = [
    {'params': model.backbone.parameters(), 'lr': LEARNING_RATE / 10}, # Backbone
    {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}     # Classifier
]
optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
print(f"Using AdamW optimizer with LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}")
print(f"Backbone LR={LEARNING_RATE / 10}, Classifier LR={LEARNING_RATE}")

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
print(f"Using CosineAnnealingLR scheduler with T_max={EPOCHS} epochs.")


# --- 8. Training Loop ---
print(f"\nStarting training for {EPOCHS} epochs on {device}...")

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
best_val_accuracy = 0.0 # Save based on best validation accuracy
best_model_wts = None
early_stopping_patience = 7 # More patience if using aggressive schedulers or complex data
epochs_no_improve = 0

for epoch in range(EPOCHS):
    epoch_start_time = time.time()

    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    correct_predictions_train = 0
    total_samples_train = 0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples_train += labels.size(0)
        correct_predictions_train += (predicted == labels).sum().item()
        train_pbar.set_postfix(loss=loss.item(), acc=correct_predictions_train/total_samples_train if total_samples_train > 0 else 0.0)

    epoch_train_loss = running_loss / total_samples_train if total_samples_train > 0 else 0.0
    epoch_train_acc = correct_predictions_train / total_samples_train if total_samples_train > 0 else 0.0
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    # --- Validation Phase ---
    model.eval()
    running_loss_val = 0.0
    correct_predictions_val = 0
    total_samples_val = 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)

    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss_val += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples_val += labels.size(0)
            correct_predictions_val += (predicted == labels).sum().item()
            val_pbar.set_postfix(loss=loss.item(), acc=correct_predictions_val/total_samples_val if total_samples_val > 0 else 0.0)

    epoch_val_loss = running_loss_val / total_samples_val if total_samples_val > 0 else 0.0
    epoch_val_acc = correct_predictions_val / total_samples_val if total_samples_val > 0 else 0.0
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{EPOCHS} | Duration: {epoch_duration:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

    # LR Scheduler Step (per epoch for CosineAnnealingLR with T_max=EPOCHS)
    if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    elif scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(epoch_val_loss)


    # --- Save Best Model & Early Stopping ---
    if epoch_val_acc > best_val_accuracy:
        print(f"  Validation accuracy improved ({best_val_accuracy:.4f} --> {epoch_val_acc:.4f}). Saving model...")
        best_val_accuracy = epoch_val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), f'best_custom_{MODEL_NAME}_model.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  Validation accuracy did not improve for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= early_stopping_patience:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs as validation accuracy did not improve for {early_stopping_patience} epochs.")
        break

print("\nTraining finished.")
if best_model_wts:
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    # model.load_state_dict(best_model_wts) # Load best model for further use if needed
else:
    print("No improvement in validation accuracy was observed, or training stopped early.")


# --- Plot training history ---
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

axs[0].plot(history['train_loss'], label='Train Loss')
axs[0].plot(history['val_loss'], label='Validation Loss')
axs[0].set_title('Model Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend(); axs[0].grid(True)

axs[1].plot(history['train_acc'], label='Train Accuracy')
axs[1].plot(history['val_acc'], label='Validation Accuracy')
axs[1].set_title('Model Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend(); axs[1].grid(True)

axs[2].plot(history['lr'], label='Learning Rate')
axs[2].set_title('Learning Rate Schedule')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Learning Rate')
axs[2].legend(); axs[2].grid(True)

plt.tight_layout()
plt.savefig(f"training_history_custom_{MODEL_NAME}.png")
plt.show()

print(f"Best model saved as: best_custom_{MODEL_NAME}_model.pth")
print(f"Training plots saved as: training_history_custom_{MODEL_NAME}.png")





