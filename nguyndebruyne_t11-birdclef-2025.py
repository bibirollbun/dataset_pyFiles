import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from torchvision.models import efficientnet_b0
import warnings
warnings.filterwarnings('ignore')


# 1. Environment Setup
torch.manual_seed(42)
device = torch.device("cpu")


# 2. Precompute RMS for Segment Selection
def compute_rms_segments(audio, sr=32000, segment_length=5):
    """Compute RMS for each 5-second segment in audio."""
    segment_samples = segment_length * sr
    segments = [audio[i:i+segment_samples] for i in range(0, len(audio), segment_samples)]
    rms_values = [np.sqrt(np.mean(segment**2)) if len(segment) > 0 else 0 for segment in segments]
    return rms_values

def precompute_rms(train_df, audio_dir, output_csv='train_segments.csv'):
    """Precompute RMS for all files and save to CSV."""
    print("Precomputing RMS for all files...")
    rms_data = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        file_path = os.path.join(audio_dir, row['filename'])
        try:
            audio, sr = librosa.load(file_path, sr=32000)
            rms_values = compute_rms_segments(audio, sr)
            for i, rms in enumerate(rms_values):
                rms_data.append([row['filename'], i, rms])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    rms_df = pd.DataFrame(rms_data, columns=['filename', 'segment_index', 'rms_value'])
    rms_df.to_csv(output_csv, index=False)
    print(f"Saved RMS data to {output_csv}")
    return rms_df


# 3. FocalBCELoss Implementation
class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# 4. Data Loading and RMS-based Filtering
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
taxonomy_df = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')

# Precompute RMS if not exists
rms_csv = 'train_segments.csv'
if not os.path.exists(rms_csv):
    rms_df = precompute_rms(train_df, '/kaggle/input/birdclef-2025/train_audio/', rms_csv)
else:
    rms_df = pd.read_csv(rms_csv)

# RMS-based filtering
RMS_THRESHOLD = 0.01
print("Filtering segments with RMS >= ", RMS_THRESHOLD)
# Keep segments with RMS >= threshold
valid_segments = rms_df[rms_df['rms_value'] >= RMS_THRESHOLD][['filename', 'segment_index']]

# Identify rare species (<20 files)
species_counts = train_df['primary_label'].value_counts()
rare_species = species_counts[species_counts < 20].index

# Keep all files for rare species
rare_files = train_df[train_df['primary_label'].isin(rare_species)]['filename']
rare_segments = rms_df[rms_df['filename'].isin(rare_files)][['filename', 'segment_index']]
# For rare species, keep segment with highest RMS if no segment >= threshold
for filename in rare_files:
    file_segments = rms_df[rms_df['filename'] == filename]
    if not any(file_segments['rms_value'] >= RMS_THRESHOLD):
        max_rms_segment = file_segments.loc[file_segments['rms_value'].idxmax()][['filename', 'segment_index']]
        valid_segments = pd.concat([valid_segments, max_rms_segment.to_frame().T], ignore_index=True)

# Downsampling
# Keep rating=0 and rating>=3, limit 30 files per common species
filtered_df = train_df[train_df['filename'].isin(valid_segments['filename'])]
filtered_df = filtered_df[(filtered_df['rating'] == 0) | (filtered_df['rating'] >= 3) | (filtered_df['primary_label'].isin(rare_species))]
common_species = species_counts[species_counts >= 20].index
common_files = []
for species in common_species:
    species_files = filtered_df[filtered_df['primary_label'] == species]['filename']
    if len(species_files) > 30:
        species_files = species_files.sample(n=30, random_state=42)
    common_files.extend(species_files)
filtered_df = filtered_df[filtered_df['filename'].isin(common_files) | filtered_df['primary_label'].isin(rare_species)]
print(f"Dataset size after RMS filtering and downsampling: {len(filtered_df)} files")

# Create segment mapping
segment_map = valid_segments.groupby('filename')['segment_index'].apply(list).to_dict()
# For files with <3 valid segments, add one random segment
for filename in filtered_df['filename']:
    valid_seg_indices = segment_map.get(filename, [])
    if len(valid_seg_indices) < 3:
        all_segments = rms_df[rms_df['filename'] == filename]['segment_index'].values
        if len(all_segments) > 0:
            random_seg = np.random.choice(all_segments)
            if random_seg not in valid_seg_indices:
                segment_map[filename].append(random_seg)


# 5. Audio Preprocessing
def preprocess_audio(audio, duration=5, sr=32000, n_mels=32):
    """Convert audio to normalized log-mel spectrogram."""
    target_length = sr * duration
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=16000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
    return log_mel_spec

def spec_augment(spec, time_mask=0.1, freq_mask=0.1):
    """Apply SpecAugment: time and frequency masking."""
    _, n_mels, n_steps = spec.shape
    t_mask_size = int(n_steps * time_mask)
    t_start = np.random.randint(0, n_steps - t_mask_size + 1)
    spec[:, :, t_start:t_start + t_mask_size] = 0
    f_mask_size = int(n_mels * freq_mask)
    f_start = np.random.randint(0, n_mels - f_mask_size + 1)
    spec[:, f_start:f_start + f_mask_size, :] = 0
    return spec

def add_random_noise(audio, std=0.005):
    """Add Gaussian noise to audio."""
    noise = np.random.normal(0, std, audio.shape)
    return audio + noise


# 6. Dataset Preparation
# Create species list and mapping
species_list = sorted(train_df['primary_label'].unique())
species_to_idx = {s: i for i, s in enumerate(species_list)}

# Create multi-label vectors
def create_multi_label_vector(row):
    label_vec = np.zeros(len(species_list))
    label_vec[species_to_idx[row['primary_label']]] = 1
    for sec_label in eval(row['secondary_labels']):
        if sec_label in species_to_idx:
            label_vec[species_to_idx[sec_label]] = 1
    return label_vec

filtered_df['multi_label'] = filtered_df.apply(create_multi_label_vector, axis=1)

# Use all data for training (no validation split)
train_files = filtered_df['filename']
train_labels = np.stack(filtered_df['multi_label'].values)

# Custom Dataset
class BirdCLEFDataset(Dataset):
    def __init__(self, files, labels, audio_dir, segment_map, duration=5, sr=32000, n_mels=32, augment=False):
        self.files = files
        self.labels = labels
        self.audio_dir = audio_dir
        self.segment_map = segment_map
        self.duration = duration
        self.sr = sr
        self.n_mels = n_mels
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files.iloc[idx]
        file_path = os.path.join(self.audio_dir, filename)
        audio, _ = librosa.load(file_path, sr=self.sr)
        # Select a valid segment
        valid_segments = self.segment_map.get(filename, [0])
        segment_idx = np.random.choice(valid_segments)
        start_sample = segment_idx * self.duration * self.sr
        audio_segment = audio[start_sample:start_sample + self.duration * self.sr]
        if len(audio_segment) < self.duration * self.sr:
            audio_segment = np.pad(audio_segment, (0, self.duration * self.sr - len(audio_segment)))
        
        if self.augment and np.random.rand() < 0.2:
            audio_segment = add_random_noise(audio_segment, std=0.005)
        
        log_mel = preprocess_audio(audio_segment, self.duration, self.sr, self.n_mels)
        log_mel = np.stack([log_mel] * 3, axis=0)  # Shape: (3, n_mels, time_steps)
        
        if self.augment:
            log_mel = spec_augment(log_mel, time_mask=0.1, freq_mask=0.1)
        
        label = self.labels[idx]
        return torch.tensor(log_mel, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Create DataLoader
train_dataset = BirdCLEFDataset(
    train_files, train_labels, '/kaggle/input/birdclef-2025/train_audio/', segment_map, augment=True
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)


# 7. Model Architecture
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=len(species_list)):
        super(EfficientNetB0, self).__init__()
        self.efficientnet = efficientnet_b0(weights=None)
        # Load pretrained weights offline
        self.efficientnet.load_state_dict(
            torch.load('/kaggle/input/efficientnet-b0/efficientnet_b0_rwightman-7f5810bc.pth')
        )
        # Replace classifier for multi-label classification
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),  # Reduced to 128 for speed
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.efficientnet(x)

model = EfficientNetB0().to(device)


# 8. Model Training
print("Training model...")
criterion = FocalBCELoss(alpha=0.25, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)
        # Mixup
        if np.random.rand() < 0.3:  # Reduced to 30%
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            perm = torch.randperm(inputs.size(0))
            mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
            mixed_targets = lam * targets + (1 - lam) * targets[perm]
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = criterion(outputs, mixed_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)

# Training loop with Early Stopping
best_train_loss = float('inf')
patience = 2
epochs_without_improvement = 0
for epoch in range(3):  # Max 3 epochs
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    
    # Save model if train loss improves
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), 'best_model.pth')
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    # Update learning rate
    scheduler.step(train_loss)
    
    # Early stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))


# 9. Inference on Test Soundscapes
def predict_test_soundscapes(model, test_dir):
    model.eval()
    submission = []
    test_files = glob.glob(f'{test_dir}/*.ogg')
    
    if not test_files:
        print("No test soundscape files found. Returning empty submission.")
        columns = ['row_id'] + species_list
        return pd.DataFrame([], columns=columns)
    
    for file in test_files:
        audio, sr = librosa.load(file, sr=32000)
        soundscape_id = os.path.basename(file).replace('.ogg', '')
        
        for i in range(0, len(audio), 5*sr):
            end_time = (i // sr) + 5
            segment = audio[i:i+5*sr]
            if len(segment) < 5*sr:
                segment = np.pad(segment, (0, 5*sr - len(segment)))
            
            log_mel = preprocess_audio(segment, n_mels=32)
            log_mel = np.stack([log_mel] * 3, axis=0)  # Shape: (3, n_mels, time_steps)
            log_mel = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(log_mel).cpu().numpy()[0]
            
            row_id = f"{soundscape_id}_{end_time}"
            submission.append([row_id] + pred.tolist())
    
    columns = ['row_id'] + species_list
    submission_df = pd.DataFrame(submission, columns=columns)
    return submission_df

print("Generating predictions...")
test_dir = '/kaggle/input/birdclef-2025/test_soundscapes'
submission_df = predict_test_soundscapes(model, test_dir)


# 10. Submission File Validation
print("Validating submission...")
sample_submission = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
assert set(submission_df.columns) == set(sample_submission.columns), "Column mismatch"
if submission_df.shape[0] > 0:
    assert submission_df.iloc[:, 1:].ge(0).all().all() and submission_df.iloc[:, 1:].le(1).all().all(), "Invalid probabilities"
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")

