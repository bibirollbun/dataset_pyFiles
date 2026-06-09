import torch.nn as nn
import numpy as np
import copy
import warnings
import torch
import librosa
import csv
import random
import os
import pandas as pd

from skimage.transform import resize
from skimage.filters import gaussian
from skimage import exposure, util
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')


# Global hyperparameters and constants
LABELS = 24  # Number of species to classify
SR = 48000  # Sampling rate
LENGTH = 10 * SR  # 10 seconds of audio
F_MIN = 24000  # Initial minimum frequency (will be updated)
F_MAX = 0  # Initial maximum frequency (will be updated)
LEARNING_RATE = 2e-4
EPOCHS = 20
N_FOLD = 5  # Number of folds for cross-validation

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


class AudioAugmentations:
    """Class containing various audio spectrogram augmentation techniques"""
    
    def __init__(self):
        # List of available augmentation functions
        self.augs = [self.add_noise, self.contrast_stretch, self.h_flip, self.v_flip]
    
    def h_flip(self, image):
        return np.stack([image[:, ::-1]] * 3)
    
    def v_flip(self, image):
        return np.stack([image[::-1, :]] * 3)
    
    def add_noise(self, image):
        noise_img = util.random_noise(image)
        return np.stack([noise_img] * 3)
    
    def contrast_stretch(self, image):
        contrast_img = exposure.rescale_intensity(image)
        return np.stack([contrast_img] * 3)
    
    def apply_random_augmentation(self, image):
        aug_func = random.choice(self.augs)
        return aug_func(image)


def spec_to_image(spec):
    """Convert spectrogram to normalized image format suitable for ResNet"""
    # Resize to dimensions compatible with ResNet input
    spec = resize(spec, (224, 400))
    
    # Normalize the spectrogram
    eps = 1e-6
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    
    # Scale to 0-255 range for image representation
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    
    return spec_scaled


def get_model():
    """Initialize and configure ResNet50 model for our classification task"""
    model = resnet50(pretrained=True)  # Load pre-trained ResNet50
    
    # Replace the final fully connected layer for our 24-class classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, LABELS)
    
    return model.to(device)


# Load training data and determine frequency range
data = pd.read_csv("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")

# Find the minimum and maximum frequency bounds from the data
for i in range(len(data)):
    current_f_min = float(data.iloc[i]['f_min'])
    current_f_max = float(data.iloc[i]['f_max'])
    
    if F_MIN > current_f_min:
        F_MIN = current_f_min
    if F_MAX < current_f_max:
        F_MAX = current_f_max

# Apply safety margins to frequency bounds
F_MIN = int(F_MIN * 0.9)
F_MAX = int(F_MAX * 1.1)

print(f"Frequency range: {F_MIN}Hz to {F_MAX}Hz")

# Extract labels and recording IDs
label_list = data['species_id'].tolist()
data_list = data['recording_id'].tolist()

# Dictionary to cache processed spectrograms
audio_data = {}


def process_audio(i):
    """Process a single audio file into a spectrogram"""
    recording_id = data_list[i]
    species_id = label_list[i]
    
    # Load audio file
    wav, sr = librosa.load(
        f'/kaggle/input/rfcx-species-audio-detection/train/{recording_id}.flac', 
        sr=None
    )
    
    # Extract the segment containing the species call
    t_min = int(data.at[i, 't_min'] * sr)
    t_max = int(data.at[i, 't_max'] * sr)
    
    # Center the segment and extract 10 seconds
    center = np.round((t_min + t_max) / 2)
    beginning = max(center - LENGTH // 2, 0)
    ending = min(beginning + LENGTH, len(wav))
    
    # Adjust beginning if segment is too short
    if ending - beginning < LENGTH:
        beginning = ending - LENGTH
    
    audio_slice = wav[int(beginning):int(ending)]
    
    # Generate mel spectrogram
    spec = librosa.feature.melspectrogram(
        y=audio_slice, 
        sr=sr, 
        fmin=F_MIN, 
        fmax=F_MAX
    )
    spec_db = librosa.power_to_db(spec, top_db=80)
    
    # Convert to image format
    img = spec_to_image(spec_db)
    
    return recording_id, img


# Process all audio files in parallel
print("Processing audio files...")
with ThreadPoolExecutor() as executor:
    results = list(tqdm(
        executor.map(process_audio, range(len(data))), 
        total=len(data)
    ))

# Store processed spectrograms in dictionary
for recording_id, img in results:
    audio_data[recording_id] = img


class AudioData(Dataset):
    """Custom PyTorch Dataset for audio spectrogram data"""
    
    def __init__(self, X, y, data_type, augmentations=None):
        self.X = X  # List of recording IDs
        self.y = y  # List of labels
        self.data_type = data_type  # "train" or "valid"
        self.audio_data = audio_data  # Cache of processed spectrograms
        self.augmentations = augmentations  # Augmentation object
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        recording_id = self.X[idx]
        label = self.y[idx]
        
        # Retrieve preprocessed spectrogram
        img = self.audio_data[recording_id]
        
        # Apply augmentations only during training
        if self.data_type == "train" and self.augmentations:
            img = self.augmentations.apply_random_augmentation(img)
        else:
            # Convert to 3-channel format (RGB-like for ResNet)
            img = np.stack((img, img, img))
        
        # Convert to tensor
        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor


# Initialize loss function and augmenter
loss_fn = nn.CrossEntropyLoss()
audio_augmenter = AudioAugmentations()


def train(model, loss_fn, train_loader, valid_loader, optimizer, scheduler):
    """Training loop with validation"""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []
    valid_losses = []
    
    for epoch in tqdm(range(1, EPOCHS + 1)):
        # Training phase
        model.train()
        batch_losses = []
        
        for batch_idx, data_batch in enumerate(train_loader):
            x, y = data_batch
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        train_losses.append(batch_losses)
        
        # Validation phase
        model.eval()
        batch_losses = []
        trace_y = []
        trace_yhat = []
        
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(valid_loader):
                x, y = data_batch
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)
                
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                
                trace_y.append(y.cpu().detach().numpy())
                trace_yhat.append(y_hat.cpu().detach().numpy())
                batch_losses.append(loss.item())
        
        valid_losses.append(batch_losses)
        
        # Calculate validation accuracy
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        
        print(f"Epoch {epoch}: "
              f"Train Loss = {np.mean(train_losses[-1]):.5f}, "
              f"Val Loss = {np.mean(valid_losses[-1]):.5f}, "
              f"Val Accuracy = {accuracy:.5f}")
        
        # Update learning rate
        scheduler.step(np.mean(valid_losses[-1]))
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# K-fold cross-validation
skf = KFold(n_splits=N_FOLD, shuffle=True, random_state=563)

for fold_id, (train_index, val_index) in enumerate(skf.split(data_list, label_list)):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_id}")
    print(f"{'='*50}")
    
    # Split data for current fold
    X_train = np.take(data_list, train_index)
    y_train = np.take(label_list, train_index, axis=0)
    X_val = np.take(data_list, val_index)
    y_val = np.take(label_list, val_index, axis=0)
    
    # Create datasets and dataloaders
    train_data = AudioData(X_train, y_train, "train", augmentations=audio_augmenter)
    valid_data = AudioData(X_val, y_val, "valid")
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True, drop_last=True)
    
    # Initialize model, optimizer, and scheduler
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3
    )
    
    # Train the model
    model = train(model, loss_fn, train_loader, valid_loader, optimizer, scheduler)
    
    # Save model weights
    torch.save(model.state_dict(), f"./model{fold_id}.pt")
    
    # Clean up to free memory
    del train_data, valid_data, train_loader, valid_loader, model
    del X_train, X_val, y_train, y_val
    torch.cuda.empty_cache()


def load_test_file(filename):
    """Load and segment test audio file into spectrograms"""
    filepath = '/kaggle/input/rfcx-species-audio-detection/test/' + filename
    wav, sr = librosa.load(filepath, sr=None)
    
    # Calculate number of 10-second segments
    segments = len(wav) / LENGTH
    segments = int(np.ceil(segments))
    
    mel_array = []
    
    for i in range(segments):
        # Extract 10-second segment
        if (i + 1) * LENGTH > len(wav):
            audio_slice = wav[len(wav) - LENGTH:len(wav)]
        else:
            audio_slice = wav[i * LENGTH:(i + 1) * LENGTH]
        
        # Generate spectrogram
        spec = librosa.feature.melspectrogram(
            y=audio_slice, 
            sr=sr, 
            fmin=F_MIN, 
            fmax=F_MAX
        )
        spec_db = librosa.power_to_db(spec, top_db=80)
        
        # Convert to image format
        img = spec_to_image(spec_db)
        mel_spec = np.stack((img, img, img))  # 3-channel format
        mel_array.append(mel_spec)
    
    return np.array(mel_array)


# Load all trained models for ensemble prediction
print("\nLoading models for ensemble prediction...")
members = []

for i in range(N_FOLD):
    print(f"Loading model from fold {i}")
    model = get_model()
    model.load_state_dict(torch.load(f'./model{i}.pt'))
    model.eval()
    members.append(model)

# Clean up model files
for i in range(N_FOLD):
    os.remove(f'./model{i}.pt')


def load_and_predict(test_file, members):
    """Load a test file and generate predictions using ensemble"""
    # Load and preprocess test data
    data = load_test_file(test_file)
    data = torch.tensor(data).float()
    
    if torch.cuda.is_available():
        data = data.cuda()
    
    # Collect predictions from all models
    output_list = []
    
    for model in members:
        with torch.no_grad():
            output = model(data)
            # Take max along segments dimension
            maxed_output = torch.max(output, dim=0)[0]
            maxed_output = maxed_output.cpu().detach()
            output_list.append(maxed_output)
    
    # Ensemble by averaging predictions
    avg_maxed_output = torch.mean(torch.stack(output_list), dim=0)
    
    # Format results
    file_id = test_file.split('.')[0]
    return [file_id] + [out.item() for out in avg_maxed_output]


def save_submission(predictions, output_file='submission.csv'):
    """Save predictions to CSV file in competition format"""
    with open(output_file, 'w', newline='') as csvfile:
        submission_writer = csv.writer(csvfile, delimiter=',')
        
        # Write header
        header = ['recording_id'] + [f's{i}' for i in range(LABELS)]
        submission_writer.writerow(header)
        
        # Write predictions
        for pred in predictions:
            submission_writer.writerow(pred)
    
    print(f"Submission saved to {output_file}")


def generate_predictions(test_files, members):
    """Generate predictions for all test files"""
    predictions = []
    
    print(f"Processing {len(test_files)} test files...")
    
    # Process test files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(load_and_predict, test_file, members)
            for test_file in test_files
        ]
        
        for future in tqdm(futures, total=len(test_files)):
            predictions.append(future.result())
    
    # Save submission file
    save_submission(predictions)
    
    return predictions


# Generate predictions on test set
test_files = os.listdir('/kaggle/input/rfcx-species-audio-detection/test/')

# Move models to GPU if available
if torch.cuda.is_available():
    members = [m.cuda() for m in members]

# Generate final predictions
predictions = generate_predictions(test_files, members)

print("\nPipeline completed successfully!")

