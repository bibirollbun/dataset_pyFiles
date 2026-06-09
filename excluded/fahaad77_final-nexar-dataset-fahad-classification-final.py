# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# Constants
FRAME_RATE = 30  # 30 fps
WINDOW_SIZE = 5  # Number of frames to use for prediction
POSITIVE_WINDOW = 1.5  # Seconds before the event to consider as positive

# Load the training data
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
print("Training data loaded:", train_df.shape)

# Fix the video path issue by ensuring proper formatting of video IDs
def get_video_path(video_id, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    base_dir = '/kaggle/input/nexar-collision-prediction'
    subfolder = 'train' if is_train else 'test'
    return os.path.join(base_dir, subfolder, f"{formatted_id}.mp4")

# Data preprocessing function to extract frames
def extract_frames(video_id, df_row, output_dir, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    video_path = get_video_path(video_id, is_train)
    target = df_row['target'] if 'target' in df_row else None
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    event_frame = None
    alert_frame = None
    
    if is_train and target == 1:
        event_time = float(df_row['time_of_event'])
        alert_time = float(df_row['time_of_alert'])
        event_frame = int(event_time * fps)
        alert_frame = int(alert_time * fps)
    
    frame_data = []
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        label = 0
        if is_train and target == 1:
            if alert_frame <= frame_idx <= event_frame:
                label = 1
            elif alert_frame - int(POSITIVE_WINDOW * fps) <= frame_idx < alert_frame:
                label = 1
        
        frame_path = os.path.join(output_dir, f"{formatted_id}_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_data.append({
            'frame_path': frame_path,
            'video_id': formatted_id,
            'frame_idx': frame_idx,
            'label': label if is_train else None
        })
    
    cap.release()
    return frame_data

# Process all videos and create a dataframe of frames
def process_videos(df, output_dir, is_train=True):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        all_frames.extend(frame_data)
    
    return pd.DataFrame(all_frames)

# Process a subset of videos for faster execution
sample_train_df = train_df.sample(10, random_state=42) if len(train_df) > 10 else train_df
train_frames_df = process_videos(sample_train_df, '/kaggle/working/train_frames', is_train=True)

# Check if we have any frames before proceeding
if len(train_frames_df) == 0:
    print("No frames were extracted. Creating dummy dataset...")
    dummy_data = []
    for i in range(100):
        dummy_data.append({
            'frame_path': f'/kaggle/working/dummy_{i:05d}.jpg',
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': i % 2
        })
    train_frames_df = pd.DataFrame(dummy_data)
    os.makedirs('/kaggle/working/dummy_images', exist_ok=True)
    for i in range(100):
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(f'/kaggle/working/dummy_{i:05d}.jpg', img)

train_frames_df.to_csv('/kaggle/working/train_frames.csv', index=False)
print(f"Processed {len(train_frames_df)} frames from {len(sample_train_df)} videos")

# Custom Dataset class
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(0, dtype=torch.float32)

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split data into train, validation, and test sets
train_val_frames, test_frames = train_test_split(train_frames_df, test_size=0.1, random_state=42)
train_frames, val_frames = train_test_split(train_val_frames, test_size=0.2, random_state=42)

# Create datasets
train_dataset = NexarDataset(train_frames, transform=train_transform)
val_dataset = NexarDataset(val_frames, transform=val_transform)
test_dataset = NexarDataset(test_frames, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Initialize ResNet-50 model
def get_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model

model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training function with additional metrics
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        epoch_precision = precision_score(train_labels, train_preds, zero_division=0)
        epoch_recall = recall_score(train_labels, train_preds, zero_division=0)
        epoch_f1 = f1_score(train_labels, train_preds, zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_epoch_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_epoch_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_epoch_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        scheduler.step(val_epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, Precision: {val_epoch_precision:.4f}, Recall: {val_epoch_recall:.4f}, F1: {val_epoch_f1:.4f}')
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            print("Saved best model!")
    
    return model

# Evaluate model on test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating on Test Set'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3)

# Evaluate on test set
evaluate_model(trained_model, test_loader, criterion)





import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import cv2
import shutil

# Modified NexarDataset class with robust error handling
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # Filter out rows with invalid frame paths
        self.df = self.df[self.df['frame_path'].apply(self._is_valid_file)]
        print(f"Filtered dataset to {len(self.df)} valid frames")
        
    def _is_valid_file(self, path):
        return os.path.isfile(path) and os.access(path, os.R_OK)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label'] if row['label'] is not None else 0.0
        
        try:
            if not self._is_valid_file(img_path):
                raise FileNotFoundError(f"Image file not found or inaccessible: {img_path}")
                
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(label, dtype=torch.float32)

# Modified test data processing
# Load test data
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
print("Test data loaded:", test_df.shape)

# Process a subset of test videos
sample_test_df = test_df.sample(2, random_state=42) if len(test_df) > 2 else test_df
test_frames_dir = '/kaggle/working/test_frames'

# Clear the test frames directory to avoid conflicts
if os.path.exists(test_frames_dir):
    shutil.rmtree(test_frames_dir)
os.makedirs(test_frames_dir, exist_ok=True)

# Process videos with additional debugging
def process_videos(df, output_dir, is_train=False):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing test videos"):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        # Validate saved frames
        valid_frames = []
        for frame in frame_data:
            if os.path.isfile(frame['frame_path']) and os.access(frame['frame_path'], os.R_OK):
                valid_frames.append(frame)
            else:
                print(f"Invalid frame file: {frame['frame_path']}")
        all_frames.extend(valid_frames)
    
    return pd.DataFrame(all_frames)

test_frames_df = process_videos(sample_test_df, test_frames_dir, is_train=False)

# Check if we have any test frames
if len(test_frames_df) == 0:
    print("No test frames were extracted. Creating dummy test data...")
    dummy_test_data = []
    os.makedirs('/kaggle/working/dummy_test_images', exist_ok=True)
    for i in range(50):
        frame_path = f'/kaggle/working/dummy_test_images/dummy_test_{i:05d}.jpg'
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(frame_path, img)
        dummy_test_data.append({
            'frame_path': frame_path,
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': None
        })
    test_frames_df = pd.DataFrame(dummy_test_data)

test_frames_df.to_csv('/kaggle/working/test_frames.csv', index=False)
print(f"Processed {len(test_frames_df)} test frames from {len(sample_test_df)} videos")

# Create test dataset and loader
test_dataset = NexarDataset(test_frames_df, transform=val_transform)
if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty after filtering invalid frames")

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)  # Set num_workers to 0

# Load best model for inference
model.load_state_dict(torch.load('/kaggle/working/best_model.pth', weights_only=True))  # Set weights_only=True
model.eval()

# Make predictions on test set with error handling
predictions = []
video_ids = []

with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(test_loader, desc='Making predictions')):
        try:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu().numpy()
            
            batch_indices = list(range(i*test_loader.batch_size, 
                                     min((i+1)*test_loader.batch_size, len(test_dataset))))
            
            actual_batch_size = min(test_loader.batch_size, len(test_dataset) - i*test_loader.batch_size)
            batch_indices = batch_indices[:actual_batch_size]
            
            batch_video_ids = [test_frames_df.iloc[idx]['video_id'] for idx in batch_indices]
            
            if isinstance(outputs, np.float32):
                outputs = np.array([outputs])
                
            if len(outputs) > len(batch_indices):
                outputs = outputs[:len(batch_indices)]
                
            predictions.extend(outputs)
            video_ids.extend(batch_video_ids)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

if not predictions:
    raise ValueError("No predictions were generated. Check test data and model inference.")

# Create a dataframe with predictions for each frame
prediction_df = pd.DataFrame({
    'video_id': video_ids,
    'prediction': predictions
})

# Aggregate predictions by video
video_predictions = prediction_df.groupby('video_id')['prediction'].max().reset_index()

# Create submission file
submission_df = pd.DataFrame({
    'id': video_predictions['video_id'],
    'target': video_predictions['prediction']
})

submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created!")
print(submission_df.head())



import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Constants
FRAME_RATE = 30  # 30 fps
WINDOW_SIZE = 5  # Number of frames to use for prediction
POSITIVE_WINDOW = 1.5  # Seconds before the event to consider as positive

# Load the training data
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
print("Training data loaded:", train_df.shape)

# Fix the video path issue by ensuring proper formatting of video IDs
def get_video_path(video_id, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    base_dir = '/kaggle/input/nexar-collision-prediction'
    subfolder = 'train' if is_train else 'test'
    return os.path.join(base_dir, subfolder, f"{formatted_id}.mp4")

# Data preprocessing function to extract frames
def extract_frames(video_id, df_row, output_dir, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    video_path = get_video_path(video_id, is_train)
    target = df_row['target'] if 'target' in df_row else None
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    event_frame = None
    alert_frame = None
    
    if is_train and target == 1:
        event_time = float(df_row['time_of_event'])
        alert_time = float(df_row['time_of_alert'])
        event_frame = int(event_time * fps)
        alert_frame = int(alert_time * fps)
    
    frame_data = []
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        label = 0
        if is_train and target == 1:
            if alert_frame <= frame_idx <= event_frame:
                label = 1
            elif alert_frame - int(POSITIVE_WINDOW * fps) <= frame_idx < alert_frame:
                label = 1
        
        frame_path = os.path.join(output_dir, f"{formatted_id}_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_data.append({
            'frame_path': frame_path,
            'video_id': formatted_id,
            'frame_idx': frame_idx,
            'label': label if is_train else None
        })
    
    cap.release()
    return frame_data

# Process all videos and create a dataframe of frames
def process_videos(df, output_dir, is_train=True):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        all_frames.extend(frame_data)
    
    return pd.DataFrame(all_frames)

# Process a subset of videos for faster execution
sample_train_df = train_df.sample(10, random_state=42) if len(train_df) > 10 else train_df
train_frames_df = process_videos(sample_train_df, '/kaggle/working/train_frames', is_train=True)

# Check if we have any frames before proceeding
if len(train_frames_df) == 0:
    print("No frames were extracted. Creating dummy dataset...")
    dummy_data = []
    for i in range(100):
        dummy_data.append({
            'frame_path': f'/kaggle/working/dummy_{i:05d}.jpg',
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': i % 2
        })
    train_frames_df = pd.DataFrame(dummy_data)
    os.makedirs('/kaggle/working/dummy_images', exist_ok=True)
    for i in range(100):
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(f'/kaggle/working/dummy_{i:05d}.jpg', img)

train_frames_df.to_csv('/kaggle/working/train_frames.csv', index=False)
print(f"Processed {len(train_frames_df)} frames from {len(sample_train_df)} videos")

# Custom Dataset class
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(0, dtype=torch.float32)

# Define transforms for ViT - Note that ViT requires a specific input size (224x224 for ViT-B/16)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split data into train, validation, and test sets
train_val_frames, test_frames = train_test_split(train_frames_df, test_size=0.1, random_state=42)
train_frames, val_frames = train_test_split(train_val_frames, test_size=0.2, random_state=42)

# Create datasets
train_dataset = NexarDataset(train_frames, transform=train_transform)
val_dataset = NexarDataset(val_frames, transform=val_transform)
test_dataset = NexarDataset(test_frames, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Initialize ViT model
def get_vit_model():
    # Load pre-trained ViT model
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    # Modify the head for binary classification
    num_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    return model

model = get_vit_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
# For ViT, we often use a lower learning rate
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training function with additional metrics
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        epoch_precision = precision_score(train_labels, train_preds, zero_division=0)
        epoch_recall = recall_score(train_labels, train_preds, zero_division=0)
        epoch_f1 = f1_score(train_labels, train_preds, zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_epoch_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_epoch_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_epoch_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        scheduler.step(val_epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, Precision: {val_epoch_precision:.4f}, Recall: {val_epoch_recall:.4f}, F1: {val_epoch_f1:.4f}')
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), '/kaggle/working/best_vit_model.pth')
            print("Saved best model!")
    
    return model

# Evaluate model on test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating on Test Set'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3)

# Evaluate on test set
evaluate_model(trained_model, test_loader, criterion)

# Modified test data processing for inference
import shutil

# Modified NexarDataset class with robust error handling
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # Filter out rows with invalid frame paths
        self.df = self.df[self.df['frame_path'].apply(self._is_valid_file)]
        print(f"Filtered dataset to {len(self.df)} valid frames")
        
    def _is_valid_file(self, path):
        return os.path.isfile(path) and os.access(path, os.R_OK)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label'] if row['label'] is not None else 0.0
        
        try:
            if not self._is_valid_file(img_path):
                raise FileNotFoundError(f"Image file not found or inaccessible: {img_path}")
                
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(label, dtype=torch.float32)

# Load test data
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
print("Test data loaded:", test_df.shape)

# Process a subset of test videos
sample_test_df = test_df.sample(2, random_state=42) if len(test_df) > 2 else test_df
test_frames_dir = '/kaggle/working/test_frames_vit'

# Clear the test frames directory to avoid conflicts
if os.path.exists(test_frames_dir):
    shutil.rmtree(test_frames_dir)
os.makedirs(test_frames_dir, exist_ok=True)

# Process videos with additional debugging
def process_videos(df, output_dir, is_train=False):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing test videos"):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        # Validate saved frames
        valid_frames = []
        for frame in frame_data:
            if os.path.isfile(frame['frame_path']) and os.access(frame['frame_path'], os.R_OK):
                valid_frames.append(frame)
            else:
                print(f"Invalid frame file: {frame['frame_path']}")
        all_frames.extend(valid_frames)
    
    return pd.DataFrame(all_frames)

test_frames_df = process_videos(sample_test_df, test_frames_dir, is_train=False)

# Check if we have any test frames
if len(test_frames_df) == 0:
    print("No test frames were extracted. Creating dummy test data...")
    dummy_test_data = []
    os.makedirs('/kaggle/working/dummy_test_images_vit', exist_ok=True)
    for i in range(50):
        frame_path = f'/kaggle/working/dummy_test_images_vit/dummy_test_{i:05d}.jpg'
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(frame_path, img)
        dummy_test_data.append({
            'frame_path': frame_path,
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': None
        })
    test_frames_df = pd.DataFrame(dummy_test_data)

test_frames_df.to_csv('/kaggle/working/test_frames_vit.csv', index=False)
print(f"Processed {len(test_frames_df)} test frames from {len(sample_test_df)} videos")

# Create test dataset and loader
test_dataset = NexarDataset(test_frames_df, transform=val_transform)
if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty after filtering invalid frames")

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)  # Set num_workers to 0

# Load best model for inference
model.load_state_dict(torch.load('/kaggle/working/best_vit_model.pth', weights_only=True))  # Set weights_only=True
model.eval()

# Make predictions on test set with error handling
predictions = []
video_ids = []

with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(test_loader, desc='Making predictions with ViT')):
        try:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu().numpy()
            
            batch_indices = list(range(i*test_loader.batch_size, 
                                     min((i+1)*test_loader.batch_size, len(test_dataset))))
            
            actual_batch_size = min(test_loader.batch_size, len(test_dataset) - i*test_loader.batch_size)
            batch_indices = batch_indices[:actual_batch_size]
            
            batch_video_ids = [test_frames_df.iloc[idx]['video_id'] for idx in batch_indices]
            
            if isinstance(outputs, np.float32):
                outputs = np.array([outputs])
                
            if len(outputs) > len(batch_indices):
                outputs = outputs[:len(batch_indices)]
                
            predictions.extend(outputs)
            video_ids.extend(batch_video_ids)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

if not predictions:
    raise ValueError("No predictions were generated. Check test data and model inference.")

# Create a dataframe with predictions for each frame
prediction_df = pd.DataFrame({
    'video_id': video_ids,
    'prediction': predictions
})

# Aggregate predictions by video
video_predictions = prediction_df.groupby('video_id')['prediction'].max().reset_index()

# Create submission file
submission_df = pd.DataFrame({
    'id': video_predictions['video_id'],
    'target': video_predictions['prediction']
})

submission_df.to_csv('/kaggle/working/vit_submission.csv', index=False)
print("ViT submission file created!")
print(submission_df.head())


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

# Constants
FRAME_RATE = 30  # 30 fps
WINDOW_SIZE = 5  # Number of frames to use for prediction
POSITIVE_WINDOW = 1.5  # Seconds before the event to consider as positive

# Load the training data
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
print("Training data loaded:", train_df.shape)

# Fix the video path issue by ensuring proper formatting of video IDs
def get_video_path(video_id, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    base_dir = '/kaggle/input/nexar-collision-prediction'
    subfolder = 'train' if is_train else 'test'
    return os.path.join(base_dir, subfolder, f"{formatted_id}.mp4")

# Data preprocessing function to extract frames
def extract_frames(video_id, df_row, output_dir, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    video_path = get_video_path(video_id, is_train)
    target = df_row['target'] if 'target' in df_row else None
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    event_frame = None
    alert_frame = None
    
    if is_train and target == 1:
        event_time = float(df_row['time_of_event'])
        alert_time = float(df_row['time_of_alert'])
        event_frame = int(event_time * fps)
        alert_frame = int(alert_time * fps)
    
    frame_data = []
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        label = 0
        if is_train and target == 1:
            if alert_frame <= frame_idx <= event_frame:
                label = 1
            elif alert_frame - int(POSITIVE_WINDOW * fps) <= frame_idx < alert_frame:
                label = 1
        
        frame_path = os.path.join(output_dir, f"{formatted_id}_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_data.append({
            'frame_path': frame_path,
            'video_id': formatted_id,
            'frame_idx': frame_idx,
            'label': label if is_train else None
        })
    
    cap.release()
    return frame_data

# Process all videos and create a dataframe of frames
def process_videos(df, output_dir, is_train=True):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        all_frames.extend(frame_data)
    
    return pd.DataFrame(all_frames)

# Process a subset of videos for faster execution
sample_train_df = train_df.sample(10, random_state=42) if len(train_df) > 10 else train_df
train_frames_df = process_videos(sample_train_df, '/kaggle/working/train_frames_efficientnet', is_train=True)

# Check if we have any frames before proceeding
if len(train_frames_df) == 0:
    print("No frames were extracted. Creating dummy dataset...")
    dummy_data = []
    for i in range(100):
        dummy_data.append({
            'frame_path': f'/kaggle/working/dummy_efficientnet_{i:05d}.jpg',
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': i % 2
        })
    train_frames_df = pd.DataFrame(dummy_data)
    os.makedirs('/kaggle/working/dummy_images_efficientnet', exist_ok=True)
    for i in range(100):
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(f'/kaggle/working/dummy_efficientnet_{i:05d}.jpg', img)

train_frames_df.to_csv('/kaggle/working/train_frames_efficientnet.csv', index=False)
print(f"Processed {len(train_frames_df)} frames from {len(sample_train_df)} videos")

# Custom Dataset class
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(0, dtype=torch.float32)

# Define transforms for EfficientNet-B1
# EfficientNet-B1 expects input images of size 240x240
train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split data into train, validation, and test sets
train_val_frames, test_frames = train_test_split(train_frames_df, test_size=0.1, random_state=42)
train_frames, val_frames = train_test_split(train_val_frames, test_size=0.2, random_state=42)

# Create datasets
train_dataset = NexarDataset(train_frames, transform=train_transform)
val_dataset = NexarDataset(val_frames, transform=val_transform)
test_dataset = NexarDataset(test_frames, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Initialize EfficientNet-B1 model
def get_efficientnet_model():
    # Load pre-trained EfficientNet-B1 model
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
    
    # Modify the classifier for binary classification
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    return model

model = get_efficientnet_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
# For EfficientNet, we often use a similar learning rate as with ViT
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training function with additional metrics
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        epoch_precision = precision_score(train_labels, train_preds, zero_division=0)
        epoch_recall = recall_score(train_labels, train_preds, zero_division=0)
        epoch_f1 = f1_score(train_labels, train_preds, zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_epoch_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_epoch_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_epoch_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        scheduler.step(val_epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, Precision: {val_epoch_precision:.4f}, Recall: {val_epoch_recall:.4f}, F1: {val_epoch_f1:.4f}')
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), '/kaggle/working/best_efficientnet_model.pth')
            print("Saved best model!")
    
    return model

# Evaluate model on test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating on Test Set'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3)

# Evaluate on test set
evaluate_model(trained_model, test_loader, criterion)

# Modified test data processing for inference
import shutil

# Modified NexarDataset class with robust error handling
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # Filter out rows with invalid frame paths
        self.df = self.df[self.df['frame_path'].apply(self._is_valid_file)]
        print(f"Filtered dataset to {len(self.df)} valid frames")
        
    def _is_valid_file(self, path):
        return os.path.isfile(path) and os.access(path, os.R_OK)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label'] if row['label'] is not None else 0.0
        
        try:
            if not self._is_valid_file(img_path):
                raise FileNotFoundError(f"Image file not found or inaccessible: {img_path}")
                
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and label
            placeholder = torch.zeros((3, 240, 240))  # Updated size for EfficientNet-B1
            return placeholder, torch.tensor(label, dtype=torch.float32)

# Load test data
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
print("Test data loaded:", test_df.shape)

# Process a subset of test videos
sample_test_df = test_df.sample(5, random_state=42) if len(test_df) > 5 else test_df
test_frames_dir = '/kaggle/working/test_frames_efficientnet'

# Clear the test frames directory to avoid conflicts
if os.path.exists(test_frames_dir):
    shutil.rmtree(test_frames_dir)
os.makedirs(test_frames_dir, exist_ok=True)

# Process videos with additional debugging
def process_videos(df, output_dir, is_train=False):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing test videos"):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        # Validate saved frames
        valid_frames = []
        for frame in frame_data:
            if os.path.isfile(frame['frame_path']) and os.access(frame['frame_path'], os.R_OK):
                valid_frames.append(frame)
            else:
                print(f"Invalid frame file: {frame['frame_path']}")
        all_frames.extend(valid_frames)
    
    return pd.DataFrame(all_frames)

test_frames_df = process_videos(sample_test_df, test_frames_dir, is_train=False)

# Check if we have any test frames
if len(test_frames_df) == 0:
    print("No test frames were extracted. Creating dummy test data...")
    dummy_test_data = []
    os.makedirs('/kaggle/working/dummy_test_images_efficientnet', exist_ok=True)
    for i in range(50):
        frame_path = f'/kaggle/working/dummy_test_images_efficientnet/dummy_test_{i:05d}.jpg'
        img = np.ones((240, 240, 3), dtype=np.uint8) * 128  # Updated size for EfficientNet-B1
        cv2.imwrite(frame_path, img)
        dummy_test_data.append({
            'frame_path': frame_path,
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': None
        })
    test_frames_df = pd.DataFrame(dummy_test_data)

test_frames_df.to_csv('/kaggle/working/test_frames_efficientnet.csv', index=False)
print(f"Processed {len(test_frames_df)} test frames from {len(sample_test_df)} videos")

# Create test dataset and loader
test_dataset = NexarDataset(test_frames_df, transform=val_transform)
if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty after filtering invalid frames")

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)  # Set num_workers to 0

# Load best model for inference
model.load_state_dict(torch.load('/kaggle/working/best_efficientnet_model.pth', weights_only=True))  # Set weights_only=True
model.eval()

# Make predictions on test set with error handling
predictions = []
video_ids = []

with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(test_loader, desc='Making predictions with EfficientNet-B1')):
        try:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu().numpy()
            
            batch_indices = list(range(i*test_loader.batch_size, 
                                     min((i+1)*test_loader.batch_size, len(test_dataset))))
            
            actual_batch_size = min(test_loader.batch_size, len(test_dataset) - i*test_loader.batch_size)
            batch_indices = batch_indices[:actual_batch_size]
            
            batch_video_ids = [test_frames_df.iloc[idx]['video_id'] for idx in batch_indices]
            
            if isinstance(outputs, np.float32):
                outputs = np.array([outputs])
                
            if len(outputs) > len(batch_indices):
                outputs = outputs[:len(batch_indices)]
                
            predictions.extend(outputs)
            video_ids.extend(batch_video_ids)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

if not predictions:
    raise ValueError("No predictions were generated. Check test data and model inference.")

# Create a dataframe with predictions for each frame
prediction_df = pd.DataFrame({
    'video_id': video_ids,
    'prediction': predictions
})

# Aggregate predictions by video
video_predictions = prediction_df.groupby('video_id')['prediction'].max().reset_index()

# Create submission file
submission_df = pd.DataFrame({
    'id': video_predictions['video_id'],
    'target': video_predictions['prediction']
})

submission_df.to_csv('/kaggle/working/efficientnet_submission.csv', index=False)
print("EfficientNet-B1 submission file created!")
print(submission_df.head())


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Constants
FRAME_RATE = 30  # 30 fps
WINDOW_SIZE = 5  # Number of frames to use for prediction
POSITIVE_WINDOW = 1.5  # Seconds before the event to consider as positive

# Load the training data
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
print("Training data loaded:", train_df.shape)

# Fix the video path issue by ensuring proper formatting of video IDs
def get_video_path(video_id, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    base_dir = '/kaggle/input/nexar-collision-prediction'
    subfolder = 'train' if is_train else 'test'
    return os.path.join(base_dir, subfolder, f"{formatted_id}.mp4")

# Data preprocessing function to extract frames
def extract_frames(video_id, df_row, output_dir, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    video_path = get_video_path(video_id, is_train)
    target = df_row['target'] if 'target' in df_row else None
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    event_frame = None
    alert_frame = None
    
    if is_train and target == 1:
        event_time = float(df_row['time_of_event'])
        alert_time = float(df_row['time_of_alert'])
        event_frame = int(event_time * fps)
        alert_frame = int(alert_time * fps)
    
    frame_data = []
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        label = 0
        if is_train and target == 1:
            if alert_frame <= frame_idx <= event_frame:
                label = 1
            elif alert_frame - int(POSITIVE_WINDOW * fps) <= frame_idx < alert_frame:
                label = 1
        
        frame_path = os.path.join(output_dir, f"{formatted_id}_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_data.append({
            'frame_path': frame_path,
            'video_id': formatted_id,
            'frame_idx': frame_idx,
            'label': label if is_train else None
        })
    
    cap.release()
    return frame_data

# Process all videos and create a dataframe of frames
def process_videos(df, output_dir, is_train=True):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        all_frames.extend(frame_data)
    
    return pd.DataFrame(all_frames)

# Process a subset of videos for faster execution
sample_train_df = train_df.sample(10, random_state=42) if len(train_df) > 10 else train_df
train_frames_df = process_videos(sample_train_df, '/kaggle/working/train_frames_mobilenet', is_train=True)

# Check if we have any frames before proceeding
if len(train_frames_df) == 0:
    print("No frames were extracted. Creating dummy dataset...")
    dummy_data = []
    for i in range(100):
        dummy_data.append({
            'frame_path': f'/kaggle/working/dummy_{i:05d}.jpg',
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': i % 2
        })
    train_frames_df = pd.DataFrame(dummy_data)
    os.makedirs('/kaggle/working/dummy_images', exist_ok=True)
    for i in range(100):
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(f'/kaggle/working/dummy_{i:05d}.jpg', img)

train_frames_df.to_csv('/kaggle/working/train_frames_mobilenet.csv', index=False)
print(f"Processed {len(train_frames_df)} frames from {len(sample_train_df)} videos")

# Custom Dataset class
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(0, dtype=torch.float32)

# Define transforms for MobileNetV2
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split data into train, validation, and test sets
train_val_frames, test_frames = train_test_split(train_frames_df, test_size=0.1, random_state=42)
train_frames, val_frames = train_test_split(train_val_frames, test_size=0.2, random_state=42)

# Create datasets
train_dataset = NexarDataset(train_frames, transform=train_transform)
val_dataset = NexarDataset(val_frames, transform=val_transform)
test_dataset = NexarDataset(test_frames, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Initialize MobileNetV2 model
def get_mobilenet_model():
    # Load pre-trained MobileNetV2 model
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Modify the classifier for binary classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    return model

model = get_mobilenet_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
# For MobileNetV2, we can use a slightly higher learning rate than ViT
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training function with additional metrics
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        epoch_precision = precision_score(train_labels, train_preds, zero_division=0)
        epoch_recall = recall_score(train_labels, train_preds, zero_division=0)
        epoch_f1 = f1_score(train_labels, train_preds, zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_epoch_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_epoch_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_epoch_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        scheduler.step(val_epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, Precision: {val_epoch_precision:.4f}, Recall: {val_epoch_recall:.4f}, F1: {val_epoch_f1:.4f}')
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), '/kaggle/working/best_mobilenet_model.pth')
            print("Saved best model!")
    
    return model

# Evaluate model on test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating on Test Set'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5)

# Evaluate on test set
evaluate_model(trained_model, test_loader, criterion)

# Modified test data processing for inference
import shutil

# Modified NexarDataset class with robust error handling
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # Filter out rows with invalid frame paths
        self.df = self.df[self.df['frame_path'].apply(self._is_valid_file)]
        print(f"Filtered dataset to {len(self.df)} valid frames")
        
    def _is_valid_file(self, path):
        return os.path.isfile(path) and os.access(path, os.R_OK)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label'] if row['label'] is not None else 0.0
        
        try:
            if not self._is_valid_file(img_path):
                raise FileNotFoundError(f"Image file not found or inaccessible: {img_path}")
                
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(label, dtype=torch.float32)

# Load test data
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
print("Test data loaded:", test_df.shape)

# Process a subset of test videos
sample_test_df = test_df.sample(2, random_state=42) if len(test_df) > 2 else test_df
test_frames_dir = '/kaggle/working/test_frames_mobilenet'

# Clear the test frames directory to avoid conflicts
if os.path.exists(test_frames_dir):
    shutil.rmtree(test_frames_dir)
os.makedirs(test_frames_dir, exist_ok=True)

# Process videos with additional debugging
def process_videos(df, output_dir, is_train=False):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing test videos"):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        # Validate saved frames
        valid_frames = []
        for frame in frame_data:
            if os.path.isfile(frame['frame_path']) and os.access(frame['frame_path'], os.R_OK):
                valid_frames.append(frame)
            else:
                print(f"Invalid frame file: {frame['frame_path']}")
        all_frames.extend(valid_frames)
    
    return pd.DataFrame(all_frames)

test_frames_df = process_videos(sample_test_df, test_frames_dir, is_train=False)

# Check if we have any test frames
if len(test_frames_df) == 0:
    print("No test frames were extracted. Creating dummy test data...")
    dummy_test_data = []
    os.makedirs('/kaggle/working/dummy_test_images_mobilenet', exist_ok=True)
    for i in range(50):
        frame_path = f'/kaggle/working/dummy_test_images_mobilenet/dummy_test_{i:05d}.jpg'
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(frame_path, img)
        dummy_test_data.append({
            'frame_path': frame_path,
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': None
        })
    test_frames_df = pd.DataFrame(dummy_test_data)

test_frames_df.to_csv('/kaggle/working/test_frames_mobilenet.csv', index=False)
print(f"Processed {len(test_frames_df)} test frames from {len(sample_test_df)} videos")

# Create test dataset and loader
test_dataset = NexarDataset(test_frames_df, transform=val_transform)
if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty after filtering invalid frames")

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)  # Set num_workers to 0

# Load best model for inference
model.load_state_dict(torch.load('/kaggle/working/best_mobilenet_model.pth', weights_only=True))  # Set weights_only=True
model.eval()

# Make predictions on test set with error handling
predictions = []
video_ids = []

with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(test_loader, desc='Making predictions with MobileNetV2')):
        try:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu().numpy()
            
            batch_indices = list(range(i*test_loader.batch_size, 
                                     min((i+1)*test_loader.batch_size, len(test_dataset))))
            
            actual_batch_size = min(test_loader.batch_size, len(test_dataset) - i*test_loader.batch_size)
            batch_indices = batch_indices[:actual_batch_size]
            
            batch_video_ids = [test_frames_df.iloc[idx]['video_id'] for idx in batch_indices]
            
            if isinstance(outputs, np.float32):
                outputs = np.array([outputs])
                
            if len(outputs) > len(batch_indices):
                outputs = outputs[:len(batch_indices)]
                
            predictions.extend(outputs)
            video_ids.extend(batch_video_ids)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

if not predictions:
    raise ValueError("No predictions were generated. Check test data and model inference.")

# Create a dataframe with predictions for each frame
prediction_df = pd.DataFrame({
    'video_id': video_ids,
    'prediction': predictions
})

# Aggregate predictions by video
video_predictions = prediction_df.groupby('video_id')['prediction'].max().reset_index()

# Create submission file
submission_df = pd.DataFrame({
    'id': video_predictions['video_id'],
    'target': video_predictions['prediction']
})

submission_df.to_csv('/kaggle/working/mobilenet_submission.csv', index=False)
print("MobileNetV2 submission file created!")
print(submission_df.head())


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torchvision.models import densenet121, DenseNet121_Weights

# Constants
FRAME_RATE = 30  # 30 fps
WINDOW_SIZE = 5  # Number of frames to use for prediction
POSITIVE_WINDOW = 1.5  # Seconds before the event to consider as positive

# Load the training data
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
print("Training data loaded:", train_df.shape)

# Fix the video path issue by ensuring proper formatting of video IDs
def get_video_path(video_id, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    base_dir = '/kaggle/input/nexar-collision-prediction'
    subfolder = 'train' if is_train else 'test'
    return os.path.join(base_dir, subfolder, f"{formatted_id}.mp4")

# Data preprocessing function to extract frames
def extract_frames(video_id, df_row, output_dir, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    video_path = get_video_path(video_id, is_train)
    target = df_row['target'] if 'target' in df_row else None
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    event_frame = None
    alert_frame = None
    
    if is_train and target == 1:
        event_time = float(df_row['time_of_event'])
        alert_time = float(df_row['time_of_alert'])
        event_frame = int(event_time * fps)
        alert_frame = int(alert_time * fps)
    
    frame_data = []
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        label = 0
        if is_train and target == 1:
            if alert_frame <= frame_idx <= event_frame:
                label = 1
            elif alert_frame - int(POSITIVE_WINDOW * fps) <= frame_idx < alert_frame:
                label = 1
        
        frame_path = os.path.join(output_dir, f"{formatted_id}_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_data.append({
            'frame_path': frame_path,
            'video_id': formatted_id,
            'frame_idx': frame_idx,
            'label': label if is_train else None
        })
    
    cap.release()
    return frame_data

# Process all videos and create a dataframe of frames
def process_videos(df, output_dir, is_train=True):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        all_frames.extend(frame_data)
    
    return pd.DataFrame(all_frames)

# Process a subset of videos for faster execution
sample_train_df = train_df.sample(10, random_state=42) if len(train_df) > 10 else train_df
train_frames_df = process_videos(sample_train_df, '/kaggle/working/train_frames_densenet', is_train=True)

# Check if we have any frames before proceeding
if len(train_frames_df) == 0:
    print("No frames were extracted. Creating dummy dataset...")
    dummy_data = []
    for i in range(100):
        dummy_data.append({
            'frame_path': f'/kaggle/working/dummy_densenet_{i:05d}.jpg',
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': i % 2
        })
    train_frames_df = pd.DataFrame(dummy_data)
    os.makedirs('/kaggle/working/dummy_images_densenet', exist_ok=True)
    for i in range(100):
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(f'/kaggle/working/dummy_densenet_{i:05d}.jpg', img)

train_frames_df.to_csv('/kaggle/working/train_frames_densenet.csv', index=False)
print(f"Processed {len(train_frames_df)} frames from {len(sample_train_df)} videos")

# Custom Dataset class
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(0, dtype=torch.float32)

# Define transforms for DenseNet
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split data into train, validation, and test sets
train_val_frames, test_frames = train_test_split(train_frames_df, test_size=0.1, random_state=42)
train_frames, val_frames = train_test_split(train_val_frames, test_size=0.2, random_state=42)

# Create datasets
train_dataset = NexarDataset(train_frames, transform=train_transform)
val_dataset = NexarDataset(val_frames, transform=val_transform)
test_dataset = NexarDataset(test_frames, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Initialize DenseNet-121 model
def get_densenet_model():
    # Load pre-trained DenseNet-121 model
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    
    # Modify the classifier for binary classification
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    return model

model = get_densenet_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
# DenseNet often works well with similar hyperparameters to ViT
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training function with additional metrics
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        epoch_precision = precision_score(train_labels, train_preds, zero_division=0)
        epoch_recall = recall_score(train_labels, train_preds, zero_division=0)
        epoch_f1 = f1_score(train_labels, train_preds, zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_epoch_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_epoch_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_epoch_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        scheduler.step(val_epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, Precision: {val_epoch_precision:.4f}, Recall: {val_epoch_recall:.4f}, F1: {val_epoch_f1:.4f}')
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), '/kaggle/working/best_densenet_model.pth')
            print("Saved best model!")
    
    return model

# Evaluate model on test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating on Test Set'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3)

# Evaluate on test set
evaluate_model(trained_model, test_loader, criterion)

# Modified test data processing for inference
import shutil

# Modified NexarDataset class with robust error handling
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # Filter out rows with invalid frame paths
        self.df = self.df[self.df['frame_path'].apply(self._is_valid_file)]
        print(f"Filtered dataset to {len(self.df)} valid frames")
        
    def _is_valid_file(self, path):
        return os.path.isfile(path) and os.access(path, os.R_OK)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label'] if row['label'] is not None else 0.0
        
        try:
            if not self._is_valid_file(img_path):
                raise FileNotFoundError(f"Image file not found or inaccessible: {img_path}")
                
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(label, dtype=torch.float32)

# Load test data
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
print("Test data loaded:", test_df.shape)

# Process a subset of test videos
sample_test_df = test_df.sample(5, random_state=42) if len(test_df) > 5 else test_df
test_frames_dir = '/kaggle/working/test_frames_densenet'

# Clear the test frames directory to avoid conflicts
if os.path.exists(test_frames_dir):
    shutil.rmtree(test_frames_dir)
os.makedirs(test_frames_dir, exist_ok=True)

# Process videos with additional debugging
test_frames_df = process_videos(sample_test_df, test_frames_dir, is_train=False)

# Check if we have any test frames
if len(test_frames_df) == 0:
    print("No test frames were extracted. Creating dummy test data...")
    dummy_test_data = []
    os.makedirs('/kaggle/working/dummy_test_images_densenet', exist_ok=True)
    for i in range(50):
        frame_path = f'/kaggle/working/dummy_test_images_densenet/dummy_test_{i:05d}.jpg'
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(frame_path, img)
        dummy_test_data.append({
            'frame_path': frame_path,
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': None
        })
    test_frames_df = pd.DataFrame(dummy_test_data)

test_frames_df.to_csv('/kaggle/working/test_frames_densenet.csv', index=False)
print(f"Processed {len(test_frames_df)} test frames from {len(sample_test_df)} videos")

# Create test dataset and loader
test_dataset = NexarDataset(test_frames_df, transform=val_transform)
if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty after filtering invalid frames")

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)  # Set num_workers to 0

# Load best model for inference
model.load_state_dict(torch.load('/kaggle/working/best_densenet_model.pth', weights_only=True))
model.eval()

# Make predictions on test set with error handling
predictions = []
video_ids = []

with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(test_loader, desc='Making predictions with DenseNet')):
        try:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu().numpy()
            
            batch_indices = list(range(i*test_loader.batch_size, 
                                     min((i+1)*test_loader.batch_size, len(test_dataset))))
            
            actual_batch_size = min(test_loader.batch_size, len(test_dataset) - i*test_loader.batch_size)
            batch_indices = batch_indices[:actual_batch_size]
            
            batch_video_ids = [test_frames_df.iloc[idx]['video_id'] for idx in batch_indices]
            
            if isinstance(outputs, np.float32):
                outputs = np.array([outputs])
                
            if len(outputs) > len(batch_indices):
                outputs = outputs[:len(batch_indices)]
                
            predictions.extend(outputs)
            video_ids.extend(batch_video_ids)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

if not predictions:
    raise ValueError("No predictions were generated. Check test data and model inference.")

# Create a dataframe with predictions for each frame
prediction_df = pd.DataFrame({
    'video_id': video_ids,
    'prediction': predictions
})

# Aggregate predictions by video
video_predictions = prediction_df.groupby('video_id')['prediction'].max().reset_index()

# Create submission file
submission_df = pd.DataFrame({
    'id': video_predictions['video_id'],
    'target': video_predictions['prediction']
})

submission_df.to_csv('/kaggle/working/densenet_submission.csv', index=False)
print("DenseNet submission file created!")
print(submission_df.head())


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torchvision.models import vgg16, VGG16_Weights

# Constants
FRAME_RATE = 30  # 30 fps
WINDOW_SIZE = 5  # Number of frames to use for prediction
POSITIVE_WINDOW = 1.5  # Seconds before the event to consider as positive

# Load the training data
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
print("Training data loaded:", train_df.shape)

# Fix the video path issue by ensuring proper formatting of video IDs
def get_video_path(video_id, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    base_dir = '/kaggle/input/nexar-collision-prediction'
    subfolder = 'train' if is_train else 'test'
    return os.path.join(base_dir, subfolder, f"{formatted_id}.mp4")

# Data preprocessing function to extract frames
def extract_frames(video_id, df_row, output_dir, is_train=True):
    formatted_id = f"{int(video_id):05d}"
    video_path = get_video_path(video_id, is_train)
    target = df_row['target'] if 'target' in df_row else None
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    event_frame = None
    alert_frame = None
    
    if is_train and target == 1:
        event_time = float(df_row['time_of_event'])
        alert_time = float(df_row['time_of_alert'])
        event_frame = int(event_time * fps)
        alert_frame = int(alert_time * fps)
    
    frame_data = []
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        label = 0
        if is_train and target == 1:
            if alert_frame <= frame_idx <= event_frame:
                label = 1
            elif alert_frame - int(POSITIVE_WINDOW * fps) <= frame_idx < alert_frame:
                label = 1
        
        frame_path = os.path.join(output_dir, f"{formatted_id}_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_data.append({
            'frame_path': frame_path,
            'video_id': formatted_id,
            'frame_idx': frame_idx,
            'label': label if is_train else None
        })
    
    cap.release()
    return frame_data

# Process all videos and create a dataframe of frames
# Modified training data processing to focus on positive cases (limited to 200)
def process_training_videos_balanced(df, output_dir, is_train=True, max_positive_videos=20):
    all_frames = []
    
    # Get positive videos (limited to max_positive_videos)
    positive_videos = df[df['target'] == 1]
    if len(positive_videos) > max_positive_videos:
        positive_videos = positive_videos.sample(max_positive_videos, random_state=42)
    
    
    print(f"Processing {len(positive_videos)} negative videos")
    
    # Process all selected negative videos
    for _, row in tqdm(positive_videos.iterrows(), total=len(positive_videos)):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        all_frames.extend(frame_data)
    
    frames_df = pd.DataFrame(all_frames)
    
    # Print class distribution
    if is_train:
        negative_frames = frames_df[frames_df['label'] == 0].shape[0]
        print(f"Extracted {negative_frames} negative frames")
    
    return frames_df

# Replace the previous processing call with this balanced version
train_frames_df = process_training_videos_balanced(train_df, '/kaggle/working/train_frames_vgg', 
                                                 is_train=True, max_positive_videos=20)


# Check if we have any frames before proceeding
if len(train_frames_df) == 0:
    print("No frames were extracted. Creating dummy dataset...")
    dummy_data = []
    for i in range(100):
        dummy_data.append({
            'frame_path': f'/kaggle/working/dummy_vgg_{i:05d}.jpg',
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': i % 2
        })
    train_frames_df = pd.DataFrame(dummy_data)
    os.makedirs('/kaggle/working/dummy_images_vgg', exist_ok=True)
    for i in range(100):
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(f'/kaggle/working/dummy_vgg_{i:05d}.jpg', img)

train_frames_df.to_csv('/kaggle/working/train_frames_vgg.csv', index=False)

# Custom Dataset class
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(0, dtype=torch.float32)

# Define transforms for VGG-16 - Standard size is 224x224
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split data into train, validation, and test sets
train_val_frames, test_frames = train_test_split(train_frames_df, test_size=0.1, random_state=42)
train_frames, val_frames = train_test_split(train_val_frames, test_size=0.2, random_state=42)

# Create datasets
train_dataset = NexarDataset(train_frames, transform=train_transform)
val_dataset = NexarDataset(val_frames, transform=val_transform)
test_dataset = NexarDataset(test_frames, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Initialize VGG-16 model
def get_vgg16_model():
    # Load pre-trained VGG16 model
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # Modify the classifier for binary classification
    # VGG16's classifier has 6 layers, we'll replace the last layer
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    return model

model = get_vgg16_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
# For VGG, we'll use a slightly higher learning rate than ViT
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training function with additional metrics
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        epoch_precision = precision_score(train_labels, train_preds, zero_division=0)
        epoch_recall = recall_score(train_labels, train_preds, zero_division=0)
        epoch_f1 = f1_score(train_labels, train_preds, zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_epoch_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_epoch_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_epoch_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        scheduler.step(val_epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, Precision: {val_epoch_precision:.4f}, Recall: {val_epoch_recall:.4f}, F1: {val_epoch_f1:.4f}')
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), '/kaggle/working/best_vgg16_model.pth')
            print("Saved best model!")
    
    return model

# Evaluate model on test set
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating on Test Set'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=3)

# Evaluate on test set
evaluate_model(trained_model, test_loader, criterion)

# Modified test data processing for inference
import shutil

# Modified NexarDataset class with robust error handling
class NexarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # Filter out rows with invalid frame paths
        self.df = self.df[self.df['frame_path'].apply(self._is_valid_file)]
        print(f"Filtered dataset to {len(self.df)} valid frames")
        
    def _is_valid_file(self, path):
        return os.path.isfile(path) and os.access(path, os.R_OK)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['frame_path']
        label = row['label'] if row['label'] is not None else 0.0
        
        try:
            if not self._is_valid_file(img_path):
                raise FileNotFoundError(f"Image file not found or inaccessible: {img_path}")
                
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(label, dtype=torch.float32)

# Load test data
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
print("Test data loaded:", test_df.shape)

# Process a subset of test videos
sample_test_df = test_df.sample(2, random_state=42) if len(test_df) > 2 else test_df
test_frames_dir = '/kaggle/working/test_frames_vgg16'

# Clear the test frames directory to avoid conflicts
if os.path.exists(test_frames_dir):
    shutil.rmtree(test_frames_dir)
os.makedirs(test_frames_dir, exist_ok=True)

# Process videos with additional debugging
def process_videos(df, output_dir, is_train=False):
    all_frames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing test videos"):
        video_id = row['id']
        frame_data = extract_frames(video_id, row, output_dir, is_train)
        # Validate saved frames
        valid_frames = []
        for frame in frame_data:
            if os.path.isfile(frame['frame_path']) and os.access(frame['frame_path'], os.R_OK):
                valid_frames.append(frame)
            else:
                print(f"Invalid frame file: {frame['frame_path']}")
        all_frames.extend(valid_frames)
    
    return pd.DataFrame(all_frames)

test_frames_df = process_videos(sample_test_df, test_frames_dir, is_train=False)

# Check if we have any test frames
if len(test_frames_df) == 0:
    print("No test frames were extracted. Creating dummy test data...")
    dummy_test_data = []
    os.makedirs('/kaggle/working/dummy_test_images_vgg16', exist_ok=True)
    for i in range(50):
        frame_path = f'/kaggle/working/dummy_test_images_vgg16/dummy_test_{i:05d}.jpg'
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(frame_path, img)
        dummy_test_data.append({
            'frame_path': frame_path,
            'video_id': f"{i % 5:05d}",
            'frame_idx': i,
            'label': None
        })
    test_frames_df = pd.DataFrame(dummy_test_data)

test_frames_df.to_csv('/kaggle/working/test_frames_vgg16.csv', index=False)
print(f"Processed {len(test_frames_df)} test frames from {len(sample_test_df)} videos")

# Create test dataset and loader
test_dataset = NexarDataset(test_frames_df, transform=val_transform)
if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty after filtering invalid frames")

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)  # Set num_workers to 0

# Load best model for inference
model.load_state_dict(torch.load('/kaggle/working/best_vgg16_model.pth', weights_only=True))  # Set weights_only=True
model.eval()

# Make predictions on test set with error handling
predictions = []
video_ids = []

with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(test_loader, desc='Making predictions with VGG-16')):
        try:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu().numpy()
            
            batch_indices = list(range(i*test_loader.batch_size, 
                                     min((i+1)*test_loader.batch_size, len(test_dataset))))
            
            actual_batch_size = min(test_loader.batch_size, len(test_dataset) - i*test_loader.batch_size)
            batch_indices = batch_indices[:actual_batch_size]
            
            batch_video_ids = [test_frames_df.iloc[idx]['video_id'] for idx in batch_indices]
            
            if isinstance(outputs, np.float32):
                outputs = np.array([outputs])
                
            if len(outputs) > len(batch_indices):
                outputs = outputs[:len(batch_indices)]
                
            predictions.extend(outputs)
            video_ids.extend(batch_video_ids)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

if not predictions:
    raise ValueError("No predictions were generated. Check test data and model inference.")

# Create a dataframe with predictions for each frame
prediction_df = pd.DataFrame({
    'video_id': video_ids,
    'prediction': predictions
})

# Aggregate predictions by video
video_predictions = prediction_df.groupby('video_id')['prediction'].max().reset_index()

# Create submission file
submission_df = pd.DataFrame({
    'id': video_predictions['video_id'],
    'target': video_predictions['prediction']
})

submission_df.to_csv('/kaggle/working/vgg16_submission.csv', index=False)
print("VGG-16 submission file created!")
print(submission_df.head())




