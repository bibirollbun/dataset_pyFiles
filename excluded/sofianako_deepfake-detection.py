# 1. Install PyTorch-compatible MTCNN (from facenet-pytorch)
# This library is essential for efficient GPU-accelerated face detection.
!pip install facenet-pytorch --quiet

# 2. Install EfficientNet implementation (often available via torch.hub, but good practice to install)
!pip install efficientnet-pytorch --quiet

# 3. Install headless version of OpenCV for video processing without GUI dependencies
!pip install opencv-python-headless --quiet


# --- 1.1. CRITICAL FIX: RESOLVE NUMPY/SCIKIT-LEARN CONFLICTS ---
# Reinstalling core data packages to a known working version to fix the 
# 'ValueError: numpy.dtype size changed' error in scikit-learn/numpy.
# This must be run before importing any scikit-learn component.

!pip install numpy==1.26.4 scikit-learn==1.2.2 --force-reinstall --quiet
!pip install facenet-pytorch efficientnet-pytorch opencv-python-headless --quiet


import os
import time
import json
import random
from PIL import Image

# Data processing and computation
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# Video processing
import cv2 

# PyTorch Core modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# MTCNN and EfficientNet implementations
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet

# === ENVIRONMENT SETUP ===

# Set a random seed for reproducibility across runs
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Optimization settings for PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Determine the device (GPU or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Check for GPU details
if device.type == 'cuda':
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

# --- 2.5. MULTIPROCESSING FIX for CUDA on Linux ---

# Check if the environment is suitable for changing the start method
if torch.cuda.is_available() and device.type == 'cuda':
    # Set the start method to 'spawn' if it hasn't been set yet
    # 'spawn' is the safest method for CUDA multiprocessing on Linux/Kaggle
    if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
        print("Setting multiprocessing start method to 'spawn' for CUDA compatibility.")
        torch.multiprocessing.set_start_method('spawn', force=True)
else:
    print("CUDA not available or not required; not setting multiprocessing start method.")


# Assume the Deepfake Detection Challenge or FaceForensics++ dataset is added as a Data Source.
# The paths are typically set up like this in Kaggle:
DATA_DIR = '../input/deepfake-detection-challenge/train_sample_videos/'
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.json')

# 1. Load metadata from the JSON file
try:
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_PATH}. Check your Kaggle Data Sources.")
    # Create an empty DataFrame to prevent subsequent code errors
    metadata_df = pd.DataFrame() 
    # NOTE: Adjust DATA_DIR and METADATA_PATH to match your chosen dataset path!
else:
    # 2. Convert raw metadata dictionary into a Pandas DataFrame for easier handling
    records = []
    for k, v in metadata.items():
        records.append({
            'file_name': k,
            'label': v['label'],
            # The 'original' key is often missing for REAL videos, so use 'N/A' as default
            'original': v.get('original', 'N/A') 
        })
    metadata_df = pd.DataFrame(records)

    # 3. Print dataset statistics
    print(f"Total videos in metadata: {len(metadata_df)}")
    print("Label distribution:")
    print(metadata_df['label'].value_counts())
    
    # 4. Filtration: Ensure we only work with video files that actually exist in the directory
    existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mp4')]
    metadata_df = metadata_df[metadata_df['file_name'].isin(existing_files)].reset_index(drop=True)
    
    print(f"\nVideos available for processing: {len(metadata_df)}")

# Display the first few rows for verification
if not metadata_df.empty:
    print(metadata_df.head())


# --- 4.1. FACE PRE-EXTRACTION UTILITY SCRIPT (FIXED WITH OPENCV SAVE) ---
# Goal: Decode videos, extract faces using MTCNN, and save faces as JPEG files 
# to resolve the RAM/multiprocessing error.

import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN

# 1. Define Paths and Constants
OUTPUT_DIR = './pre_extracted_faces_dataset/'
FACE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'faces')
os.makedirs(FACE_OUTPUT_DIR, exist_ok=True)

PRE_EXTRACT_FRAMES = 15 
FACE_SIZE = 256 

def save_faces_from_video(video_path: str, video_name: str, label: str, frames_to_save: int, mtcnn_model):
    """
    Decodes video, extracts faces using MTCNN, and saves using OpenCV to avoid PIL errors.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return

    # Safely generate indices
    if frame_count < frames_to_save:
        frame_indices = np.arange(frame_count)
    else:
        frame_indices = np.linspace(0, frame_count - 1, frames_to_save, dtype=int)
    
    for k, i in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
            
        # Convert to RGB for MTCNN detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # 1. Detect faces
        boxes, _ = mtcnn_model.detect(frame_pil)
        
        if boxes is not None and len(boxes) > 0:
            # Take the first face
            box = boxes[0].astype(int)
            
            # Ensure box coordinates are within image boundaries
            width, height = frame_pil.size
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(width, box[2])
            y2 = min(height, box[3])
            
            # Check if crop area is valid
            if x2 > x1 and y2 > y1:
                # 2. Crop using PIL (Safe and easy)
                face_img_pil = frame_pil.crop((x1, y1, x2, y2))
                
                # 3. Resize using PIL
                face_img_pil = face_img_pil.resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS)
                
                # 4. Save using OpenCV (bypasses PIL error)
                # Convert PIL (RGB) back to Numpy (RGB) then to OpenCV (BGR)
                face_img_np = np.array(face_img_pil)
                face_img_bgr = cv2.cvtColor(face_img_np, cv2.COLOR_RGB2BGR)
                
                output_filename = f"{video_name.split('.')[0]}_{k}_{label}.jpg"
                output_path = os.path.join(FACE_OUTPUT_DIR, output_filename)
                
                # Write file using OpenCV (Quality 95)
                cv2.imwrite(output_path, face_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
    cap.release()


# --- Main Execution for Pre-Extraction ---

# Re-initialize MTCNN 
mtcnn_pre_extract = MTCNN(
    image_size=FACE_SIZE, margin=0, min_face_size=20, 
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False, device=device
)

# Combine train_df and val_df 
try:
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True) 
except NameError:
    full_df = metadata_df.copy()

print(f"Starting face extraction for {len(full_df)} videos using OpenCV saver...")

for index, row in tqdm(full_df.iterrows(), total=len(full_df)):
    video_path = os.path.join(DATA_DIR, row['file_name'])
    
    save_faces_from_video(
        video_path, 
        row['file_name'], 
        row['label'], 
        frames_to_save=PRE_EXTRACT_FRAMES,
        mtcnn_model=mtcnn_pre_extract
    )

print("Face extraction complete. New faces are saved in: ./pre_extracted_faces_dataset/faces/")


# --- 6. DATA SPLIT AND DATALOADER INITIALIZATION (IMAGE MODE) ---

from sklearn.model_selection import train_test_split
from torchvision import transforms # Import transforms here if not done in Cell 2

# Define NEW Data Paths
NEW_DATA_DIR = './pre_extracted_faces_dataset/faces/' 

# 1. Create a DataFrame from the saved images
image_files = os.listdir(NEW_DATA_DIR)
image_files = [f for f in image_files if f.endswith('.jpg')] 

# The file name format is: videoName_frameIndex_LABEL.jpg
new_metadata_df = pd.DataFrame({
    'file_name': image_files,
    # Extract 'REAL' or 'FAKE' from the file name (e.g., last element before .jpg)
    'label': [f.split('_')[-1].split('.')[0] for f in image_files] 
})

print(f"Total extracted face images: {len(new_metadata_df)}")
print("Label distribution in image dataset:")
print(new_metadata_df['label'].value_counts())

# Split the image metadata into training and validation sets
train_df, val_df = train_test_split(
    new_metadata_df, 
    test_size=0.1, # Use a smaller test split if the total number of frames is huge
    random_state=SEED, 
    stratify=new_metadata_df['label']
)

# Define Image Transformation (Normalization)
IMG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(), # Converts to tensor and scales to [0, 1]
    # Normalization (0.5 mean/std for range [-1, 1])
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])


# Define Hyperparameters
BATCH_SIZE = 32 # Can now be much larger
NUM_WORKERS = 0

# 2. Define the new Dataset Class (DeepFakeImageDataset)
class DeepFakeImageDataset(Dataset):
    """
    Dataset for loading pre-extracted face images directly from disk.
    """
    def __init__(self, metadata_df: pd.DataFrame, root_dir: str, transform=IMG_TRANSFORM):
        self.metadata = metadata_df.copy()
        self.root_dir = root_dir
        self.transform = transform 

        # Convert label strings to numeric format (0 for REAL, 1 for FAKE)
        self.metadata['label_numeric'] = self.metadata['label'].apply(
            lambda x: 1.0 if x == 'FAKE' else 0.0 # Use 1.0/0.0 for BCE Loss
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        file_name = row['file_name']
        label = row['label_numeric']
        image_path = os.path.join(self.root_dir, file_name)
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        return image_tensor, torch.tensor(label, dtype=torch.float32)

# 3. Initialize DataLoaders
train_dataset = DeepFakeImageDataset(train_df, NEW_DATA_DIR, transform=IMG_TRANSFORM)
val_dataset = DeepFakeImageDataset(val_df, NEW_DATA_DIR, transform=IMG_TRANSFORM)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"\nImage DataLoaders initialized successfully with NUM_WORKERS={NUM_WORKERS}.")


def setup_efficientnet(model_name: str = 'efficientnet-b4') -> nn.Module:
    """
    Loads a pre-trained EfficientNet model and adapts its final layer 
    for binary classification.
    
    Args:
        model_name (str): Name of the EfficientNet variant (e.g., 'efficientnet-b4').
        
    Returns:
        nn.Module: The configured model ready for training.
    """
    
    print(f"Loading pre-trained model: {model_name}...")
    
    try:
        # Load model weights pre-trained on ImageNet
        # The 'from_pretrained' function handles downloading the weights
        model = EfficientNet.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading EfficientNet from PyPI/Cache. Trying local hub or custom path. Error: {e}")
        # Fallback: if you have a local copy of weights, load them here
        model = EfficientNet.from_name(model_name)
        # You may need to manually load state_dict here: model.load_state_dict(...)

    # Freeze the convolutional base layers (optional, but speeds up early training)
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # Get the number of input features for the final classification layer
    num_ftrs = model._fc.in_features
    
    # Replace the final fully-connected layer (_fc) with a new Sequential layer
    # Output: 1 neuron (for binary classification probability)
    model._fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid() # Sigmoid squashes the output to [0, 1] (probability of being FAKE)
    )
    
    # Move the model to the defined device (GPU)
    model = model.to(device)
    
    print("Model loaded and adapted successfully.")
    return model

# Initialize the model
model = setup_efficientnet(model_name='efficientnet-b4')

# Print the model architecture (last few layers) for verification
print("\nFinal layer architecture:")
print(model._fc)


# Binary Cross-Entropy Loss (BCE) is standard for two-class probability output
criterion = nn.BCELoss() 

# Adam optimizer is a robust choice for deep learning models
# We only pass parameters that require gradient updates (i.e., parameters that are not frozen)
LEARNING_RATE = 1e-4 # Standard starting learning rate
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Optional: Learning Rate Scheduler (helps stabilize training and reach better optima)
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Reduces LR if validation loss doesn't improve for 'patience' epochs
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

print(f"Criterion: {criterion.__class__.__name__}")
print(f"Optimizer: {optimizer.__class__.__name__} with LR={LEARNING_RATE}")


# --- 9. SIMPLIFIED TRAINING AND VALIDATION LOOPS (Image Mode) ---

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    """Runs a single training epoch for image data."""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        # inputs shape: (Batch_Size, C, H, W) - now images only
        inputs = inputs.to(device)
        labels = labels.to(device) 

        optimizer.zero_grad()
        
        # Forward pass (one image per prediction)
        outputs = model(inputs).squeeze(1) # outputs shape: (Batch_Size)
        
        # Loss calculation (direct loss on batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0) # Multiply by batch size

    return running_loss / len(loader.dataset)


def validate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Runs a single validation epoch for image data."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs).squeeze(1) # outputs shape: (Batch_Size)

            # Loss calculation
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # Accuracy calculation
            predicted_labels = (outputs > 0.5).float() 
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    avg_accuracy = correct_predictions / total_samples
    return avg_loss, avg_accuracy


NUM_EPOCHS = 15 # Define the number of full passes over the dataset

best_val_accuracy = 0.0
history = {
    'train_loss': [],
    'val_loss': [],
    'val_acc': []
}

print(f"Starting training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    
    # Training step
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    history['train_loss'].append(train_loss)
    
    # Validation step
    val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_accuracy)
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | Duration: {epoch_duration:.2f}s")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    
    # Step the learning rate scheduler
    scheduler.step(val_loss)
    
    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        # Save the model state
        torch.save(model.state_dict(), f'best_deepfake_model_epoch_{epoch+1}.pth')
        print(f"Model saved! New best accuracy: {best_val_accuracy:.4f}")
        
print("\nTraining completed.")


import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN # Need MTCNN instance for visualization

# Define paths for visualization (use one sample video file name) adylbeequz.mp4 aelfnikyqj
SAMPLE_VIDEO_PATH = os.path.join(DATA_DIR, 'adylbeequz.mp4') 

# Re-initialize MTCNN for visualization (post_process=False is better for raw detection display)
mtcnn_vis = MTCNN(
    image_size=FACE_SIZE, margin=0, min_face_size=20, 
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False, device=device
)

def visualize_face_detection(video_path: str, frame_index: int = 5):
    """
    Shows the original frame and the detected face crop for a given video.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Could not read frame {frame_index} from video {os.path.basename(video_path)}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    # 1. Detect faces (gets bounding box coordinates)
    boxes, _ = mtcnn_vis.detect(frame_pil)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # --- Left Plot: Raw Frame with Bounding Box ---
    ax[0].imshow(frame_rgb)
    ax[0].set_title(f"Raw Frame {frame_index}")
    ax[0].axis('off')

    if boxes is not None and len(boxes) > 0:
        # Draw the bounding box for the first detected face
        box = boxes[0].astype(int)
        ax[0].plot([box[0], box[2], box[2], box[0], box[0]], 
                   [box[1], box[1], box[3], box[3], box[1]], 
                   color='red', linewidth=3)
        
        # 2. Extract and crop the face
        face_tensor = mtcnn_vis.extract(frame_pil, boxes[0:1], save_path=None)
        # --- Right Plot: Extracted and Processed Face ---
        if face_tensor is not None:
            # Denormalize tensor back to image array for display
            # Крок 1: Permute (C, H, W -> H, W, C) і перемістити на CPU
            face_img_array = face_tensor.permute(1, 2, 0).cpu().numpy()
            
            # --- КРОК 2: ЗВОРОТНА СТАНДАРТИЗАЦІЯ (ДЕНОРМАЛІЗАЦІЯ) ---
            # Визначаємо параметри ImageNet
            # mean = np.array([0.485, 0.456, 0.406])
            # std = np.array([0.229, 0.224, 0.225])
            
            # # Зворотна стандартизація: x = x' * std + mean
            # face_img_array = face_img_array * std + mean
            
            # # КРОК 3: Обрізка для гарантії [0, 1]
            # face_img_array = np.clip(face_img_array, 0, 1)
            if face_img_array.max() <= 1.0:
                face_img_array = face_img_array * 255
            # Крок 4: Перетворення до [0, 255] і цілих чисел (uint8)
            face_img = face_img_array.astype(np.uint8)
            
            ax[1].imshow(face_img)
            ax[1].set_title(f"Extracted Face ({FACE_SIZE}x{FACE_SIZE})")
            ax[1].axis('off')
        else:
            ax[1].set_title("No face extracted after cropping")
            
    else:
        ax[0].set_title(f"Raw Frame {frame_index} (No Face Detected)")
        ax[1].set_title("No Face Detected")
        
    plt.tight_layout()
    plt.savefig('face_detection_example.png')
    plt.show()

# Run the visualization utility (requires a valid video path)
visualize_face_detection(SAMPLE_VIDEO_PATH) 
# NOTE: Uncomment the line above and ensure SAMPLE_VIDEO_PATH is correct after the file upload.


# --- 13. MODEL EVALUATION AND CONFUSION MATRIX ---

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import seaborn as sns

def evaluate_model(model: nn.Module, loader: DataLoader, model_path: str, device: torch.device):
    """
    Loads the trained model, evaluates it on the DataLoader, and generates metrics.
    """
    # Load the best weights saved during training
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}. Ensure training completed and file exists.")
        return

    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze(1) # Probability outputs (0 to 1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(outputs.cpu().tolist())

    # --- Metrics Calculation ---
    
    # Binary predictions (using 0.5 threshold)
    binary_predictions = (np.array(all_predictions) > 0.5).astype(int)
    
    # 1. Accuracy
    acc = accuracy_score(all_labels, binary_predictions)
    print(f"\nFinal Validation Accuracy: {acc:.4f}")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, binary_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['REAL (0)', 'FAKE (1)'], yticklabels=['REAL (0)', 'FAKE (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 3. ROC-AUC Score
    try:
        roc_auc = roc_auc_score(all_labels, all_predictions)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        # 4. ROC Curve Plot
        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.show()
    except ValueError:
        print("ROC AUC calculation skipped (not enough samples in one class).")

# --- Execution ---
# NOTE: Replace 'best_deepfake_model_epoch_X.pth' with the actual file name saved in cell 10.
BEST_MODEL_PATH = 'best_deepfake_model_epoch_9.pth' 
evaluate_model(model, val_loader, BEST_MODEL_PATH, device)


def plot_training_history(history: dict):
    """
    Plots the loss and accuracy curves based on the saved history dictionary.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (BCE)')
    plt.legend()
    
    # 2. Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# --- Execution ---
# NOTE: This requires the 'history' dictionary from cell 10 to be accessible.
plot_training_history(history)


def visualize_comparative_predictions(model, dataloader, device, num_images=5):
    """
    Візуалізує та порівнює зображення, класифіковані моделлю як FAKE та REAL.
    
    Args:
        model: Навчена модель (EfficientNet).
        dataloader: DataLoader з валідаційними даними.
        device: Пристрій (cuda або cpu).
        num_images: Кількість зображень кожного типу для показу.
    """
    model.eval()
    
    # Списки для збереження прикладів
    pred_fake_imgs = []
    pred_fake_infos = [] # (True Label, Probability)
    
    pred_real_imgs = []
    pred_real_infos = []
    
    print("Збір прикладів для візуалізації...")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # labels: 1.0 = FAKE, 0.0 = REAL
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).view(-1) # Отримуємо ймовірності [0, 1]
            preds = (probs > 0.5).float()           # Порогова класифікація
            
            # Проходимо по батчу
            for i in range(len(preds)):
                # Якщо назбирали достатньо обох типів, виходимо
                if len(pred_fake_imgs) >= num_images and len(pred_real_imgs) >= num_images:
                    break
                
                # Денормалізація зображення для показу
                # Зворотна формула до transforms.Normalize([0.5...], [0.5...])
                # pixel = input * 0.5 + 0.5
                img_tensor = inputs[i].cpu()
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = img_np * 0.5 + 0.5
                img_np = np.clip(img_np, 0, 1) # Гарантуємо діапазон [0, 1]
                
                prob = probs[i].item()
                true_label = labels[i].item()
                
                # Розподіляємо за ПЕРЕДБАЧЕННЯМ моделі
                if preds[i] == 1.0: # Модель каже: FAKE
                    if len(pred_fake_imgs) < num_images:
                        pred_fake_imgs.append(img_np)
                        pred_fake_infos.append((true_label, prob))
                else: # Модель каже: REAL
                    if len(pred_real_imgs) < num_images:
                        pred_real_imgs.append(img_np)
                        pred_real_infos.append((true_label, prob))
            
            if len(pred_fake_imgs) >= num_images and len(pred_real_imgs) >= num_images:
                break
    
    # --- Побудова графіків ---
    fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Рядок 1: Зображення, передбачені як FAKE
    for i in range(num_images):
        ax = axes[0, i]
        if i < len(pred_fake_imgs):
            ax.imshow(pred_fake_imgs[i])
            
            true_lbl = pred_fake_infos[i][0]
            prob = pred_fake_infos[i][1]
            
            # Колір заголовка: Зелений, якщо прогноз вірний, Червоний - якщо ні
            # Прогноз тут завжди FAKE (1.0). Якщо True Label == 1.0, то вірно.
            is_correct = (true_lbl == 1.0)
            color = 'green' if is_correct else 'red'
            true_str = "FAKE" if true_lbl == 1.0 else "REAL"
            
            ax.set_title(f"Pred: FAKE\nTrue: {true_str}\nProb: {prob:.2f}", color=color, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "Not found", ha='center')
        ax.axis('off')
    
    axes[0, 0].set_ylabel("Predicted: FAKE", rotation=90, size='large', labelpad=10)

    # Рядок 2: Зображення, передбачені як REAL
    for i in range(num_images):
        ax = axes[1, i]
        if i < len(pred_real_imgs):
            ax.imshow(pred_real_imgs[i])
            
            true_lbl = pred_real_infos[i][0]
            prob = pred_real_infos[i][1]
            
            # Прогноз тут завжди REAL (0.0). Якщо True Label == 0.0, то вірно.
            is_correct = (true_lbl == 0.0)
            color = 'green' if is_correct else 'red'
            true_str = "FAKE" if true_lbl == 1.0 else "REAL"
            
            ax.set_title(f"Pred: REAL\nTrue: {true_str}\nProb: {prob:.2f}", color=color, fontweight='bold')
        else:
            ax.text(0.5, 0.5, "Not found", ha='center')
        ax.axis('off')

    axes[1, 0].set_ylabel("Predicted: REAL", rotation=90, size='large', labelpad=10)
    
    plt.suptitle("Model Predictions Comparison: FAKE vs REAL", fontsize=16)
    plt.show()

# --- ЗАПУСК ВІЗУАЛІЗАЦІЇ ---
# Переконайтеся, що 'model' та 'val_loader' вже визначені та модель навчена
try:
    visualize_comparative_predictions(model, val_loader, device, num_images=5)
except NameError:
    print("Помилка: Переконайтеся, що ви запустили всі попередні комірки (визначення моделі, dataloader тощо).")

