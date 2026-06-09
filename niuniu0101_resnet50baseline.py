# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# --- Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
# Note: ImageFolder is imported but CustomCassavaDataset is preferred for this setup.
from torchvision.datasets import ImageFolder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import json 
import shutil 

# --- Configuration ---
class Config:
    # Adjust paths based on Kaggle Notebook environment
    DATA_ROOT = '/kaggle/input/cassava-leaf-disease-classification' 
    TRAIN_CSV = os.path.join(DATA_ROOT, 'train.csv')
    TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, 'train_images') 
    PROCESSED_TRAIN_DIR = os.path.join('/kaggle/working', 'processed_train_images') 

    IMAGE_SIZE = 384 
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 5 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

print(f"Using device: {Config.DEVICE}")


# --- Data Preparation Utility (Optional) ---
# This function is used if you want to physically move/copy files for ImageFolder structure.
# For CustomCassavaDataset, it's not strictly necessary.
def prepare_data_for_imagefolder(train_csv_path, train_images_dir, output_dir):
    """
    Reads the train.csv and organizes images into class-specific subfolders
    for torchvision.datasets.ImageFolder.
    """
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"'{output_dir}' already exists and is not empty. Skipping data preparation.")
        return

    print(f"Preparing data for ImageFolder structure in '{output_dir}'...")
    df = pd.read_csv(train_csv_path)

    for class_id in range(Config.NUM_CLASSES):
        os.makedirs(os.path.join(output_dir, str(class_id)), exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Organizing images"):
        img_filename = row['image_id']
        label = str(row['label'])
        
        src_path = os.path.join(train_images_dir, img_filename)
        dst_path = os.path.join(output_dir, label, img_filename)

        if not os.path.exists(src_path):
            print(f"Warning: Image {src_path} not found. Skipping.")
            continue
        
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {e}")
            
    print("Data preparation complete.")

# --- Custom Dataset Definition ---
class CustomCassavaDataset(Dataset):
    def __init__(self, image_ids, labels, img_dir, transform=None):
        self.image_ids = image_ids.tolist()
        self.labels = labels.tolist()
        self.img_dir = img_dir
        self.transform = transform
        
        self.label_map = {label: i for i, label in enumerate(sorted(list(set(self.labels))))}
        print(f"Dataset initialized with {len(self.image_ids)} samples.")
        print(f"Label map: {self.label_map}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.label_map[label]

# --- Data Transforms ---
train_transforms = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet means and stds
])

val_transforms = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- Model Definition (ResNet50 Baseline) ---
def get_resnet50_baseline_model(num_classes=Config.NUM_CLASSES):
    """
    Loads a pre-trained ResNet50 model and modifies its final classification layer.
    """
    print("11")
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_accuracy = 0.0
    
    model.to(device) 
    best_model_save_path = os.path.join('/kaggle/working', "resnet50_baseline_best_model.pth")

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() 

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}")

        # Corrected unpacking here
        val_loss, val_accuracy, val_f1, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f} F1: {val_f1:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Saved best model with accuracy: {best_val_accuracy:.4f}")

    print("Training complete!")

# --- Evaluation Function ---
def evaluate_model(model, data_loader, criterion, device):
    model.eval() 
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad(): 
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    f1 = f1_score(all_labels, all_predictions, average='macro') 
    cm = confusion_matrix(all_labels, all_predictions)

    return avg_loss, accuracy, f1, all_labels, all_predictions, cm


# --- Main Execution - Step 1: Create Dataset and DataLoader ---
df_train = pd.read_csv(Config.TRAIN_CSV)
train_img_ids, val_img_ids, train_labels, val_labels = train_test_split(
    df_train['image_id'], df_train['label'],
    test_size=0.2, stratify=df_train['label'], random_state=Config.RANDOM_SEED
)

train_dataset = CustomCassavaDataset(
    image_ids=train_img_ids,
    labels=train_labels,
    img_dir=Config.TRAIN_IMAGES_DIR, 
    transform=train_transforms
)
val_dataset = CustomCassavaDataset(
    image_ids=val_img_ids,
    labels=val_labels,
    img_dir=Config.TRAIN_IMAGES_DIR,
    transform=val_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count() // 2, 
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE * 2, 
    shuffle=False,
    num_workers=os.cpu_count() // 2,
    pin_memory=True
)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# --- Main Execution - Step 2: Initialize Model, Loss Function, and Optimizer ---
model = get_resnet50_baseline_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

# --- Main Execution - Step 3: Train the model ---
print("--- Starting Model Training ---")
train_model(model, train_loader, val_loader, criterion, optimizer, Config.NUM_EPOCHS, Config.DEVICE)





# --- Main Execution - Step 4: Load the best model and evaluate on the validation set one last time ---
print("\n--- Final Evaluation of Best Model ---")
best_model = get_resnet50_baseline_model()
best_model_path = os.path.join('/kaggle/working', "resnet50_baseline_best_model.pth")

if os.path.exists(best_model_path):
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(Config.DEVICE) 

    val_loss, val_accuracy, val_f1, all_labels_final, all_predictions_final, cm_final = evaluate_model(
        best_model, val_loader, criterion, Config.DEVICE
    )
    print(f"Best Model Val Loss: {val_loss:.4f}")
    print(f"Best Model Val Accuracy: {val_accuracy:.4f}")
    print(f"Best Model Val F1-Score (Macro): {val_f1:.4f}")
    print("Confusion Matrix:\n", cm_final)
else:
    print(f"Best model not found at {best_model_path}. Please ensure training completed successfully.")

print("\nResNet50 Baseline experiment complete.")


# --- 1. Imports for this independent module ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

# --- 2. Define ALL Paths and Configurations for this Test Module (Independent) ---
# IMPORTANT: Adjust these paths if your dataset location differs in your Kaggle environment.
# These paths are assumed to be consistent with standard Kaggle competition data mounting.
DATA_ROOT_FOR_INFERENCE = '/kaggle/input/cassava-leaf-disease-classification'
TEST_IMAGES_DIR_FOR_INFERENCE = os.path.join(DATA_ROOT_FOR_INFERENCE, 'test_images')
SAMPLE_SUBMISSION_CSV_FOR_INFERENCE = os.path.join(DATA_ROOT_FOR_INFERENCE, 'sample_submission.csv')
BEST_MODEL_PATH_FOR_INFERENCE = os.path.join('/kaggle/working', 'resnet50_baseline_best_model.pth') # Explicit model path

IMAGE_SIZE_FOR_INFERENCE = 384
BATCH_SIZE_INFERENCE = 64 # Use a larger batch size for inference
NUM_CLASSES_FOR_INFERENCE = 5
DEVICE_FOR_INFERENCE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device for inference: {DEVICE_FOR_INFERENCE}")
print(f"Loading test images from: {TEST_IMAGES_DIR_FOR_INFERENCE}")
print(f"Loading sample submission from: {SAMPLE_SUBMISSION_CSV_FOR_INFERENCE}")
print(f"Loading best model from: {BEST_MODEL_PATH_FOR_INFERENCE}")


# --- 3. Define the Model Architecture (Must exactly match the trained model) ---
# This definition is self-contained within this block.
def get_resnet50_model_for_inference(num_classes=NUM_CLASSES_FOR_INFERENCE):
    model = models.resnet50(weights=None) # No ImageNet weights here; we'll load custom ones
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- 4. Define Test Dataset Class ---
# This definition is self-contained within this block.
class TestCassavaDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform=None):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_name # Return image tensor and its ID

# --- 5. Define Test Transforms ---
# These transforms are self-contained within this block.
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_FOR_INFERENCE, IMAGE_SIZE_FOR_INFERENCE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet means and stds
])

# --- 6. Main Inference Logic ---
print("\n--- Starting Test Inference ---")

# Load the model structure
model = get_resnet50_model_for_inference()

# Load the trained weights from the explicitly defined path
if os.path.exists(BEST_MODEL_PATH_FOR_INFERENCE):
    # map_location ensures it loads correctly whether on CPU or GPU
    model.load_state_dict(torch.load(BEST_MODEL_PATH_FOR_INFERENCE, map_location=DEVICE_FOR_INFERENCE))
    model.to(DEVICE_FOR_INFERENCE)
    model.eval() # Set model to evaluation mode
    print(f"Model successfully loaded from {BEST_MODEL_PATH_FOR_INFERENCE}")
else:
    print(f"Error: Best model weights not found at {BEST_MODEL_PATH_FOR_INFERENCE}.")
    print("Please ensure your training run successfully saved the model in /kaggle/working/.")
    # Exit the script if the model is not found, as inference cannot proceed.
    import sys
    sys.exit("Model file not found. Exiting.")


# Load test image IDs from the sample submission file
submission_df_template = pd.read_csv(SAMPLE_SUBMISSION_CSV_FOR_INFERENCE)
test_image_ids = submission_df_template['image_id'].tolist()

# Create test dataset and DataLoader
test_dataset = TestCassavaDataset(
    image_ids=test_image_ids,
    img_dir=TEST_IMAGES_DIR_FOR_INFERENCE, # Using the explicitly defined test images directory
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_INFERENCE,
    shuffle=False,
    num_workers=os.cpu_count() // 2, # Use half of CPU cores for data loading
    pin_memory=True
)

all_test_predictions = []

with torch.no_grad(): # Disable gradient calculation for inference
    for inputs, _ in tqdm(test_loader, desc="Predicting on test set"): # _ for image_ids; we just need inputs here
        inputs = inputs.to(DEVICE_FOR_INFERENCE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_test_predictions.extend(predicted.cpu().numpy())

# Create the final submission DataFrame
# Kaggle expects the submission.csv to have image_ids in the same order as sample_submission.csv.
# Since we loaded test_image_ids from sample_submission.csv and process them sequentially,
# the predictions will already be in the correct order corresponding to these image_ids.
submission_df = pd.DataFrame({
    'image_id': test_image_ids, # Use the original order of image_ids from the template
    'label': all_test_predictions
})

# Save the submission file to /kaggle/working/
submission_file_path = os.path.join('/kaggle/working', 'submission.csv')
submission_df.to_csv(submission_file_path, index=False)

print(f"\nSubmission file saved to: {submission_file_path}")
print("First 5 rows of generated submission.csv:")
print(submission_df.head())

print("\nFully Independent Test Module execution complete.")




