%%capture
!pip install albumentations
!pip install --upgrade albumentations
!pip install timm
!pip install --upgrade timm
!pip install kaggle
!pip install kagglehub
!pip install pandas


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from albumentations.pytorch import ToTensorV2
import timm
import albumentations as albu
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data._utils.collate import default_collate
import kagglehub


# configuring the path of Kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
# Remark that have to import Kaggle API key in the same path with notebook

# Path to competition dataset
path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
print("Path to dataset files:", path)


# Model Constants
BATCH_SIZE = 8
SEED = 42
DINO_SIZE = 518
EFFNET_SIZE = 384
HIERA_SIZE = 224

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the path to the directory you want to inspect
target_path = "/root/.cache/kagglehub/datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4"

def list_directory_contents(path):
    """
    Lists all files and subdirectories in the specified path.
    """
    print(f"--- Contents of: {path} ---")
    try:
        # Get a list of all entries in the directory
        contents = os.listdir(path)
        if not contents:
            print("The directory is empty.")
        else:
            for item in contents:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print(f"DIRECTORY: {item}")
                else:
                    print(f"FILE:      {item}")
    except FileNotFoundError:
        print(f"Error: The directory was not found at {path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function with the target path
list_directory_contents(target_path)


# Model-specific augmentations
def get_train_transform(size):
    return albu.Compose([
        albu.LongestMaxSize(size),
        albu.PadIfNeeded(size, size, border_mode=0),
        albu.HorizontalFlip(p=0.7),  # Increased from 0.5 to 0.7
        albu.VerticalFlip(p=0.7),    # Increased from 0.5 to 0.7
        albu.Rotate(limit=180, p=0.9),  # Widened from 90 to 180, p from 0.7 to 0.9
        albu.Affine(
            translate_percent=(-0.2, 0.2),  # Increased translation range
            scale=(0.6, 1.4),                 # Wider scaling range
            rotate=(-60, 60),                 # Increased rotation range
            shear=(-15, 15),                  # Added shear for additional distortion
            p=0.9
        ),
        albu.RandomBrightnessContrast(
            brightness_limit=(-0.4, 0.4),  # Increased from 0.3 to 0.4
            contrast_limit=(-0.4, 0.4),    # Increased from 0.3 to 0.4
            p=0.9                          # Increased from 0.8 to 0.9
        ),
        albu.HueSaturationValue(
            hue_shift_limit=30,             # Increased from 20 to 30
            sat_shift_limit=40,             # Increased from 30 to 40
            val_shift_limit=30,             # Increased from 20 to 30
            p=0.9                          # Increased from 0.7 to 0.9
        ),
        albu.GaussNoise(
            std_range=(0.1, 0.5),         # Stronger noise
            p=0.9
        ),
        albu.GaussianBlur(
            blur_limit=(3, 9),             # Increased from (3, 7) to (3, 9)
            p=0.8                          # Increased from 0.6 to 0.8
        ),
        albu.RandomGamma(
            gamma_limit=(60, 140),         # Expanded from (80, 120) to (60, 140)
            p=0.7                          # Increased from 0.5 to 0.7
        ),
        albu.CLAHE(
            clip_limit=(2.0, 6.0),         # Expanded from 4.0 to (2.0, 6.0) for variability
            tile_grid_size=(8, 8),
            p=0.7                          # Increased from 0.5 to 0.7
        ),
        albu.OneOf([
            albu.OpticalDistortion(distort_limit=0.1, p=1.0),            # Increased from 0.05 to 0.1
            albu.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),  # Increased from 0.05 to 0.1
            albu.ElasticTransform()                                      # Added elastic transform
        ], p=0.5),                                                       # Increased from 0.3 to 0.5
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albu.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.7),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.9),
        ToTensorV2(),
    ])

def get_val_transform(size):
    return albu.Compose([
        albu.LongestMaxSize(size),
        albu.PadIfNeeded(size, size, border_mode=0),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albu.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.7),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.9),
        ToTensorV2(),
    ])



class MultiSizeDataset(Dataset):
    def __init__(self, df, data_dir, is_train=True):
        self.df = df
        self.data_dir = data_dir
        self.is_train = is_train
        self.transforms = {
            'dinov2': get_train_transform(DINO_SIZE) if is_train else get_val_transform(DINO_SIZE),
            'hiera': get_train_transform(HIERA_SIZE) if is_train else get_val_transform(HIERA_SIZE),
            'effnet': get_train_transform(EFFNET_SIZE) if is_train else get_val_transform(EFFNET_SIZE),
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['file_name']
        label = self.df.iloc[idx]['label']
        img_path = os.path.join(self.data_dir, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            max_size = max(DINO_SIZE, EFFNET_SIZE, HIERA_SIZE)
            image = np.zeros((max_size, max_size, 3), dtype=np.uint8)

        transformed = {
            model: transform(image=image)['image']
            for model, transform in self.transforms.items()
        }
        return transformed, label

# Data preparation
# Data preparation
train_csv_path = "/root/.cache/kagglehub/datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4/train.csv"
train_df = pd.read_csv(train_csv_path)
train_data_dir = "/root/.cache/kagglehub/datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4"
# train_df = train_df.sample(frac=0.05)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df['label'])

train_dataset = MultiSizeDataset(train_df, train_data_dir, is_train=True)
val_dataset = MultiSizeDataset(val_df, train_data_dir, is_train=False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# Model Constants
FINE_TUNE_EPOCHS = 5                                             # Example for illustration, actually I saved model then loaded model and continue fined-tune around 80 Epochs

# Learning rate schedule
FINE_TUNE_LRS = [5e-5, 5e-5, 5e-5, 1e-5, 1e-5, 1e-5, 5e-6, 5e-6] # Fine-tuning schedule

def lrfn_factory(lr_list):
    """Factory function to create learning rate schedulers"""
    def lr_fn(epoch):
        return lr_list[epoch] if epoch < len(lr_list) else lr_list[-1]
    return lr_fn


# Modified train_phase with gradient clipping, stability checks, and early stopping
def train_phase(model, train_loader, val_loader, optimizers, schedulers, 
               loss_fn, device, total_epochs=FINE_TUNE_EPOCHS, checkpoint_path='best_model.pth',
               dinov2_path='best_dinov2.pth', hiera_path='best_hiera.pth', effnet_path='best_effnet.pth',
               patience=10):
    model.to(device)
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    max_grad_norm = 1.0  # Gradient clipping threshold
    epochs_no_improve = 0

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        print("-" * 50)
        
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # Zero gradients
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = sum(loss_fn(out, labels) for out in outputs) / 3.0
            
            # Backward pass with gradient clipping
            scaler.scale(loss).backward()
            for optimizer in optimizers.values():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            train_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(torch.mean(torch.stack(outputs), dim=0), 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        # Step schedulers after epoch
        for scheduler in schedulers.values():
            scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = sum(loss_fn(out, labels) for out in outputs) / 3.0
                
                # Check for NaN
                if torch.isnan(loss):
                    print("Warning: NaN loss detected, skipping batch")
                    continue
                
                val_loss += loss.item()
                _, predicted = torch.max(torch.mean(torch.stack(outputs), dim=0), 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('nan')
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        # Save best model
        if not np.isnan(val_acc) and val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
                'best_acc': best_acc,
                'epoch': epoch
            }, checkpoint_path)
            # Save individual models
            torch.save(model.dinov2.state_dict(), dinov2_path)
            torch.save(model.hiera.state_dict(), hiera_path)
            torch.save(model.effnet.state_dict(), effnet_path)
            print(f"Saved models with accuracy: {best_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}: no improvement for {patience} epochs.")
                break
    
    return history


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
    else:
        # Handle case where checkpoint is direct state dict
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model

class EnsembleModel(nn.Module):
    def __init__(self, dinov2_path=None, hiera_path=None, effnet_path=None):
        super().__init__()
        # Use valid model names for timm.create_model()
        self.dinov2 = self._create_model(
            'vit_small_patch14_dinov2.lvd142m',  # Correct model name
            dinov2_path,
            num_classes=2,
            img_size=DINO_SIZE
        )
        self.hiera = self._create_model(
            'hiera_small_224',  # Correct model name
            hiera_path,
            num_classes=2
        )
        self.effnet = self._create_model(
            'tf_efficientnetv2_s.in21k_ft_in1k',  # Correct model name
            effnet_path,
            num_classes=2
        )

    def _create_model(self, model_name, checkpoint_path, num_classes, **kwargs):
        # Create the model using timm
        model = timm.create_model(
            model_name,
            pretrained=checkpoint_path is None,  # Use pretrained weights if no checkpoint
            num_classes=num_classes,
            **kwargs
        )
        
        # Load checkpoint if provided
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            
            # Handle different checkpoint formats
            if 'model_state' in state_dict:
                state_dict = state_dict['model_state']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded {model_name} from {checkpoint_path}")
        
        return model

    def forward(self, x_dict):
        return [
            self.dinov2(x_dict['dinov2']),
            self.hiera(x_dict['hiera']),
            self.effnet(x_dict['effnet'])
        ]


# Load pre-trained model
print("Loading pre-trained model...")
model = EnsembleModel(
    dinov2_path=None,  # Path to DINOv2 checkpoint
    hiera_path=None,    # Path to Hiera checkpoint
    effnet_path=None   # Path to EfficientNet checkpoint
)
# If you have a saved checkpoint and want to resume training,
# uncomment the two lines below and comment out the 'model = EnsembleModel(...)' line above.
# checkpoint_path = 'fine_tuned_ensemble.pth'
# model = load_checkpoint(model, checkpoint_path, DEVICE)

# Set up fine-tuning components
optimizers = {
    'dinov2': AdamW(model.dinov2.parameters(), lr=FINE_TUNE_LRS[0], weight_decay=1e-4),
    'hiera': AdamW(model.hiera.parameters(), lr=FINE_TUNE_LRS[0], weight_decay=1e-4),
    'effnet': AdamW(model.effnet.parameters(), lr=FINE_TUNE_LRS[0], weight_decay=1e-4)
}

schedulers = {
    name: LambdaLR(optimizer, lr_lambda=lrfn_factory(FINE_TUNE_LRS))
    for name, optimizer in optimizers.items()
}

loss_fn = nn.CrossEntropyLoss()

# Start fine-tuning
print("Starting fine-tuning...")
history = train_phase(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    optimizers=optimizers,
    schedulers=schedulers,
    loss_fn=loss_fn,
    device=DEVICE,
    total_epochs=FINE_TUNE_EPOCHS,
    checkpoint_path='fine_tuned_ensemble.pth',
    dinov2_path='best_dinov2.pth',  # Path to save DINOv2
    hiera_path='best_hiera.pth',    # Path to save Hiera
    effnet_path='best_effnet.pth',   # Path to save EfficientNet
    patience=3,  # Stop if validation loss doesn't improve for 10 epochs
)

print("Fine-tuning complete.")


# Load pre-trained model
print("Loading pre-trained model...")
model = EnsembleModel(
    dinov2_path='best_dinov2.pth', # Path to DINOv2 checkpoint
    hiera_path='best_hiera.pth',   # Path to Hiera checkpoint
    effnet_path='best_effnet.pth'  # Path to EfficientNet checkpoin
)

# Load ensemble checkpoint if it exists
checkpoint_path = 'fine_tuned_ensemble.pth'
model = load_checkpoint(model, checkpoint_path, DEVICE)

# Set up fine-tuning components
optimizers = {
    'dinov2': AdamW(model.dinov2.parameters(), lr=FINE_TUNE_LRS[0], weight_decay=1e-4),
    'hiera': AdamW(model.hiera.parameters(), lr=FINE_TUNE_LRS[0], weight_decay=1e-4),
    'effnet': AdamW(model.effnet.parameters(), lr=FINE_TUNE_LRS[0], weight_decay=1e-4)
}

schedulers = {
    name: LambdaLR(optimizer, lr_lambda=lrfn_factory(FINE_TUNE_LRS))
    for name, optimizer in optimizers.items()
}

loss_fn = nn.CrossEntropyLoss()

# Start fine-tuning
print("Starting fine-tuning...")
history = train_phase(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    optimizers=optimizers,
    schedulers=schedulers,
    loss_fn=loss_fn,
    device=DEVICE,
    total_epochs=FINE_TUNE_EPOCHS,
    checkpoint_path='fine_tuned_ensemble.pth',
    dinov2_path='best_dinov2.pth',  # Path to save DINOv2
    hiera_path='best_hiera.pth',    # Path to save Hiera
    effnet_path='best_effnet.pth',   # Path to save EfficientNet
    patience=3,  # Stop if validation loss doesn't improve for 10 epochs
)

print("Fine-tuning complete.")


def visualize_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Training/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
visualize_history(history)


# Load the test CSV for predictions
test_csv_path = "/root/.cache/kagglehub/datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4/test.csv"
test_df = pd.read_csv(test_csv_path)

# Define the base directory where test images are stored
test_data_dir = "/root/.cache/kagglehub/datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4/test_data_v2"

# Update image paths in the test dataframe
# The 'id' column contains paths like 'test_data_v2/1a2d9fd3e21b4266aea1f66b30aed157.jpg'
# Extract the filename and join with test_data_dir
test_df['file_name'] = test_df['id'].apply(lambda x: os.path.join(test_data_dir, os.path.basename(x)))


def get_val_transform(size):
    return albu.Compose([
        albu.LongestMaxSize(size),
        albu.PadIfNeeded(size, size, border_mode=0),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# Update MultiSizeDataset to handle test data without labels
class MultiSizeDataset(Dataset):
    def __init__(self, df, data_dir, is_train=True):
        self.df = df
        self.data_dir = data_dir
        self.is_train = is_train
        self.transforms = {
            'dinov2': get_train_transform(DINO_SIZE) if is_train else get_val_transform(DINO_SIZE),
            'hiera': get_train_transform(HIERA_SIZE) if is_train else get_val_transform(HIERA_SIZE),
            'effnet': get_train_transform(EFFNET_SIZE) if is_train else get_val_transform(EFFNET_SIZE),
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['file_name']
        label = self.df.iloc[idx].get('label', -1)  # Use .get() for safety
        img_path = filename  # Use the pre-computed full path directly

        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            max_size = max(DINO_SIZE, EFFNET_SIZE, HIERA_SIZE)
            image = np.zeros((max_size, max_size, 3), dtype=np.uint8)

        transformed = {
            model: transform(image=image)['image']
            for model, transform in self.transforms.items()
        }
        return transformed, label

# Custom collate function to handle None labels
def custom_collate(batch):
    # Unzip the batch into inputs and labels
    inputs = [item[0] for item in batch]  # List of transformed dictionaries
    labels = [item[1] for item in batch]  # List of labels (may contain None for test)
    
    # Collate only the inputs (dictionaries of tensors)
    collated_inputs = {}
    for key in inputs[0].keys():  # Assume all inputs have the same keys
        collated_inputs[key] = default_collate([d[key] for d in inputs])
    
    return collated_inputs, labels  # Return collated inputs and list of labels

# Create test dataset using updated MultiSizeDataset
test_dataset = MultiSizeDataset(test_df, test_data_dir, is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate)


# Load the fine-tuned ensemble model
model = EnsembleModel(
    dinov2_path='best_dinov2.pth',
    hiera_path='best_hiera.pth',
    effnet_path='best_effnet.pth'
)
model = load_checkpoint(model, 'fine_tuned_ensemble.pth', DEVICE)
model.eval()

# Perform inference on test images
predictions = []
model.to(DEVICE)

# Modified prediction loop
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_dataloader, desc="Predicting")):
        inputs, labels = batch # Labels are -1 in this case
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        try:
            outputs = model(inputs)
            ensemble_output = torch.mean(torch.stack(outputs), dim=0)
            predicted_labels = torch.argmax(ensemble_output, dim=-1).cpu().numpy()
            
            # Get corresponding IDs for the batch
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + len(predicted_labels)
            batch_ids = test_df.loc[start_idx:end_idx-1, 'id'].values
            
            for id_, label in zip(batch_ids, predicted_labels):
                predictions.append((id_, label))
                
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Fallback for failed batch
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_ids = test_df.loc[start_idx:end_idx-1, 'id'].values
            for id_ in batch_ids:
                predictions.append((id_, 0))


# Create a DataFrame for submission
submission_df = pd.DataFrame(predictions, columns=["id", "label"])

# Ensure all test IDs are included (handle missing predictions)
missing_ids = set(test_df['id']) - set(submission_df['id'])
if missing_ids:
    print(f"Warning: {len(missing_ids)} images failed to process. Assigning default label 0.")
    missing_predictions = [(id_, 0) for id_ in missing_ids]
    missing_df = pd.DataFrame(missing_predictions, columns=["id", "label"])
    submission_df = pd.concat([submission_df, missing_df], ignore_index=True)

# Sort by ID to match original order
submission_df = submission_df.sort_values(by='id').reset_index(drop=True)

# Save to CSV for submission
submission_csv_path = "submission.csv"
submission_df.to_csv(submission_csv_path, index=False)
print(f"Submission file saved at {submission_csv_path}")

# Display label distribution and sample
print("\nLabel distribution:")
print(submission_df['label'].value_counts())
print("\nSubmission sample:")
print(submission_df.head())

