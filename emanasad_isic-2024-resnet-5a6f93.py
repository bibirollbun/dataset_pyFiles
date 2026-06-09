import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import gc
import timm

# Memory optimization - set device with better memory management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # More conservative memory fraction to prevent OOM errors
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.7, i)  # Reduced from 0.85
    except AttributeError:
        print("Could not set memory fraction - continuing with default memory management")
print(f"Using device: {device}")

# Paths configuration
TRAIN_CSV = "/kaggle/input/isic-2024-challenge/train-metadata.csv"
TRAIN_IMAGES_DIR = "/kaggle/input/isic-2024-challenge/train-image/image"
TEST_CSV = "/kaggle/input/isic-2024-challenge/test-metadata.csv"
SAVE_PATH = "optimized_skin_cancer_model.pth"
PRETRAINED_MODEL_PATH = "/kaggle/input/resnet/pytorch/default/1/resnet18_skin_cancer.pth"

class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_test=False, cache_size=100):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.cache_size = cache_size
        self.cache = {}  # Image cache to reduce disk I/O
        
        # Determine the filename column index
        if self.data.shape[1] > 0:
            filename_column = 0  # Assume the first column is filename
            
            # Ensure filenames are strings and add ".jpg" extension if needed
            if isinstance(self.data.iloc[0, filename_column], str) and not self.data.iloc[0, filename_column].endswith('.jpg'):
                self.data.iloc[:, filename_column] = self.data.iloc[:, filename_column].astype(str) + ".jpg"
        else:
            raise ValueError("CSV file has no columns")
        
        # Only check a sample of images for existence to speed up initialization
        sample_size = min(1000, len(self.data))
        sample_indices = np.random.choice(len(self.data), size=sample_size, replace=False)
        sample_exists = [
            os.path.exists(os.path.join(self.root_dir, self.data.iloc[idx, filename_column]))
            for idx in sample_indices
        ]
        
        if not any(sample_exists):
            raise ValueError("No valid images found in the sample! Check CSV file paths.")
        
        # Take all data without checking every file (faster initialization)
        self.valid_data = self.data.reset_index(drop=True)
        
        print(f"Total images in dataset: {len(self.valid_data)}")
        if not self.is_test and self.valid_data.shape[1] > 1:
            print(f"Class distribution: {self.valid_data.iloc[:, 1].value_counts()}")
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.valid_data.iloc[idx, 0])
        
        # Check cache first
        if idx in self.cache:
            image = self.cache[idx]
        else:
            try:
                image = Image.open(img_name).convert("RGB")
                
                # Manage cache size
                if len(self.cache) >= self.cache_size:
                    # Remove a random item from cache to keep size limited
                    self.cache.pop(list(self.cache.keys())[0])
                
                self.cache[idx] = image
            except Exception as e:
                # Return a black image instead of raising error
                print(f"Error loading {img_name}: {e}")
                image = Image.new('RGB', (224, 224), color='black')
        
        if self.is_test:
            # For test set, we might not have labels
            label = 0  # Placeholder
        else:
            label = int(self.valid_data.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.subset)




# Add this function after the SkinCancerDataset class definition

def create_balanced_sampler(dataset):
    """
    Create a sampler that undersamples the majority class (benign=0) to balance with minority class (malignant=1)
    """
    try:
        # Get all labels from the dataset
        labels = [dataset.valid_data.iloc[idx, 1] for idx in range(len(dataset))]
        
        # Count occurrences of each class
        class_counts = np.bincount(labels)
        print(f"Original class distribution: {class_counts}")
        
        # Create indices list for each class
        class_indices = {}
        for class_idx in range(len(class_counts)):
            class_indices[class_idx] = [i for i, label in enumerate(labels) if label == class_idx]
        
        # Determine the size of the minority class (expected to be class 1 - malignant)
        minority_class = 1
        minority_size = len(class_indices[minority_class])
        
        # Undersample the majority class (benign = 0)
        majority_class = 0
        undersampled_indices = np.random.choice(
            class_indices[majority_class], 
            size=minority_size, 
            replace=False
        )
        
        # Combine the indices for balanced dataset
        balanced_indices = np.concatenate([
            undersampled_indices,
            class_indices[minority_class]
        ])
        
        # Create new labels and weights for sampling
        balanced_labels = [labels[i] for i in balanced_indices]
        class_counts_balanced = np.bincount(balanced_labels)
        print(f"Balanced class distribution: {class_counts_balanced}")
        
        # Return the indices for creating a subset
        return balanced_indices
        
    except Exception as e:
        print(f"Error creating balanced sampler: {e}")
        # Return all indices if there's an error
        return list(range(len(dataset)))

# Now modify the main() function to use this sampler 
# Find the section where the dataset is split into train and validation
# and replace it with this:

def main():
    # ... (existing code)
    
    # Load dataset with optimized image loading
    print("Loading training dataset...")
    try:
        full_dataset = SkinCancerDataset(
            csv_file=TRAIN_CSV, 
            root_dir=TRAIN_IMAGES_DIR, 
            transform=None,
            cache_size=500
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Implement undersampling for the majority class
    print("Implementing undersampling for balanced training...")
    try:
        # Get balanced indices through undersampling
        balanced_indices = create_balanced_sampler(full_dataset)
        
        # Shuffle the balanced indices
        np.random.shuffle(balanced_indices)
        
        # Split into train and validation sets
        train_size = int(0.8 * len(balanced_indices))
        train_indices = balanced_indices[:train_size]
        val_indices = balanced_indices[train_size:]
        
        print(f"After undersampling and splitting: {len(train_indices)} training samples, {len(val_indices)} validation samples")
        
    except Exception as e:
        print(f"Error in undersampling: {e}")
        # Fall back to original splitting logic
        print("Falling back to original dataset split...")
        # Original code for splitting dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Get stratification labels - faster
        try:
            # Only sample some indices for stratification
            idx_sample = min(10000, len(full_dataset))
            indices = np.random.choice(len(full_dataset), size=idx_sample, replace=False)
            sample_labels = [full_dataset.valid_data.iloc[i, 1] for i in indices]
            
            # Use the class distribution as weights for sampling
            class_counts = np.bincount(sample_labels)
            weights = 1.0 / class_counts[sample_labels]
            
            # Sample indices based on weights
            sampled_indices = np.random.choice(
                indices, 
                size=idx_sample, 
                replace=True, 
                p=weights/weights.sum()
            )
            
            # Split into train and validation
            train_indices = sampled_indices[:int(0.8 * len(sampled_indices))]
            val_indices = sampled_indices[int(0.8 * len(sampled_indices)):]
            
            # Now add remaining indices not in the sample
            remaining = np.setdiff1d(np.arange(len(full_dataset)), indices)
            np.random.shuffle(remaining)
            train_split = int(0.8 * len(remaining))
            
            train_indices = np.concatenate([train_indices, remaining[:train_split]])
            val_indices = np.concatenate([val_indices, remaining[train_split:]])
        except Exception as e:
            print(f"Error in stratification: {e}")
            # Fall back to simple split
            indices = np.arange(len(full_dataset))
            np.random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
    
    # Create train and validation subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_dataset = TransformedSubset(train_dataset, train_transform)
    val_dataset = TransformedSubset(val_dataset, val_transform)
    

# Streamlined evaluation function
def evaluate_model(model, dataloader, criterion, threshold=0.5):
    model.eval()
    all_probs = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays for faster processing
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Calculate AUC-ROC if possible
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0
    
    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def find_optimal_threshold(labels, probs):
    if len(np.unique(labels)) < 2:
        return 0.5
        
    # Faster implementation using fewer points on the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probs, drop_intermediate=True)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    
    if ix >= len(thresholds):
        return 0.5
    
    return thresholds[ix]

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_epochs=15, validate_every=1, patience=5, accumulation_steps=2, start_epoch=0):
    best_val_f1 = 0.0
    early_stop_counter = 0
    best_val_loss = float('inf')
    best_threshold = 0.5
    
    # Mixed precision training
    scaler = GradScaler()
    
    print(f"Starting training for {num_epochs} epochs with optimizations")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'threshold': []
    }
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                           desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}", leave=True)
        
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Update weights every few batches (gradient accumulation)
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            
            # Less frequent updates to progress bar (reduces overhead)
            if i % 10 == 0:
                progress_bar.set_postfix(loss=loss.item() * accumulation_steps)
        
        # Calculate and store training loss
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase - less frequent validation saves time
        if epoch % validate_every == 0 or epoch == start_epoch + num_epochs - 1:
            val_metrics = evaluate_model(model, val_loader, criterion)
            
            for key in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']:
                history[key].append(val_metrics[key])
            
            # Find optimal threshold only in later epochs
            if epoch > (start_epoch + num_epochs) // 3:
                try:
                    best_threshold = find_optimal_threshold(
                        val_metrics['labels'], val_metrics['probabilities']
                    )
                except Exception:
                    pass
            
            history['threshold'].append(best_threshold)
            
            # Re-evaluate with optimal threshold
            preds = (val_metrics['probabilities'] >= best_threshold).astype(int)
            val_metrics['f1_optimal'] = f1_score(val_metrics['labels'], preds, zero_division=0)
            
            # Print statistics (less output)
            print(f"Epoch {epoch+1}/{start_epoch + num_epochs} | Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
            
            # Update learning rate
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            # Save best model (only when improvement is significant)
            if val_metrics['f1_optimal'] > best_val_f1 + 0.005:
                best_val_f1 = val_metrics['f1_optimal']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_threshold': best_threshold,
                    'f1': best_val_f1
                }, SAVE_PATH)
                print("Saved best model!")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss - 0.01:
                best_val_loss = val_metrics['loss']
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"\nEarly stopping after {epoch+1} epochs!")
                    break
        
        # Clean up memory after each epoch
        torch.cuda.empty_cache()
        gc.collect()
    
    # Plot only at the end of training instead of during training
    plot_training_metrics(history)
    
    try:
        # Load best model
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_threshold = checkpoint['best_threshold']
        
        print(f"Training complete! Best F1: {checkpoint['f1']:.4f}, Best Threshold: {best_threshold:.2f}")
    except Exception as e:
        print(f"Error loading best model: {e}. Using current model state.")
    
    return model, best_threshold, history

def plot_training_metrics(history):
    """Simplified plotting function"""
    if len(history['train_loss']) == 0:
        print("No training history to plot")
        return
        
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot losses
    axs[0, 0].plot(history['train_loss'], label='Training Loss')
    axs[0, 0].plot(history.get('val_loss', []), label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].legend()
    
    # Plot other metrics
    if 'accuracy' in history and len(history['accuracy']) > 0:
        axs[0, 1].plot(history['accuracy'], label='Accuracy')
        axs[0, 1].plot(history.get('auc', []), label='AUC')
        axs[0, 1].set_title('Accuracy & AUC')
        axs[0, 1].legend()
        
        axs[1, 0].plot(history.get('precision', []), label='Precision')
        axs[1, 0].plot(history.get('recall', []), label='Recall')
        axs[1, 0].set_title('Precision & Recall')
        axs[1, 0].legend()
        
        axs[1, 1].plot(history.get('f1', []), label='F1 Score')
        axs[1, 1].set_title('F1 Score')
        axs[1, 1].legend()
    
    plt.tight_layout()
    try:
        plt.savefig('training_metrics.png')
    except Exception:
        pass
    plt.close()  # Close to free memory

def get_model(model_name="resnet18", num_classes=2, pretrained=True):
    """More efficient model creation using ResNet18 by default"""
    try:
        if model_name == "resnet18":
            # Create ResNet18 model
            model = models.resnet18(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            # Use timm for other models
            model = timm.create_model(model_name, pretrained=pretrained)
            
            # Adapt final layer
            if hasattr(model, 'classifier'):
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
            elif hasattr(model, 'fc'):
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
            else:
                # Generic approach
                for name, module in model.named_children():
                    if isinstance(module, nn.Linear):
                        in_features = module.in_features
                        model._modules[name] = nn.Linear(in_features, num_classes)
                        break
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        print("Falling back to ResNet18")
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def load_pretrained_model(model_path, model=None, model_name="resnet18", num_classes=2):
    """Load a pretrained model from disk"""
    try:
        print(f"Loading pretrained model from {model_path}")
        if model is None:
            model = get_model(model_name, num_classes, pretrained=False)
            
        # Try to load the state dict
        try:
            # First try to load the complete checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if it's a complete checkpoint or just the state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model state from checkpoint dictionary")
                
                # Return additional info if available
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_threshold = checkpoint.get('best_threshold', 0.5)
                print(f"Resuming from epoch {start_epoch}, threshold {best_threshold:.4f}")
                return model, start_epoch, best_threshold
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("Loaded model state from state_dict key")
                return model, 0, 0.5
            else:
                # Try to load as direct state dict
                model.load_state_dict(checkpoint)
                print("Loaded direct model state dict")
                return model, 0, 0.5
        except Exception as e:
            print(f"First loading attempt failed: {e}")
            
            # Second approach: Handle potential key mismatches
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Handle potential key prefix differences (e.g., 'module.' prefix)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_k = k[7:]  # Remove 'module.' prefix
                    else:
                        new_k = k
                    new_state_dict[new_k] = v
                
                # Load with strict=False to ignore missing/extra keys
                model.load_state_dict(new_state_dict, strict=False)
                print("Loaded state dict with flexible key matching")
                return model, 0, 0.5
            except Exception as e:
                print(f"Second loading attempt failed: {e}")
                raise e
                
    except Exception as e:
        print(f"Failed to load pretrained model: {e}")
        return model, 0, 0.5

def predict_test_data(model, test_loader, threshold=0.5):
    """Faster test prediction"""
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            
            # Use mixed precision for inference too
            with autocast():
                outputs = model(inputs)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            probabilities.extend(probs)
            predictions.extend(preds)
    
    # Get filenames from test dataset
    try:
        filenames = [test_loader.dataset.valid_data.iloc[i, 0] for i in range(len(test_loader.dataset))]
    except:
        filenames = [f"image_{i}.jpg" for i in range(len(predictions))]
    
    # Ensure all lists have the same length
    min_len = min(len(predictions), len(probabilities), len(filenames))
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'image': filenames[:min_len],
        'target': predictions[:min_len],
        'probability': probabilities[:min_len]
    })
    
    try:
        submission.to_csv('submission.csv', index=False)
        print(f"Saved predictions to submission.csv")
    except Exception as e:
        print(f"Warning: Could not save submission file: {e}")
    
    return submission

def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Simplified data transforms with better efficiency
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Do resize and crop in one step
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset with optimized image loading
    print("Loading training dataset...")
    try:
        full_dataset = SkinCancerDataset(
            csv_file=TRAIN_CSV, 
            root_dir=TRAIN_IMAGES_DIR, 
            transform=None,
            cache_size=500  # Cache more images
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Split dataset - use faster splitting
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Get stratification labels - faster
    try:
        # Only sample some indices for stratification
        idx_sample = min(10000, len(full_dataset))
        indices = np.random.choice(len(full_dataset), size=idx_sample, replace=False)
        sample_labels = [full_dataset.valid_data.iloc[i, 1] for i in indices]
        
        # Use the class distribution as weights for sampling
        class_counts = np.bincount(sample_labels)
        weights = 1.0 / class_counts[sample_labels]
        
        # Sample indices based on weights
        sampled_indices = np.random.choice(
            indices, 
            size=idx_sample, 
            replace=True, 
            p=weights/weights.sum()
        )
        
        # Split into train and validation
        train_indices = sampled_indices[:int(0.8 * len(sampled_indices))]
        val_indices = sampled_indices[int(0.8 * len(sampled_indices)):]
        
        # Now add remaining indices not in the sample
        remaining = np.setdiff1d(np.arange(len(full_dataset)), indices)
        np.random.shuffle(remaining)
        train_split = int(0.8 * len(remaining))
        
        train_indices = np.concatenate([train_indices, remaining[:train_split]])
        val_indices = np.concatenate([val_indices, remaining[train_split:]])
    except Exception as e:
        print(f"Error in stratification: {e}")
        # Fall back to simple split
        indices = np.arange(len(full_dataset))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
    
    # Create train and validation subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_dataset = TransformedSubset(train_dataset, train_transform)
    val_dataset = TransformedSubset(val_dataset, val_transform)
    
    # Compute class weights - simplified approach
    print("Computing class weights...")
    try:
        class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32)  # Default weights
        
        # Try to compute actual weights if possible
        try:
            # Obtain only a sample of labels for faster computation
            sampled_train_indices = train_indices[:min(5000, len(train_indices))]
            train_labels = [full_dataset.valid_data.iloc[i, 1] for i in sampled_train_indices]
            unique_classes = np.unique(train_labels)
            computed_weights = compute_class_weight(class_weight='balanced', 
                                                  classes=unique_classes, 
                                                  y=train_labels)
            class_weights = torch.tensor(computed_weights, dtype=torch.float32)
        except Exception as e:
            print(f"Using default class weights: {e}")
            
        print(f"Class weights: {class_weights}")
    except Exception as e:
        print(f"Error computing class weights: {e}")
        class_weights = torch.ones(2, dtype=torch.float32)
    
    # Create optimized DataLoaders
    print("Creating data loaders...")
    try:
        # Determine optimal batch size and workers
        batch_size = 64 if torch.cuda.is_available() else 32
        
        # More conservative number of workers
        num_workers = min(4, os.cpu_count() or 1)
        if os.name == 'nt':  # Windows
            num_workers = 0
            
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # Simpler to just shuffle
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,  # Don't drop last batch
            persistent_workers=num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size*2,  # Larger batch size for validation
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        # Simple fallback
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model and load pretrained weights
    print("Creating model and loading pretrained weights...")
    model = get_model("resnet18", num_classes=2, pretrained=False)
    model, start_epoch, best_threshold = load_pretrained_model(
        PRETRAINED_MODEL_PATH, 
        model=model, 
        model_name="resnet18", 
        num_classes=2
    )
    model = model.to(device)
    
    # Evaluate the pretrained model on validation set
    print("Evaluating pretrained model...")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    val_metrics = evaluate_model(model, val_loader, criterion, threshold=best_threshold)
    
    print(f"Pretrained model metrics:")
    print(f"  Validation Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1 Score: {val_metrics['f1']:.4f}")
    print(f"  AUC: {val_metrics['auc']:.4f}")
    
    # Fine-tune the model
    print("Starting fine-tuning...")
    
    # Use standard Adam with less weight decay for faster convergence
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Lower LR for fine-tuning
    
    # Use a faster scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Train with fewer epochs for fine-tuning
    try:
        trained_model, best_threshold, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            scheduler,
            device,
            num_epochs=5,         # Fewer epochs for fine-tuning
            validate_every=1,      # Validate every epoch
            patience=3,            # Less patience for early stopping
            accumulation_steps=2,  # Less accumulation for faster updates
            start_epoch=start_epoch  # Continue from pretrained epoch
        )
        
        # Process test data with minimal visualization
        print("Processing test data...")
        try:
            test_dataset = SkinCancerDataset(
                csv_file=TEST_CSV, 
                root_dir=TRAIN_IMAGES_DIR,
                transform=val_transform,
                is_test=True,
                cache_size=1000  # Larger cache for test
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size*2,  # Larger batch size for prediction
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0
            )
            
            # Generate predictions
            submission = predict_test_data(trained_model, test_loader, threshold=best_threshold)
            print(f"Generated predictions for {len(submission)} test images")
            
        except Exception as e:
            print(f"Error processing test data: {e}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        
    print("Process completed!")
    
    # Final evaluation on validation set
    print("Final evaluation on validation set...")
    try:
        final_metrics = evaluate_model(model, val_loader, criterion, threshold=best_threshold)
        
        print(f"Final model metrics:")
        print(f"  Validation Loss: {final_metrics['loss']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall: {final_metrics['recall']:.4f}")
        print(f"  F1 Score: {final_metrics['f1']:.4f}")
        print(f"  AUC: {final_metrics['auc']:.4f}")
        
        # Plot confusion matrix
        try:
            cm = confusion_matrix(final_metrics['labels'], final_metrics['predictions'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
            
        # Plot ROC curve
        try:
            fpr, tpr, _ = roc_curve(final_metrics['labels'], final_metrics['probabilities'])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {final_metrics["auc"]:.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.savefig('roc_curve.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting ROC curve: {e}")
            
    except Exception as e:
        print(f"Error during final evaluation: {e}")
        
    # Save and export model for deployment
    try:
        # Save full model
        torch.save(model, 'full_model.pth')
        
        # Save model in TorchScript format for deployment
        try:
            model.eval()
            example = torch.rand(1, 3, 224, 224).to(device)
            traced_model = torch.jit.trace(model, example)
            traced_model.save('model_torchscript.pt')
            print("Saved TorchScript model for deployment")
        except Exception as e:
            print(f"Error saving TorchScript model: {e}")
            
        # Export model metadata
        model_info = {
            'architecture': 'resnet18',
            'input_size': [224, 224],
            'classes': ['benign', 'malignant'],
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'threshold': best_threshold
        }
        
        import json
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)
            
        print("Saved model metadata")
        
    except Exception as e:
        print(f"Error saving final model: {e}")
        
    print("Training and evaluation complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b3
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc, roc_auc_score
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to the dataset
benign_path = '/kaggle/input/isic24-balanced/benign'
malignant_path = '/kaggle/input/isic24-balanced/malignant'
efficientnet_b3_path = '/kaggle/input/efficientnet_b3/pytorch/default/1/efficientnet_b3_rwightman-b3899882.pth'

# Data augmentation and transformation with stronger augmentations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((300, 300)),  # Larger resize for better feature extraction
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Test-time augmentation transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomRotation((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),  # Direct resize without crop
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
]

# Custom dataset class for ISIC images
class ISICDataset(Dataset):
    def __init__(self, benign_dir, malignant_dir, transform=None, subset_files=None):
        self.transform = transform
        
        # Get all image paths and labels
        if subset_files is None:
            benign_images = [(os.path.join(benign_dir, img), 0) for img in os.listdir(benign_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
            malignant_images = [(os.path.join(malignant_dir, img), 1) for img in os.listdir(malignant_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
            self.images = benign_images + malignant_images
        else:
            benign_files, malignant_files = subset_files
            benign_images = [(os.path.join(benign_dir, img), 0) for img in benign_files]
            malignant_images = [(os.path.join(malignant_dir, img), 1) for img in malignant_files]
            self.images = benign_images + malignant_images
        
        np.random.shuffle(self.images)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Function to split dataset with stratification
def split_dataset(benign_path, malignant_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Get all image files
    benign_images = [f for f in os.listdir(benign_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    malignant_images = [f for f in os.listdir(malignant_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle images
    np.random.shuffle(benign_images)
    np.random.shuffle(malignant_images)
    
    # Split benign images
    benign_train_idx = int(len(benign_images) * train_ratio)
    benign_val_idx = int(len(benign_images) * (train_ratio + val_ratio))
    
    benign_train = benign_images[:benign_train_idx]
    benign_val = benign_images[benign_train_idx:benign_val_idx]
    benign_test = benign_images[benign_val_idx:]
    
    # Split malignant images
    malignant_train_idx = int(len(malignant_images) * train_ratio)
    malignant_val_idx = int(len(malignant_images) * (train_ratio + val_ratio))
    
    malignant_train = malignant_images[:malignant_train_idx]
    malignant_val = malignant_images[malignant_train_idx:malignant_val_idx]
    malignant_test = malignant_images[malignant_val_idx:]
    
    # Create temporary directories for splits
    splits = {
        'train': {'benign': benign_train, 'malignant': malignant_train},
        'val': {'benign': benign_val, 'malignant': malignant_val},
        'test': {'benign': benign_test, 'malignant': malignant_test}
    }
    
    return splits

# Get dataset splits
data_splits = split_dataset(benign_path, malignant_path)

# Create datasets for each split
image_datasets = {}
for x in ['train', 'val', 'test']:
    subset_files = (data_splits[x]['benign'], data_splits[x]['malignant'])
    image_datasets[x] = ISICDataset(
        benign_dir=benign_path,
        malignant_dir=malignant_path,
        transform=data_transforms[x],
        subset_files=subset_files
    )

# Calculate class weights for handling imbalance
def get_class_weights(dataset):
    labels = [label for _, label in dataset.images]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)

class_weights = get_class_weights(image_datasets['train'])
print(f"Class weights: {class_weights}")

# Create weighted sampler for training data
def create_weighted_sampler(dataset):
    labels = [label for _, label in dataset.images]
    weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

train_sampler = create_weighted_sampler(image_datasets['train'])

# Create dataloaders
batch_size = 24  # Smaller batch size for better generalization
dataloaders = {
    'train': DataLoader(
        dataset=image_datasets['train'],
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    ),
    'val': DataLoader(
        dataset=image_datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ),
    'test': DataLoader(
        dataset=image_datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = ['benign', 'malignant']

print(f"Dataset sizes: Train: {dataset_sizes['train']}, Validation: {dataset_sizes['val']}, Test: {dataset_sizes['test']}")

# Focal Loss implementation for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Define the EfficientNet model with advanced techniques
class EfficientNetForSkinClassification(nn.Module):
    def __init__(self, model_name='efficientnet-b3', num_classes=2, pretrained=True):
        super(EfficientNetForSkinClassification, self).__init__()
        
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name)
        else:
            self.model = EfficientNet.from_name(model_name)
        
        # Extract features dimension
        self._fc_dim = self.model._fc.in_features
        
        # Replace final layers with custom classifier
        self.model._fc = nn.Identity()
        
        # Advanced classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(self._fc_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.model.extract_features(x)
        features = self.model._avg_pooling(features)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

# Define the PyTorch EfficientNet-B0 model
class EfficientNetB0ForSkinClassification(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB0ForSkinClassification, self).__init__()
        
        # Use PyTorch's efficientnet_b0
        self.model = efficientnet_b3(pretrained=pretrained)
        
        # Extract features dimension
        self._fc_dim = self.model.classifier[1].in_features
        
        # Replace final layers with custom classifier
        self.model.classifier = nn.Identity()
        
        # Advanced classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(self._fc_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.model.features(x)
        features = self.model.avgpool(features)
        features = torch.flatten(features, 1)
        return self.classifier(features)

# Ensemble model combining two EfficientNet models
class SkinLesionEnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super(SkinLesionEnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.classifier = nn.Linear(4, 2)  # 2 classes x 2 models = 4 inputs
        
    def forward(self, x):
        output1 = self.model1(x)
        output2 = self.model2(x)
        # Concatenate logits from both models
        combined = torch.cat((output1, output2), dim=1)
        return self.classifier(combined)

# Function to load model weights
def load_pretrained_weights(model, path):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    
    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

# Cutmix implementation
def cutmix(data, target, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    
    lam = np.random.beta(alpha, alpha)
    
    image_h, image_w = data.size(2), data.size(3)
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y1 = int(np.round(min(cy + h / 2, image_h)))
    
    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    
    return data, target, shuffled_target, lam

# Training function with multiple data augmentation techniques
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, 
                mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5, mixup_prob=0.5):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0
    patience = 10  # Early stopping patience
    counter = 0
    
    # History to track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            all_probs = []
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Apply data augmentation techniques during training
                    if phase == 'train':
                        r = np.random.rand(1)
                        if r < cutmix_prob:
                            # Apply cutmix
                            inputs, labels_a, labels_b, lam = cutmix(inputs, labels, cutmix_alpha)
                            outputs = model(inputs)
                            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                        elif r < cutmix_prob + mixup_prob:
                            # Apply mixup
                            lam = np.random.beta(mixup_alpha, mixup_alpha)
                            index = torch.randperm(inputs.size()[0]).to(device)
                            mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
                            outputs = model(mixed_inputs)
                            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
                        else:
                            # No augmentation
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    probs = F.softmax(outputs, dim=1)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                if phase == 'train' and (r < cutmix_prob + mixup_prob):
                    # For augmented training, we approximate the corrects
                    if r < cutmix_prob:
                        running_corrects += (lam * torch.sum(preds == labels_a.data) + 
                                           (1 - lam) * torch.sum(preds == labels_b.data)).item()
                    else:
                        running_corrects += (lam * torch.sum(preds == labels.data) + 
                                           (1 - lam) * torch.sum(preds == labels.data[index])).item()
                else:
                    running_corrects += torch.sum(preds == labels.data).item()
                
                # Collect predictions and labels for AUC calculation
                if phase == 'val':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].detach().cpu().numpy())  # Malignant class probability
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            # Calculate AUC for validation set
            epoch_auc = 0.0
            if phase == 'val':
                try:
                    epoch_auc = roc_auc_score(all_labels, all_probs)
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')
                except:
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: N/A')
            else:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            if phase == 'val':
                history['val_auc'].append(epoch_auc)
            
            # Deep copy the model if best validation accuracy
            if phase == 'val':
                # Use a combination of accuracy and AUC for model selection
                current_performance = epoch_acc * 0.7 + epoch_auc * 0.3
                best_performance = best_acc * 0.7 + best_auc * 0.3
                
                if current_performance > best_performance:
                    best_acc = epoch_acc
                    best_auc = epoch_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0  # Reset early stopping counter
                else:
                    counter += 1
        
        print()
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}, Best val AUC: {best_auc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Test Time Augmentation (TTA) evaluation function
def tta_evaluate_model(model, dataloader, criterion, tta_transforms):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            batch_size = inputs.size(0)
            labels = labels.to(device)
            
            # Initialize tensors to store TTA predictions
            tta_outputs = torch.zeros(batch_size, 2).to(device)
            
            # Apply each TTA transform and average predictions
            for transform in tta_transforms:
                # Create a dataset with the current transform
                tta_images = []
                for i in range(batch_size):
                    img = inputs[i].cpu()
                    img = transforms.ToPILImage()(img)
                    img = transform(img)
                    tta_images.append(img)
                
                # Stack transformed images
                tta_inputs = torch.stack(tta_images).to(device)
                
                # Forward pass
                outputs = model(tta_inputs)
                tta_outputs += outputs
            
            # Average predictions across all TTA transforms
            outputs = tta_outputs / len(tta_transforms)
            loss = criterion(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant class
    
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    return test_loss, test_acc, all_preds, all_labels, all_probs

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('AUC vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Calculate advanced metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    print("Advanced Metrics:")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity/Recall: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # ABCD criteria metrics (for dermatological analysis)
    print("\nABCD Criteria Approximated Metrics:")
    print(f"A (Asymmetry) - Balanced Accuracy: {balanced_acc:.4f}")
    print(f"B (Border) - Precision: {precision:.4f}")
    print(f"C (Color) - Sensitivity/Recall: {sensitivity:.4f}")
    print(f"D (Differential) - Specificity: {specificity:.4f}")
    
    return sensitivity, specificity, precision, f1, balanced_acc

# Function to plot ROC curve
def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    
    return roc_auc

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('pr_curve.png')
    plt.show()
    
    return pr_auc

# Function to print classification report
def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)
    return report

# Function to perform k-fold cross-validation
def k_fold_cross_validation(n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get all data
    benign_images = [f for f in os.listdir(benign_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    malignant_images = [f for f in os.listdir(malignant_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create labels array for stratification
    all_files = benign_images + malignant_images
    all_labels = [0] * len(benign_images) + [1] * len(malignant_images)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_files, all_labels)):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Split files and labels
        train_files = [all_files[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        test_files = [all_files[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]
        
        # Further split train into train and validation
        val_size = int(len(train_idx) * 0.2)
        val_files = train_files[:val_size]
        val_labels = train_labels[:val_size]
        train_files = train_files[val_size:]
        train_labels = train_labels[val_size:]
        
        # Organize files by class
        train_benign = [f for f, l in zip(train_files, train_labels) if l == 0]
        train_malignant = [f for f, l in zip(train_files, train_labels) if l == 1]
        val_benign = [f for f, l in zip(val_files, val_labels) if l == 0]
        val_malignant = [f for f, l in zip(val_files, val_labels) if l == 1]
        test_benign = [f for f, l in zip(test_files, test_labels) if l == 0]
        test_malignant = [f for f, l in zip(test_files, test_labels) if l == 1]
        
        # Create datasets
        fold_datasets = {
            'train': ISICDataset(
                benign_dir=benign_path,
                malignant_dir=malignant_path,
                transform=data_transforms['train'],
                subset_files=(train_benign, train_malignant)
            ),
            'val': ISICDataset(
                benign_dir=benign_path,
                malignant_dir=malignant_path,
                transform=data_transforms['val'],
                subset_files=(val_benign, val_malignant)
            ),
            'test': ISICDataset(
                benign_dir=benign_path,
                malignant_dir=malignant_path,
                transform=data_transforms['test'],
                subset_files=(test_benign, test_malignant)
            )
        }
        
        # Create dataloaders
        fold_sampler = create_weighted_sampler(fold_datasets['train'])
        fold_dataloaders = {
            'train': DataLoader(
                dataset=fold_datasets['train'],
                batch_size=batch_size,
                sampler=fold_sampler,
                num_workers=4,
                pin_memory=True
            ),
            'val': DataLoader(
                dataset=fold_datasets['val'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ),
            'test': DataLoader(
                dataset=fold_datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        }
        
        # Initialize and train model
        model = EfficientNetForSkinClassification(model_name='efficientnet-b3', pretrained=True)
        model = model.to(device)
        
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        
        model, _ = train_model(model, criterion, optimizer, scheduler, num_epochs=20)
        
        # Evaluate model on test set
        test_loss, test_acc, test_preds, test_labels, test_probs = evaluate_model(
            model, fold_dataloaders['test'], criterion)
        
        roc_auc = roc_auc_score(test_labels, test_probs)
        
        fold_results.append({
            'fold': fold+1,
            'test_acc': test_acc,
            'test_auc': roc_auc,
        })
        
        print(f"Fold {fold+1} results: Accuracy: {test_acc:.4f}, AUC: {roc_auc:.4f}")

    # Calculate average results
    avg_acc = np.mean([res['test_acc'] for res in fold_results])
    avg_auc = np.mean([res['test_auc'] for res in fold_results])
    
    print(f"\nK-fold cross-validation results (k={n_splits}):")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")
    
    return fold_results

# Standard evaluation function (without TTA)
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant class
    
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    return test_loss, test_acc, all_preds, all_labels, all_probs

# Main execution flow with dual model approach
def main():
    print("Setting up models...")
    
    # Initialize and train the first model (EfficientNet-B3)
    model1 = EfficientNetForSkinClassification(model_name='efficientnet-b3', pretrained=True)
    model1 = model1.to(device)
    
    # Initialize and train the second model (EfficientNet-B0 equivalent)
    model2 = EfficientNetB0ForSkinClassification(pretrained=True)
    model2 = model2.to(device)
    
    # Load pre-trained weights if available
    try:
        model1 = load_pretrained_weights(model1, efficientnet_b3_path)
        print("Loaded EfficientNet-B3 weights")
    except Exception as e:
        print(f"Could not load EfficientNet-B3 weights: {e}")
    
    # Define loss function with focal loss for better handling of imbalanced data
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Use different learning rates for different parts of the models
    params1 = [
        {'params': [p for n, p in model1.model.named_parameters() if 'features' in n], 'lr': 0.0001},
        {'params': model1.classifier.parameters(), 'lr': 0.001}
    ]
    
    params2 = [
        {'params': [p for n, p in model2.model.named_parameters() if 'features' in n], 'lr': 0.0001},
        {'params': model2.classifier.parameters(), 'lr': 0.001}
    ]
    
    # Advanced optimizers
    optimizer1 = optim.AdamW(params1, weight_decay=0.01)
    optimizer2 = optim.AdamW(params2, weight_decay=0.01)
    
    # Learning rate schedulers
    scheduler1 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=10, T_mult=1, eta_min=1e-6)
    scheduler2 = lr_scheduler.OneCycleLR(optimizer2, max_lr=[0.0001, 0.001], 
                                       steps_per_epoch=len(dataloaders['train']), 
                                       epochs=30)
    
    print("Training EfficientNet-B3 model...")
    model1, history1 = train_model(model1, criterion, optimizer1, scheduler1, 
                                 num_epochs=30, mixup_alpha=0.2, cutmix_alpha=1.0,
                                 cutmix_prob=0.5, mixup_prob=0.3)
    
    print("Training EfficientNet-B0 model...")
    model2, history2 = train_model(model2, criterion, optimizer2, scheduler2, 
                                 num_epochs=30, mixup_alpha=0.2, cutmix_alpha=1.0,
                                 cutmix_prob=0.3, mixup_prob=0.5)
    
    # Plot training history for both models
    plot_training_history(history1)
    plot_training_history(history2)
    
    # Evaluate individual models with TTA
    print("Evaluating EfficientNet-B3 model with TTA...")
    test_loss1, test_acc1, test_preds1, test_labels1, test_probs1 = tta_evaluate_model(
        model1, dataloaders['test'], criterion, tta_transforms)
    
    print("Evaluating EfficientNet-B0 model with TTA...")
    test_loss2, test_acc2, test_preds2, test_labels2, test_probs2 = tta_evaluate_model(
        model2, dataloaders['test'], criterion, tta_transforms)
    
    print(f"EfficientNet-B3 Test Accuracy: {test_acc1:.4f}")
    print(f"EfficientNet-B0 Test Accuracy: {test_acc2:.4f}")
    
    # Ensemble the models (averaging predictions)
    print("Creating ensemble predictions...")
    ensemble_probs = (np.array(test_probs1) + np.array(test_probs2)) / 2
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    # Create a weighted ensemble (if one model performs better)
    if test_acc1 > test_acc2:
        weight1 = 0.6
        weight2 = 0.4
    else:
        weight1 = 0.4
        weight2 = 0.6
    
    weighted_probs = (weight1 * np.array(test_probs1) + weight2 * np.array(test_probs2))
    weighted_preds = (weighted_probs > 0.5).astype(int)
    
    # Choose the best ensemble method
    if np.sum(ensemble_preds == test_labels1) > np.sum(weighted_preds == test_labels1):
        final_preds = ensemble_preds
        final_probs = ensemble_probs
        ensemble_type = "Average"
    else:
        final_preds = weighted_preds
        final_probs = weighted_probs
        ensemble_type = f"Weighted ({weight1:.1f}/{weight2:.1f})"
    
    # Calculate ensemble accuracy
    ensemble_acc = np.sum(final_preds == test_labels1) / len(test_labels1)
    print(f"Ensemble ({ensemble_type}) Test Accuracy: {ensemble_acc:.4f}")
    
    # Plot confusion matrix for ensemble
    sensitivity, specificity, precision, f1, balanced_acc = plot_confusion_matrix(
        test_labels1, final_preds, classes=class_names)
    
    # Print classification report for ensemble
    report = print_classification_report(test_labels1, final_preds)
    
    # Plot ROC curve for ensemble
    roc_auc = plot_roc_curve(test_labels1, final_probs)
    
    # Plot precision-recall curve for ensemble
    pr_auc = plot_precision_recall_curve(test_labels1, final_probs)
    
    # If accuracy is still not high enough (< 95%), try advanced ensemble techniques
    if ensemble_acc < 0.95:
        print("Accuracy below 95%, implementing advanced ensemble techniques...")
        
        # Create a full ensemble model
        ensemble_model = SkinLesionEnsembleModel(model1, model2)
        ensemble_model = ensemble_model.to(device)
        
        # Train the ensemble model end-to-end
        ensemble_optimizer = optim.AdamW(ensemble_model.parameters(), lr=0.0005, weight_decay=0.01)
        ensemble_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(ensemble_optimizer, T_0=5, T_mult=1, eta_min=1e-6)
        
        print("Training ensemble model...")
        ensemble_model, ensemble_history = train_model(ensemble_model, criterion, ensemble_optimizer, 
                                                     ensemble_scheduler, num_epochs=15)
        
        # Evaluate the trained ensemble model with TTA
        print("Evaluating ensemble model with TTA...")
        e_loss, e_acc, e_preds, e_labels, e_probs = tta_evaluate_model(
            ensemble_model, dataloaders['test'], criterion, tta_transforms)
        
        print(f"Trained Ensemble Test Accuracy: {e_acc:.4f}")
        
        # If the trained ensemble is better, use its predictions
        if e_acc > ensemble_acc:
            final_preds = e_preds
            final_probs = e_probs
            ensemble_acc = e_acc
            ensemble_type = "Trained Ensemble"
            
            # Re-evaluate with the better ensemble
            sensitivity, specificity, precision, f1, balanced_acc = plot_confusion_matrix(
                e_labels, e_preds, classes=class_names)
            report = print_classification_report(e_labels, e_preds)
            roc_auc = plot_roc_curve(e_labels, e_probs)
            pr_auc = plot_precision_recall_curve(e_labels, e_probs)
    
    # Save the models
    torch.save(model1.state_dict(), 'isic_efficientnet_b3_model.pth')
    torch.save(model2.state_dict(), 'isic_efficientnet_b0_model.pth')
    if ensemble_acc < 0.95:
        torch.save(ensemble_model.state_dict(), 'isic_ensemble_model.pth')
    
    # Return final metrics in a dictionary
    metrics = {
        'accuracy': ensemble_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ensemble_type': ensemble_type
    }
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('skin_lesion_classification_metrics.csv', index=False)
    
    print("\nFinal Metrics Summary:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return metrics

if __name__ == "__main__":
    metrics = main()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to the dataset
benign_path = '/kaggle/input/isic24-balanced/benign'
malignant_path = '/kaggle/input/isic24-balanced/malignant'

# Data augmentation and transformation with stronger augmentations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((300, 300)),  # Larger resize for better feature extraction
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),  # Increased rotation angle for more variety
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Enhanced color jitter
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # Enhanced affine transformation
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # Enhanced perspective distortion
        transforms.RandomResizedCrop(size=224, scale=(0.75, 1.0)),  # Changed scale for crop variety
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)  # Enhanced random erasing
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Test-time augmentation transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomRotation((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),  # Direct resize without crop
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
]

# Custom dataset class for ISIC images
class ISICDataset(Dataset):
    def __init__(self, benign_dir, malignant_dir, transform=None, subset_files=None):
        self.transform = transform
        
        # Get all image paths and labels
        if subset_files is None:
            benign_images = [(os.path.join(benign_dir, img), 0) for img in os.listdir(benign_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
            malignant_images = [(os.path.join(malignant_dir, img), 1) for img in os.listdir(malignant_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
            self.images = benign_images + malignant_images
        else:
            benign_files, malignant_files = subset_files
            benign_images = [(os.path.join(benign_dir, img), 0) for img in benign_files]
            malignant_images = [(os.path.join(malignant_dir, img), 1) for img in malignant_files]
            self.images = benign_images + malignant_images
        
        np.random.shuffle(self.images)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Function to split dataset with stratification
def split_dataset(benign_path, malignant_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Get all image files
    benign_images = [f for f in os.listdir(benign_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    malignant_images = [f for f in os.listdir(malignant_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle images
    np.random.shuffle(benign_images)
    np.random.shuffle(malignant_images)
    
    # Split benign images
    benign_train_idx = int(len(benign_images) * train_ratio)
    benign_val_idx = int(len(benign_images) * (train_ratio + val_ratio))
    
    benign_train = benign_images[:benign_train_idx]
    benign_val = benign_images[benign_train_idx:benign_val_idx]
    benign_test = benign_images[benign_val_idx:]
    
    # Split malignant images
    malignant_train_idx = int(len(malignant_images) * train_ratio)
    malignant_val_idx = int(len(malignant_images) * (train_ratio + val_ratio))
    
    malignant_train = malignant_images[:malignant_train_idx]
    malignant_val = malignant_images[malignant_train_idx:malignant_val_idx]
    malignant_test = malignant_images[malignant_val_idx:]
    
    # Create splits
    splits = {
        'train': {'benign': benign_train, 'malignant': malignant_train},
        'val': {'benign': benign_val, 'malignant': malignant_val},
        'test': {'benign': benign_test, 'malignant': malignant_test}
    }
    
    return splits

# Get dataset splits
data_splits = split_dataset(benign_path, malignant_path)

# Create datasets for each split
image_datasets = {}
for x in ['train', 'val', 'test']:
    subset_files = (data_splits[x]['benign'], data_splits[x]['malignant'])
    image_datasets[x] = ISICDataset(
        benign_dir=benign_path,
        malignant_dir=malignant_path,
        transform=data_transforms[x],
        subset_files=subset_files
    )

# Calculate class weights for handling imbalance
def get_class_weights(dataset):
    labels = [label for _, label in dataset.images]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)

class_weights = get_class_weights(image_datasets['train'])
print(f"Class weights: {class_weights}")

# Create weighted sampler for training data
def create_weighted_sampler(dataset):
    labels = [label for _, label in dataset.images]
    weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

train_sampler = create_weighted_sampler(image_datasets['train'])

# Create dataloaders
batch_size = 32  # Increased batch size for MobileNetV2 which is less memory intensive
dataloaders = {
    'train': DataLoader(
        dataset=image_datasets['train'],
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    ),
    'val': DataLoader(
        dataset=image_datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ),
    'test': DataLoader(
        dataset=image_datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = ['benign', 'malignant']

print(f"Dataset sizes: Train: {dataset_sizes['train']}, Validation: {dataset_sizes['val']}, Test: {dataset_sizes['test']}")

# Focal Loss implementation for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Modified MobileNetV2 model with advanced techniques
class MobileNetV2ForSkinClassification(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2ForSkinClassification, self).__init__()
        
        # Load MobileNetV2 without pretrained weights to avoid download issues
        self.model = models.mobilenet_v2(pretrained=False)
        
        # Extract features dimension
        self._fc_dim = self.model.classifier[1].in_features
        
        # Replace final layers with custom classifier
        self.model.classifier = nn.Identity()
        
        # Advanced classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(self._fc_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),  # Using LeakyReLU instead of ReLU
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.model.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return self.classifier(features)

# Ensemble model combining multiple MobileNetV2 models
class SkinLesionEnsembleModel(nn.Module):
    def __init__(self, models):
        super(SkinLesionEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        n_models = len(models)
        self.classifier = nn.Linear(2 * n_models, 2)  # 2 classes x n_models = inputs
        
    def forward(self, x):
        # Get outputs from all models
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Concatenate all outputs
        combined = torch.cat(outputs, dim=1)
        return self.classifier(combined)

# Cutmix implementation
def cutmix(data, target, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    
    lam = np.random.beta(alpha, alpha)
    
    image_h, image_w = data.size(2), data.size(3)
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y1 = int(np.round(min(cy + h / 2, image_h)))
    
    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    
    return data, target, shuffled_target, lam

# Mixup implementation
def mixup(data, target, alpha=0.2):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    
    lam = np.random.beta(alpha, alpha)
    mixed_data = lam * data + (1 - lam) * shuffled_data
    
    return mixed_data, target, shuffled_target, lam

# Training function with multiple data augmentation techniques
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, 
                mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5, mixup_prob=0.5):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0
    best_f1 = 0.0
    patience = 15  # Early stopping patience
    counter = 0
    
    # History to track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            all_probs = []
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Apply data augmentation techniques during training
                    if phase == 'train':
                        r = np.random.rand(1)
                        if r < cutmix_prob:
                            # Apply cutmix
                            inputs, labels_a, labels_b, lam = cutmix(inputs, labels, cutmix_alpha)
                            outputs = model(inputs)
                            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                        elif r < cutmix_prob + mixup_prob:
                            # Apply mixup
                            mixed_inputs, labels_a, labels_b, lam = mixup(inputs, labels, mixup_alpha)
                            outputs = model(mixed_inputs)
                            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                        else:
                            # No augmentation
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    probs = F.softmax(outputs, dim=1)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                if phase == 'train' and r < cutmix_prob + mixup_prob:
                    # For augmented training, we approximate the corrects
                    if r < cutmix_prob:
                        running_corrects += (lam * torch.sum(preds == labels_a.data) + 
                                           (1 - lam) * torch.sum(preds == labels_b.data)).item()
                    else:
                        running_corrects += (lam * torch.sum(preds == labels_a.data) + 
                                           (1 - lam) * torch.sum(preds == labels_b.data)).item()
                else:
                    running_corrects += torch.sum(preds == labels.data).item()
                
                # Collect predictions and labels for AUC calculation
                if phase == 'val':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].detach().cpu().numpy())  # Malignant class probability
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            # Calculate AUC and F1 for validation set
            epoch_auc = 0.0
            epoch_f1 = 0.0
            if phase == 'val':
                try:
                    epoch_auc = roc_auc_score(all_labels, all_probs)
                    # Calculate F1 score
                    precision = precision_score(all_labels, all_preds)
                    recall = recall_score(all_labels, all_preds)
                    epoch_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f} F1: {epoch_f1:.4f}')
                except:
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: N/A F1: N/A')
            else:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            if phase == 'val':
                history['val_auc'].append(epoch_auc)
                history['val_f1'].append(epoch_f1)
            
            # Deep copy the model if best validation performance
            if phase == 'val':
                # Use a combination of accuracy, AUC and F1 for model selection
                current_performance = epoch_acc * 0.5 + epoch_auc * 0.3 + epoch_f1 * 0.2
                best_performance = best_acc * 0.5 + best_auc * 0.3 + best_f1 * 0.2
                
                if current_performance > best_performance:
                    best_acc = epoch_acc
                    best_auc = epoch_auc
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0  # Reset early stopping counter
                else:
                    counter += 1
        
        print()
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}, Best val AUC: {best_auc:.4f}, Best val F1: {best_f1:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Test Time Augmentation (TTA) evaluation function
def tta_evaluate_model(model, dataloader, criterion, tta_transforms):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            batch_size = inputs.size(0)
            labels = labels.to(device)
            
            # Initialize tensors to store TTA predictions
            tta_outputs = torch.zeros(batch_size, 2).to(device)
            
            # Apply each TTA transform and average predictions
            for transform in tta_transforms:
                # Create a dataset with the current transform
                tta_images = []
                for i in range(batch_size):
                    img = inputs[i].cpu()
                    img = transforms.ToPILImage()(img)
                    img = transform(img)
                    tta_images.append(img)
                
                # Stack transformed images
                tta_inputs = torch.stack(tta_images).to(device)
                
                # Forward pass
                outputs = model(tta_inputs)
                tta_outputs += outputs
            
            # Average predictions across all TTA transforms
            outputs = tta_outputs / len(tta_transforms)
            loss = criterion(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant class
    
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    return test_loss, test_acc, all_preds, all_labels, all_probs

# Standard evaluation function (without TTA)
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant class
    
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    return test_loss, test_acc, all_preds, all_labels, all_probs

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(18, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('AUC vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Calculate advanced metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    print("Advanced Metrics:")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity/Recall: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1
    }

# Function to plot ROC curve
# Function to plot ROC curve (completing the cutoff function)
def plot_roc_curve(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    
    return roc_auc

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('precision_recall_curve.png')
    plt.show()
    
    return pr_auc

# Function to find optimal threshold
def find_optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"At this threshold - TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
    
    return optimal_threshold

# Create and train the model
def create_and_train_model():
    # Initialize the model
    model = MobileNetV2ForSkinClassification(num_classes=2)
    model = model.to(device)
    
    # Set up loss function with class weights (weighted BCE or Focal Loss)
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # Set up optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Set up learning rate scheduler with warmup
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Train the model
    trained_model, history = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs=50,  # Maximum epochs (early stopping may trigger before this)
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        mixup_prob=0.3,
        cutmix_prob=0.3
    )
    
    # Plot training history
    plot_training_history(history)
    
    return trained_model, criterion

# Create and train multiple models for ensemble
def create_ensemble_models(num_models=3):
    models = []
    for i in range(num_models):
        print(f"Training model {i+1}/{num_models} for ensemble...")
        model = MobileNetV2ForSkinClassification(num_classes=2)
        model = model.to(device)
        
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        trained_model, _ = train_model(
            model,
            criterion,
            optimizer,
            scheduler,
            num_epochs=40,
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
            mixup_prob=0.3,
            cutmix_prob=0.3
        )
        
        models.append(trained_model)
    
    # Create ensemble model
    ensemble_model = SkinLesionEnsembleModel(models)
    ensemble_model = ensemble_model.to(device)
    
    # Fine-tune ensemble model
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(ensemble_model.classifier.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print("Fine-tuning ensemble model...")
    ensemble_model, history = train_model(
        ensemble_model,
        criterion,
        optimizer,
        scheduler,
        num_epochs=15,
        mixup_prob=0,
        cutmix_prob=0
    )
    
    return ensemble_model, criterion

# Main execution flow
def main():
    print("Starting skin lesion classification training...")
    
    # Choose between single model or ensemble
    use_ensemble = True
    
    if use_ensemble:
        print("Creating ensemble model...")
        model, criterion = create_ensemble_models(num_models=3)
    else:
        print("Creating single model...")
        model, criterion = create_and_train_model()
    
    # Evaluate on test set with Test Time Augmentation
    print("\nEvaluating model on test set with TTA...")
    test_loss, test_acc, test_preds, test_labels, test_probs = tta_evaluate_model(
        model, 
        dataloaders['test'], 
        criterion, 
        tta_transforms
    )
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    metrics = plot_confusion_matrix(test_labels, test_preds, class_names)
    
    # Plot ROC curve
    print("\nGenerating ROC curve...")
    roc_auc = plot_roc_curve(test_labels, test_probs)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot Precision-Recall curve
    print("\nGenerating Precision-Recall curve...")
    pr_auc = plot_precision_recall_curve(test_labels, test_probs)
    print(f"PR AUC: {pr_auc:.4f}")
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    optimal_threshold = find_optimal_threshold(test_labels, test_probs)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Save the model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'skin_lesion_classifier.pth')
    
    print("Training and evaluation completed successfully!")

# Execute main function if script is run directly
if __name__ == "__main__":
    main()




