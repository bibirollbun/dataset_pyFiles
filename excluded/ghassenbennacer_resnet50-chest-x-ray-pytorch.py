import pandas as pd 
from glob import glob
import os 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
import time
from sklearn.metrics import roc_auc_score
from datetime import datetime


metadata_path="/kaggle/input/grand-xray-slam-division-a/train1.csv"
image_folder="/kaggle/input/grand-xray-slam-division-a/train1"
df=pd.read_csv(metadata_path)


df


print(df.isnull().sum())


df.info()


CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices'
]

label_counts = df[CONDITIONS].sum().sort_values(ascending=False)
print("Label distribution:")
print(label_counts)

# Visualize distribution
plt.figure(figsize=(12, 6))
label_counts.plot(kind='bar')
plt.title('Distribution of Thoracic Conditions')
plt.xticks(rotation=45)
plt.show()


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_folder, conditions, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)  # Clean the indices
        self.image_folder = image_folder  # Folder where images are stored
        self.conditions = conditions      # List of condition names
        self.transform = transform
        
        print(f"Dataset created with {len(self.dataframe)} samples")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get the row using iloc (this ensures we get the idx-th row regardless of index values)
        row = self.dataframe.iloc[idx]
        img_filename = row['Image_name']
        img_path = os.path.join(self.image_folder, img_filename)
        if not os.path.exists(img_path):
            print(f" File not found: {img_path}")
            image = Image.new('RGB', (224, 224), color=0)
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f" Error loading {img_path}: {e}")
                image = Image.new('RGB', (224, 224), color=0)
        label_values = []
        for condition in self.conditions:
            label_values.append(float(row[condition]))
        
        label = torch.tensor(label_values, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        
        return image, label


from torchvision import transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(20),  # Random rotation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Define transforms for validation and testing (no augmentation)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def detect_institution_from_filename(image_name):
    """
    Extract institution from filename pattern using string prefix approach
    CheXpert: 00000001_001_001.jpg (starts with 00 or other patterns < 10)
    MIMIC: 10001217_001_001.jpg (starts with 10)  
    NIH: 20009281_016_000.jpg (starts with 20)
    """
    first_part = image_name.split('_')[0]
    
    if first_part.startswith('10'):
        return 'MIMIC' 
    elif first_part.startswith('20'):
        return 'NIH'
    else:
        return 'CheXpert'

def create_stratified_subset(df, CONDITIONS, target_size=10000, random_state=42):
    """
    Create subset maintaining exact label and institution proportions
    """
    np.random.seed(random_state)
    
    # Create working copy and add institution column from filename
    df = df.copy()
    df['Institution'] = df['Image_name'].apply(detect_institution_from_filename)
    
    # Calculate sampling ratio
    total_size = len(df)
    sampling_ratio = target_size / total_size
    
    print(f"Original dataset size: {total_size}")
    print(f"Target subset size: {target_size}")
    print(f"Sampling ratio: {sampling_ratio:.4f}")
    print("-" * 50)
    
    # Show institution distribution
    print("Institution Distribution (from filename analysis):")
    inst_counts = df['Institution'].value_counts()
    for inst, count in inst_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {inst}: {count:,} samples ({pct:.1f}%)")
    print("-" * 50)
    
    # Create label pattern for multi-label stratification
    df['label_pattern'] = df[CONDITIONS].apply(
        lambda row: ''.join(row.astype(str)), axis=1
    )
    
    # Group by institution and label pattern
    grouped = df.groupby(['Institution', 'label_pattern'])
    
    print(f"Total unique strata (Institution + Label Pattern): {len(grouped)}")
    
    subset_parts = []
    skipped_strata = 0
    
    for (institution, pattern), group in grouped:
        # Calculate target size for this stratum
        stratum_target = max(1, int(len(group) * sampling_ratio))
        actual_sample = min(stratum_target, len(group))
        
        if actual_sample > 0:
            sampled = group.sample(n=actual_sample, random_state=random_state)
            subset_parts.append(sampled)
        else:
            skipped_strata += 1
    
    print(f"Strata sampled: {len(subset_parts)}")
    print(f"Strata skipped (too small): {skipped_strata}")
    
    # Combine all parts
    subset_df = pd.concat(subset_parts).reset_index(drop=True)
    
    # Clean up temporary columns
    subset_df = subset_df.drop(['Institution', 'label_pattern'], axis=1)
    df = df.drop(['Institution', 'label_pattern'], axis=1)
    
    print(f"Final subset size: {len(subset_df)}")
    
    return subset_df

def validate_subset_distributions(original_df, subset_df, CONDITIONS):
    """
    Compare distributions between original and subset datasets
    """
    print("DISTRIBUTION VALIDATION")
    print("=" * 60)
    
    # Add institution columns for both datasets using filename method
    for df_name, df in [("original", original_df), ("subset", subset_df)]:
        df['Institution'] = df['Image_name'].apply(detect_institution_from_filename)
    
    # 1. Label Distribution Comparison
    print("1. LABEL DISTRIBUTION COMPARISON:")
    print("-" * 40)
    print(f"{'Condition':<25} {'Original %':<12} {'Subset %':<12} {'Difference':<12}")
    print("-" * 40)
    
    max_diff = 0
    for condition in CONDITIONS:
        orig_pct = original_df[condition].mean() * 100
        subset_pct = subset_df[condition].mean() * 100
        diff = abs(orig_pct - subset_pct)
        max_diff = max(max_diff, diff)
        
        print(f"{condition:<25} {orig_pct:<12.2f} {subset_pct:<12.2f} {diff:<12.2f}")
    
    print(f"\nMaximum label difference: {max_diff:.2f}%")
    
    # 2. Institution Distribution Comparison
    print(f"\n2. INSTITUTION DISTRIBUTION COMPARISON:")
    print("-" * 40)
    
    orig_inst = original_df['Institution'].value_counts(normalize=True) * 100
    subset_inst = subset_df['Institution'].value_counts(normalize=True) * 100
    
    print(f"{'Institution':<15} {'Original %':<12} {'Subset %':<12} {'Difference':<12}")
    print("-" * 40)
    
    for institution in ['CheXpert', 'MIMIC', 'NIH']:
        orig_pct = orig_inst.get(institution, 0)
        subset_pct = subset_inst.get(institution, 0)
        diff = abs(orig_pct - subset_pct)
        print(f"{institution:<15} {orig_pct:<12.2f} {subset_pct:<12.2f} {diff:<12.2f}")
    
    # 3. Multi-label Pattern Analysis
    print(f"\n3. MULTI-LABEL PATTERN ANALYSIS:")
    print("-" * 40)
    
    # Count number of positive conditions per image
    original_df['num_conditions'] = original_df[CONDITIONS].sum(axis=1)
    subset_df['num_conditions'] = subset_df[CONDITIONS].sum(axis=1)
    
    orig_pattern = original_df['num_conditions'].value_counts(normalize=True).sort_index() * 100
    subset_pattern = subset_df['num_conditions'].value_counts(normalize=True).sort_index() * 100
    
    print(f"{'# Conditions':<15} {'Original %':<12} {'Subset %':<12} {'Difference':<12}")
    print("-" * 40)
    
    all_counts = set(orig_pattern.index) | set(subset_pattern.index)
    for count in sorted(all_counts):
        orig_pct = orig_pattern.get(count, 0)
        subset_pct = subset_pattern.get(count, 0)
        diff = abs(orig_pct - subset_pct)
        print(f"{count:<15} {orig_pct:<12.2f} {subset_pct:<12.2f} {diff:<12.2f}")
    
    # 4. Summary Statistics
    print(f"\n4. SUMMARY STATISTICS:")
    print("-" * 40)
    print(f"Original dataset size: {len(original_df):,}")
    print(f"Subset size: {len(subset_df):,}")
    print(f"Sampling ratio: {len(subset_df)/len(original_df):.4f}")
    print(f"Average conditions per image (Original): {original_df[CONDITIONS].sum(axis=1).mean():.2f}")
    print(f"Average conditions per image (Subset): {subset_df[CONDITIONS].sum(axis=1).mean():.2f}")
    
    # Clean up temporary columns
    for df in [original_df, subset_df]:
        df.drop(['Institution', 'num_conditions'], axis=1, inplace=True)
    
    print(f"\n5. QUALITY ASSESSMENT:")
    print("-" * 40)
    if max_diff < 1.0:
        print("EXCELLENT: Label distributions match very closely (<1% difference)")
    elif max_diff < 2.0:
        print("GOOD: Label distributions acceptable (<2% difference)")
    elif max_diff < 5.0:
        print("CAUTION: Some label distributions differ significantly (2-5% difference)")
    else:
        print("WARNING: Large distribution differences (>5%). Consider larger subset or different sampling strategy.")

# Test the detection function first
print("Testing institution detection function:")
test_filenames = ['00000001_001_001.jpg', '10001217_001_001.jpg', '20009281_016_000.jpg']
for filename in test_filenames:
    result = detect_institution_from_filename(filename)
    print(f"{filename} -> {result}")

# Test on actual data
df['Institution_Test'] = df['Image_name'].apply(detect_institution_from_filename)
print(f"\nActual institution distribution in your data:")
print(df['Institution_Test'].value_counts())
print(f"Percentages:")
print(df['Institution_Test'].value_counts(normalize=True) * 100)

print("\n" + "="*60)
print("Creating stratified subset with corrected institution detection...")
train_subset = create_stratified_subset(df, CONDITIONS, target_size=10000, random_state=42)

print("\nValidating subset quality...")
validate_subset_distributions(df, train_subset, CONDITIONS)

print(f"\nSubset created successfully!")
print(f"Use 'train_subset' for your experiments.")
print(f"Original shape: {df.shape}")
print(f"Subset shape: {train_subset.shape}")

# Clean up the test column
df = df.drop('Institution_Test', axis=1)


train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)


train_dataset = ChestXrayDataset(train_df, image_folder, CONDITIONS, train_transform)
val_dataset = ChestXrayDataset(val_df,image_folder, CONDITIONS,  transform=val_test_transform)
#test_dataset = ChestXrayDataset(test_df, transform=val_test_transform)
train_loader = DataLoader(
    train_dataset, 
    batch_size=64,       
    shuffle=True,
    num_workers=4,       
    pin_memory=True,       
    persistent_workers=True, 
    prefetch_factor=2    
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=64,        
    shuffle=False,
    num_workers=4,        
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


test_images, test_labels = next(iter(train_loader))
print(f"Batch shape: {test_images.shape}")
print(f"Labels shape: {test_labels.shape}")
print(f"Image range: {test_images.min():.3f} to {test_images.max():.3f}")


def show_samples(dataloader, conditions, num_samples=8, figsize=(16, 12)):
    """
    Display sample images with their labels from the dataloader
    """
    # Get one batch
    images, labels = next(iter(dataloader))
    
    # Take only the requested number of samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Calculate grid dimensions
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        
        # Denormalize the image for display
        # Reverse ImageNet normalization
        image = images[idx]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Denormalize
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy and transpose for matplotlib (H, W, C)
        image_np = image.permute(1, 2, 0).numpy()
        
        # Display image
        ax.imshow(image_np)
        ax.axis('off')
        
        # Get positive conditions for this sample
        sample_labels = labels[idx]
        positive_conditions = [conditions[i] for i, val in enumerate(sample_labels) if val > 0.5]
        
        # Create title with positive conditions
        if positive_conditions:
            title = ', '.join(positive_conditions[:3]) 
            if len(positive_conditions) > 3:
                title += f' (+{len(positive_conditions)-3} more)'
        else:
            title = 'No positive findings'
        
        ax.set_title(title, fontsize=10, pad=5)
    

    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.show()


show_samples(train_loader, CONDITIONS, num_samples=8)


"""
# Load pre-trained DenseNet121
model = models.densenet121(pretrained=True)

# Modify the classifier layer for multi-label classification
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(CONDITIONS))

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Move the model to the GPU(s)
model = model.cuda()
"""


"""
from efficientnet_pytorch import EfficientNet 
model = EfficientNet.from_pretrained('efficientnet-b4')

# Get number of features in the last layer
num_features = model._fc.in_features

# Replace the classifier with your own
model._fc = nn.Linear(num_features, len(CONDITIONS))

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Move model to GPU(s)
model = model.cuda()
"""



model_resnet = models.resnet50(pretrained=True)
num_features = model_resnet.fc.in_features
model_resnet.fc = nn.Sequential(
    nn.Dropout(p=0.3), 
    nn.Linear(num_features, len(CONDITIONS))
)
# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for ResNet-50!")
    model_resnet = nn.DataParallel(model_resnet)
# Move the model to the GPU(s)
model_resnet = model_resnet.cuda()
model=model_resnet



"""
def calculate_class_weights_from_dataloader(train_loader, num_classes):

    class_counts = torch.zeros(num_classes)
    total_samples = 0
    
    for _, labels in train_loader:
        # labels shape: [batch_size, num_classes]
        class_counts += labels.sum(dim=0)
        total_samples += labels.shape[0]
    
    # Calculate frequencies
    class_frequencies = class_counts / total_samples
    
    # Calculate pos_weights (higher weight for rare classes)
    pos_weights = (1.0 - class_frequencies) / (class_frequencies + 1e-8)
    
    print("Class frequencies:", class_frequencies)
    print("Pos weights:", pos_weights)
    
    return pos_weights.cuda()

# Use with your data
num_classes = len(CONDITIONS)  # Based on your dataset
pos_weights = calculate_class_weights_from_dataloader(train_loader, num_classes)
"""


import torch.optim as optim
import torch.nn as nn

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Enhanced training function with detailed logging
    """
    # Track training history
    train_losses = []
    val_losses = []
    val_aucs = []
    best_val_auc = 0.0
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                current_loss = train_loss / train_batches
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {current_loss:.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Store for AUC calculation
                predictions = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.append(predictions)
                all_labels.append(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate validation AUC
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        try:
            val_auc_scores = []
            for i in range(len(CONDITIONS)):
                if len(np.unique(all_labels[:, i])) > 1:  
                    auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                    val_auc_scores.append(auc)
                else:
                    val_auc_scores.append(0.5)  
            
            mean_val_auc = np.mean(val_auc_scores)
            val_aucs.append(mean_val_auc)
            
        except Exception as e:
            print(f"Warning: Could not calculate AUC - {e}")
            mean_val_auc = 0.0
            val_aucs.append(0.0)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val AUC: {mean_val_auc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  New best model saved! AUC: {best_val_auc:.4f}")
        
        print("-" * 70)
    
    print(f"Training completed!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs,
        'best_val_auc': best_val_auc
    }

def evaluate_model(model, test_loader, CONDITIONS, device='cuda'):
    """
    Enhanced evaluation function with detailed metrics
    """
    print("Starting model evaluation...")
    print(f"Test batches: {len(test_loader)}")
    
    model.eval()
    all_labels = []
    all_predictions = []
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Convert logits to probabilities
            predictions = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    eval_time = time.time() - eval_start_time
    
    # Combine all predictions and labels
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    print(f"Evaluation completed in {eval_time:.1f}s")
    print(f"Total samples evaluated: {len(all_predictions)}")
    print("-" * 70)
    
    # Calculate AUC for each condition
    auc_scores = []
    print("Per-condition AUC scores:")
    
    for i, condition in enumerate(CONDITIONS):
        try:
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
            else:
                auc = 0.5
                print(f"  Warning: {condition} has only one class in test set")
            
            auc_scores.append(auc)
            print(f"  {condition:<25}: {auc:.4f}")
            
        except Exception as e:
            print(f"  Error calculating AUC for {condition}: {e}")
            auc_scores.append(0.5)
    
    mean_auc = np.mean(auc_scores)
    print("-" * 70)
    print(f"Mean AUC: {mean_auc:.4f}")
    
    # Additional statistics
    print(f"\nLabel distribution in test set:")
    for i, condition in enumerate(CONDITIONS):
        pos_count = np.sum(all_labels[:, i])
        pos_pct = (pos_count / len(all_labels)) * 100
        print(f"  {condition:<25}: {pos_count:>5} ({pos_pct:>4.1f}%)")
    
    return {
        'auc_scores': auc_scores,
        'mean_auc': mean_auc,
        'predictions': all_predictions,
        'labels': all_labels
    }

def plot_training_history(history):
    """
    Plot training curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # AUC curve
    ax2.plot(history['val_aucs'], label='Validation AUC', color='green')
    ax2.set_title('Validation AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


# Training
history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)

# Plot training curves
plot_training_history(history)

# Evaluation
#results = evaluate_model(model, test_loader, CONDITIONS)


def create_test_dataset_from_folder(test_folder, CONDITIONS, transform):
    """
    Create test dataset directly from test folder images
    """
    # Get all image files from test folder
    image_files = []
    for file in os.listdir(test_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(file)
    
    # Sort to ensure consistent ordering
    image_files.sort()
    
    # Create DataFrame with image names
    test_df = pd.DataFrame({'Image_name': image_files})
    
    # Add dummy labels (required for dataset compatibility)
    for condition in CONDITIONS:
        test_df[condition] = 0
    
    print(f"Found {len(test_df)} test images")
    
    # Create dataset
    test_dataset = ChestXrayDataset(test_df, test_folder, CONDITIONS, transform)
    
    return test_dataset, test_df

def generate_test_predictions(model, test_loader, device='cuda'):
    """
    Generate predictions for all test images
    """
    print("Generating predictions...")
    
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, *_) in enumerate(test_loader): 
            images = images.to(device)
            outputs = model(images)
            
            # Convert logits to probabilities using sigmoid
            predictions = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.append(predictions)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Combine all predictions
    all_predictions = np.vstack(all_predictions)
    print(f"Generated predictions for {len(all_predictions)} images")
    
    return all_predictions

def create_submission_csv(test_df, predictions, CONDITIONS, output_path, use_binary=False, threshold=0.5):
    """
    Create final submission CSV file
    """
    print("Creating submission CSV...")
    
    submission = pd.DataFrame()
    submission['Image_name'] = test_df['Image_name'].values
    
    # Add prediction columns
    for i, condition in enumerate(CONDITIONS):
        if use_binary:
            # Convert probabilities to binary (0 or 1)
            submission[condition] = (predictions[:, i] > threshold).astype(int)
        else:
            # Keep as probabilities (competition requirement)
            submission[condition] = predictions[:, i]
    
    # Show sample output for verification
    print(f"Sample predictions for first image:")
    print(f"  Image: {submission.iloc[0]['Image_name']}")
    for condition in CONDITIONS:
        print(f"  {condition}: {submission.iloc[0][condition]:.4f}")
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    
    return submission

def generate_submission(model, test_folder, CONDITIONS, val_test_transform, output_path, device='cuda'):
    """
    Complete pipeline: test folder -> predictions -> CSV submission
    """
    print("Starting submission generation...")
    print("=" * 50)
    
    test_dataset, test_df = create_test_dataset_from_folder(test_folder, CONDITIONS, val_test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,  
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Step 3: Generate predictions
    predictions = generate_test_predictions(model, test_loader, device)
    
    # Step 4: Create submission CSV (probabilities by default)
    submission = create_submission_csv(test_df, predictions, CONDITIONS, output_path)
    
    print("=" * 50)
    print("Submission generation completed!")
    print(f"Final submission shape: {submission.shape}")
    print(f"Columns: {list(submission.columns)}")
    
    return submission

# Usage example:
def run_final_submission(model, CONDITIONS, val_test_transform):
    """
    Generate final competition submission
    """
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Define paths
    test_folder = "/kaggle/input/grand-xray-slam-division-a/test1"
    
    # Create timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'/kaggle/working/submission_{timestamp}.csv'
    
    # Generate submission with probabilities (correct format)
    submission = generate_submission(
        model=model,
        test_folder=test_folder,
        CONDITIONS=CONDITIONS,
        val_test_transform=val_test_transform,
        output_path=output_path,
        device='cuda'
    )
    
    print(f"\n FINAL SUBMISSION: {output_path}")
    return submission


# This will create your final competition submission
final_submission = run_final_submission(model, CONDITIONS, val_test_transform)













