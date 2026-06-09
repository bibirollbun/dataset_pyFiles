# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Load the training data
train_df = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/train1.csv')
print(f"Training data shape: {train_df.shape}")

# Display column names to identify the correct image column
print("\nColumn names in train_df:")
print(train_df.columns.tolist())

# Display first few rows
print("\nFirst 3 rows:")
print(train_df.head(3))

# Define the labels for the competition
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
          'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
          'Lung Opacity', 'No Finding', 'Pleural Effusion', 
          'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

# Find the correct column name for the image filename
image_col = None
possible_names = ['Image_Name', 'Image_Name', 'ImageName', 'image_name', 'filename', 'Image_name', 'Image']
for col in possible_names:
    if col in train_df.columns:
        image_col = col
        break

if image_col is None:
    # If none of the expected names are found, use the first column that seems like it could be image names
    for col in train_df.columns:
        if any(term in col.lower() for term in ['image', 'file', 'name']):
            image_col = col
            break
    if image_col is None:
        image_col = train_df.columns[0]  # Use first column as fallback

print(f"\nUsing column '{image_col}' for image names")


# Check for missing values
print("\nMissing values:")
print(train_df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(train_df.describe())

# Visualize the distribution of labels
plt.figure(figsize=(20, 10))
label_counts = train_df[labels].sum().sort_values(ascending=False)
sns.barplot(x=label_counts.values, y=label_counts.index)
plt.title('Distribution of Thoracic Conditions')
plt.xlabel('Count')
plt.ylabel('Condition')
plt.tight_layout()
plt.savefig('/kaggle/working/label_distribution.png')
plt.show()

# Check co-occurrence of conditions
plt.figure(figsize=(15, 12))
correlation_matrix = train_df[labels].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Thoracic Conditions')
plt.tight_layout()
plt.savefig('/kaggle/working/label_correlation.png')
plt.show()

# Distribution of views (if available)
if 'ViewPosition' in train_df.columns:
    plt.figure(figsize=(12, 6))
    view_counts = train_df['ViewPosition'].value_counts()
    plt.pie(view_counts.values, labels=view_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of View Positions')
    plt.savefig('/kaggle/working/view_distribution.png')
    plt.show()

# Age distribution (if available)
if 'Age' in train_df.columns:
    plt.figure(figsize=(12, 6))
    sns.histplot(train_df['Age'].dropna(), bins=30, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('/kaggle/working/age_distribution.png')
    plt.show()

# Sex distribution (if available)
if 'Sex' in train_df.columns:
    plt.figure(figsize=(8, 6))
    sex_counts = train_df['Sex'].value_counts()
    sns.barplot(x=sex_counts.index, y=sex_counts.values)
    plt.title('Sex Distribution')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.savefig('/kaggle/working/sex_distribution.png')
    plt.show()


# Sample and display some images (limited to save memory)
def display_sample_images(df, num_images=6):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    sample_indices = np.random.choice(len(df), num_images, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Get the correct image path
        img_filename = df.iloc[idx][image_col]
        img_path = os.path.join('/kaggle/input/grand-xray-slam-division-a/train1', img_filename)
        
        try:
            img = Image.open(img_path)
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Image: {img_filename}")
            
            # Show positive labels
            positive_labels = [label for label in labels if df.iloc[idx][label] == 1]
            if positive_labels:
                axes[i].set_xlabel(f"Labels: {', '.join(positive_labels)}")
            
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading\n{img_filename}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/sample_images.png')
    plt.show()

# Display sample images
display_sample_images(train_df, num_images=6)


class ChestXRayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False, image_col=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        
        # Determine the image column name
        if image_col is None:
            # Auto-detect image column
            possible_names = ['Image_Name', 'Image_Name', 'ImageName', 'image_name', 'filename', 'Image_name']
            for col in possible_names:
                if col in df.columns:
                    self.image_col = col
                    break
            else:
                # If none found, use first column
                self.image_col = df.columns[0]
        else:
            self.image_col = image_col
        
        # For training data, we have labels
        if not is_test:
            self.labels = df[labels].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx][self.image_col])
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (256, 256), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, img_name
        else:
            labels = torch.FloatTensor(self.labels[idx])
            return image, labels

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ChestXRayModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ChestXRayModel, self).__init__()
        
        # Use EfficientNet as base model
        self.base_model = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace the classifier
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()  # Remove original classifier
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

# Create model
model = ChestXRayModel(num_classes=14)
model = model.to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")


# For demonstration, we'll use a small subset of data
sample_size = min(2000, len(train_df))  # Use smaller sample for demonstration
sample_df = train_df.sample(sample_size, random_state=42)

# Split data into train and validation
train_data, val_data = train_test_split(
    sample_df, test_size=0.2, random_state=42, stratify=sample_df[labels].sum(axis=1)
)

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")

# Create datasets and dataloaders
train_dataset = ChestXRayDataset(train_data, '/kaggle/input/grand-xray-slam-division-a/train1', 
                                transform=train_transform, image_col=image_col)
val_dataset = ChestXRayDataset(val_data, '/kaggle/input/grand-xray-slam-division-a/train1', 
                              transform=val_transform, image_col=image_col)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# Calculate class weights for imbalanced data
def calculate_class_weights(df, labels):
    class_weights = []
    for label in labels:
        positive = df[label].sum()
        negative = len(df) - positive
        weight = negative / (positive + 1e-6)  # Add small epsilon to avoid division by zero
        class_weights.append(weight)
    
    return torch.FloatTensor(class_weights).to(device)

# Loss function and optimizer
class_weights = calculate_class_weights(train_df, labels)
criterion = nn.BCELoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, targets in tqdm(loader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    # Calculate AUC for each class
    auc_scores = []
    for i in range(all_targets.shape[1]):
        try:
            # Check if we have both positive and negative samples
            if len(np.unique(all_targets[:, i])) > 1:
                auc = roc_auc_score(all_targets[:, i], all_outputs[:, i])
                auc_scores.append(auc)
            else:
                auc_scores.append(0.5)  # Neutral score for constant targets
        except:
            auc_scores.append(0.5)
    
    return running_loss / len(loader.dataset), np.mean(auc_scores), auc_scores

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device):
    best_auc = 0.0
    train_losses = []
    val_losses = []
    val_aucs = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, mean_auc, class_aucs = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_aucs.append(mean_auc)
        
        # Update scheduler
        scheduler.step(mean_auc)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {mean_auc:.4f}')
        
        # Print class-wise AUC for top 5 classes
        class_auc_df = pd.DataFrame({'Class': labels, 'AUC': class_aucs})
        class_auc_df = class_auc_df.sort_values('AUC', ascending=False)
        print("Top 5 classes by AUC:")
        for i, row in class_auc_df.head().iterrows():
            print(f'  {row["Class"]}: {row["AUC"]:.4f}')
        
        # Save best model
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            print(f'New best model saved with AUC: {best_auc:.4f}')
    
    return train_losses, val_losses, val_aucs

# Train the model for a few epochs (using sample data)
print("Starting training with sample data...")
train_losses, val_losses, val_aucs = train_model(
    model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=3, device=device
)


# Enhanced training with more epochs and better regularization
def enhanced_training():
    # Reload the best model
    model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
    
    # Use a more aggressive learning rate schedule
    enhanced_optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    enhanced_scheduler = optim.lr_scheduler.CosineAnnealingLR(enhanced_optimizer, T_max=5)
    
    # Use focal loss for better handling of class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
        
        def forward(self, inputs, targets):
            BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            
            if self.reduction == 'mean':
                return torch.mean(F_loss)
            elif self.reduction == 'sum':
                return torch.sum(F_loss)
            else:
                return F_loss
    
    focal_criterion = FocalLoss()
    
    print("Starting enhanced training with focal loss...")
    enhanced_losses, enhanced_val_losses, enhanced_val_aucs = train_model(
        model, train_loader, val_loader, enhanced_optimizer, focal_criterion, enhanced_scheduler, epochs=5, device=device
    )
    
    return enhanced_losses, enhanced_val_losses, enhanced_val_aucs

# Uncomment to run enhanced training
# enhanced_losses, enhanced_val_losses, enhanced_val_aucs = enhanced_training()
# plot_training_progress(enhanced_losses, enhanced_val_losses, enhanced_val_aucs)


def plot_training_progress(train_losses, val_losses, val_aucs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot AUC
    ax2.plot(val_aucs, label='Validation AUC', color='green', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_progress.png')
    plt.show()

# Plot training progress
plot_training_progress(train_losses, val_losses, val_aucs)

# Save training metrics
metrics_df = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_auc': val_aucs
})
metrics_df.to_csv('/kaggle/working/training_metrics.csv', index=False)
print("Training metrics saved to /kaggle/working/training_metrics.csv")


# Create ensemble of models for better performance
def create_ensemble(models_list, test_loader, device):
    all_predictions = []
    
    for model in models_list:
        model.eval()
        model_predictions = []
        
        with torch.no_grad():
            for images, names in tqdm(test_loader, desc="Model predictions"):
                images = images.to(device)
                outputs = model(images)
                model_predictions.append(outputs.cpu().numpy())
        
        all_predictions.append(np.concatenate(model_predictions))
    
    # Average predictions from all models
    ensemble_pred = np.mean(all_predictions, axis=0)
    return ensemble_pred

# Create multiple models with different architectures
def create_different_models():
    models_list = []
    
    # EfficientNet-B0
    model1 = ChestXRayModel(num_classes=14)
    models_list.append(model1)
    
    # DenseNet-121
    class DenseNetModel(nn.Module):
        def __init__(self, num_classes=14):
            super(DenseNetModel, self).__init__()
            self.base_model = models.densenet121(pretrained=True)
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.base_model(x)
    
    model2 = DenseNetModel(num_classes=14)
    models_list.append(model2)
    
    # ResNet-50
    class ResNetModel(nn.Module):
        def __init__(self, num_classes=14):
            super(ResNetModel, self).__init__()
            self.base_model = models.resnet50(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.base_model(x)
    
    model3 = ResNetModel(num_classes=14)
    models_list.append(model3)
    
    return models_list

# Train ensemble (commented out for time)
# print("Creating model ensemble...")
# ensemble_models = create_different_models()
# for i, model in enumerate(ensemble_models):
#     model = model.to(device)
#     print(f"Training model {i+1}/{len(ensemble_models)}")
#     train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=2, device=device)
#     torch.save(model.state_dict(), f'/kaggle/working/model_{i+1}.pth')


# Enhanced submission with calibration and post-processing
def create_enhanced_submission(model, test_dir, transform, device, sample_submission_path, image_col):
    # Load sample submission
    sample_submission = pd.read_csv(sample_submission_path)
    sample_image_col = 'Image_name' if 'Image_name' in sample_submission.columns else sample_submission.columns[0]
    
    # Create test dataset
    test_df = pd.DataFrame({image_col: sample_submission[sample_image_col]})
    test_dataset = ChestXRayDataset(test_df, test_dir, transform=transform, is_test=True, image_col=image_col)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    model.eval()
    predictions = []
    image_names = []
    
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Creating enhanced predictions"):
            images = images.to(device)
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())
            image_names.extend(names)
    
    predictions = np.concatenate(predictions)
    
    # Apply temperature scaling calibration
    def temperature_scale(logits, temperature=0.8):
        return logits ** (1/temperature)
    
    calibrated_predictions = temperature_scale(predictions, temperature=0.8)
    
    # Apply label correlation adjustment (if conditions are correlated)
    correlation_matrix = train_df[labels].corr().values
    adjusted_predictions = np.dot(calibrated_predictions, correlation_matrix)
    adjusted_predictions = np.clip(adjusted_predictions, 0, 1)  # Ensure valid probabilities
    
    # Create submission dataframe
    submission_df = pd.DataFrame(adjusted_predictions, columns=labels)
    submission_df.insert(0, sample_image_col, image_names)
    submission_df.columns = sample_submission.columns
    
    return submission_df

# Create enhanced submission
print("Creating enhanced submission with calibration...")
enhanced_submission = create_enhanced_submission(
    model, 
    '/kaggle/input/grand-xray-slam-division-a/test1', 
    val_transform, 
    device,
    '/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv',
    image_col=image_col
)

# Save enhanced submission
enhanced_submission.to_csv('/kaggle/working/enhanced_submission.csv', index=False)
print("Enhanced submission file created at /kaggle/working/enhanced_submission.csv")


# Create the final optimized submission
def create_final_submission():
    # Load the best model
    model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
    
    # Create submission with test-time augmentation
    def predict_with_tta(model, test_loader, device, n_augmentations=3):
        model.eval()
        all_predictions = []
        
        # Define TTA transformations
        tta_transforms = [
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
        
        image_names = []
        for images, names in tqdm(test_loader, desc="TTA predictions"):
            batch_predictions = []
            
            for tta_transform in tta_transforms[:n_augmentations]:
                # Apply different augmentation
                augmented_images = torch.stack([tta_transform(Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))) 
                                              for img in images])
                
                augmented_images = augmented_images.to(device)
                with torch.no_grad():
                    outputs = model(augmented_images)
                    batch_predictions.append(outputs.cpu().numpy())
            
            # Average predictions from different augmentations
            avg_predictions = np.mean(batch_predictions, axis=0)
            all_predictions.append(avg_predictions)
            image_names.extend(names)
        
        return np.concatenate(all_predictions), image_names
    
    # Load test data
    sample_submission = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv')
    sample_image_col = 'Image_name' if 'Image_name' in sample_submission.columns else sample_submission.columns[0]
    
    test_df = pd.DataFrame({image_col: sample_submission[sample_image_col]})
    test_dataset = ChestXRayDataset(test_df, '/kaggle/input/grand-xray-slam-division-a/test1', 
                                   transform=val_transform, is_test=True, image_col=image_col)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Get TTA predictions
    tta_predictions, image_names = predict_with_tta(model, test_loader, device, n_augmentations=2)
    
    # Create final submission
    final_submission = pd.DataFrame(tta_predictions, columns=labels)
    final_submission.insert(0, sample_image_col, image_names)
    final_submission.columns = sample_submission.columns
    
    return final_submission

# Create final submission with TTA
print("Creating final submission with Test-Time Augmentation...")
final_submission = create_final_submission()
final_submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Final submission file created at /kaggle/working/submission.csv")

# Final summary
print("=" * 70)
print("FINAL SUBMISSION READY!")
print("=" * 70)
print("Available submission files:")
print("1. /kaggle/working/submission.csv - Basic predictions")
print("2. /kaggle/working/enhanced_submission.csv - With calibration")
print("3. /kaggle/working/submission.csv - With TTA (Recommended)")
print("=" * 70)
print("Recommendation: Use final_submission.csv for your competition entry!")
print("=" * 70)




