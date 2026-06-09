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


import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data paths
BASE_DIR = '/kaggle/input/2025-bamboo-summer-competiton-dl-pr'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')
TEST_CSV = os.path.join(BASE_DIR, 'test.csv')
SAMPLE_SUBMISSION = os.path.join(BASE_DIR, 'sample_submission.csv')

# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION)

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Number of classes: {train_df['label'].nunique()}")

# Display class distribution
class_counts = train_df['label'].value_counts()
print("\nClass distribution:")
print(class_counts)

# Visualize class distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='label', order=class_counts.index)
plt.title('Distribution of Butterfly Species')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Label encoding
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Custom Dataset class
class ButterflyDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, is_test=False):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['filename']
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, img_name
        else:
            label = self.dataframe.iloc[idx]['label_encoded']
            return image, label

# Data augmentation and preprocessing
def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Model definition using EfficientNet
class ButterflyClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0'):
        super(ButterflyClassifier, self).__init__()
        
        # Load pre-trained model
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.backbone.classifier[1].in_features, num_classes)
            )
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=True)
            self.backbone.heads = nn.Linear(self.backbone.heads.head.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    best_model_wts = model.state_dict()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
        
        scheduler.step()
        print()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs

# Cross-validation training
def cross_validation_training(train_df, num_folds=5):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label_encoded'])):
        print(f"\nFold {fold + 1}/{num_folds}")
        print("=" * 50)
        
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        train_dataset = ButterflyDataset(train_fold, TRAIN_DIR, get_transforms('train'))
        val_dataset = ButterflyDataset(val_fold, TRAIN_DIR, get_transforms('val'))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Initialize model
        model = ButterflyClassifier(num_classes, 'efficientnet_b0').to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Train model
        model, _, _, _, _ = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15)
        
        # Evaluate on validation set
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        fold_accuracy = accuracy_score(val_labels, val_preds)
        fold_accuracies.append(fold_accuracy)
        fold_models.append(model)
        
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
    
    print(f"\nCross-validation mean accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies) * 2:.4f})")
    return fold_models, fold_accuracies

# Ensemble prediction
def ensemble_predict(models, test_loader, label_encoder):
    all_predictions = []
    
    for model in models:
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, filenames in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        all_predictions.append(np.vstack(predictions))
    
    # Average predictions across all models
    ensemble_predictions = np.mean(all_predictions, axis=0)
    final_predictions = np.argmax(ensemble_predictions, axis=1)
    
    # Convert back to original labels
    final_labels = label_encoder.inverse_transform(final_predictions)
    
    return final_labels

# Main execution
if __name__ == "__main__":
    # Train models using cross-validation
    print("Starting cross-validation training...")
    fold_models, fold_accuracies = cross_validation_training(train_df, num_folds=5)
    
    # Create test dataset
    test_dataset = ButterflyDataset(test_df, TEST_DIR, get_transforms('val'), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Make ensemble predictions
    print("Making ensemble predictions...")
    ensemble_predictions = ensemble_predict(fold_models, test_loader, label_encoder)
    
    # Create submission file
    submission = pd.DataFrame({
        'filename': test_df['filename'],
        'label': ensemble_predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")
    
    # Display sample predictions
    print("\nSample predictions:")
    print(submission.head(10))
    
    # Save the best model
    best_fold_idx = np.argmax(fold_accuracies)
    torch.save(fold_models[best_fold_idx].state_dict(), 'best_butterfly_model.pth')
    print(f"Best model saved (Fold {best_fold_idx + 1} with accuracy {fold_accuracies[best_fold_idx]:.4f})")

# Additional utility functions for post-processing and analysis
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_predictions(model, val_loader, label_encoder):
    """Analyze model predictions and create confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to original labels
    pred_labels = label_encoder.inverse_transform(all_preds)
    true_labels = label_encoder.inverse_transform(all_labels)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels))
    
    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Test Time Augmentation (TTA) for better predictions
def tta_predict(model, test_loader, label_encoder, num_tta=5):
    """Test Time Augmentation for more robust predictions"""
    model.eval()
    all_tta_predictions = []
    
    tta_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for tta_idx in range(num_tta):
        predictions = []
        
        with torch.no_grad():
            for inputs, filenames in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        all_tta_predictions.append(np.vstack(predictions))
    
    # Average TTA predictions
    tta_ensemble = np.mean(all_tta_predictions, axis=0)
    final_predictions = np.argmax(tta_ensemble, axis=1)
    final_labels = label_encoder.inverse_transform(final_predictions)
    
    return final_labels




