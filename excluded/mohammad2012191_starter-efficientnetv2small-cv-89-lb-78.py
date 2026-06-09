import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Model configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_FOLDS = 5
NUM_CLASSES = 7
PATIENCE = 5  # Early stopping patience

# Paths
TRAIN_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'
TEST_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test'
TRAIN_CSV = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
SUBMISSION_CSV = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/dummy_sub.csv'


# Load train labels
train_df = pd.read_csv(TRAIN_CSV)
print(f'Train data shape: {train_df.shape}')
print(f'Classes: {train_df["label"].unique()}')
print(f'Class distribution:\n{train_df["label"].value_counts()}')

# Encode labels
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])

# Load test filenames
test_files = sorted(os.listdir(TEST_DIR))
print(f'Number of test images: {len(test_files)}')


class SheepDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.is_test:
            img_name = self.df[idx]
            img_path = os.path.join(self.img_dir, img_name)
        else:
            img_name = self.df.iloc[idx]['filename']
            img_path = os.path.join(self.img_dir, img_name)
            label = self.df.iloc[idx]['encoded_label']
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image, img_name
        else:
            return image, label


# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation and test transforms
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def create_model(num_classes):
    # Load pretrained EfficientNetV2-S
    model = models.efficientnet_v2_s(pretrained=True)
    
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Linear(num_features, num_classes)
    return model.to(device)


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1


def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1, np.array(all_probs), np.array(all_labels)


# Initialize stratified k-fold
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Store models for ensemble
fold_models = []

# Initialize OOF predictions
oof_predictions = np.zeros((len(train_df), NUM_CLASSES))
oof_labels = np.zeros(len(train_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['encoded_label'])):
    print(f'\n{"="*50}')
    print(f'Fold {fold + 1}/{NUM_FOLDS}')
    print(f'{"="*50}')
    
    # Split data
    train_data = train_df.iloc[train_idx].reset_index(drop=True)
    val_data = train_df.iloc[val_idx].reset_index(drop=True)
    
    # Create datasets
    train_dataset = SheepDataset(train_data, TRAIN_DIR, transform=train_transform)
    val_dataset = SheepDataset(val_data, TRAIN_DIR, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model, criterion, optimizer
    model = create_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validate
        val_loss, val_f1, val_probs, val_true_labels = validate_epoch(model, val_loader, criterion)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            
            # Save best model to disk
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            }, f'best_model_fold_{fold}.pth')
        
    # Load best model
    model.load_state_dict(best_model_state)
    fold_models.append(model)
    
    # Get OOF predictions for best model
    _, _, best_val_probs, best_val_labels = validate_epoch(model, val_loader, criterion)
    oof_predictions[val_idx] = best_val_probs
    oof_labels[val_idx] = best_val_labels
    
    print(f'\nFold {fold + 1} Best Val F1: {best_val_f1:.4f}')


# Calculate OOF predictions
oof_preds = np.argmax(oof_predictions, axis=1)

# Overall OOF F1 score
oof_f1 = f1_score(oof_labels, oof_preds, average='macro')
print(f'\nOverall OOF F1 Score: {oof_f1:.4f}')


# Create test dataset
test_dataset = SheepDataset(test_files, TEST_DIR, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Ensemble predictions
all_predictions = []

print('\nGenerating test predictions...')
for i, model in enumerate(fold_models):
    print(f'Processing fold {i+1}/{NUM_FOLDS}')
    model.eval()
    fold_predictions = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f'Fold {i+1} Test Inference'):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            fold_predictions.append(probabilities.cpu().numpy())
    
    fold_predictions = np.concatenate(fold_predictions, axis=0)
    all_predictions.append(fold_predictions)

# Average predictions across folds
avg_predictions = np.mean(all_predictions, axis=0)
final_predictions = np.argmax(avg_predictions, axis=1)

# Decode predictions
predicted_labels = label_encoder.inverse_transform(final_predictions)


# Create submission dataframe
submission_df = pd.DataFrame({
    'filename': test_files,
    'label': predicted_labels
})

# Save submission
submission_df.to_csv('submission.csv', index=False)
print(f'\nSubmission saved! Shape: {submission_df.shape}')
print(submission_df.head())

# Save test predictions with probabilities
test_pred_df = pd.DataFrame({
    'filename': test_files,
    'pred_label': predicted_labels
})

# Add probability columns
for i, class_name in enumerate(label_encoder.classes_):
    test_pred_df[f'prob_{class_name}'] = avg_predictions[:, i]

test_pred_df.to_csv('test_predictions_with_probs.csv', index=False)
print(f'\nTest predictions with probabilities saved to test_predictions_with_probs.csv')

