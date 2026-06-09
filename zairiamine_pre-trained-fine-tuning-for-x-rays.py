# Import necessary libraries
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm
from skmultilearn.model_selection import iterative_train_test_split
from collections import Counter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load and analyze data
try:
    train_df = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/train1.csv')
    print(f"Loaded train1.csv with {len(train_df)} rows")
except FileNotFoundError:
    print("Error: train1.csv not found. Ensure dataset is attached.")
    raise

label_columns = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# Check for missing columns
missing_cols = [col for col in label_columns if col not in train_df.columns]
if missing_cols:
    print(f"Error: Missing columns in train1.csv: {missing_cols}")
    raise KeyError(f"Missing columns: {missing_cols}")

# --- Data Visualization ---

# 1. Label Distribution Plot
print("\nGenerating Label Distribution Plot...")
label_counts = train_df[label_columns].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
plt.title('Distribution of Disease Labels in Training Data')
plt.xlabel('Disease Label')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Disease Correlation Plot
print("\nGenerating Disease Correlation Plot...")
plt.figure(figsize=(10, 8))
correlation_matrix = train_df[label_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation between Disease Labels')
plt.show()


# Perform iterative stratified (multi-label) split
X = train_df[['Image_name']]
y = train_df[label_columns].values

X_train, y_train, X_val, y_val = iterative_train_test_split(X.values, y, test_size=0.2)

train_data = pd.DataFrame(X_train, columns=['Image_name'])
train_data[label_columns] = y_train

val_data = pd.DataFrame(X_val, columns=['Image_name'])
val_data[label_columns] = y_val

print(f"Train samples after stratified split: {len(train_data)}")
print(f"Validation samples after stratified split: {len(val_data)}")
print("\nDistribution of labels in training set:", Counter(np.where(y_train == 1)[1]))
print("Distribution of labels in validation set:", Counter(np.where(y_val == 1)[1]))


# Define Dataset and DataLoader classes
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_size=(224, 224), is_test=False, transforms=None):
        self.df = df
        self.img_size = img_size
        self.is_test = is_test
        self.label_columns = label_columns
        self.image_dir = '/kaggle/input/grand-xray-slam-division-a/train1/' if not is_test else '/kaggle/input/grand-xray-slam-division-a/test1/'
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image_name'])
        
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}. Returning black image.")
            img = np.zeros(self.img_size, dtype=np.uint8)
        
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if self.transforms:
            img = self.transforms(img)
        
        if not self.is_test:
            labels = row[self.label_columns].values.astype(np.float32)
            return img, torch.tensor(labels)
        
        return img

# Define image transformations with data augmentation for training
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define transformations for validation/testing (no augmentation)
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create DataLoader instances
batch_size = 32
train_dataset = ChestXRayDataset(train_data, img_size=(224, 224), transforms=train_transforms)
val_dataset = ChestXRayDataset(val_data, img_size=(224, 224), transforms=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print("PyTorch DataLoaders created successfully.")


# Calculate class weights for Weighted BCE Loss
print("Calculating class weights...")
total_samples = len(train_data)
positive_counts = train_data[label_columns].sum()
negative_counts = total_samples - positive_counts
pos_weight = negative_counts / positive_counts
pos_weight = torch.tensor(pos_weight.values, dtype=torch.float).to(device)

print("Calculated positive weights:", pos_weight)


# Build and fine-tune DenseNet-121
def create_model(num_classes=14, dropout_rate=0.5):
    # Load pretrained DenseNet-121
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    # Freeze all layers except the last dense block and the classifier
    for name, param in model.named_parameters():
        if 'denseblock4' not in name and 'norm5' not in name:
            param.requires_grad = False

    # Get the number of input features for the classifier
    num_ftrs = model.classifier.in_features
    
    # Redefine the classifier to be a sequential block with Dropout and a linear layer
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

model = create_model().to(device)
# The criterion is now initialized with the calculated pos_weight
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=3
)
print("Model, Loss, Optimizer, and Scheduler created.")


# Train model with Early Stopping
num_epochs = 30
best_val_auc = 0.0
patience = 5
epochs_no_improve = 0
early_stop = False

# Lists to store training history
train_loss_history = []
val_loss_history = []
train_auc_history = []
val_auc_history = []

for epoch in range(num_epochs):
    if early_stop:
        print("Early stopping triggered.")
        break

    # --- Training loop ---
    model.train()
    running_train_loss = 0.0
    train_preds, train_labels = [], []
    train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} Training")
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * images.size(0)
        train_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        train_labels.append(labels.detach().cpu().numpy())
        train_loop.set_postfix(loss=loss.item())
    epoch_train_loss = running_train_loss / len(train_data)
    train_loss_history.append(epoch_train_loss)
    train_auc = roc_auc_score(np.vstack(train_labels), np.vstack(train_preds), average='macro')
    train_auc_history.append(train_auc)


    # --- Validation loop ---
    model.eval()
    running_val_loss = 0.0
    val_preds, val_labels = [], []
    val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} Validation")
    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)
            val_preds.append(torch.sigmoid(outputs).cpu().numpy())
            val_labels.append(labels.cpu().numpy())
    epoch_val_loss = running_val_loss / len(val_data)
    val_loss_history.append(epoch_val_loss)
    val_preds = np.vstack(val_preds)
    val_labels = np.vstack(val_labels)
    val_auc = roc_auc_score(val_labels, val_preds, average='macro')
    val_auc_history.append(val_auc)

    # Log metrics
    print(f"\nEpoch {epoch+1}/{num_epochs} Complete: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    # Early stopping and saving the best model
    if val_auc > best_val_auc:
        print(f"Validation AUC improved from {best_val_auc:.4f} to {val_auc:.4f}. Saving model...")
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'best_model.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"Validation AUC did not improve. Patience: {epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            early_stop = True

    # Step the scheduler
    scheduler.step(val_auc)

print("Training complete.")


# Plot training history for Loss and AUC
epochs_ran = len(train_loss_history)
epochs = range(1, epochs_ran + 1)

# Plot Loss History
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss_history, 'b', label='Training Loss')
plt.plot(epochs, val_loss_history, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot AUC History
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_auc_history, 'b', label='Training AUC')
plt.plot(epochs, val_auc_history, 'r', label='Validation AUC')
plt.title('Training and Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.show()


# Generate submission file
try:
    sample_submission = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv')
    print(f"Loaded sample_submission_1.csv with {len(sample_submission)} rows")
except FileNotFoundError:
    print("Error: sample_submission_1.csv not found.")
    raise

# Create test dataset and loader
test_dataset = ChestXRayDataset(sample_submission, img_size=(224, 224), is_test=True, transforms=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load the best model and set to evaluation mode
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generate predictions
predictions = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Generating predictions"):
        images = images.to(device)
        outputs = model(images)
        batch_preds = torch.sigmoid(outputs).cpu().numpy()
        predictions.append(batch_preds)

# Combine predictions and create submission DataFrame
predictions = np.vstack(predictions)
predictions = predictions[:len(sample_submission)]
submission_df = sample_submission.copy()
submission_df[label_columns] = predictions

# Save the submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

