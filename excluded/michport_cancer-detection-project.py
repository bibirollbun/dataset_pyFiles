import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# Define paths
data_dir = '/kaggle/input/histopathologic-cancer-detection/'
train_labels_path = os.path.join(data_dir, 'train_labels.csv')
train_images_dir = os.path.join(data_dir, 'train')
test_images_dir = os.path.join(data_dir, 'test')

# Load labels
train_labels_df = pd.read_csv(train_labels_path)

# Basic data description
print(f"Training labels shape: {train_labels_df.shape}")
print("\nFirst 5 rows of training labels:")
print(train_labels_df.head().to_markdown(index=False))
print("\nMissing values in training labels:")
print(train_labels_df.isnull().sum())

# Check for duplicates
print("\nDuplicate rows in training labels:")
print(train_labels_df[train_labels_df.duplicated(keep=False)])

# Class distribution
class_counts = train_labels_df['label'].value_counts()
print("\nClass distribution:")
print(class_counts)

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=train_labels_df)
plt.title('Class Distribution (0: Non-Cancerous, 1: Cancerous)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Display sample images
def display_sample_images(df, image_dir, num_samples=3):
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*4, 8))
    for i, label in enumerate([0, 1]):
        label_df = df[df['label'] == label].sample(num_samples, random_state=42)
        for j, row in enumerate(label_df.itertuples()):
            img_path = os.path.join(image_dir, f'{row.id}.tif')
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].set_title(f'Label: {label}')
            axes[i, j].axis('off')
    plt.suptitle('Sample Images (Top: Non-Cancerous, Bottom: Cancerous)')
    plt.show()

display_sample_images(train_labels_df, train_images_dir)

# Image statistics
sample_image = Image.open(os.path.join(train_images_dir, f'{train_labels_df["id"].iloc[0]}.tif'))
print(f"\nSample image dimensions: {sample_image.size}")
print(f"Sample image mode: {sample_image.mode}")

# Pixel intensity distribution (histograms for multiple samples)
def plot_pixel_intensity(image_dir, sample_ids, title):
    fig, axes = plt.subplots(1, len(sample_ids), figsize=(4 * len(sample_ids), 5))
    if len(sample_ids) == 1: # Handle single subplot case
        axes = [axes]
    for i, img_id in enumerate(sample_ids):
        img_path = os.path.join(image_dir, f'{img_id}.tif')
        img = np.array(Image.open(img_path))
        axes[i].hist(img.flatten(), bins=50, color='skyblue', alpha=0.7)
        axes[i].set_title(f'Image {img_id[:5]}')
        axes[i].set_xlabel('Pixel Intensity')
        axes[i].set_ylabel('Frequency')
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

sample_ids = train_labels_df['id'].sample(3, random_state=42).values
plot_pixel_intensity(train_images_dir, sample_ids, 'Pixel Intensity Distribution for Sample Images')

# Check pixel value range (min/max)
min_pixel_values = []
max_pixel_values = []
sample_ids_check = train_labels_df['id'].sample(50, random_state=42).values # Check more samples for range
for img_id in sample_ids_check:
    img_path = os.path.join(train_images_dir, f'{img_id}.tif')
    img = np.array(Image.open(img_path))
    min_pixel_values.append(np.min(img))
    max_pixel_values.append(np.max(img))

print(f"\nMin pixel value across {len(sample_ids_check)} samples: {np.min(min_pixel_values)}")
print(f"Max pixel value across {len(sample_ids_check)} samples: {np.max(max_pixel_values)}")

# Summarize image similarity (simple approach: average pixel values or standard deviation)
# This is a very basic form of similarity, more advanced methods would use embeddings or perceptual hashing.
avg_pixel_values_by_label = defaultdict(list)
for index, row in train_labels_df.sample(100, random_state=42).iterrows(): # Sample 100 images for this
    img_path = os.path.join(train_images_dir, f'{row["id"]}.tif')
    img = np.array(Image.open(img_path))
    avg_pixel_values_by_label[row['label']].append(np.mean(img))

print("\nAverage pixel values for sample images by class:")
for label, values in avg_pixel_values_by_label.items():
    print(f"Label {label}: Mean={np.mean(values):.2f}, Std Dev={np.std(values):.2f}")

# Check image sizes (confirming they are all 96x96 as stated)
unique_image_sizes = set()
for i, img_id in enumerate(train_labels_df['id'].sample(100, random_state=42).values): # Check a subset
    img_path = os.path.join(train_images_dir, f'{img_id}.tif')
    with Image.open(img_path) as img:
        unique_image_sizes.add(img.size)
    if i > 100: # Limit check to prevent slow execution
        break

print(f"\nUnique image sizes observed: {unique_image_sizes}")
if len(unique_image_sizes) == 1 and list(unique_image_sizes)[0] == (96, 96):
    print("All sample images conform to the expected 96x96 dimensions.")
else:
    print("Image sizes vary or are not all 96x96.")


# Enhanced data transformations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), # Added vertical flip
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Added color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset
class CancerDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1] if 'label' in self.dataframe.columns else 0
        img_path = os.path.join(self.image_dir, f'{img_name}.tif')
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Split dataset
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_labels_df, test_size=0.2, stratify=train_labels_df['label'], random_state=42)

# Create datasets and loaders
train_dataset = CancerDataset(train_df, train_images_dir, transform=train_transforms)
val_dataset = CancerDataset(val_df, train_images_dir, transform=val_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self, num_filters=32, dropout_rate=0.5):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output size: 48x48

            nn.Conv2d(num_filters, num_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output size: 24x24

            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.Conv2d(num_filters*4, num_filters*4, kernel_size=3, padding=1), # Added another conv layer
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output size: 12x12

            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters*8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output size: 6x6
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 96, 96)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128), # Added another dense layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Hyperparameter tuning with early stopping
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, patience=3): # Increased patience
    train_losses, val_losses, val_accuracies, val_aucs = [], [], [], []
    best_auc, best_model = 0.0, None
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss, correct, total, all_preds, all_labels, all_probs = 0.0, 0, 0, [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_auc = roc_auc_score(all_labels, all_probs)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_aucs.append(val_auc)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
            # Reset patience counter only if AUC improves significantly (e.g., > 0.001)
            epochs_no_improve = 0 # Reset on any improvement
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation AUC for {patience} epochs.")
                break
    return train_losses, val_losses, val_accuracies, val_aucs, best_model, all_labels, all_preds


# Try different hyperparameters
hyperparams = [
    {'num_filters': 32, 'dropout_rate': 0.5, 'lr': 0.001, 'batch_size': 64, 'epochs': 15}, # Increased epochs for potentially better convergence
    {'num_filters': 64, 'dropout_rate': 0.4, 'lr': 0.0008, 'batch_size': 32, 'epochs': 15}, # Adjusted dropout and LR
    {'num_filters': 48, 'dropout_rate': 0.5, 'lr': 0.001, 'batch_size': 128, 'epochs': 15} # New set of parameters
]

results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Calculate class weights for imbalanced dataset
class_counts = train_labels_df['label'].value_counts()
# Weight for class 0 (non-cancerous) = Total samples / (2 * Count of class 0)
# Weight for class 1 (cancerous) = Total samples / (2 * Count of class 1)
# Here, I use max_count / count_of_each_class for simplicity in PyTorch's CrossEntropyLoss
weight_for_0 = class_counts[1] / class_counts[0]
weight_for_1 = 1.0 # Or class_counts[0] / class_counts[1] if 0 is the minority class. Here, 1 is the minority class.
# Given class 0 is majority (130908) and class 1 is minority (89117)
# I want to give higher weight to the minority class.
class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float).to(device)
# Alternatively, consider balanced weights: N_samples / (N_classes * N_samples_class_i)
total_samples = len(train_labels_df)
class_0_weight = total_samples / (2 * class_counts[0])
class_1_weight = total_samples / (2 * class_counts[1])
class_weights = torch.tensor([class_0_weight, class_1_weight], dtype=torch.float).to(device)
print(f"Calculated class weights: {class_weights.cpu().numpy()}")


for i, params in enumerate(hyperparams):
    print(f"\n--- Training with Hyperparameter Set {i+1}: {params} ---")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
    model = DeepCNN(num_filters=params['num_filters'], dropout_rate=params['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Apply class weights
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    train_losses, val_losses, val_accuracies, val_aucs, best_model, all_labels, all_preds = train_and_evaluate(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=params['epochs']
    )
    results.append({
        'params': params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_aucs': val_aucs,
        'best_model': best_model,
        'final_val_labels': all_labels,
        'final_val_preds': all_preds
    })


# Plot results
plt.figure(figsize=(18, 12))

for i, result in enumerate(results):
    params_str = ", ".join([f"{k}:{v}" for k, v in result['params'].items()])

    # Plot Training and Validation Loss
    plt.subplot(2, 3, 1)
    plt.plot(result['train_losses'], label=f"Train Loss (Set {i+1})")
    plt.plot(result['val_losses'], label=f"Val Loss (Set {i+1})")
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Validation Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(result['val_accuracies'], label=f"Val Accuracy (Set {i+1})")
    plt.title('Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Validation AUC
    plt.subplot(2, 3, 3)
    plt.plot(result['val_aucs'], label=f"Val AUC (Set {i+1})")
    plt.title('Validation AUC', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Display detailed results for each hyperparameter set
print("\n--- Detailed Results per Hyperparameter Set ---")
for i, result in enumerate(results):
    params = result['params']
    final_val_accuracy = result['val_accuracies'][-1] if result['val_accuracies'] else 'N/A'
    final_val_auc = result['val_aucs'][-1] if result['val_aucs'] else 'N/A'
    print(f"\nHyperparameter Set {i+1}:")
    print(f"  Parameters: {params}")
    print(f"  Final Validation Accuracy: {final_val_accuracy:.2f}%")
    print(f"  Final Validation AUC: {final_val_auc:.4f}")

    # Display Confusion Matrix and Classification Report for the last epoch's predictions
    if result['final_val_labels'] and result['final_val_preds']:
        cm = confusion_matrix(result['final_val_labels'], result['final_val_preds'])
        print("\n  Confusion Matrix:")
        print(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']).to_markdown())

        print("\n  Classification Report:")
        print(classification_report(result['final_val_labels'], result['final_val_preds'], target_names=['Non-Cancerous (0)', 'Cancerous (1)']))

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix for Set {i+1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

# Hyperparameter Optimization Summary
print("\n--- Hyperparameter Optimization Summary ---")
best_overall_auc = 0.0
best_overall_params = None
best_overall_model_state = None

for i, result in enumerate(results):
    current_best_auc_for_set = max(result['val_aucs'])
    print(f"Set {i+1} (Params: {result['params']}): Max Val AUC = {current_best_auc_for_set:.4f}")
    if current_best_auc_for_set > best_overall_auc:
        best_overall_auc = current_best_auc_for_set
        best_overall_params = result['params']
        best_overall_model_state = result['best_model']

print(f"\nBest performing model parameters: {best_overall_params}")
print(f"Achieved best Validation AUC: {best_overall_auc:.4f}")


# Create submission file
from torch.utils.data import DataLoader

best_result = max(results, key=lambda x: max(x['val_aucs']))  # Choose model with highest AUC
best_model_state = best_result['best_model']
model = DeepCNN(num_filters=best_result['params']['num_filters'],
                dropout_rate=best_result['params']['dropout_rate']).to(device)
model.load_state_dict(best_model_state)
model.eval()

test_df = pd.DataFrame({'id': [f.split('.tif')[0] for f in os.listdir(test_images_dir)]})
test_dataset = CancerDataset(test_df, test_images_dir, transform=val_transforms) # Using val_transforms for test set
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Make predictions
predictions = []
image_ids = test_df['id'].values

with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability of class 1
        predictions.extend(probs)

submission_df = pd.DataFrame({'id': image_ids, 'label': predictions})

submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")

