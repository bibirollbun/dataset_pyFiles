import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

train=pd.read_csv('/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/Train_ISBI2025.csv')
test=pd.read_csv('/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/Test_ISBI2025.csv')


train['label'].value_counts()


# Add image path column with the correct path
def add_image_path(row):
    return os.path.join('/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/isbi2025-ps3c-train-dataset', row['image_name'])

train['image_path'] = train.apply(add_image_path, axis=1)


import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Get unique classes
classes = train["label"].unique()

# Number of samples per class
num_samples = 5

# Create a figure to display images
plt.figure(figsize=(15, len(classes) * 3))

for row_idx, class_name in enumerate(classes):
    # Sample 5 images from the current class
    sample_images = train[train["label"] == class_name].sample(n=num_samples, random_state=42)
    
    for col_idx, (_, row) in enumerate(sample_images.iterrows()):
        img_path = row["image_path"]  # Path to the image
        label = row["label"]  # Class label
        
        # Load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Plot the image
        plt.subplot(len(classes), num_samples, row_idx * num_samples + col_idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Class: {label}", fontsize=10)

# Display the images
plt.tight_layout()
plt.show()



import pandas as pd

# Assuming train_df is your DataFrame and it has a column named "label"
train = train[train['label'] != 'bothcells']



train['label'].value_counts()


train.head()


import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Train-test split and label mapping code remains the same
train_df, val_df = train_test_split(
    train, 
    test_size=0.2,
    stratify=train['label'],
    random_state=42
)

label_to_idx = {label: idx for idx, label in enumerate(sorted(train['label'].unique()))}

# Oversample unhealthy class
unhealthy_class = train_df[train_df['label'] == 'unhealthy']
augmented_unhealthy = resample(
    unhealthy_class,
    replace=True,
    n_samples=10 * len(unhealthy_class),
    random_state=42
)

balanced_train_df = pd.concat([train_df, augmented_unhealthy])

# Define transforms
train_transform = A.Compose([
    A.Resize(224, 224),
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class ImageDataset(Dataset):
    def __init__(self, dataframe, label_to_idx, transform=None):
        self.dataframe = dataframe
        self.label_to_idx = label_to_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = self.dataframe.iloc[idx]['image_path']
        
        # Error handling for missing files
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found.")
        
        # Read image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get label
        label = self.dataframe.iloc[idx]['label']
        label_idx = self.label_to_idx.get(label, -1)
        
        return image, torch.tensor(label_idx, dtype=torch.long)

# Create datasets
train_dataset = ImageDataset(
    balanced_train_df, 
    label_to_idx, 
    transform=train_transform
)

val_dataset = ImageDataset(
    val_df, 
    label_to_idx, 
    transform=val_transform
)

# Create DataLoaders
BATCH_SIZE = 128
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Verify the data
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of unique classes: {len(label_to_idx)}")
print("\nLabel to Index Mapping:")
for label, idx in label_to_idx.items():
    print(f"{label}: {idx}")

# Verify the first batch
for images, labels in train_loader:
    print("\nFirst batch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")
    break


assert train_dataset.label_to_idx == val_dataset.label_to_idx, "Label encodings for train and validation sets do not match!"



import matplotlib.pyplot as plt

# Count the occurrences of each class in the train_df
class_counts = balanced_train_df['label'].value_counts()

# Plot the histogram
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Class Distribution in train_df')
plt.xlabel('Class Labels')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, f1_score

def train_model_with_tqdm(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses, val_losses = [], []
    val_accuracies = []
    val_f1_scores = []

    print("Starting training with progress tracking...\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=True)
        
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=f"{running_loss / len(train_loader):.4f}")
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=True)

        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                # Calculate validation accuracy and collect predictions/labels for metrics
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracy = correct / total * 100
        val_accuracies.append(val_accuracy)

        # Calculate confusion matrix and F1 score
        cm = confusion_matrix(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        val_f1_scores.append(f1)

        scheduler.step(epoch_val_loss)  # Use validation loss for scheduler

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()  # Save best model state
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= 6:  # Early stopping patience
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            model.load_state_dict(best_model_state)  # Restore the best model state
            break

        print(f"Epoch {epoch + 1} complete.")
        print(f"Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Validation Confusion Matrix:\n{cm}")
        print(f"Validation F1-Score: {f1:.4f}\n")

    print("Training complete.")
    return model, train_losses, val_losses, val_accuracies



import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Learning rate scheduler
class CustomReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.1, patience=3, cooldown=2, min_lr=1e-6, verbose=False):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')
        self.bad_epochs = 0
        self.cooldown_counter = 0

    def step(self, val_loss):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs > self.patience:
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                if new_lr < param_group['lr']:
                    param_group['lr'] = new_lr
                    self.cooldown_counter = self.cooldown
                    if self.verbose:
                        print(f"Reducing learning rate to {new_lr:.6f}")
            self.bad_epochs = 0

        if self.verbose:
            current_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            print(f"Current learning rates: {current_lrs}")



import torch
import torch.nn as nn
from torchvision import models

class CustomResNeXt(nn.Module):
    def __init__(self, model_name, num_classes, cardinality=32, depth=29, width_per_group=4):
        """
        Args:
            model_name (str): Name of the ResNeXt model (e.g., 'resnext50_32x4d').
            num_classes (int): Number of output classes.
            cardinality (int): Number of groups in ResNeXt. Default is 32 for ResNeXt-50.
            depth (int): Depth of the ResNeXt model.
            width_per_group (int): Width per group.
        """
        super(CustomResNeXt, self).__init__()

        # Load ResNeXt model with pretrained weights
        if model_name == 'resnext50_32x4d':
            self.resnext = models.resnext50_32x4d(pretrained=True)
        elif model_name == 'resnext101_32x8d':
            self.resnext = models.resnext101_32x8d(pretrained=True)
        else:
            raise ValueError(f"Model {model_name} not supported")

        # Freeze all layers initially
        for param in self.resnext.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 4 blocks (you may adjust the number of blocks based on the model architecture)
        num_blocks = len(self.resnext.layer4)
        
        # Unfreeze the last 4 blocks
        for param in self.resnext.layer4.parameters():
            param.requires_grad = True
        
        # Unfreeze the final fully connected layer (fc)
        for param in self.resnext.fc.parameters():
            param.requires_grad = True

        # Store the number of input features from the original fully connected layer
        self.num_features = self.resnext.fc.in_features
        
        # Replace the original fully connected layer
        self.resnext.fc = nn.Identity()

        # Custom layers to replace the fully connected layer
        self.custom_layers = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Pass input through the ResNeXt backbone layers
        x = self.resnext(x)
        
        # Pass the output through the custom layers
        x = self.custom_layers(x)
        
        return x



# # Initialize model, criterion, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = len(train['label'].unique())
# model = CustomResNeXt(model_name='resnext101_32x8d', num_classes=3)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
# scheduler = CustomReduceLROnPlateau(optimizer)

# # Example usage remains the same
# model, train_losses, val_losses, val_accuracies = train_model_with_tqdm(
#     model, 
#     train_loader, 
#     val_loader, 
#     criterion, 
#     optimizer, 
#     scheduler,
#     num_epochs=30
# )

# # Plot training history
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()



# # Inference Code on Test Dataset using the trained model

# import torch
# import numpy as np
# import random
# from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
# import pandas as pd
# import os
# from PIL import Image
# from torch.utils.data import Dataset

# # Change working directory to '/kaggle/working'
# os.chdir(r'/kaggle/working')

# # Save the model weights (after training)
# # Assume 'model' is your trained model
# model_save_path = 'ResNeXt101.pth'
# torch.save(model.state_dict(), model_save_path)

# print(f"Model weights saved at {model_save_path}")

# # Set random seeds for reproducibility
# def set_seeds(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# class TestImageDataset(Dataset):
#     def __init__(self, dataframe, transform=None):
#         self.dataframe = dataframe
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.dataframe)
    
#     def __getitem__(self, idx):
#         img_path = self.dataframe.iloc[idx]['image_path']
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"Image file {img_path} not found.")
        
#         with Image.open(img_path) as img:
#             image = img.convert('RGB')
        
#         if self.transform:
#             random.seed(idx)
#             torch.manual_seed(idx)
#             image = self.transform(image)
            
#         return image

# def predict(model, dataloader, device='cuda', idx_to_label=None):
#     if idx_to_label is None:
#         raise ValueError("idx_to_label mapping must be provided")
        
#     model.eval()
#     predictions = []
    
#     with torch.no_grad():
#         for batch_idx, images in enumerate(tqdm(dataloader, desc="Making Predictions")):
#             images = images.to(device)
#             outputs = model(images)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             _, predicted_idxs = torch.max(probs, 1)
#             batch_predictions = [idx_to_label[idx.item()] for idx in predicted_idxs]
#             predictions.extend(batch_predictions)
    
#     return predictions

# def main(weights_path, model, output_path, test_csv_path, test_images_path, transform, batch_size):
#     # Set seeds for reproducibility
#     set_seeds(42)
    
#     # Load test data
#     test_df = pd.read_csv(test_csv_path)
#     test_df['image_path'] = test_df['image_name'].apply(
#         lambda x: os.path.join(test_images_path, x)
#     )
    
#     # Create dataset and dataloader
#     test_dataset = TestImageDataset(test_df, transform=transform)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True,
#         worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
#     )
    
#     # Load model and weights
#     model.load_state_dict(torch.load(weights_path, map_location='cuda', weights_only=True))
#     model = model.to('cuda')
    
#     # Define label mapping
#     LABEL_ENCODING = {'healthy': 0, 'rubbish': 1, 'unhealthy': 2}
#     IDX_TO_LABEL = {v: k for k, v in LABEL_ENCODING.items()}
    
#     # Get predictions
#     predictions = predict(model, test_loader, device='cuda', idx_to_label=IDX_TO_LABEL)
    
#     # Save predictions
#     test_df['label'] = predictions
#     output_df = test_df[['image_name', 'label']]
#     output_df.to_csv(output_path, index=False)
    
#     print(f"Predictions saved to '{output_path}'")
#     return output_df

# weights_path = '/kaggle/working/ResNeXt101.pth'
# test_csv_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/Test_ISBI2025.csv'
# test_images_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/isbi2025-ps3c-test-dataset'
# output_path = 'ResNeXt101.csv'
# batch_size = 128
# val_transform = val_transform

# # Replace with your custom model class
# model = CustomResNeXt(model_name='resnext101_32x8d', num_classes=3)

# test_df = main(weights_path, model, output_path, test_csv_path, test_images_path, val_transform, batch_size)



import torch
import torch.nn as nn
import timm
import torch.nn.functional as F  # For functional API

class CustomSEResNeXt50(nn.Module):
    def __init__(self, num_classes):
        super(CustomSEResNeXt50, self).__init__()
        
        # Load a pretrained SE-ResNeXt50 model from timm
        self.resnet = timm.create_model('seresnext50_32x4d', pretrained=True)
        
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last two blocks (layer3 and layer4) for fine-tuning
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Remove the original classification head by setting it to Identity
        self.resnet.fc = nn.Identity()
        
        # Get the number of features from the backbone
        self.num_features = self.resnet.num_features
        
        # Create new classification layers with a single 512-unit layer
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features using the SE-ResNeXt backbone
        features = self.resnet.forward_features(x)
        
        # If features have spatial dimensions, apply global average pooling
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # Flatten the pooled features to shape [B, C]
        features = features.flatten(1)
        
        # Pass extracted features through the custom classifier
        x = self.classifier(features)
        
        return x



# # Initialize model, criterion, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = len(train['label'].unique())
# model = CustomSEResNeXt50(num_classes).to(device)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
# scheduler = CustomReduceLROnPlateau(optimizer)

# # Example usage remains the same
# model, train_losses, val_losses, val_accuracies = train_model_with_tqdm(
#     model, 
#     train_loader, 
#     val_loader, 
#     criterion, 
#     optimizer, 
#     scheduler,
#     num_epochs=10 # Increase number of epochs for deeper learning
# )

# # Plot training history
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()



# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from tqdm.notebook import tqdm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import random

# # Change working directory to '/kaggle/working'
# os.chdir(r'/kaggle/working')

# # Save the model weights (after training)
# # Assume 'model' is your trained model
# model_save_path = 'CustomSEResNeXt50.pth'
# torch.save(model.state_dict(), model_save_path)

# print(f"Model weights saved at {model_save_path}")

# # Set random seeds for reproducibility
# def set_seeds(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # Test dataset class
# class TestImageDataset(Dataset):
#     def __init__(self, dataframe, transform=None):
#         self.data = dataframe
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         # Get image path directly from dataframe
#         img_path = self.data['image_path'].iloc[idx]
        
#         # Load the image
#         image = Image.open(img_path).convert("RGB")
        
#         # Convert PIL image to NumPy array
#         image = np.array(image)
        
#         # Apply transformations
#         if self.transform:
#             transformed = self.transform(image=image)
#             image = transformed["image"]

#         return image
# # Prediction function
# def predict(model, dataloader, device='cuda', idx_to_label=None):
#     if idx_to_label is None:
#         raise ValueError("idx_to_label mapping must be provided")
        
#     model.eval()
#     predictions = []
    
#     with torch.no_grad():
#         for batch_idx, images in enumerate(tqdm(dataloader, desc="Making Predictions")):
#             images = images.to(device)
#             outputs = model(images)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             _, predicted_idxs = torch.max(probs, 1)
#             batch_predictions = [idx_to_label[idx.item()] for idx in predicted_idxs]
#             predictions.extend(batch_predictions)
    
#     return predictions

# # Main inference function
# def main(weights_path, model, output_path, test_csv_path, test_images_path, transform, batch_size):
#     # Set seeds for reproducibility
#     set_seeds(42)
    
#     # Load test data
#     test_df = pd.read_csv(test_csv_path)
#     test_df['image_path'] = test_df['image_name'].apply(
#         lambda x: os.path.join(test_images_path, x)
#     )
    
#     # Create dataset and dataloader
#     test_dataset = TestImageDataset(test_df, transform=transform)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True,
#         worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
#     )
    
#     # Load model and weights
#     model.load_state_dict(torch.load(weights_path, map_location='cuda', weights_only=True))
#     model = model.to('cuda')
    
#     # Define label mapping
#     LABEL_ENCODING = {'healthy': 0, 'rubbish': 1, 'unhealthy': 2}
#     IDX_TO_LABEL = {v: k for k, v in LABEL_ENCODING.items()}
    
#     # Get predictions
#     predictions = predict(model, test_loader, device='cuda', idx_to_label=IDX_TO_LABEL)
    
#     # Save predictions
#     test_df['label'] = predictions
#     output_df = test_df[['image_name', 'label']]
#     output_df.to_csv(output_path, index=False)
    
#     print(f"Predictions saved to '{output_path}'")
#     return output_df

# # Test image transforms
# val_transform = A.Compose([
#     A.Resize(224, 224),
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensorV2(),
# ])

# # Paths
# weights_path = '/kaggle/working/CustomSEResNeXt50.pth'
# test_csv_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/Test_ISBI2025.csv'
# test_images_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/isbi2025-ps3c-test-dataset'
# output_path = 'CustomSEResNeXt50.csv'
# batch_size = 64
# val_transform = val_transform

# # Replace with your custom model class
# model = CustomSEResNeXt50(num_classes).to(device)

# test_df = main(weights_path, model, output_path, test_csv_path, test_images_path, val_transform, batch_size)



import torch
import torch.nn as nn
import timm
import torch.nn.functional as F  # For functional API

class CustomSEResNeXt101(nn.Module):
    def __init__(self, num_classes):
        super(CustomSEResNeXt101, self).__init__()
        
        # Load a pretrained SE-ResNeXt50 model from timm
        self.resnet = timm.create_model('seresnext101_32x4d', pretrained=True)
        
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last two blocks (layer3 and layer4) for fine-tuning
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Remove the original classification head by setting it to Identity
        self.resnet.fc = nn.Identity()
        
        # Get the number of features from the backbone
        self.num_features = self.resnet.num_features
        
        # Create new classification layers with a single 512-unit layer
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features using the SE-ResNeXt backbone
        features = self.resnet.forward_features(x)
        
        # If features have spatial dimensions, apply global average pooling
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # Flatten the pooled features to shape [B, C]
        features = features.flatten(1)
        
        # Pass extracted features through the custom classifier
        x = self.classifier(features)
        
        return x



# # Initialize model, criterion, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = len(train['label'].unique())
# model = CustomSEResNeXt101(num_classes).to(device)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
# scheduler = CustomReduceLROnPlateau(optimizer)

# # Example usage remains the same
# model, train_losses, val_losses, val_accuracies = train_model_with_tqdm(
#     model, 
#     train_loader, 
#     val_loader, 
#     criterion, 
#     optimizer, 
#     scheduler,
#     num_epochs=10
# )

# # Plot training history
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()



# import torch
# import numpy as np
# import random
# from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
# import pandas as pd
# import os
# from PIL import Image
# from torch.utils.data import Dataset

# # Change working directory to '/kaggle/working'
# os.chdir(r'/kaggle/working')

# # Save the model weights (after training)
# # Assume 'model' is your trained model
# model_save_path = 'CustomSEResNeXt101.pth'
# torch.save(model.state_dict(), model_save_path)

# print(f"Model weights saved at {model_save_path}")

# # Set random seeds for reproducibility
# def set_seeds(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # Test dataset class
# class TestImageDataset(Dataset):
#     def __init__(self, dataframe, transform=None):
#         self.data = dataframe
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         # Get image path directly from dataframe
#         img_path = self.data['image_path'].iloc[idx]
        
#         # Load the image
#         image = Image.open(img_path).convert("RGB")
        
#         # Convert PIL image to NumPy array
#         image = np.array(image)
        
#         # Apply transformations
#         if self.transform:
#             transformed = self.transform(image=image)
#             image = transformed["image"]

#         return image
# # Prediction function
# def predict(model, dataloader, device='cuda', idx_to_label=None):
#     if idx_to_label is None:
#         raise ValueError("idx_to_label mapping must be provided")
        
#     model.eval()
#     predictions = []
    
#     with torch.no_grad():
#         for batch_idx, images in enumerate(tqdm(dataloader, desc="Making Predictions")):
#             images = images.to(device)
#             outputs = model(images)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             _, predicted_idxs = torch.max(probs, 1)
#             batch_predictions = [idx_to_label[idx.item()] for idx in predicted_idxs]
#             predictions.extend(batch_predictions)
    
#     return predictions

# # Main inference function
# def main(weights_path, model, output_path, test_csv_path, test_images_path, transform, batch_size):
#     # Set seeds for reproducibility
#     set_seeds(42)
    
#     # Load test data
#     test_df = pd.read_csv(test_csv_path)
#     test_df['image_path'] = test_df['image_name'].apply(
#         lambda x: os.path.join(test_images_path, x)
#     )
    
#     # Create dataset and dataloader
#     test_dataset = TestImageDataset(test_df, transform=transform)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True,
#         worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
#     )
    
#     # Load model and weights
#     model.load_state_dict(torch.load(weights_path, map_location='cuda', weights_only=True))
#     model = model.to('cuda')
    
#     # Define label mapping
#     LABEL_ENCODING = {'healthy': 0, 'rubbish': 1, 'unhealthy': 2}
#     IDX_TO_LABEL = {v: k for k, v in LABEL_ENCODING.items()}
    
#     # Get predictions
#     predictions = predict(model, test_loader, device='cuda', idx_to_label=IDX_TO_LABEL)
    
#     # Save predictions
#     test_df['label'] = predictions
#     output_df = test_df[['image_name', 'label']]
#     output_df.to_csv(output_path, index=False)
    
#     print(f"Predictions saved to '{output_path}'")
#     return output_df

# # Test image transforms
# val_transform = A.Compose([
#     A.Resize(224, 224),
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensorV2(),
# ])

# weights_path = '/kaggle/working/CustomSEResNeXt101.pth'
# test_csv_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/Test_ISBI2025.csv'
# test_images_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/isbi2025-ps3c-test-dataset'
# output_path = 'CustomSEResNeXt101.csv'
# batch_size = 64
# val_transform = val_transform

# # Replace with your custom model class
# model = CustomSEResNeXt101(num_classes).to(device)

# test_df = main(weights_path, model, output_path, test_csv_path, test_images_path, val_transform, batch_size)



import torch
import torch.nn as nn
import torchvision.models as models

class CustomWideResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomWideResNet, self).__init__()
        
        # Load a pretrained WideResNet50_2 model from torchvision
        self.resnet = models.wide_resnet50_2(pretrained=True)
        
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last two blocks (layer3 and layer4) for fine-tuning
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Obtain the number of input features for the original fc layer
        num_features = self.resnet.fc.in_features
        
        # Remove the original classification head by replacing it with Identity
        self.resnet.fc = nn.Identity()
        
        # Create new classification layers with a single 512-unit layer
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Pass input through the modified WideResNet backbone.
        # Because we replaced fc with Identity, this returns the flattened features.
        features = self.resnet(x)
        
        # Pass extracted features through the custom classifier
        x = self.classifier(features)
        
        return x



# # Initialize model, criterion, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = len(train['label'].unique())
# model = CustomWideResNet(num_classes).to(device)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
# scheduler = CustomReduceLROnPlateau(optimizer)

# # Example usage remains the same
# model, train_losses, val_losses, val_accuracies = train_model_with_tqdm(
#     model, 
#     train_loader, 
#     val_loader, 
#     criterion, 
#     optimizer, 
#     scheduler,
#     num_epochs=10
# )

# # Plot training history
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()


# import torch
# import numpy as np
# import random
# from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
# import pandas as pd
# import os
# from PIL import Image
# from torch.utils.data import Dataset

# # Change working directory to '/kaggle/working'
# os.chdir(r'/kaggle/working')

# # Save the model weights (after training)
# # Assume 'model' is your trained model
# model_save_path = 'CustomWideResNet50.pth'
# torch.save(model.state_dict(), model_save_path)

# print(f"Model weights saved at {model_save_path}")

# # Set random seeds for reproducibility
# def set_seeds(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # Test dataset class
# class TestImageDataset(Dataset):
#     def __init__(self, dataframe, transform=None):
#         self.data = dataframe
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         # Get image path directly from dataframe
#         img_path = self.data['image_path'].iloc[idx]
        
#         # Load the image
#         image = Image.open(img_path).convert("RGB")
        
#         # Convert PIL image to NumPy array
#         image = np.array(image)
        
#         # Apply transformations
#         if self.transform:
#             transformed = self.transform(image=image)
#             image = transformed["image"]

#         return image
# # Prediction function
# def predict(model, dataloader, device='cuda', idx_to_label=None):
#     if idx_to_label is None:
#         raise ValueError("idx_to_label mapping must be provided")
        
#     model.eval()
#     predictions = []
    
#     with torch.no_grad():
#         for batch_idx, images in enumerate(tqdm(dataloader, desc="Making Predictions")):
#             images = images.to(device)
#             outputs = model(images)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             _, predicted_idxs = torch.max(probs, 1)
#             batch_predictions = [idx_to_label[idx.item()] for idx in predicted_idxs]
#             predictions.extend(batch_predictions)
    
#     return predictions

# # Main inference function
# def main(weights_path, model, output_path, test_csv_path, test_images_path, transform, batch_size):
#     # Set seeds for reproducibility
#     set_seeds(42)
    
#     # Load test data
#     test_df = pd.read_csv(test_csv_path)
#     test_df['image_path'] = test_df['image_name'].apply(
#         lambda x: os.path.join(test_images_path, x)
#     )
    
#     # Create dataset and dataloader
#     test_dataset = TestImageDataset(test_df, transform=transform)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True,
#         worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
#     )
    
#     # Load model and weights
#     model.load_state_dict(torch.load(weights_path, map_location='cuda', weights_only=True))
#     model = model.to('cuda')
    
#     # Define label mapping
#     LABEL_ENCODING = {'healthy': 0, 'rubbish': 1, 'unhealthy': 2}
#     IDX_TO_LABEL = {v: k for k, v in LABEL_ENCODING.items()}
    
#     # Get predictions
#     predictions = predict(model, test_loader, device='cuda', idx_to_label=IDX_TO_LABEL)
    
#     # Save predictions
#     test_df['label'] = predictions
#     output_df = test_df[['image_name', 'label']]
#     output_df.to_csv(output_path, index=False)
    
#     print(f"Predictions saved to '{output_path}'")
#     return output_df

# # Test image transforms
# val_transform = A.Compose([
#     A.Resize(224, 224),
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensorV2(),
# ])

# weights_path = '/kaggle/working/CustomWideResNet50.pth'
# test_csv_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/Test_ISBI2025.csv'
# test_images_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/isbi2025-ps3c-test-dataset'
# output_path = 'CustomWideResNet50.csv'
# batch_size = 64
# val_transform = val_transform

# # Replace with your custom model class
# model = CustomWideResNet(num_classes).to(device)

# test_df = main(weights_path, model, output_path, test_csv_path, test_images_path, val_transform, batch_size)



import torch
import torch.nn as nn
import torchvision.models as models

class CustomWideResNet101(nn.Module):
    def __init__(self, num_classes):
        super(CustomWideResNet101, self).__init__()
        
        # Load a pretrained WideResNet101_2 model from torchvision
        self.resnet = models.wide_resnet101_2(pretrained=True)
        
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last two blocks (layer3 and layer4) for fine-tuning
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Obtain the number of input features for the original fc layer
        num_features = self.resnet.fc.in_features
        
        # Remove the original classification head by replacing it with Identity
        self.resnet.fc = nn.Identity()
        
        # Create new classification layers with a single 512-unit layer
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Pass input through the modified WideResNet101_2 backbone.
        features = self.resnet(x)
        
        # Pass extracted features through the custom classifier
        x = self.classifier(features)
        
        return x



# # Initialize model, criterion, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = len(train['label'].unique())
# model = CustomWideResNet101(num_classes).to(device)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
# scheduler = CustomReduceLROnPlateau(optimizer)

# # Example usage remains the same
# model, train_losses, val_losses, val_accuracies = train_model_with_tqdm(
#     model, 
#     train_loader, 
#     val_loader, 
#     criterion, 
#     optimizer, 
#     scheduler,
#     num_epochs=30
# )

# # Plot training history
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()


# import torch
# import numpy as np
# import random
# from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
# import pandas as pd
# import os
# from PIL import Image
# from torch.utils.data import Dataset

# # Change working directory to '/kaggle/working'
# os.chdir(r'/kaggle/working')

# # Save the model weights (after training)
# # Assume 'model' is your trained model
# model_save_path = 'CustomWideResNet101.pth'
# torch.save(model.state_dict(), model_save_path)

# print(f"Model weights saved at {model_save_path}")

# # Set random seeds for reproducibility
# def set_seeds(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # Test dataset class
# class TestImageDataset(Dataset):
#     def __init__(self, dataframe, transform=None):
#         self.data = dataframe
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         # Get image path directly from dataframe
#         img_path = self.data['image_path'].iloc[idx]
        
#         # Load the image
#         image = Image.open(img_path).convert("RGB")
        
#         # Convert PIL image to NumPy array
#         image = np.array(image)
        
#         # Apply transformations
#         if self.transform:
#             transformed = self.transform(image=image)
#             image = transformed["image"]

#         return image
# # Prediction function
# def predict(model, dataloader, device='cuda', idx_to_label=None):
#     if idx_to_label is None:
#         raise ValueError("idx_to_label mapping must be provided")
        
#     model.eval()
#     predictions = []
    
#     with torch.no_grad():
#         for batch_idx, images in enumerate(tqdm(dataloader, desc="Making Predictions")):
#             images = images.to(device)
#             outputs = model(images)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             _, predicted_idxs = torch.max(probs, 1)
#             batch_predictions = [idx_to_label[idx.item()] for idx in predicted_idxs]
#             predictions.extend(batch_predictions)
    
#     return predictions

# # Main inference function
# def main(weights_path, model, output_path, test_csv_path, test_images_path, transform, batch_size):
#     # Set seeds for reproducibility
#     set_seeds(42)
    
#     # Load test data
#     test_df = pd.read_csv(test_csv_path)
#     test_df['image_path'] = test_df['image_name'].apply(
#         lambda x: os.path.join(test_images_path, x)
#     )
    
#     # Create dataset and dataloader
#     test_dataset = TestImageDataset(test_df, transform=transform)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True,
#         worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
#     )
    
#     # Load model and weights
#     model.load_state_dict(torch.load(weights_path, map_location='cuda', weights_only=True))
#     model = model.to('cuda')
    
#     # Define label mapping
#     LABEL_ENCODING = {'healthy': 0, 'rubbish': 1, 'unhealthy': 2}
#     IDX_TO_LABEL = {v: k for k, v in LABEL_ENCODING.items()}
    
#     # Get predictions
#     predictions = predict(model, test_loader, device='cuda', idx_to_label=IDX_TO_LABEL)
    
#     # Save predictions
#     test_df['label'] = predictions
#     output_df = test_df[['image_name', 'label']]
#     output_df.to_csv(output_path, index=False)
    
#     print(f"Predictions saved to '{output_path}'")
#     return output_df

# # Test image transforms
# val_transform = A.Compose([
#     A.Resize(224, 224),
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensorV2(),
# ])

# weights_path = '/kaggle/working/CustomWideResNet101.pth'
# test_csv_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/Test_ISBI2025.csv'
# test_images_path = '/kaggle/input/isbi-pap-smear-cell-classification-challenge/ISBI Pap Smear Cell Classification Challenge/isbi2025-ps3c-test-dataset'
# output_path = 'CustomWideResNet101.csv'
# batch_size = 64
# val_transform = val_transform

# # Replace with your custom model class
# model = CustomWideResNet101(num_classes).to(device)

# test_df = main(weights_path, model, output_path, test_csv_path, test_images_path, val_transform, batch_size)





