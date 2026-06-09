# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/open-data-day-2025-dates-types-classification/train'):
    print('Number of entries in train data :' , len(filenames))

for dirname, _, filenames in os.walk('/kaggle/input/open-data-day-2025-dates-types-classification/test'):
    print('Number of entries in test data :' , len(filenames))



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_labels_df = pd.read_csv('/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv')
train_labels_df.label.value_counts()


# new_train_labels_path
import os
import pandas as pd

# Define the path to your parent folder
parent_folder = "/kaggle/input/new-data/New Data"

# Initialize an empty list to store the file data
data = []

# Iterate through each subfolder in the parent folder
for label in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, label)
    if os.path.isdir(subfolder_path):
        # For each file in the subfolder
        for file in os.listdir(subfolder_path):
            # Check if the file is an image (optional: you can adjust extensions)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append([file, label])

# Create a DataFrame
df = pd.DataFrame(data, columns=['filename', 'label'])

# TEST : otal -> 76 row
#df.label.value_counts()

# Save the DataFrame to a CSV file
csv_path = "new_train_labels_path.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file has been saved to {csv_path}")



# new_train_dir
import os
import shutil

# Define the source parent folder and the destination folder
source_folder = "/kaggle/input/new-data/New Data"
dest_folder = "new_train_dir"

# Create the destination folder if it doesn't exist
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Loop through each subfolder in the source folder
for label in os.listdir(source_folder):
    subfolder_path = os.path.join(source_folder, label)
    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            # Check if the file is an image (adjust extensions as needed)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                source_file = os.path.join(subfolder_path, file)
                dest_file = os.path.join(dest_folder, file)
                shutil.copy2(source_file, dest_file)

print("All images have been copied to", dest_folder)



# final_train_labels_path.csv
import pandas as pd

# Read the two CSV files
df1 = pd.read_csv('/kaggle/working/new_train_labels_path.csv')
df2 = pd.read_csv('/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv')

# Combine the dataframes
final_df = pd.concat([df1, df2], ignore_index=True)

# Reorder columns to ensure they are ['filename', 'label']
final_df = final_df[['filename', 'label']]

# Save the final dataframe to a CSV file
final_df.to_csv('final_train_labels_path.csv', index=False)

print("Final CSV file created with", len(final_df), "rows.")


# final_train_dir
import os
import shutil

# Define your source directories (update these paths accordingly)
source_dir1 = "/kaggle/input/open-data-day-2025-dates-types-classification/train"   # Directory with 432 images
source_dir2 = "/kaggle/working/new_train_dir"  # Directory with 76 images

# Define the destination directory
destination_dir = "final_train_dir"

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List of source directories
source_dirs = [source_dir1, source_dir2]

# Copy image files from each source directory into the destination directory
for src_dir in source_dirs:
    for filename in os.listdir(src_dir):
        # Check if the file is an image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(destination_dir, filename)
            # Copy the file to the destination directory
            shutil.copy2(src_file, dest_file)

print(f"All files have been copied to '{destination_dir}'.")



more_than_one_TRAIN = [
    '73b05b0f.jpg',
    '7568324d.jpg',
    '56fa4cc4.jpg',
    '92ce537b.png',
    'feda0fb2.jpg',
    '3c8e617a.jpg',
    'eb6cf482.jpg',
    'a1d3ec0f.jpg',
    'f10703e9.png',
    'bdaed8f6.jpg',
    'bf3fad35.jpg',
    'e17feef7.jpg',
    '04cdb8ab.jpg',
    '9cb951b7.jpg',
    '55262933.jpg'
]

print(len(more_than_one_TRAIN))
# Select the first 10 images from the list
selected_files = more_than_one_TRAIN[:15]

# Create a grid of subplots: 2 rows x 5 columns
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 6))

# Loop over the selected files and display each image in the grid
for ax, file in zip(axes.flatten(), selected_files):
    img_path = '/kaggle/input/open-data-day-2025-dates-types-classification/train/' + file
    #print(img_path)
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')  # Hide axes

plt.tight_layout()
plt.show()


more_than_one_TEST = ['0b59e0e8.jpg',
 'ff70ccdd.jpg',
 'bdc24c08.jpg',
 'fd4cfbda.png',
 '39a7f607.jpg',
 '13ea1bc3.png',
 '2161559c.png',
 '1febe1ee.png',
 '3065c632.png',
 'b7e512b0.png',
 '7b5c08aa.png',
 '01365354.png',
 'a9a92b90.png',
 'ff79ca51.png',
 '68e583f1.jpg',
 '86bfd634.png',
 'f5b6aed6.jpg',
 '5b8792cc.png',
 '0138dbd9.png',
 'c21e139d.png',
 '0d11422b.png',
 '79418931.jpg',
 '7ff9f8f1.jpg',
 'f53df200.png',
 '7179631f.png',
 '8a8a7e05.png',
 '9e676438.jpg',
 '0fc62b58.jpg',
 'bdcb7414.jpg',
 '7e201c3a.png',
 '41e5cda5.png',
 '9066e26f.jpg',
 'ab31ec82.jpg',
 'e7bad807.png',
 'aee05a0a.png',
 '2c866c3e.png',
 '080bd6a6.png',
 '887ab7e6.jpg',
 '654e9c05.png',
 '0c3e33ce.jpg',
 '76fc1521.png',
 '0eff9f76.png',
 'f251aac1.jpg',
 '3cac486e.png',
 '71016962.png',
 '2461ccb7.png',
 '738f535d.png',
 '1cc1867e.jpg',
 'f2fc4e83.jpg',
 '6b23a97a.png',
 '28728d90.jpg',
 '5431f0d5.png',
 '57ac35c5.png',
 'bb76314f.png',
 'dd1821c4.png',
 'd48d1717.jpg',
 'ae9e2470.jpg',
 '89930623.jpg',
 'e9a900ab.png',
 '01bbf2a0.jpg',
 '291c8c1c.jpg',
 'dedc060b.jpg',
 '1beec184.jpg',
 'e1b793bf.jpg',
 'e2671fe2.jpg',
 '76b600a4.png',
 '0eae02a8.png',
 '21c321ea.png',
 'f38338af.png',
 '2c496a0c.png',
 '0e3f2e72.jpg',
 '18b225c1.png',
 'e847ca37.png',
 '1c529c39.jpg',
 '5e994df7.jpg',
 '42489bf6.jpg',
 '35b1d162.png',
 '6a551b31.png',
 '65c456d3.jpg',
 '3598e77f.jpg',
 '19e2536a.jpg',
 '0582db09.jpg',
 '8ae471ad.jpg',
 'da0ab651.jpg',
 'a73a10d2.png',
 '8b647321.jpg',
 '76c4be4c.png',
 'f545357e.png',
 'a70bc65a.jpg',
 '03d33d36.png',
 '9789fc6b.png',
 '3f9b55cd.png']


print(len(more_than_one_TEST))

# Select the first 10 images from the list
#selected_files = random.sample(more_than_one_TEST, 10)
selected_files = more_than_one_TEST[:20]

# Create a grid of subplots: 2 rows x 5 columns
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 6))

# Loop over the selected files and display each image in the grid
for ax, file in zip(axes.flatten(), selected_files):
    img_pth = '/kaggle/input/open-data-day-2025-dates-types-classification/test/' + file
    img = mpimg.imread(img_pth)
    ax.imshow(img)
    ax.axis('off')  # Hide axes

plt.tight_layout()
plt.show()


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.io import read_image
from PIL import Image

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# Configuration
class CFG:
    img_size = 224
    batch_size = 16
    num_workers = 4
    lr = 3e-4
    weight_decay = 1e-5
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7
    class_names = ['Sokari', 'Meneifi', 'Ajwa', 'Shaishe', 'Nabtat Ali', 'Sugaey', 'Medjool']
    models_to_train = ['efficientnet_b0', 'resnet50', 'convnext_small']
    n_folds = 5
    save_path = 'model_weights/'

# Create save directory
os.makedirs(CFG.save_path, exist_ok=True)

# Custom Dataset
class DateDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Handle both jpg and png files
        try:
            # Attempt to open with PIL for better compatibility
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading image: {img_path}")
            # Return a black image as fallback
            image = Image.new('RGB', (CFG.img_size, CFG.img_size))
        
        if self.transform:
            image = self.transform(image)
        
        if self.test:
            return image
        else:
            label = self.labels[idx]
            return image, label

# Data Augmentation
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.RandomResizedCrop(CFG.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Add random perspective for more variety
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # Occasionally convert to grayscale to make model robust to color variations
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Add random erasing for robustness
        transforms.RandomErasing(p=0.2),
    ])

def get_valid_transforms():
    return transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Model Creation
def create_model(model_name, pretrained=True):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, CFG.num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, CFG.num_classes)
    elif model_name == 'convnext_small':
        model = models.convnext_small(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, CFG.num_classes)
    
    return model.to(CFG.device)

# Training Function
def train_epoch(model, dataloader, criterion, optimizer, scheduler=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(CFG.device), labels.to(CFG.device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': train_loss / (pbar.n + 1), 'acc': 100. * correct / total})
    
    if scheduler is not None:
        scheduler.step()
        
    return train_loss / len(dataloader), 100. * correct / total

# Validation Function
def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Validation Loss: {val_loss / len(dataloader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CFG.class_names))
    
    return val_loss / len(dataloader), val_acc, val_f1

# Test Time Augmentation
def tta_predict(model, image, n_augs=5):
    model.eval()
    
    # Basic transformation
    basic_transform = get_valid_transforms()
    image_tensor = basic_transform(image).unsqueeze(0).to(CFG.device)
    
    # TTA transformations
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((CFG.img_size, CFG.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) for _ in range(n_augs)
    ]
    
    # Get predictions
    with torch.no_grad():
        # Base prediction
        base_outputs = model(image_tensor)
        
        # TTA predictions
        for transform in tta_transforms:
            aug_image = transform(image).unsqueeze(0).to(CFG.device)
            aug_outputs = model(aug_image)
            base_outputs += aug_outputs
        
        # Average predictions
        base_outputs /= (n_augs + 1)
    
    return base_outputs

# Main Training Loop with K-Fold
def run_training():
    # Load data from the provided paths
    print("Preparing data...")
    
    # Set paths from provided locations
    train_dir = "/kaggle/working/final_train_dir"
    test_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/test"
    train_labels_path = "/kaggle/working/final_train_labels_path.csv"
    
    # Load the training labels
    df = pd.read_csv(train_labels_path)
    
    # Create label mapping
    label_map = {label: idx for idx, label in enumerate(CFG.class_names)}
    
    # Get image paths and labels using x_col='filename', y_col='label'
    all_image_paths = [os.path.join(train_dir, image_name) for image_name in df['filename']]
    all_labels = [label_map[label] for label in df['label']]
    
    # Count class distribution from the dataset
    class_counts = [0] * len(CFG.class_names)
    for label in all_labels:
        class_counts[label] += 1
    
    # Verify data loading
    print(f"Loaded {len(all_image_paths)} images with labels")
    for class_name, count in zip(CFG.class_names, class_counts):
        print(f"{class_name}: {count} images")
    
    # Class weights for handling imbalance - now using actual counts from data
    total_samples = sum(class_counts)
    # Calculate inverse frequency weights
    class_weights = torch.FloatTensor([(total_samples / count) for count in class_counts]).to(CFG.device)
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print("Class weights:", class_weights.cpu().numpy())
    
    # Initialize results storage
    results = []
    
    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=42)
    
    for model_name in CFG.models_to_train:
        print(f"\nTraining model: {model_name}")
        
        fold_val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_image_paths, all_labels)):
            print(f"\nTraining fold {fold+1}/{CFG.n_folds}")
            
            # Split data
            train_image_paths = [all_image_paths[i] for i in train_idx]
            train_labels = [all_labels[i] for i in train_idx]
            val_image_paths = [all_image_paths[i] for i in val_idx]
            val_labels = [all_labels[i] for i in val_idx]
            
            # Create datasets and dataloaders
            train_dataset = DateDataset(train_image_paths, train_labels, transform=get_train_transforms())
            val_dataset = DateDataset(val_image_paths, val_labels, transform=get_valid_transforms())
            
            train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, 
                                     num_workers=CFG.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False,
                                   num_workers=CFG.num_workers, pin_memory=True)
            
            # Create model
            model = create_model(model_name)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.num_epochs)
            
            # Training loop
            best_val_f1 = 0
            
            for epoch in range(CFG.num_epochs):
                print(f"\nEpoch {epoch+1}/{CFG.num_epochs}")
                
                # Train
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
                
                # Validate
                val_loss, val_acc, val_f1 = validate(model, val_loader, criterion)
                
                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), f"{CFG.save_path}/{model_name}_fold{fold+1}.pth")
                    print(f"Saved best model with F1: {best_val_f1:.4f}")
            
            fold_val_scores.append(best_val_f1)
            
        # Model results across folds
        avg_val_score = np.mean(fold_val_scores)
        results.append({
            'model': model_name,
            'avg_f1': avg_val_score,
            'fold_scores': fold_val_scores
        })
        
        print(f"\n{model_name} - Average F1 Score: {avg_val_score:.4f}")
    
    # Print final results
    print("\n===== Final Results =====")
    for res in results:
        print(f"{res['model']}: {res['avg_f1']:.4f}")

# Ensemble Prediction Function
def ensemble_predict(image_path):
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        print(f"Error loading image: {image_path}")
        image = Image.new('RGB', (CFG.img_size, CFG.img_size))
    
    all_predictions = []
    
    # For each model type
    for model_name in CFG.models_to_train:
        fold_predictions = []
        
        # For each fold
        for fold in range(CFG.n_folds):
            # Load model
            model = create_model(model_name, pretrained=False)
            model.load_state_dict(torch.load(f"{CFG.save_path}/{model_name}_fold{fold+1}.pth"))
            model.eval()
            
            # Get TTA predictions
            outputs = tta_predict(model, image)
            fold_predictions.append(outputs)
        
        # Average predictions across folds
        model_prediction = torch.mean(torch.stack(fold_predictions), dim=0)
        all_predictions.append(model_prediction)
    
    # Average predictions across models
    final_prediction = torch.mean(torch.stack(all_predictions), dim=0)
    pred_class = torch.argmax(final_prediction).item()
    
    return pred_class, CFG.class_names[pred_class]

def predict_test_set():
    """Generate predictions for the test set and create a submission file"""
    test_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/test"
    
    # Get all test images
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    
    print(f"Found {len(test_images)} test images. Generating predictions...")
    
    # Store predictions
    predictions = []
    
    for img_name in tqdm(test_images, desc="Predicting"):
        img_path = os.path.join(test_dir, img_name)
        _, pred_label = ensemble_predict(img_path)
        predictions.append({"filename": img_name, "label": pred_label})
    
    # Create submission dataframe
    submission_df = pd.DataFrame(predictions)
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission file created at {submission_path}")
    return submission_df

if __name__ == "__main__":
    # Run training
    run_training()
    
    # Generate test predictions and create submission file
    submission_df = predict_test_set()
    
    # Display submission preview
    print("\nSubmission Preview:")
    print(submission_df.head())





submission_df




