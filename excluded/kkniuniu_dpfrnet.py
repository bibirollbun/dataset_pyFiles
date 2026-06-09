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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder # Included for completeness, but CustomCassavaDataset is primarily used
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import json 
import shutil 

# --- Configuration ---
class Config:
    # --- Dataset Paths ---
    DATA_ROOT = '/kaggle/input/cassava-leaf-disease-classification' 
    TRAIN_CSV = os.path.join(DATA_ROOT, 'train.csv')
    TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, 'train_images') 
    
    TEST_CSV = os.path.join(DATA_ROOT, 'sample_submission.csv') # Path to sample_submission.csv for test image IDs
    TEST_IMAGES_DIR = os.path.join(DATA_ROOT, 'test_images') # Path to test_images folder

    # Directory for processed data or saved models (must be writable, e.g., /kaggle/working/)
    PROCESSED_TRAIN_DIR = os.path.join('/kaggle/working', 'processed_train_images') 

    # --- Model and Training Parameters ---
    IMAGE_SIZE = 384 # Input image size for the model
    BATCH_SIZE = 32
    NUM_EPOCHS = 10 # Training from scratch might require more epochs
    LEARNING_RATE = 1e-4 # Training from scratch might require adjusted LR
    NUM_CLASSES = 5 # Number of disease classes (0, 1, 2, 3, 4)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42

    # --- Feature Map Sizes for ASDA integration (based on IMAGE_SIZE=384 and ResNet50 architecture) ---
    # These are approximate spatial dimensions after each ResNet layer
    # ResNet50 input (e.g., 3x384x384) -> conv1/maxpool -> ~96x96
    # layer1 output: 256 channels @ 96x96
    # layer2 output: 512 channels @ 48x48
    # layer3 output: 1024 channels @ 24x24
    # layer4 output: 2048 channels @ 12x12
    RESNET_FM_SIZES = {
        'layer1': (96, 96), 
        'layer2': (48, 48), 
        'layer3': (24, 24), 
        'layer4': (12, 12)
    }

# Set random seeds for reproducibility
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

print(f"Using device: {Config.DEVICE}")

# --- Data Preparation Utility (Optional - for ImageFolder structure) ---
def prepare_data_for_imagefolder(train_csv_path, train_images_dir, output_dir):
    """
    Reads the train.csv and organizes images into class-specific subfolders
    for torchvision.datasets.ImageFolder. This can consume significant disk space.
    """
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"'{output_dir}' already exists and is not empty. Skipping data preparation.")
        return

    print(f"Preparing data for ImageFolder structure in '{output_dir}'...")
    df = pd.read_csv(train_csv_path)

    for class_id in range(Config.NUM_CLASSES):
        os.makedirs(os.path.join(output_dir, str(class_id)), exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Organizing images"):
        img_filename = row['image_id']
        label = str(row['label'])
        
        src_path = os.path.join(train_images_dir, img_filename)
        dst_path = os.path.join(output_dir, label, img_filename)

        if not os.path.exists(src_path):
            print(f"Warning: Image {src_path} not found. Skipping.")
            continue
        
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {e}")
            
    print("Data preparation complete.")

# --- Custom Dataset for Training and Validation (Reads from TRAIN_IMAGES_DIR) ---
class CustomCassavaDataset(Dataset):
    def __init__(self, image_ids, labels, img_dir, transform=None):
        self.image_ids = image_ids.tolist()
        self.labels = labels.tolist()
        self.img_dir = img_dir
        self.transform = transform
        
        self.label_map = {label: i for i, label in enumerate(sorted(list(set(self.labels))))}
        print(f"Dataset initialized with {len(self.image_ids)} samples.")
        print(f"Label map: {self.label_map}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.label_map[label]

# --- Custom Dataset for Test (Reads from TEST_IMAGES_DIR) ---
class TestCassavaDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform=None):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_name # Return image tensor and its ID for submission

# --- Data Transforms ---
train_transforms = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet means and stds
])

val_transforms = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- CCIA Module Definition ---
class CCIA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCIA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.channel_interaction_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) 
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y) 
        
        y = y.unsqueeze(1) # (b, 1, c)
        y = self.channel_interaction_conv(y) 
        y = y.squeeze(1) # (b, c)

        y = self.sigmoid(y).view(b, c, 1, 1) 
        return x * y.expand_as(x) 

# --- ASDA Module Definition ---
class ASDA(nn.Module):
    def __init__(self, channel, input_H, input_W, reduction_ratio=4):
        super(ASDA, self).__init__()
        self.input_H = input_H
        self.input_W = input_W

        # Multi-branch Local Feature Extraction
        self.conv_3x3 = nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1, bias=False)
        self.conv_5x5 = nn.Conv2d(channel, channel // 2, kernel_size=5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Feature Concatenation & Reduction
        self.conv_1x1_reduce = nn.Conv2d(channel, 1, kernel_size=1, bias=False) 

        # Detail Enhancement & Spatial Reshaping
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) 

        self.fc_spatial1 = nn.Linear(4 * 4, (4 * 4) // reduction_ratio, bias=False)
        self.fc_spatial2 = nn.Linear((4 * 4) // reduction_ratio, input_H * input_W, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        f_3x3 = self.relu(self.conv_3x3(x))
        f_5x5 = self.relu(self.conv_5x5(x))
        
        f_local = torch.cat([f_3x3, f_5x5], dim=1) 

        f_spatial_pre = self.conv_1x1_reduce(f_local) 

        f_pooled = self.adaptive_pool(f_spatial_pre) 
        
        f_pooled = f_pooled.view(b, -1) 
        
        f_linear = self.relu(self.fc_spatial1(f_pooled))
        spatial_weights = self.fc_spatial2(f_linear).view(b, 1, self.input_H, self.input_W) 

        spatial_weights = self.sigmoid(spatial_weights)
        
        return x * spatial_weights.expand_as(x)

# --- NEW: Fusion Block for Parallel Attention and 1x1 Conv Fusion ---
class FusionBlock(nn.Module):
    def __init__(self, in_channels, H, W, use_ccia=False, use_asda=False):
        super(FusionBlock, self).__init__()
        self.use_ccia = use_ccia
        self.use_asda = use_asda
        
        # Instantiate attention modules based on flags
        if self.use_ccia:
            self.ccia = CCIA(channel=in_channels)
        if self.use_asda:
            self.asda = ASDA(channel=in_channels, input_H=H, input_W=W)
            
        # Determine output channels for concatenation
        # Original features (in_channels) + CCIA output (in_channels) + ASDA output (in_channels)
        concat_channels = in_channels 
        if self.use_ccia: concat_channels += in_channels
        if self.use_asda: concat_channels += in_channels

        # 1x1 Convolution for fusion if any attention module is used
        if self.use_ccia or self.use_asda:
            self.fusion_conv = nn.Conv2d(concat_channels, in_channels, kernel_size=1, bias=False)
        else:
            self.fusion_conv = nn.Identity() # If no attention, just pass through

    def forward(self, x_original):
        # Collect features to concatenate
        features_to_concat = [x_original]
        
        x_ccia = x_original
        if self.use_ccia:
            x_ccia = self.ccia(x_original)
            features_to_concat.append(x_ccia)
            
        x_asda = x_original
        if self.use_asda:
            x_asda = self.asda(x_original)
            features_to_concat.append(x_asda)
        
        # Concatenate features
        # If only original is present (no attention), it will be a list of one item, concat will still work
        x_combined = torch.cat(features_to_concat, dim=1)
        
        # Apply fusion convolution
        x_fused = self.fusion_conv(x_combined)
        
        return x_fused


# --- Modified ResNet50 Classifier Model Definition (incorporates FusionBlock) ---
class ResNet50_Classifier(nn.Module):
    # weights_init_type parameter controls backbone initialization
    def __init__(self, num_classes=Config.NUM_CLASSES, use_ccia=False, use_asda=False, weights_init_type='imagenet'):
        super(ResNet50_Classifier, self).__init__()
        
        # Initialize ResNet50 backbone
        if weights_init_type == 'imagenet':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 
            print("ResNet50 backbone initialized with ImageNet pretrained weights.")
        elif weights_init_type == 'random': 
            self.resnet = models.resnet50(weights=None) 
            print("ResNet50 backbone initialized with random weights (training from scratch).")
        else: 
            self.resnet = models.resnet50(weights=None)
            print(f"ResNet50 backbone initialized with random weights (unknown init type: {weights_init_type}).")

        self.use_ccia = use_ccia
        self.use_asda = use_asda

        # Remove original FC layer
        self.resnet.fc = nn.Identity() 

        # --- Instantiate FusionBlocks for each ResNet layer output ---
        # The channels for ResNet50 layers are 256, 512, 1024, 2048 respectively
        self.fusion_block1 = FusionBlock(256, Config.RESNET_FM_SIZES['layer1'][0], Config.RESNET_FM_SIZES['layer1'][1], use_ccia, use_asda)
        self.fusion_block2 = FusionBlock(512, Config.RESNET_FM_SIZES['layer2'][0], Config.RESNET_FM_SIZES['layer2'][1], use_ccia, use_asda)
        self.fusion_block3 = FusionBlock(1024, Config.RESNET_FM_SIZES['layer3'][0], Config.RESNET_FM_SIZES['layer3'][1], use_ccia, use_asda)
        self.fusion_block4 = FusionBlock(2048, Config.RESNET_FM_SIZES['layer4'][0], Config.RESNET_FM_SIZES['layer4'][1], use_ccia, use_asda)
        
        # Final classification layer
        self.fc = nn.Linear(2048, num_classes) 

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Apply each layer and then its corresponding FusionBlock
        x = self.resnet.layer1(x)
        x = self.fusion_block1(x)

        x = self.resnet.layer2(x)
        x = self.fusion_block2(x)
        
        x = self.resnet.layer3(x)
        x = self.fusion_block3(x)

        x = self.resnet.layer4(x)
        x = self.fusion_block4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name="model"):
    best_val_accuracy = 0.0
    model.to(device) 
    best_model_save_path = os.path.join('/kaggle/working', f"{model_name}_best_model.pth")

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}")

        val_loss, val_accuracy, val_f1, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f} F1: {val_f1:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Saved best model with accuracy: {best_val_accuracy:.4f}")
    print("Training complete!")

# --- Evaluation Function ---
def evaluate_model(model, data_loader, criterion, device):
    model.eval() 
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad(): 
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    f1 = f1_score(all_labels, all_predictions, average='macro') 
    cm = confusion_matrix(all_labels, all_predictions)
    return avg_loss, accuracy, f1, all_labels, all_predictions, cm

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Step 0: Data Loading and Splitting (for Training & Validation) ---
    print("\n--- Preparing Training and Validation Data ---")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    train_img_ids, val_img_ids, train_labels, val_labels = train_test_split(
        df_train['image_id'], df_train['label'],
        test_size=0.2, stratify=df_train['label'], random_state=Config.RANDOM_SEED
    )

    train_dataset = CustomCassavaDataset(
        image_ids=train_img_ids,
        labels=train_labels,
        img_dir=Config.TRAIN_IMAGES_DIR, # Uses TRAIN_IMAGES_DIR for training data
        transform=train_transforms
    )
    val_dataset = CustomCassavaDataset(
        image_ids=val_img_ids,
        labels=val_labels,
        img_dir=Config.TRAIN_IMAGES_DIR, # Uses TRAIN_IMAGES_DIR for validation data
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2, 
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


    # --- Step 1: Initialize and Train ResNet50 + CCIA + ASDA (DPFR-Net) Model ---
    print("\n--- Initializing and Training ResNet50 + CCIA + ASDA (DPFR-Net) Model ---")
    # Instantiate the model with CCIA and ASDA enabled, and random weights for backbone
    # This now uses the FusionBlock for parallel attention and 1x1 conv fusion
    model_dpfr_net = ResNet50_Classifier(num_classes=Config.NUM_CLASSES, use_ccia=True, use_asda=True, weights_init_type='random') 
    criterion_dpfr_net = nn.CrossEntropyLoss()
    optimizer_dpfr_net = optim.Adam(model_dpfr_net.parameters(), lr=Config.LEARNING_RATE)

    # Train the combined model
    train_model(model_dpfr_net, train_loader, val_loader, criterion_dpfr_net, optimizer_dpfr_net, Config.NUM_EPOCHS, Config.DEVICE, model_name="resnet50_dpfr_net")


    # --- Step 2: Final Evaluation of ResNet50 + CCIA + ASDA (DPFR-Net) & Submission Generation ---
    print("\n--- Final Evaluation of ResNet50 + CCIA + ASDA (DPFR-Net) Model & Submission Generation ---")
    
    # Instantiate the same model architecture for loading weights
    # weights_init_type='random' to match how it was trained and avoid downloading ImageNet weights
    best_model_dpfr_net = ResNet50_Classifier(num_classes=Config.NUM_CLASSES, use_ccia=True, use_asda=True, weights_init_type='random') 
    best_model_dpfr_net_path = os.path.join('/kaggle/working', "resnet50_dpfr_net_best_model.pth") 

    if not os.path.exists(best_model_dpfr_net_path):
        print(f"Error: Best ResNet50 + CCIA + ASDA (DPFR-Net) model not found at {best_model_dpfr_net_path}. Cannot proceed with evaluation or submission.")
    else:
        best_model_dpfr_net.load_state_dict(torch.load(best_model_dpfr_net_path, map_location=Config.DEVICE))
        best_model_dpfr_net.to(Config.DEVICE) 
        best_model_dpfr_net.eval() # Set model to evaluation mode

        # --- Evaluate on Validation Set ---
        print("\n--- Evaluating ResNet50 + CCIA + ASDA (DPFR-Net) on Validation Set ---")
        val_loss_dpfr_net, val_accuracy_dpfr_net, val_f1_dpfr_net, _, _, cm_dpfr_net = evaluate_model(
            best_model_dpfr_net, val_loader, criterion_dpfr_net, Config.DEVICE
        )
        print(f"Best ResNet50 + CCIA + ASDA (DPFR-Net) Val Loss: {val_loss_dpfr_net:.4f}")
        print(f"Best ResNet50 + CCIA + ASDA (DPFR-Net) Val Accuracy: {val_accuracy_dpfr_net:.4f}")
        print(f"Best ResNet50 + CCIA + ASDA (DPFR-Net) Val F1-Score (Macro): {val_f1_dpfr_net:.4f}")
        print("Confusion Matrix:\n", cm_dpfr_net)

        # --- Generate Submission File for Test Data ---
        print("\n--- Generating Submission File for ResNet50 + CCIA + ASDA (DPFR-Net) ---")

        test_transforms_submission = transforms.Compose([ 
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load test image IDs from sample_submission.csv
        submission_df_template = pd.read_csv(Config.TEST_CSV)
        test_image_ids = submission_df_template['image_id'].tolist()

        # Create Test Dataset and DataLoader
        test_dataset = TestCassavaDataset(
            image_ids=test_image_ids,
            img_dir=Config.TEST_IMAGES_DIR, # Uses Config.TEST_IMAGES_DIR for test data
            transform=test_transforms_submission
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2, 
            shuffle=False,
            num_workers=os.cpu_count() // 2,
            pin_memory=True
        )

        all_test_predictions_dpfr_net = []

        with torch.no_grad(): 
            for inputs, _ in tqdm(test_loader, desc="Predicting on test set with DPFR-Net"): 
                inputs = inputs.to(Config.DEVICE)
                outputs = best_model_dpfr_net(inputs) # Use the DPFR-Net model here
                _, predicted = torch.max(outputs.data, 1)
                all_test_predictions_dpfr_net.extend(predicted.cpu().numpy())

        # Create the final submission DataFrame
        submission_df_dpfr_net = pd.DataFrame({
            'image_id': test_image_ids, 
            'label': all_test_predictions_dpfr_net
        })

        # Save the submission file to /kaggle/working/
        submission_file_path_dpfr_net = os.path.join('/kaggle/working', 'submission.csv') # Unique name for DPFR-Net submission
        submission_df_dpfr_net.to_csv(submission_file_path_dpfr_net, index=False)

        print(f"\nSubmission file for ResNet50 + CCIA + ASDA (DPFR-Net) saved to: {submission_file_path_dpfr_net}")
        print("First 5 rows of submission_dpfr_net.csv:")
        print(submission_df_dpfr_net.head())

    print("\nResNet50 + CCIA + ASDA (DPFR-Net) experiment complete.")

