from IPython.display import Image, display

# Display image directly
display(Image('/kaggle/input/satellite-sites/Sites/attention_-11.012927_-59.108915_20250523_151051.png'))
display(Image('/kaggle/input/satellite-sites/Sites/attention_-14.700835_-71.391091_20250523_160726.png'))


import pandas as pd
import requests
import os
import time
from PIL import Image
from io import BytesIO
import numpy as np

# Configuration
BASE_PATH = '/home/ubuntu/openai/train_data'
POS_FOLDER = os.path.join(BASE_PATH, 'pos')
NEG_FOLDER = os.path.join(BASE_PATH, 'neg')
ZOOM_LEVEL = 17
IMAGE_SIZE = 640  # Google Maps static API default size

# Positive sample patterns to look for in name column
POSITIVE_PATTERNS = ['ACR', 'cir', 'sq', 'oct', 'ovl', 'lin', 'zj', 'mv']

def create_folders():
    """Create pos and neg folders if they don't exist"""
    os.makedirs(POS_FOLDER, exist_ok=True)
    os.makedirs(NEG_FOLDER, exist_ok=True)
    print(f"Created/verified folders: {POS_FOLDER}, {NEG_FOLDER}")

def download_satellite_image(lat, lon, filename, folder):
    """
    Download satellite image from Google Maps Static API
    Note: You'll need a Google Maps API key for this to work
    """
    # Google Maps Static API URL
    # You need to replace 'YOUR_API_KEY' with your actual Google Maps API key
    api_key = 'YOUR_GOOGLE_API_KEY'  # Replace with your actual API key
    
    url = f"https://maps.googleapis.com/maps/api/staticmap?"
    params = {
        'center': f"{lat},{lon}",
        'zoom': ZOOM_LEVEL,
        'size': f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        'maptype': 'satellite',
        'key': api_key
    }
    
    # Construct full URL
    full_url = url + '&'.join([f"{k}={v}" for k, v in params.items()])
    
    try:
        response = requests.get(full_url, timeout=30)
        response.raise_for_status()
        
        # Save image
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded: {filename}")
        return True
        
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def process_negative_samples():
    """Process submit.csv to get negative samples (type='other')"""
    print("Processing negative samples from submit.csv...")
    
    # Read submit.csv
    submit_df = pd.read_csv('submit.csv')
    print(f"Total rows in submit.csv: {len(submit_df)}")
    
    # Filter for 'other' type
    neg_samples = submit_df[submit_df['type'] == 'other'].copy()
    print(f"Found {len(neg_samples)} negative samples (type='other')")
    
    # Download images for negative samples
    success_count = 0
    for idx, row in neg_samples.iterrows():
        lat = row['y']  # y column contains latitude
        lon = row['x']  # x column contains longitude
        
        filename = f"neg_{idx}_{lat:.6f}_{lon:.6f}.jpg"
        
        if download_satellite_image(lat, lon, filename, NEG_FOLDER):
            success_count += 1
        
        # Add delay to avoid rate limiting
        time.sleep(0.1)
        
        # Progress update every 50 downloads
        if (success_count + 1) % 50 == 0:
            print(f"Progress: {success_count}/{len(neg_samples)} negative samples downloaded")
    
    print(f"Completed negative samples: {success_count}/{len(neg_samples)} downloaded successfully")
    return success_count

def process_positive_samples():
    """Process amazon_geoglyphs_sites.csv to get positive samples"""
    print("Processing positive samples from amazon_geoglyphs_sites.csv...")
    
    # Read amazon geoglyphs file
    geoglyphs_df = pd.read_csv('amazon_geoglyphs_sites.csv')
    print(f"Total rows in amazon_geoglyphs_sites.csv: {len(geoglyphs_df)}")
    
    # Filter for positive patterns in name column
    pattern_mask = geoglyphs_df['name'].str.contains('|'.join(POSITIVE_PATTERNS), case=False, na=False)
    pos_samples = geoglyphs_df[pattern_mask].copy()
    print(f"Found {len(pos_samples)} positive samples with patterns: {POSITIVE_PATTERNS}")
    
    # Show breakdown by pattern
    for pattern in POSITIVE_PATTERNS:
        count = geoglyphs_df['name'].str.contains(pattern, case=False, na=False).sum()
        print(f"  - '{pattern}': {count} samples")
    
    # Download images for positive samples
    success_count = 0
    for idx, row in pos_samples.iterrows():
        # Convert latitude/longitude to float if they're strings
        try:
            lat = float(row['latitude'])
            lon = float(row['longitude'])
        except (ValueError, TypeError):
            print(f"Skipping row {idx}: Invalid coordinates")
            continue
        
        name = str(row['name']).replace('/', '_').replace('\\', '_')  # Clean filename
        filename = f"pos_{name}_{lat:.6f}_{lon:.6f}.jpg"
        
        if download_satellite_image(lat, lon, filename, POS_FOLDER):
            success_count += 1
        
        # Add delay to avoid rate limiting
        time.sleep(0.1)
        
        # Progress update every 25 downloads
        if (success_count + 1) % 25 == 0:
            print(f"Progress: {success_count}/{len(pos_samples)} positive samples downloaded")
    
    print(f"Completed positive samples: {success_count}/{len(pos_samples)} downloaded successfully")
    return success_count

def main():
    """Main function to orchestrate the download process"""
    print("Starting satellite image download process...")
    print(f"Target folders: {POS_FOLDER}, {NEG_FOLDER}")
    print(f"Zoom level: {ZOOM_LEVEL}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("-" * 50)
    
    # Create folders
    create_folders()
    
    # Check existing files
    existing_pos = len([f for f in os.listdir(POS_FOLDER) if f.endswith('.jpg')]) if os.path.exists(POS_FOLDER) else 0
    existing_neg = len([f for f in os.listdir(NEG_FOLDER) if f.endswith('.jpg')]) if os.path.exists(NEG_FOLDER) else 0
    
    print(f"Existing images - Positive: {existing_pos}, Negative: {existing_neg}")
    print("-" * 50)
    
    # Process negative samples
    neg_downloaded = process_negative_samples()
    print("-" * 50)
    
    # Process positive samples  
    pos_downloaded = process_positive_samples()
    print("-" * 50)
    
    # Final summary
    print("DOWNLOAD SUMMARY:")
    print(f"Positive samples downloaded: {pos_downloaded}")
    print(f"Negative samples downloaded: {neg_downloaded}")
    print(f"Total new images: {pos_downloaded + neg_downloaded}")
    
    # Final counts
    final_pos = len([f for f in os.listdir(POS_FOLDER) if f.endswith('.jpg')])
    final_neg = len([f for f in os.listdir(NEG_FOLDER) if f.endswith('.jpg')])
    print(f"Final image counts - Positive: {final_pos}, Negative: {final_neg}")

# Alternative function without API (for testing file processing)
def test_data_processing():
    """Test the data processing without downloading images"""
    print("TESTING DATA PROCESSING (NO DOWNLOADS)")
    print("-" * 50)
    
    # Test negative samples
    submit_df = pd.read_csv('submit.csv')
    neg_samples = submit_df[submit_df['type'] == 'other']
    print(f"Negative samples found: {len(neg_samples)}")
    print("Sample negative coordinates:")
    for i in range(min(5, len(neg_samples))):
        row = neg_samples.iloc[i]
        print(f"  {i+1}. Lat: {row['y']}, Lon: {row['x']}")
    
    print("-" * 30)
    
    # Test positive samples
    geoglyphs_df = pd.read_csv('amazon_geoglyphs_sites.csv')
    pattern_mask = geoglyphs_df['name'].str.contains('|'.join(POSITIVE_PATTERNS), case=False, na=False)
    pos_samples = geoglyphs_df[pattern_mask]
    print(f"Positive samples found: {len(pos_samples)}")
    print("Sample positive coordinates:")
    for i in range(min(5, len(pos_samples))):
        row = pos_samples.iloc[i]
        print(f"  {i+1}. Name: {row['name']}, Lat: {row['latitude']}, Lon: {row['longitude']}")

if __name__ == "__main__":
    # IMPORTANT: Before running main(), you need to:
    # 1. Get a Google Maps Static API key from Google Cloud Console
    # 2. Replace 'YOUR_API_KEY' in the download_satellite_image function
    
    # For testing data processing first:
    test_data_processing()
    
    # Uncomment the line below to run the full download process:
    main()


import os
import shutil
import pandas as pd
import numpy as np
from urllib.parse import unquote
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import warnings
import math
import random
import cv2
warnings.filterwarnings('ignore')

# Set working directory
os.chdir('/home/ubuntu/openai')

# ====== ORIGINAL DATASET CLASS (from your code) ======
class SatelliteImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.float32),
                'path': image_path
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.float32),
                'path': image_path
            }

# ====== ENHANCED ATTENTION MODULES ======
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleAttention, self).__init__()
        self.scale1 = nn.Conv2d(in_channels, in_channels//4, 1)
        self.scale2 = nn.Conv2d(in_channels, in_channels//4, 3, padding=1)
        self.scale3 = nn.Conv2d(in_channels, in_channels//4, 5, padding=2)
        self.scale4 = nn.Conv2d(in_channels, in_channels//4, 7, padding=3)
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)
        
        multi_scale = torch.cat([s1, s2, s3, s4], dim=1)
        attention_weight = self.attention(multi_scale)
        
        return x * attention_weight

# ====== ENHANCED EFFICIENTNET MODEL ======
class EnhancedAttentionEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, num_classes=2):
        super(EnhancedAttentionEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dims = [f.shape[1] for f in features]
            print(f"Feature dimensions: {feature_dims}")
        
        # Add attention modules at multiple scales
        self.attention_modules = nn.ModuleList([
            CBAM(dim) for dim in feature_dims[-3:]  # Last 3 feature levels
        ])
        
        self.multi_scale_attention = MultiScaleAttention(feature_dims[-1])
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Store attention maps for visualization
        self.attention_maps = {}
        
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Apply attention to the last 3 feature levels
        attended_features = []
        for i, (feat, attn) in enumerate(zip(features[-3:], self.attention_modules)):
            attended = attn(feat)
            attended_features.append(attended)
            self.attention_maps[f'level_{i}'] = attended
        
        # Apply multi-scale attention to the final feature map
        final_features = self.multi_scale_attention(features[-1])
        self.attention_maps['multi_scale'] = final_features
        
        # Global pooling and classification
        pooled = self.global_pool(final_features)
        pooled = pooled.view(pooled.size(0), -1)
        output = self.classifier(pooled)
        
        return output

# ====== DATASET PREPARATION (from your original code) ======
def prepare_train_val_split(train_data_path='/home/ubuntu/openai/train_data', val_split=0.2):
    """Create train/validation split from pos and neg folders."""
    print("ğŸ”„ Preparing train/validation split from pos and neg folders...")
    
    pos_source = os.path.join(train_data_path, 'pos')
    neg_source = os.path.join(train_data_path, 'neg')
    
    if not os.path.exists(pos_source):
        print(f"â�Œ Error: {pos_source} directory not found!")
        return False
    if not os.path.exists(neg_source):
        print(f"â�Œ Error: {neg_source} directory not found!")
        return False
    
    # Create destination directories
    os.makedirs('dataset/train/positive', exist_ok=True)
    os.makedirs('dataset/train/negative', exist_ok=True)
    os.makedirs('dataset/val/positive', exist_ok=True)
    os.makedirs('dataset/val/negative', exist_ok=True)
    
    random.seed(42)
    
    # Process positive images
    pos_files = [f for f in os.listdir(pos_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(pos_files)
    
    pos_val_count = int(len(pos_files) * val_split)
    pos_train_files = pos_files[pos_val_count:]
    pos_val_files = pos_files[:pos_val_count]
    
    print(f"ğŸ“Š Positive images: {len(pos_files)} total, {len(pos_train_files)} train, {len(pos_val_files)} val")
    
    # Copy files
    for filename in pos_train_files:
        src = os.path.join(pos_source, filename)
        dst = os.path.join('dataset/train/positive', filename)
        shutil.copy2(src, dst)
    
    for filename in pos_val_files:
        src = os.path.join(pos_source, filename)
        dst = os.path.join('dataset/val/positive', filename)
        shutil.copy2(src, dst)
    
    # Process negative images
    neg_files = [f for f in os.listdir(neg_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(neg_files)
    
    neg_val_count = int(len(neg_files) * val_split)
    neg_train_files = neg_files[neg_val_count:]
    neg_val_files = neg_files[:neg_val_count]
    
    print(f"ğŸ“Š Negative images: {len(neg_files)} total, {len(neg_train_files)} train, {len(neg_val_files)} val")
    
    for filename in neg_train_files:
        src = os.path.join(neg_source, filename)
        dst = os.path.join('dataset/train/negative', filename)
        shutil.copy2(src, dst)
    
    for filename in neg_val_files:
        src = os.path.join(neg_source, filename)
        dst = os.path.join('dataset/val/negative', filename)
        shutil.copy2(src, dst)
    
    print(f"âœ… Dataset split completed!")
    return True

def load_dataset():
    """Load and prepare the dataset for training."""
    print("ğŸ“‚ Loading dataset...")
    
    # Enhanced data transforms with more augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Slightly larger for random crop
        transforms.RandomCrop(224),     # Random crop for better data diversity
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),  # Increased rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),  # Occasionally convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def collect_images_and_labels(base_dir, transform):
        image_paths = []
        labels = []
        
        # Positive images (label = 1)
        pos_dir = os.path.join(base_dir, 'positive')
        if os.path.exists(pos_dir):
            for filename in os.listdir(pos_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(pos_dir, filename))
                    labels.append(1)
        
        # Negative images (label = 0)
        neg_dir = os.path.join(base_dir, 'negative')
        if os.path.exists(neg_dir):
            for filename in os.listdir(neg_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(neg_dir, filename))
                    labels.append(0)
        
        return image_paths, labels
    
    train_paths, train_labels = collect_images_and_labels('dataset/train', train_transform)
    train_dataset = SatelliteImageDataset(train_paths, train_labels, train_transform)
    
    val_paths, val_labels = collect_images_and_labels('dataset/val', val_transform)
    val_dataset = SatelliteImageDataset(val_paths, val_labels, val_transform)
    
    print(f"ğŸ“Š Training set: {len(train_dataset)} images")
    print(f"   - Positive: {sum(train_labels)} images")
    print(f"   - Negative: {len(train_labels) - sum(train_labels)} images")
    
    print(f"ğŸ“Š Validation set: {len(val_dataset)} images")
    print(f"   - Positive: {sum(val_labels)} images")
    print(f"   - Negative: {len(val_labels) - sum(val_labels)} images")
    
    return train_dataset, val_dataset

# ====== ENHANCED VISUALIZATION ======
def visualize_enhanced_attention(model, images, labels, paths, epoch, batch_idx, output_dir='./'):
    """Enhanced attention visualization with multiple attention maps"""
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    
    num_samples = min(len(images), 2)  # Limit samples due to more visualizations
    
    # Create subplot grid: original, multiple attention maps, overlay
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    
    for i in range(num_samples):
        # Unnormalize image for display
        img = images[i].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original - Label: {int(labels[i].item())}")
        axes[i, 0].axis('off')
        
        # Multiple attention visualizations
        attention_types = ['level_0', 'level_1', 'level_2', 'multi_scale']
        for j, att_type in enumerate(attention_types):
            if att_type in model.attention_maps and i < model.attention_maps[att_type].shape[0]:
                att_map = model.attention_maps[att_type][i]
                
                if len(att_map.shape) == 3:  # [C, H, W]
                    att_map = att_map.mean(dim=0)  # Average over channels
                
                att_map = att_map.cpu().numpy()
                att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
                
                # Resize to match image size
                att_map_resized = cv2.resize(att_map, (img.shape[1], img.shape[0]))
                
                axes[i, j+1].imshow(att_map_resized, cmap='jet')
                axes[i, j+1].set_title(f"Attention {att_type}")
                axes[i, j+1].axis('off')
            else:
                axes[i, j+1].axis('off')
        
        # Final overlay
        if 'multi_scale' in model.attention_maps and i < model.attention_maps['multi_scale'].shape[0]:
            final_att = model.attention_maps['multi_scale'][i].mean(dim=0).cpu().numpy()
            final_att = (final_att - final_att.min()) / (final_att.max() - final_att.min() + 1e-8)
            final_att_resized = cv2.resize(final_att, (img.shape[1], img.shape[0]))
            
            axes[i, 5].imshow(img)
            im = axes[i, 5].imshow(final_att_resized, cmap='jet', alpha=0.6)
            axes[i, 5].set_title(f"Final Overlay - Pred: {probs[i]:.3f}")
            axes[i, 5].axis('off')
        else:
            axes[i, 5].axis('off')
    
    plt.tight_layout()
    fig.suptitle(f"Enhanced Attention Visualization - Epoch {epoch}, Batch {batch_idx}", fontsize=16, y=0.98)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"enhanced_attention_ep{epoch}_batch{batch_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"ğŸ’¾ Saved enhanced attention visualization to {save_path}")

# ====== TRAINING FUNCTION ======
def train_enhanced_model(model, train_loader, val_loader, num_epochs=20):
    """Train enhanced attention model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"ğŸš€ Using device: {device}")
    
    model = model.to(device)
    
    # Use focal loss for better handling of class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.ce_loss = nn.CrossEntropyLoss()
        
        def forward(self, inputs, targets):
            ce_loss = self.ce_loss(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
    
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    attention_output_dir = './enhanced_attention_visualizations'
    os.makedirs(attention_output_dir, exist_ok=True)
    
    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"\nğŸ”„ Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        train_pbar = tqdm(train_loader, desc="Training Enhanced Model")
        for batch_idx, batch in enumerate(train_pbar):
            images = batch['image'].to(device)
            labels = batch['label'].long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
            
            # Visualize attention every 20 batches (reduced from 100)
            if (batch_idx + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    visualize_enhanced_attention(
                        model, images[:2], labels[:2], 
                        [batch['path'][k] for k in range(min(2, len(images)))],
                        epoch + 1, batch_idx + 1, attention_output_dir
                    )
                model.train()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_predictions
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, batch in enumerate(val_pbar):
                images = batch['image'].to(device)
                labels = batch['label'].long().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_enhanced_attention_model.pth')
            print(f"ğŸ’¾ New best model saved! Val Acc: {val_acc:.4f}")
        
        scheduler.step()
    
    # Load best model
    model.load_state_dict(torch.load('best_enhanced_attention_model.pth'))
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    
    return model

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_enhanced_model(model, val_loader):
    """Evaluate the enhanced model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].long().to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nğŸ“Š Enhanced Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nğŸ“ˆ Confusion Matrix:")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Enhanced Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1

# ====== MAIN FUNCTION ======
def main():
    """Main training pipeline with enhanced attention."""
    print("ğŸŒ� Enhanced Satellite Image Classifier with Advanced Attention")
    print("=" * 80)
    
    # Step 1: Check data
    train_data_path = '/home/ubuntu/openai/train_data'
    pos_path = os.path.join(train_data_path, 'pos')
    neg_path = os.path.join(train_data_path, 'neg')
    
    if not os.path.exists(pos_path) or not os.path.exists(neg_path):
        print("â�Œ Error: pos or neg directories not found!")
        return
    
    pos_count = len([f for f in os.listdir(pos_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    neg_count = len([f for f in os.listdir(neg_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"ğŸ“Š Found {pos_count} positive and {neg_count} negative images")
    
    if pos_count == 0 or neg_count == 0:
        print("â�Œ Error: Need both positive and negative images!")
        return
    
    # Step 2: Prepare dataset
    if not prepare_train_val_split(train_data_path):
        print("â�Œ Failed to prepare dataset split!")
        return
    
    # Step 3: Load dataset
    train_dataset, val_dataset = load_dataset()
    
    if len(train_dataset) == 0:
        print("â�Œ No training data found!")
        return
    
    # Step 4: Create data loaders
    batch_size = 12  # Reduced batch size for enhanced model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Step 5: Create enhanced model
    print("ğŸ�—ï¸� Creating Enhanced Attention EfficientNet...")
    model = EnhancedAttentionEfficientNet('efficientnet_b0', pretrained=True, num_classes=2)
    
    # Step 6: Train enhanced model
    trained_model = train_enhanced_model(model, train_loader, val_loader, num_epochs=50)
    
    # Step 7: Evaluate enhanced model
    accuracy, precision, recall, f1 = evaluate_enhanced_model(trained_model, val_loader)
    
    print("\nğŸ�‰ Enhanced training completed successfully!")
    print(f"ğŸ“� Best model saved as 'best_enhanced_attention_model.pth'")
    print(f"ğŸ“ˆ Training plots saved as 'enhanced_training_history.png'")
    print(f"ğŸ�¨ Attention visualizations in './enhanced_attention_visualizations/'")

if __name__ == "__main__":
    main()


"""
Archaeological Site Detection along Rivers with Enhanced Attention Model
======================================================================

This script uses the enhanced EfficientNet model with advanced attention mechanisms
to detect potential archaeological sites along river segments.

Requirements:
- torch
- torchvision
- timm
- PIL (Pillow)
- pandas
- numpy
- requests
- folium
- geopandas
- shapely
- opencv-python (cv2)

Usage:
1. Ensure you have the trained model file 'best_enhanced_attention_model.pth'
2. Set your Google Maps API key
3. Provide river segments data (GeoJSON or pickled data)
4. Run the script
"""

import math
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image, ImageDraw, ImageFont
import requests
import time
import json
import pickle
from datetime import datetime
import geopandas as gpd
from shapely.geometry import LineString, Point
import folium
from folium import plugins
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Model settings
    MODEL_PATH = 'best_enhanced_attention_model.pth'
    MODEL_NAME = 'efficientnet_b0'
    
    # Google Maps API settings
    GOOGLE_MAPS_API_KEY = 'YOUR_GOOGLE_API_KEY'  # Replace with your key
    ZOOM_LEVEL = 17
    IMAGE_SIZE = 640
    MAP_TYPE = 'satellite'
    
    # Calculate optimal sampling distance based on image coverage
    METERS_PER_PIXEL = 156543.03392 * math.cos(0 * math.pi / 180) / (2 ** ZOOM_LEVEL)
    IMAGE_COVERAGE_M = IMAGE_SIZE * METERS_PER_PIXEL
    
    # Detection settings with optimized sampling
    CONFIDENCE_THRESHOLD = 0.7
    OVERLAP_PERCENTAGE = 20
    SAMPLE_DISTANCE_M = int(IMAGE_COVERAGE_M * (1 - OVERLAP_PERCENTAGE / 100))
    
    MAX_RIVERS_TO_PROCESS = 100
    MAX_POINTS_PER_RIVER = 50
    
    # API rate limiting
    REQUESTS_PER_MINUTE = 50
    DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE
    
    # Output settings
    OUTPUT_DIR = './archaeological_detections_enhanced_2'
    IMAGES_DIR = os.path.join(OUTPUT_DIR, 'detected_sites')
    CSV_OUTPUT = os.path.join(OUTPUT_DIR, 'archaeological_detections_enhanced.csv')
    MAP_OUTPUT = os.path.join(OUTPUT_DIR, 'detection_map_enhanced.html')
    ATTENTION_VIZ_DIR = os.path.join(OUTPUT_DIR, 'attention_visualizations')
    
    # Input data
    RIVERS_DATA_PATH = 'extracted_rivers.geojson'
    RIVER_CACHE_PATH = './river_cache'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.IMAGES_DIR, exist_ok=True)
os.makedirs(Config.ATTENTION_VIZ_DIR, exist_ok=True)

# ====== ENHANCED ATTENTION MODULES (from training script) ======
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleAttention, self).__init__()
        self.scale1 = nn.Conv2d(in_channels, in_channels//4, 1)
        self.scale2 = nn.Conv2d(in_channels, in_channels//4, 3, padding=1)
        self.scale3 = nn.Conv2d(in_channels, in_channels//4, 5, padding=2)
        self.scale4 = nn.Conv2d(in_channels, in_channels//4, 7, padding=3)
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)
        
        multi_scale = torch.cat([s1, s2, s3, s4], dim=1)
        attention_weight = self.attention(multi_scale)
        
        return x * attention_weight

# ====== ENHANCED EFFICIENTNET MODEL (from training script) ======
class EnhancedAttentionEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, num_classes=2):
        super(EnhancedAttentionEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dims = [f.shape[1] for f in features]
        
        # Add attention modules at multiple scales
        self.attention_modules = nn.ModuleList([
            CBAM(dim) for dim in feature_dims[-3:]
        ])
        
        self.multi_scale_attention = MultiScaleAttention(feature_dims[-1])
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Store attention maps for visualization
        self.attention_maps = {}
        
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Apply attention to the last 3 feature levels
        attended_features = []
        for i, (feat, attn) in enumerate(zip(features[-3:], self.attention_modules)):
            attended = attn(feat)
            attended_features.append(attended)
            self.attention_maps[f'level_{i}'] = attended
        
        # Apply multi-scale attention to the final feature map
        final_features = self.multi_scale_attention(features[-1])
        self.attention_maps['multi_scale'] = final_features
        
        # Global pooling and classification
        pooled = self.global_pool(final_features)
        pooled = pooled.view(pooled.size(0), -1)
        output = self.classifier(pooled)
        
        return output

def load_enhanced_model():
    """
    Load the trained enhanced attention EfficientNet model
    """
    print(f"Loading enhanced model from {Config.MODEL_PATH}...")
    
    model = EnhancedAttentionEfficientNet(Config.MODEL_NAME, pretrained=True, num_classes=2)
    
    if os.path.exists(Config.MODEL_PATH):
        try:
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
            print("âœ… Enhanced model loaded successfully")
        except Exception as e:
            print(f"â�Œ Error loading model: {e}")
            print("Using randomly initialized model")
    else:
        print(f"âš ï¸� Model file not found at {Config.MODEL_PATH}")
        print("Using randomly initialized model")
    
    model.to(Config.DEVICE)
    model.eval()
    
    return model

def visualize_detection_attention(model, image, lat, lon, confidence, save_path):
    """
    Visualize attention maps for a detection
    """
    model.eval()
    device = Config.DEVICE
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor_image = transform(image).unsqueeze(0).to(device)
    
    # Get prediction with attention maps
    with torch.no_grad():
        _ = model(tensor_image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original\nLat: {lat:.4f}, Lon: {lon:.4f}")
    axes[0].axis('off')
    
    # Attention visualizations
    attention_types = ['level_0', 'level_1', 'level_2', 'multi_scale']
    for i, att_type in enumerate(attention_types):
        if att_type in model.attention_maps:
            att_map = model.attention_maps[att_type][0]
            
            if len(att_map.shape) == 3:
                att_map = att_map.mean(dim=0)
            
            att_map = att_map.cpu().numpy()
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
            
            # Resize to match image size
            att_map_resized = cv2.resize(att_map, (image.width, image.height))
            
            axes[i+1].imshow(att_map_resized, cmap='jet')
            axes[i+1].set_title(f"{att_type}")
            axes[i+1].axis('off')
    
    # Final overlay
    if 'multi_scale' in model.attention_maps:
        final_att = model.attention_maps['multi_scale'][0].mean(dim=0).cpu().numpy()
        final_att = (final_att - final_att.min()) / (final_att.max() - final_att.min() + 1e-8)
        final_att_resized = cv2.resize(final_att, (image.width, image.height))
        
        axes[5].imshow(image)
        axes[5].imshow(final_att_resized, cmap='jet', alpha=0.6)
        axes[5].set_title(f"Overlay\nConfidence: {confidence:.3f}")
        axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def remove_clustered_detections(detections_list, min_distance_m=300):
    """
    Remove detections that are too close to each other
    """
    if len(detections_list) <= 1:
        return detections_list
    
    print(f"\nğŸ”� Removing clustered detections (min distance: {min_distance_m}m)...")
    
    # Sort by confidence (highest first)
    detections_sorted = sorted(detections_list, key=lambda x: x['confidence'], reverse=True)
    filtered_detections = []
    
    for detection in detections_sorted:
        lat, lon = detection['latitude'], detection['longitude']
        
        # Check if this detection is too close to any already filtered detection
        is_too_close = False
        for filtered_det in filtered_detections:
            f_lat, f_lon = filtered_det['latitude'], filtered_det['longitude']
            
            # Calculate approximate distance
            lat_diff = (lat - f_lat) * 111000
            lon_diff = (lon - f_lon) * 111000 * math.cos(math.radians(lat))
            distance = math.sqrt(lat_diff**2 + lon_diff**2)
            
            if distance < min_distance_m:
                is_too_close = True
                break
        
        if not is_too_close:
            filtered_detections.append(detection)
    
    removed_count = len(detections_list) - len(filtered_detections)
    print(f"   - Removed {removed_count} clustered detections")
    print(f"   - Kept {len(filtered_detections)} unique detections")
    
    return filtered_detections

def load_river_segments():
    """
    Load river segments from pickle cache files or GeoJSON
    """
    river_segments = []
    
    # Primary method: Load from cache directory (pkl files)
    if os.path.exists(Config.RIVER_CACHE_PATH):
        print(f"Loading river segments from cache directory: {Config.RIVER_CACHE_PATH}")
        cache_files = [f for f in os.listdir(Config.RIVER_CACHE_PATH) if f.endswith('_rivers.pkl')]
        
        if cache_files:
            print(f"Found {len(cache_files)} cache files to process...")
            
            for cache_file in sorted(cache_files):
                try:
                    cache_path = os.path.join(Config.RIVER_CACHE_PATH, cache_file)
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                        
                    rivers = data.get('rivers', [])
                    names = data.get('river_names', [])
                    
                    print(f"  Processing {cache_file}: {len(rivers)} rivers")
                    
                    for i, (river_coords, name) in enumerate(zip(rivers, names)):
                        if len(river_coords) >= 2:
                            river_segments.append({
                                'id': f"{cache_file.replace('_rivers.pkl', '')}_{i}",
                                'name': name if name and name != f"River_{i+1}" else f"River_{cache_file}_{i}",
                                'coordinates': river_coords,
                                'source_file': cache_file
                            })
                        
                except Exception as e:
                    print(f"  â�Œ Error loading {cache_file}: {e}")
                    continue
            
            print(f"âœ… Loaded {len(river_segments)} river segments from {len(cache_files)} cache files")
    
    # Fallback: Try to load from GeoJSON
    if not river_segments and os.path.exists(Config.RIVERS_DATA_PATH):
        print(f"Fallback: Loading river segments from {Config.RIVERS_DATA_PATH}...")
        try:
            gdf = gpd.read_file(Config.RIVERS_DATA_PATH)
            
            for idx, row in gdf.iterrows():
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)
                    lat_lon_coords = [(lat, lon) for lon, lat in coords]
                    river_segments.append({
                        'id': row.get('id', idx),
                        'name': row.get('name', f'River_{idx}'),
                        'coordinates': lat_lon_coords,
                        'source_file': 'geojson'
                    })
            
            print(f"âœ… Loaded {len(river_segments)} river segments from GeoJSON")
            
        except Exception as e:
            print(f"â�Œ Error loading GeoJSON: {e}")
    
    if not river_segments:
        print("â�Œ No river segments found!")
        return []
    
    print(f"\nğŸ“Š River segments summary:")
    print(f"  - Total segments: {len(river_segments)}")
    
    return river_segments

def sample_points_along_river_optimized(river_coords, distance_m=None):
    """
    Sample points along a river at optimal intervals
    """
    if distance_m is None:
        distance_m = Config.SAMPLE_DISTANCE_M
    
    if len(river_coords) < 2:
        return []
    
    # Convert to GeoDataFrame for distance calculations
    line = LineString([(lon, lat) for lat, lon in river_coords])
    gdf = gpd.GeoDataFrame([1], geometry=[line], crs='EPSG:4326')
    gdf_utm = gdf.to_crs('EPSG:32719')  # UTM 19S for Amazon region
    
    utm_line = gdf_utm.geometry.iloc[0]
    total_length = utm_line.length
    
    # Calculate number of sample points
    num_points = int(total_length / distance_m) + 1
    
    sample_points = []
    processed_locations = []
    
    for i in range(num_points):
        distance_along = min(i * distance_m, total_length)
        
        if distance_along <= total_length:
            point_utm = utm_line.interpolate(distance_along)
            point_gdf = gpd.GeoDataFrame([1], geometry=[point_utm], crs='EPSG:32719')
            point_wgs84 = point_gdf.to_crs('EPSG:4326').geometry.iloc[0]
            
            lat, lon = point_wgs84.y, point_wgs84.x
            
            # Check for duplicates
            is_duplicate = False
            for prev_lat, prev_lon in processed_locations:
                lat_diff = abs(lat - prev_lat) * 111000
                lon_diff = abs(lon - prev_lon) * 111000 * math.cos(math.radians(lat))
                approx_distance = math.sqrt(lat_diff**2 + lon_diff**2)
                
                if approx_distance < distance_m * 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                sample_points.append((lat, lon))
                processed_locations.append((lat, lon))
    
    return sample_points

def download_satellite_image(lat, lon):
    """
    Download satellite image from Google Maps API
    """
    if Config.GOOGLE_MAPS_API_KEY == 'YOUR_GOOGLE_MAPS_API_KEY':
        print("âš ï¸� Please set your Google Maps API key in the config")
        return None
    
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        'center': f"{lat},{lon}",
        'zoom': Config.ZOOM_LEVEL,
        'size': f"{Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}",
        'maptype': Config.MAP_TYPE,
        'key': Config.GOOGLE_MAPS_API_KEY,
        'format': 'png'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        image = Image.open(requests.get(base_url, params=params, stream=True).raw)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        print(f"Error downloading image for {lat}, {lon}: {e}")
        return None

def preprocess_image(image):
    """
    Preprocess image for model inference
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        tensor_image = transform(image).unsqueeze(0)
        return tensor_image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        black_image = Image.new('RGB', (224, 224), (0, 0, 0))
        tensor_image = transform(black_image).unsqueeze(0)
        return tensor_image

def classify_image(model, image):
    """
    Classify image using the enhanced model
    """
    try:
        with torch.no_grad():
            tensor_image = preprocess_image(image).to(Config.DEVICE)
            outputs = model(tensor_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0][1].cpu().item()
            predicted_class = 1 if confidence > 0.5 else 0
            
            return predicted_class, confidence
    
    except Exception as e:
        print(f"Error classifying image: {e}")
        return 0, 0.0

def add_confidence_text_to_image(image, confidence):
    """
    Add confidence score text to the image
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    confidence_text = f"Confidence: {confidence:.3f}"
    
    font = None
    try:
        font_paths = [
            "arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf"
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 20)
                break
            except (OSError, IOError):
                continue
                
    except Exception:
        pass
    
    if font is not None:
        try:
            text_bbox = draw.textbbox((0, 0), confidence_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except Exception:
            try:
                text_width, text_height = draw.textsize(confidence_text, font=font)
            except Exception:
                text_width = len(confidence_text) * 12
                text_height = 20
    else:
        font = ImageFont.load_default()
        text_width = len(confidence_text) * 8
        text_height = 16
    
    x = 10
    y = 10
    
    padding = 5
    rect_coords = [
        (x - padding, y - padding),
        (x + text_width + padding, y + text_height + padding)
    ]
    draw.rectangle(rect_coords, fill='black', outline='white', width=1)
    draw.text((x, y), confidence_text, fill='white', font=font)
    
    return img_copy

def save_detection(image, lat, lon, confidence, river_info, detections_list, model):
    """
    Save positive detection with image and attention visualization
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"site_{lat:.6f}_{lon:.6f}_{timestamp}_conf{confidence:.3f}.png"
    filepath = os.path.join(Config.IMAGES_DIR, filename)
    
    # Add confidence text to image
    annotated_image = add_confidence_text_to_image(image, confidence)
    annotated_image.save(filepath)
    
    # Save attention visualization
    try:
        import matplotlib.pyplot as plt
        att_filename = f"attention_{lat:.6f}_{lon:.6f}_{timestamp}.png"
        att_filepath = os.path.join(Config.ATTENTION_VIZ_DIR, att_filename)
        visualize_detection_attention(model, image, lat, lon, confidence, att_filepath)
    except Exception as e:
        print(f"Could not save attention visualization: {e}")
    
    # Add to detections list
    detection_record = {
        'filename': filename,
        'latitude': lat,
        'longitude': lon,
        'confidence': confidence,
        'river_id': river_info['id'],
        'river_name': river_info['name'],
        'timestamp': datetime.now().isoformat(),
        'detection_date': datetime.now().strftime("%Y-%m-%d"),
        'detection_time': datetime.now().strftime("%H:%M:%S")
    }
    
    detections_list.append(detection_record)
    
    print(f"ğŸ’¾ Saved detection: {filename} (confidence: {confidence:.3f})")

def create_detection_map(detections_df, river_segments):
    """
    Create an interactive map showing all detections
    """
    if detections_df.empty:
        print("No detections to map")
        return None
    
    center_lat = detections_df['latitude'].mean()
    center_lon = detections_df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add satellite tile layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add terrain layer
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
        attr='Stamen Terrain',
        name='Stamen Terrain',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Create feature groups
    detections_group = folium.FeatureGroup(name="Archaeological Detections")
    rivers_group = folium.FeatureGroup(name="River Segments")
    
    # Add detections
    for idx, row in detections_df.iterrows():
        if row['confidence'] >= 0.9:
            color = 'red'
            confidence_level = 'Very High'
        elif row['confidence'] >= 0.8:
            color = 'orange'
            confidence_level = 'High'
        elif row['confidence'] >= 0.7:
            color = 'yellow'
            confidence_level = 'Medium'
        else:
            color = 'lightgreen'
            confidence_level = 'Low'
        
        popup_text = f"""
        <b>Archaeological Site Detection</b><br>
        <b>Confidence:</b> {row['confidence']:.3f} ({confidence_level})<br>
        <b>River:</b> {row['river_name']}<br>
        <b>Coordinates:</b> {row['latitude']:.6f}, {row['longitude']:.6f}<br>
        <b>Date:</b> {row['detection_date']}<br>
        <b>Time:</b> {row['detection_time']}<br>
        <b>Image:</b> {row['filename']}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            fillColor=color,
            fillOpacity=0.8,
            weight=2
        ).add_to(detections_group)
    
    # Add river segments (sample for performance)
    display_rivers = river_segments[:100] if len(river_segments) > 100 else river_segments
    
    for river in display_rivers:
        if len(river['coordinates']) >= 2:
            folium.PolyLine(
                locations=river['coordinates'],
                popup=f"<b>River:</b> {river['name']}",
                color='blue',
                weight=2,
                opacity=0.6
            ).add_to(rivers_group)
    
    detections_group.add_to(m)
    rivers_group.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    plugins.MiniMap().add_to(m)
    plugins.MeasureControl(primary_length_unit='kilometers').add_to(m)
    plugins.Fullscreen().add_to(m)
    
    return m

def analyze_coverage_efficiency(detections_list):
    """
    Analyze the efficiency of the detection coverage
    """
    if len(detections_list) < 2:
        return
    
    print(f"\nğŸ“Š Coverage Efficiency Analysis:")
    
    distances = []
    coords = [(d['latitude'], d['longitude']) for d in detections_list]
    
    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]
        
        lat_diff = (lat2 - lat1) * 111000
        lon_diff = (lon2 - lon1) * 111000 * math.cos(math.radians(lat1))
        distance = math.sqrt(lat_diff**2 + lon_diff**2)
        distances.append(distance)
    
    if distances:
        avg_distance = sum(distances) / len(distances)
        min_distance = min(distances)
        max_distance = max(distances)
        
        print(f"   - Average distance between detections: {avg_distance:.0f}m")
        print(f"   - Minimum distance between detections: {min_distance:.0f}m")
        print(f"   - Maximum distance between detections: {max_distance:.0f}m")
        print(f"   - Image coverage area: {Config.IMAGE_COVERAGE_M:.0f}m Ã— {Config.IMAGE_COVERAGE_M:.0f}m")
        
        overlap_count = sum(1 for d in distances if d < Config.IMAGE_COVERAGE_M * 0.5)
        print(f"   - Potential overlapping detections: {overlap_count}/{len(distances)}")

def main():
    """
    Main function for enhanced archaeological site detection
    """
    print("ğŸ�›ï¸� Archaeological Site Detection with Enhanced Attention Model")
    print("=" * 70)
    
    # Display optimization info
    print(f"ğŸ”§ Detection Settings:")
    print(f"   - Model: Enhanced EfficientNet with Advanced Attention")
    print(f"   - Image coverage: {Config.IMAGE_COVERAGE_M:.0f}m Ã— {Config.IMAGE_COVERAGE_M:.0f}m")
    print(f"   - Sampling distance: {Config.SAMPLE_DISTANCE_M}m")
    print(f"   - Target overlap: {Config.OVERLAP_PERCENTAGE}%")
    print(f"   - Confidence threshold: {Config.CONFIDENCE_THRESHOLD}")
    
    # Check Google Maps API key
    if Config.GOOGLE_MAPS_API_KEY == 'YOUR_GOOGLE_MAPS_API_KEY':
        print("âš ï¸� WARNING: Please set your Google Maps API key in Config.GOOGLE_MAPS_API_KEY")
        return
    
    # Load model and data
    model = load_enhanced_model()
    river_segments = load_river_segments()
    if not river_segments:
        return
    
    # Limit rivers for processing
    if Config.MAX_RIVERS_TO_PROCESS and len(river_segments) > Config.MAX_RIVERS_TO_PROCESS:
        print(f"âš ï¸� Limiting to {Config.MAX_RIVERS_TO_PROCESS} rivers (out of {len(river_segments)})")
        import random
        random.seed(42)
        river_segments = random.sample(river_segments, Config.MAX_RIVERS_TO_PROCESS)
    
    # Initialize tracking
    detections_list = []
    total_images_processed = 0
    total_detections = 0
    
    print(f"\nğŸ”� Starting enhanced detection process...")
    
    # Process rivers
    for river_idx, river in enumerate(river_segments):
        print(f"\nğŸŒŠ Processing river {river_idx + 1}/{len(river_segments)}: {river['name']}")
        
        # Use optimized sampling
        sample_points = sample_points_along_river_optimized(river['coordinates'])
        
        # Limit points per river
        if Config.MAX_POINTS_PER_RIVER and len(sample_points) > Config.MAX_POINTS_PER_RIVER:
            step = len(sample_points) // Config.MAX_POINTS_PER_RIVER
            sample_points = sample_points[::step][:Config.MAX_POINTS_PER_RIVER]
        
        print(f"   ğŸ“� Sampling {len(sample_points)} points")
        
        if not sample_points:
            continue
        
        # Process each sample point
        for point_idx, (lat, lon) in enumerate(sample_points):
            if point_idx % 5 == 0:
                print(f"   Processing point {point_idx + 1}/{len(sample_points)}")
            
            # Download and classify
            image = download_satellite_image(lat, lon)
            if image is None:
                continue
            
            predicted_class, confidence = classify_image(model, image)
            total_images_processed += 1
            
            # Check for detection
            if predicted_class == 1 and confidence >= Config.CONFIDENCE_THRESHOLD:
                print(f"   ğŸ�¯ DETECTION! Lat: {lat:.6f}, Lon: {lon:.6f}, Conf: {confidence:.3f}")
                save_detection(image, lat, lon, confidence, river, detections_list, model)
                total_detections += 1
            
            # Rate limiting
            time.sleep(Config.DELAY_BETWEEN_REQUESTS)
        
        print(f"   âœ… River completed: {len([d for d in detections_list if d['river_id'] == river['id']])} detections")
    
    # Process results
    if detections_list:
        print(f"\nğŸ”� Post-processing detections...")
        
        # Analyze coverage efficiency
        analyze_coverage_efficiency(detections_list)
        
        # Remove clustered detections
        filtered_detections = remove_clustered_detections(detections_list, min_distance_m=300)
        
        # Save filtered results
        detections_df = pd.DataFrame(filtered_detections)
        detections_df.to_csv(Config.CSV_OUTPUT, index=False)
        print(f"\nğŸ’¾ Saved {len(filtered_detections)} unique detections to {Config.CSV_OUTPUT}")
        
        # Create map
        print("ğŸ—ºï¸� Creating detection map...")
        detection_map = create_detection_map(detections_df, river_segments)
        if detection_map:
            detection_map.save(Config.MAP_OUTPUT)
            print(f"ğŸ’¾ Detection map saved to {Config.MAP_OUTPUT}")
        
        # Final statistics
        print(f"\nğŸ“ˆ Final Results:")
        print(f"   - Images processed: {total_images_processed}")
        print(f"   - Raw detections: {total_detections}")
        print(f"   - Unique sites: {len(filtered_detections)}")
        print(f"   - Detection rate: {total_detections/total_images_processed*100:.2f}%")
        print(f"   - Average confidence: {detections_df['confidence'].mean():.3f}")
        print(f"   - Clustering reduction: {total_detections - len(filtered_detections)} duplicates removed")
        
        # Confidence distribution
        high_conf = len(detections_df[detections_df['confidence'] >= 0.9])
        med_conf = len(detections_df[(detections_df['confidence'] >= 0.7) & (detections_df['confidence'] < 0.9)])
        
        print(f"\nğŸ“Š Confidence Distribution:")
        print(f"   - High confidence (â‰¥0.9): {high_conf}")
        print(f"   - Medium confidence (0.7-0.9): {med_conf}")
        
        print(f"\nğŸ�¨ Attention visualizations saved in: {Config.ATTENTION_VIZ_DIR}")
        
    else:
        print(f"\nâ�Œ No sites detected above threshold {Config.CONFIDENCE_THRESHOLD}")
    
    print(f"\nâœ… Enhanced detection completed!")
    print(f"ğŸ“� All results saved in: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()

