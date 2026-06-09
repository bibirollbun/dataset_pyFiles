# Step 1: Set up paths
from pathlib import Path
import os

DATA_DIR = Path('/kaggle/input/petfinder-adoption-prediction')
OUTPUT_DIR = Path('/kaggle/working')

print(f" Data directory: {DATA_DIR}")
print(f" Output directory: {OUTPUT_DIR}")


# Step 2: Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import warnings
warnings.filterwarnings('ignore')

print(" All packages imported successfully!")
print(f" PyTorch version: {torch.__version__}")
print(f" CUDA available: {torch.cuda.is_available()}")


# Step 3: Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ðŸ“± Device: {device}")

print("ðŸ”¨ Loading ResNet50 model...")
model = resnet50(weights='DEFAULT')
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

print(" Model loaded successfully!")
print(f" Feature dimension: 2048")


# Step 4: Define image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

print(" Image preprocessing pipeline ready")


# Step 5: Extract features from sample images
train_images = DATA_DIR / 'train_images'
train_img_list = list(train_images.glob('*.jpg'))[:100]  # First 100 images

print(f"Extracting features from {len(train_img_list)} images...")

features_list = []
for i, img_path in enumerate(train_img_list):
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model(img_tensor).squeeze().cpu().numpy()
        features_list.append(features)
        
        if (i + 1) % 20 == 0:
            print(f"   Processed {i+1}/{len(train_img_list)} images")
    except Exception as e:
        print(f"   Error processing {img_path.name}: {e}")

features_array = np.array(features_list)
print(f"\n Extracted {len(features_array)} feature vectors")
print(f" Feature shape: {features_array.shape}")


# Step 6: Analyze features
print(" Feature Statistics:")
print(f"   Mean: {features_array.mean():.4f}")
print(f"   Std: {features_array.std():.4f}")
print(f"   Min: {features_array.min():.4f}")
print(f"   Max: {features_array.max():.4f}")

# Per-feature statistics
feature_means = features_array.mean(axis=0)
feature_stds = features_array.std(axis=0)

print(f"\n Per-Feature Statistics:")
print(f"   Mean of means: {feature_means.mean():.4f}")
print(f"   Mean of stds: {feature_stds.mean():.4f}")


# Step 7: Visualize features
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature mean distribution
axes[0].hist(feature_means, bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('Feature Mean Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Mean Value')
axes[0].set_ylabel('Frequency')
axes[0].grid(alpha=0.3)

# Feature std distribution
axes[1].hist(feature_stds, bins=50, color='lightcoral', edgecolor='black')
axes[1].set_title('Feature Std Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Std Value')
axes[1].set_ylabel('Frequency')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print(" Feature analysis visualization saved")


# Step 8: Save features
np.save(OUTPUT_DIR / 'features.npy', features_array)
print(f" Features saved to features.npy")

# Save feature statistics
stats = {
    'num_images': len(features_array),
    'feature_dim': features_array.shape[1],
    'mean': float(features_array.mean()),
    'std': float(features_array.std()),
    'min': float(features_array.min()),
    'max': float(features_array.max())
}

import json
with open(OUTPUT_DIR / 'feature_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"Feature statistics saved to feature_stats.json")


# Step 9: Summary
print("\n" + "="*80)
print(" FEATURE EXTRACTION SUMMARY")
print("="*80)
print(f"\nImages processed: {len(features_array)}")
print(f"Feature dimension: {features_array.shape[1]}")
print(f"Total features: {features_array.size:,}")
print(f"\nFeature statistics:")
print(f"   Mean: {features_array.mean():.4f}")
print(f"   Std: {features_array.std():.4f}")
print(f"   Range: [{features_array.min():.4f}, {features_array.max():.4f}]")
print(f"\nOutput files:")
print(f"   - features.npy")
print(f"   - feature_stats.json")
print(f"   - feature_analysis.png")
print("\n" + "="*80)
print(" FEATURE EXTRACTION COMPLETE!")
print("="*80)

