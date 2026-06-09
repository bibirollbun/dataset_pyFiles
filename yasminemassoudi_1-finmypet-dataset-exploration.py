# Step 1: Set up paths for Kaggle
from pathlib import Path
import os

# Kaggle data path
DATA_DIR = Path('/kaggle/input/petfinder-adoption-prediction')

print(f" Data directory: {DATA_DIR}")
print(f" Exists: {DATA_DIR.exists()}")

# List contents
if DATA_DIR.exists():
    print("\n Contents:")
    for item in sorted(DATA_DIR.iterdir()):
        print(f"   - {item.name}")


# Step 2: Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print(" All packages imported successfully!")


# Step 3: Load data
train_csv = DATA_DIR / 'train/train.csv'
test_csv = DATA_DIR / 'test/test.csv'

print(" Loading CSV files...")
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

print(f"Train dataset: {len(train_df)} records")
print(f" Test dataset: {len(test_df)} records")
print(f"\n Train columns: {list(train_df.columns)}")


# Step 4: Display sample data
print(" Train Dataset Sample:")
print(train_df.head())
print(f"\n Dataset Info:")
print(train_df.info())


# Step 5: Check images
train_images = DATA_DIR / 'train_images'
test_images = DATA_DIR / 'test_images'

print(" Checking images...")
train_img_list = list(train_images.glob('*.jpg'))
test_img_list = list(test_images.glob('*.jpg'))

print(f" Train images: {len(train_img_list)}")
print(f" Test images: {len(test_img_list)}")


# Step 6: Create visualizations
print(" Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Type distribution
if 'Type' in train_df.columns:
    train_df['Type'].value_counts().plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Pet Type Distribution')
    axes[0, 0].set_xlabel('Type')
    axes[0, 0].set_ylabel('Count')

# Age distribution
if 'Age' in train_df.columns:
    train_df['Age'].hist(bins=30, ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Age Distribution')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Count')

# Gender distribution
if 'Gender' in train_df.columns:
    train_df['Gender'].value_counts().plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Gender Distribution')
    axes[1, 0].set_xlabel('Gender')
    axes[1, 0].set_ylabel('Count')

# Adoption speed
if 'AdoptionSpeed' in train_df.columns:
    train_df['AdoptionSpeed'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1], color='gold')
    axes[1, 1].set_title('Adoption Speed Distribution')
    axes[1, 1].set_xlabel('Adoption Speed')
    axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('distributions.png', dpi=100, bbox_inches='tight')
plt.show()

print("âœ… Distributions saved")


# Step 7: Display sample images
print("ðŸ“¸ Displaying sample images...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, ax in enumerate(axes.flat):
    if idx < len(train_img_list):
        try:
            img = Image.open(train_img_list[idx])
            ax.imshow(img)
            ax.set_title(f'Image {idx+1}')
            ax.axis('off')
        except:
            ax.text(0.5, 0.5, 'Error loading image', ha='center', va='center')
            ax.axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=100, bbox_inches='tight')
plt.show()

print("âœ… Sample images displayed")


# Step 8: Summary statistics
print("\n" + "="*80)
print("ðŸ“Š DATASET EXPLORATION SUMMARY")
print("="*80)
print(f"\nTrain records: {len(train_df):,}")
print(f"Test records: {len(test_df):,}")
print(f"Train images: {len(train_img_list):,}")
print(f"Test images: {len(test_img_list):,}")
print(f"\nFeatures: {len(train_df.columns)}")
print(f"\nMissing values:\n{train_df.isnull().sum()}")
print("\n" + "="*80)
print("âœ… DATASET EXPLORATION COMPLETE!")
print("="*80)

