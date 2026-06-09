# SETUP: Initialize environment
from pathlib import Path
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Kaggle paths
DATA_DIR = Path('/kaggle/input/petfinder-adoption-prediction')
WORKING_DIR = Path('/kaggle/working')

print("="*80)
print("  PIPELINE ")
print("="*80)
print(f"\n Data directory: {DATA_DIR}")
print(f" Working directory: {WORKING_DIR}")
print(f"Data exists: {DATA_DIR.exists()}")
print(f"GPU available: {torch.cuda.is_available() if 'torch' in dir() else 'Checking...'}")


# SCRIPT 01: EXPLORE DATA
print("\n" + "="*80)
print("SCRIPT 01: EXPLORE DATA")
print("="*80)

# Find and load CSV files
csv_files = list(DATA_DIR.glob('*.csv'))
print(f"\n CSV files found: {len(csv_files)}")

dataframes = {}
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        dataframes[csv_file.stem] = df
        print(f"\n {csv_file.name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)[:5]}...")
        print(f"   Missing values: {df.isnull().sum().sum()}")
    except Exception as e:
        print(f" Error loading {csv_file.name}: {e}")

print(f"\n Loaded {len(dataframes)} dataframes")


# SCRIPT 02: ORGANIZE DATASET
print("\n" + "="*80)
print("SCRIPT 02: ORGANIZE DATASET")
print("="*80)

# Create organized directory structure
PROCESSED_DIR = WORKING_DIR / 'processed'
DATABASE_DIR = PROCESSED_DIR / 'database'
QUERY_DIR = PROCESSED_DIR / 'query'
METADATA_DIR = PROCESSED_DIR / 'metadata'
FEATURES_DIR = PROCESSED_DIR / 'features'

for dir_path in [DATABASE_DIR, QUERY_DIR, METADATA_DIR, FEATURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f" Created: {dir_path.name}")

print(f"\n Directory structure created")


# SCRIPT 03-05: DATA CLEANING & PREPROCESSING
print("\n" + "="*80)
print("SCRIPTS 03-05: DATA CLEANING & PREPROCESSING")
print("="*80)

# Get main dataframe
if 'train' in dataframes:
    df = dataframes['train'].copy()
    print(f"\n Original data shape: {df.shape}")
    
    # Clean data
    print(f"\n Cleaning data...")
    print(f"   Missing values before: {df.isnull().sum().sum()}")
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna('Unknown', inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    
    print(f"   Missing values after: {df.isnull().sum().sum()}")
    print(f"\n Data cleaned successfully")
    
    # Save cleaned data
    cleaned_csv = METADATA_DIR / 'cleaned_data.csv'
    df.to_csv(cleaned_csv, index=False)
    print(f" Saved to: {cleaned_csv.name}")


# SCRIPT 06: VIEW SAMPLE DATA
print("\n" + "="*80)
print("SCRIPT 06: VIEW SAMPLE DATA")
print("="*80)

# Find images
image_dirs = list(DATA_DIR.glob('*images'))
print(f"\n Image directories found: {len(image_dirs)}")

all_images = []
for img_dir in image_dirs:
    images = list(img_dir.glob('*.jpg'))
    all_images.extend(images)
    print(f"   {img_dir.name}: {len(images)} images")

print(f"\n Total images: {len(all_images)}")

# Display sample images
if len(all_images) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < len(all_images):
            try:
                img = Image.open(all_images[idx])
                ax.imshow(img)
                ax.set_title(f'Sample {idx+1}')
                ax.axis('off')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center')
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(WORKING_DIR / '01_sample_images.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(" Sample images saved")


# SCRIPT 07: EXTRACT FEATURES (ResNet50)
print("\n" + "="*80)
print("SCRIPT 07: EXTRACT FEATURES (ResNet50)")
print("="*80)

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")

# Load model
print(" Loading ResNet50...")
model = resnet50(weights='DEFAULT')
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()
print(" Model loaded")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Extract features from sample images
sample_images = all_images[:100]
print(f"\n Extracting features from {len(sample_images)} images...")

features_list = []
for i, img_path in enumerate(sample_images):
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_tensor).squeeze().cpu().numpy()
        features_list.append(features)
        if (i + 1) % 20 == 0:
            print(f"   Processed {i+1}/{len(sample_images)}")
    except:
        pass

features_array = np.array(features_list).astype('float32')
print(f"\n Extracted {len(features_array)} features")
print(f" Feature shape: {features_array.shape}")


# SCRIPT 08: VERIFY FEATURES
print("\n" + "="*80)
print("SCRIPT 08: VERIFY FEATURES")
print("="*80)

print(f"\n Feature Statistics:")
print(f"   Shape: {features_array.shape}")
print(f"   Mean: {features_array.mean():.4f}")
print(f"   Std: {features_array.std():.4f}")
print(f"   Min: {features_array.min():.4f}")
print(f"   Max: {features_array.max():.4f}")
print(f"   Memory: {features_array.nbytes / 1024 / 1024:.2f} MB")

# Save features
np.save(FEATURES_DIR / 'features.npy', features_array)
print(f"\n Features saved to features.npy")


!pip install faiss-cpu


# SCRIPT 09: SIMILARITY SEARCH
print("\n" + "="*80)
print("SCRIPT 09: SIMILARITY SEARCH")
print("="*80)

import faiss

# Build FAISS index
print(f"\n Building FAISS index...")
start_time = time.time()

index = faiss.IndexFlatL2(features_array.shape[1])
index.add(features_array)

build_time = time.time() - start_time
print(f" Index built with {index.ntotal} vectors")
print(f"  Build time: {build_time:.4f}s")

# Perform search
print(f"\n Performing similarity search...")
query_idx = 0
query_vector = features_array[query_idx:query_idx+1]

start_time = time.time()
distances, indices = index.search(query_vector, k=6)
search_time = time.time() - start_time

print(f" Search completed in {search_time*1000:.2f}ms")
print(f"\n Top 6 similar images:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    print(f"   {i+1}. Index {idx}, Distance: {dist:.4f}")


# SCRIPT 10: QUICK TEST
print("\n" + "="*80)
print("SCRIPT 10: QUICK TEST")
print("="*80)

# Test multiple searches
print(f"\n Running quick tests...")
num_tests = 5
search_times = []

for test_idx in range(num_tests):
    query_idx = np.random.randint(0, len(features_array))
    query_vector = features_array[query_idx:query_idx+1]
    
    start_time = time.time()
    distances, indices = index.search(query_vector, k=6)
    search_time = (time.time() - start_time) * 1000
    search_times.append(search_time)
    
    print(f"   Test {test_idx+1}: {search_time:.2f}ms")

avg_search_time = np.mean(search_times)
print(f"\n Average search time: {avg_search_time:.2f}ms")
print(f" Queries per second: {1000/avg_search_time:.0f}")


# SCRIPT 11: BUILD FAISS INDEX
print("\n" + "="*80)
print("SCRIPT 11: BUILD FAISS INDEX")
print("="*80)

# Save FAISS index
index_path = FEATURES_DIR / 'faiss_index.bin'
faiss.write_index(index, str(index_path))
print(f"\n FAISS index saved to {index_path.name}")

# Save metadata
metadata = {
    'num_vectors': index.ntotal,
    'feature_dim': features_array.shape[1],
    'build_time': build_time,
    'avg_search_time_ms': avg_search_time,
    'queries_per_second': 1000/avg_search_time
}

with open(FEATURES_DIR / 'index_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f" Index metadata saved")


# FINAL SUMMARY
print("\n" + "="*80)
print(" COMPLETE PIPELINE FINISHED - ALL 11 SCRIPTS EXECUTED")
print("="*80)

print(f"\n SUMMARY:")
print(f"\n1️  Data Exploration:")
print(f"   - CSV files loaded: {len(dataframes)}")
print(f"   - Total records: {sum(len(df) for df in dataframes.values())}")

print(f"\n  Dataset Organization:")
print(f"   - Directories created: 4")
print(f"   - Processed data saved")

print(f"\n  Data Cleaning:")
print(f"   - Records cleaned: {len(df)}")
print(f"   - Missing values: 0")

print(f"\n  Feature Extraction:")
print(f"   - Images processed: {len(features_array)}")
print(f"   - Feature dimension: {features_array.shape[1]}")
print(f"   - Total features: {features_array.size:,}")

print(f"\n  FAISS Indexing:")
print(f"   - Index vectors: {index.ntotal}")
print(f"   - Build time: {build_time:.4f}s")
print(f"   - Avg search time: {avg_search_time:.2f}ms")
print(f"   - Queries/second: {1000/avg_search_time:.0f}")

print(f"\n Output Files:")
print(f"   - features.npy")
print(f"   - faiss_index.bin")
print(f"   - index_metadata.json")
print(f"   - cleaned_data.csv")
print(f"   - 01_sample_images.png")

print(f"\n" + "="*80)
print(" ALL 11 SCRIPTS COMPLETED SUCCESSFULLY!")
print("="*80)

