#import os

#data_path = '/kaggle/input/isic-2024-challenge'  # Use the actual folder name you see

#print("Top-level contents:")
#print(os.listdir(data_path))

# If there's a 'data' folder inside, list its contents:
#data_folder = os.path.join(data_path, 'data')
#if os.path.exists(data_folder):
#    print("\nContents of 'data' folder:")
#    print(os.listdir(data_folder))
#else:
 #   print("\nNo 'data' folder found in the dataset.")



# Kaggle Notebook: Process ISIC Images into Flattened CSV

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm  # progress bar

# 1. Paths (adjust dataset folder name if different)
DATA_DIR    = '/kaggle/input/isic-2024-challenge'
META_CSV    = os.path.join(DATA_DIR, 'train-metadata.csv')
IMAGE_DIR   = os.path.join(DATA_DIR, 'train-image', 'image')

# 2. Load metadata and sample IDs
df = pd.read_csv(META_CSV, low_memory=False)
positive_ids = df[df['target'] == 1]['isic_id'].tolist()
negative_ids = df[df['target'] == 0]['isic_id'].tolist()
negative_ids = random.sample(negative_ids, len(positive_ids))

# 3. Combine and prepare labels
ids    = positive_ids + negative_ids
labels = [1]*len(positive_ids) + [0]*len(negative_ids)

# 4. Process each image: convert to 128×128 grayscale & flatten
data = []
for isic_id, label in tqdm(zip(ids, labels), total=len(ids)):
    found = False
    for ext in ('.jpg', '.png'):
        path = os.path.join(IMAGE_DIR, f"{isic_id}{ext}")
        if os.path.exists(path):
            img = Image.open(path).convert('L').resize((128, 128))
            pixels = np.array(img).flatten().tolist()
            data.append([isic_id, label] + pixels)
            found = True
            break
    if not found:
        print(f"⚠️ Missing image for ID: {isic_id}")

# 5. Build DataFrame & export
cols = ['isic_id', 'label'] + [f'pixel_{i}' for i in range(128*128)]
df_images = pd.DataFrame(data, columns=cols)
df_images.to_csv('image_data_128.csv', index=False)

# 6. Preview
df_images.head()


import pandas as pd
train_meta = pd.read_csv(
    '/kaggle/input/isic-2024-challenge/train-metadata.csv',
    low_memory=False
)
test_meta = pd.read_csv(
    '/kaggle/input/isic-2024-challenge/test-metadata.csv',
    low_memory=False
)
common_cols = train_meta.columns.intersection(test_meta.columns).tolist()
train_meta_filtered = train_meta[common_cols]

print("Filtered training metadata columns:")
print(train_meta_filtered.columns)
train_meta_filtered = train_meta_filtered[test_meta.columns.intersection(common_cols)]





