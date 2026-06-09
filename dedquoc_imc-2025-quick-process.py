# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import matplotlib.pyplot as plt
from PIL import Image

# Define dataset path and folder to load
base_path = "/kaggle/input/image-matching-challenge-2025/test"
folder_name = "ETs"  # Change this to any folder you want to explore
folder_path = os.path.join(base_path, folder_name)

# Get all PNG image paths in the folder
image_paths = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

print(f"Found {len(image_paths)} images in folder: {folder_name}")

# Lazy load and preview first 5 images (adjust if needed)
n_show = 5
plt.figure(figsize=(15, 3))
for i, img_name in enumerate(image_paths[:n_show]):
    img_path = os.path.join(folder_path, img_name)
    img = Image.open(img_path)
    
    plt.subplot(1, n_show, i + 1)
    plt.imshow(img)
    plt.title(img_name, fontsize=8)
    plt.axis("off")
plt.suptitle(f"Preview: {folder_name}", fontsize=14)
plt.show()



!pip install open-clip-torch
!pip install tqdm
!pip install transformers
!pip install --quiet git+https://github.com/openai/CLIP.git


import torch
import clip
from PIL import Image
from tqdm import tqdm
import os

# Load CLIP model (ViT-B/32 is a good balance of speed and performance)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def extract_clip_embeddings(folder_path, image_names):
    embeddings = []
    names = []

    for name in tqdm(image_names, desc="Embedding images"):
        path = os.path.join(folder_path, name)
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image).cpu().squeeze(0)
        
        embeddings.append(embedding)
        names.append(name)

    embeddings_tensor = torch.stack(embeddings)
    return names, embeddings_tensor



# Choose test folder
folder_name = "ETs"
folder_path = f"/kaggle/input/image-matching-challenge-2025/test/{folder_name}"
image_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

# Extract CLIP embeddings
names, embs = extract_clip_embeddings(folder_path, image_names)
print(f"Extracted shape: {embs.shape} (num_images, 512)")


import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Load CLIP model (ViT-B/32 is lightweight)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Pick a test folder to avoid OOM — e.g., ETs
folder_path = "/kaggle/input/image-matching-challenge-2025/test/ETs"
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

# Extract CLIP features (image embeddings)
image_features = {}
with torch.no_grad():
    for image_name in tqdm(image_files, desc="Extracting CLIP features"):
        img_path = os.path.join(folder_path, image_name)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        features = model.encode_image(image)
        image_features[image_name] = features.squeeze(0).cpu()  # shape: (512,)

print(f"Extracted features for {len(image_features)} images in '{folder_path.split('/')[-1]}'")



from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import numpy as np

# Stack all features into matrix
feature_matrix = torch.stack(list(image_features.values())).numpy()
feature_matrix = normalize(feature_matrix)  # L2 normalization for cosine distance

# DBSCAN with cosine metric (eps controls granularity)
clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(feature_matrix)

# Get image -> cluster label mapping
labels = clustering.labels_  # -1 = outlier
image_to_cluster = dict(zip(image_features.keys(), labels))

# Print clustering result
from collections import Counter
print(Counter(labels))


import pandas as pd

dataset_name = 'ETs'
submission_rows = []

for image_name, cluster_id in image_to_cluster.items():
    if cluster_id == -1:
        scene_label = 'outliers'
        rot = ';'.join(['nan'] * 9)
        trans = ';'.join(['nan'] * 3)
    else:
        scene_label = f'cluster{cluster_id + 1}'
        rot = ';'.join(['nan'] * 9)
        trans = ';'.join(['nan'] * 3)

    submission_rows.append({
        'dataset': dataset_name,
        'scene': scene_label,
        'image': image_name,
        'rotation_matrix': rot,
        'translation_vector': trans
    })

submission_df = pd.DataFrame(submission_rows)
submission_df.head()

# Added line to save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
submission_df.head()

