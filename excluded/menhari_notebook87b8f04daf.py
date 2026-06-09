import os
import torch
import pandas as pd
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
BATCH_SIZE = 64
DATA_ROOT = "/kaggle/input/h690"   # dataset root
OUTPUT_PATH = "submission.csv"

# =========================================================
# LOAD IMAGE PATHS
# =========================================================
# Recursively collect all image files (jpg, png, jpeg)
image_paths = []
for root, dirs, files in os.walk(DATA_ROOT):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, f))

print(f"âœ… Found {len(image_paths)} images.")

# Create dataframe
metadata = pd.DataFrame({
    "image": [os.path.basename(p) for p in image_paths],
    "path": image_paths
})

# =========================================================
# MODEL + FEATURE EXTRACTOR
# =========================================================
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()  # remove classification layer
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_features(paths):
    features = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Extracting features"):
        batch_paths = paths[i:i+BATCH_SIZE]
        imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        imgs = torch.stack(imgs).to(DEVICE)
        feats = model(imgs).cpu().numpy()
        features.append(feats)
    return np.vstack(features)

# =========================================================
# FEATURE EXTRACTION
# =========================================================
features = extract_features(metadata["path"].values)
features = normalize(features)

# =========================================================
# SIMPLE CLUSTERING (cosine similarity)
# =========================================================
sim = cosine_similarity(features)
threshold = 0.75  # tweakable
groups = []
visited = set()

for i in tqdm(range(len(metadata)), desc="Clustering"):
    if i in visited:
        continue
    neighbors = np.where(sim[i] > threshold)[0]
    cluster = list(neighbors)
    for n in neighbors:
        visited.add(n)
    groups.append(cluster)

# Map each image to a group/component ID
image_to_group = {}
for group_id, cluster in enumerate(groups):
    for idx in cluster:
        image_to_group[metadata.iloc[idx]["image"]] = group_id

# =========================================================
# SAVE SUBMISSION
# =========================================================
submission = pd.DataFrame({
    "image": metadata["image"],
    "component": metadata["image"].map(image_to_group)
})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"ðŸŽ‰ Submission saved to {OUTPUT_PATH}")


