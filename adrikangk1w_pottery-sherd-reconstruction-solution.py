import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


BASE_DIR = '/kaggle/input/h690/h690/h690'
CSV_PATH = os.path.join(BASE_DIR, 'jd_sherds_info.csv')
IMG_DIR = os.path.join(BASE_DIR, 'sherd_images')

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded metadata. Total sherds: {len(df)}")
    display(df.head())
else:
    print(f"ERROR: CSV file not found at {CSV_PATH}. Please check the path.")


class FeatureExtractor:
    def __init__(self):
        # We use the model up to the average pooling layer to get a 2048-d vector
        print("Loading ResNet50 model...")
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_deep_features(self, img_path):
        """Extracts 2048-d deep features from the image."""
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = self.preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = self.model(img_t)
            return emb.cpu().numpy().flatten()
        except Exception as e:
            return np.zeros(2048)

    def get_color_features(self, img_path):
        """Extracts HSV color histogram."""
        try:
            img = cv2.imread(img_path)
            if img is None: return np.zeros(48)
            
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # We focus on Hue (Color) and Value (Lightness)
            hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 2, 3], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except Exception as e:
            return np.zeros(48)

extractor = FeatureExtractor()


features = []
image_ids = df['image_id'].tolist()
valid_indices = []

print("Starting feature extraction...")
for idx, img_id in enumerate(tqdm(image_ids)):
    img_filename = f"{img_id}.jpg"
    img_path = os.path.join(IMG_DIR, img_filename)
    
    if not os.path.exists(img_path):
        continue
        
    deep_feat = extractor.get_deep_features(img_path)
    color_feat = extractor.get_color_features(img_path)

    norm_deep = np.linalg.norm(deep_feat)
    if norm_deep > 0:
        deep_feat = deep_feat / norm_deep
        
    combined = np.concatenate([deep_feat, color_feat * 0.5])
    
    features.append(combined)
    valid_indices.append(idx)

features_matrix = np.array(features)
print(f"Feature Extraction Complete. Matrix Shape: {features_matrix.shape}")


df['Assembly Group'] = -1
group_id_counter = 1

units = df['unit'].unique()

print(f"Clustering across {len(units)} units...")

for unit in tqdm(units):
    unit_mask = df['unit'] == unit
    unit_indices = df.index[unit_mask]
    
    local_feats = []
    local_df_indices = []
    
    for i in unit_indices:
        if i in valid_indices:
            feat_idx = valid_indices.index(i)
            local_feats.append(features_matrix[feat_idx])
            local_df_indices.append(i)
            
    if len(local_feats) == 0:
        continue
        
    local_feats = np.array(local_feats)

    if len(local_feats) == 1:
        df.loc[local_df_indices[0], 'Assembly Group'] = f"AssemblyGroup{group_id_counter}"
        group_id_counter += 1
        continue
        
    sim_matrix = cosine_similarity(local_feats)
    dist_matrix = 1 - sim_matrix
    dist_matrix[dist_matrix < 0] = 0
    
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.25, 
        metric='precomputed',
        linkage='average'
    )
    
    labels = clustering.fit_predict(dist_matrix)
    
    for lbl in np.unique(labels):
        members_mask = (labels == lbl)
        member_indices = [local_df_indices[j] for j in range(len(members_mask)) if members_mask[j]]
        
        df.loc[member_indices, 'Assembly Group'] = f"AssemblyGroup{group_id_counter}"
        group_id_counter += 1

print(f"Clustering Complete. Total Assembly Groups created: {group_id_counter - 1}")


missed_mask = df['Assembly Group'] == -1
if missed_mask.any():
    print(f"Warning: {missed_mask.sum()} images were not clustered. Assigning unique groups.")
    for idx in df[missed_mask].index:
        df.loc[idx, 'Assembly Group'] = f"AssemblyGroup{group_id_counter}"
        group_id_counter += 1

valid_df = df.iloc[valid_indices].copy()

submission = valid_df[['image_id', 'Assembly Group']].copy()
print(f"Before removing duplicates: {len(submission)} rows")
submission = submission.drop_duplicates(subset=['image_id'], keep='first')
print(f"After removing duplicates: {len(submission)} rows")

submission.columns = ['image_id', 'assembly_id']
submission = submission.sort_values('image_id').reset_index(drop=True)

duplicates = submission['image_id'].duplicated().sum()
if duplicates > 0:
    print(f"⚠️ WARNING: Found {duplicates} duplicate image_ids!")
    submission = submission.drop_duplicates(subset=['image_id'], keep='first')
else:
    print("✅ No duplicate image_ids found.")

submission.to_csv('submission.csv', index=False)
print(f"\n✅ Submission saved to 'submission.csv' with {len(submission)} rows.")
print(f"Expected: 35159 rows")
print(f"Difference: {35159 - len(submission)} rows")

display(submission.head(10))
print(f"\nTotal images in submission: {len(submission)}")
print(f"Unique image_ids: {submission['image_id'].nunique()}")
print(f"Unique assembly_ids: {submission['assembly_id'].nunique()}")

