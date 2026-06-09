""" Imports """
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models, datasets
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ViTModel, ViTFeatureExtractor
from tqdm import tqdm


sns.set(style="whitegrid")


root_dir ="/kaggle/input/animal-clef-2025"
metadata_path = "/kaggle/input/animal-clef-2025/metadata.csv"


metadata_df = pd.read_csv(metadata_path)
metadata_df


metadata_df.info()


metadata_df.isna().sum()


metadata_df[metadata_df["split"] == "database"].isna().sum()


metadata_df[metadata_df["split"] == "query"].isna().sum()


# Count images per class and split
class_distribution = metadata_df.groupby(["dataset", "split"]).size().unstack(fill_value=0)

# Plot
plt.figure(figsize=(12, 6))
ax = class_distribution.plot(kind="bar", colormap="coolwarm", figsize=(12, 6), edgecolor="black")

# Add value labels on bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),  
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Labels and title
plt.xlabel("Class (Dataset Column)")
plt.ylabel("Number of Images")
plt.title("Number of Images in Database vs Query per Class")
plt.legend(title="Split")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show plot
plt.show()


# Visualizing the count of images v/s orientation

plt.figure(figsize=(8, 5))
sns.countplot(data=metadata_df, x="orientation", order=metadata_df["orientation"].value_counts().index)
plt.xlabel("Orientation")
plt.ylabel("Count")
plt.title("Image Count per Orientation")
plt.show()



# Visualizing no of images in database v/s query

plt.figure(figsize=(6, 5))
sns.countplot(data=metadata_df, x="split")
plt.xlabel("Dataset Split")
plt.ylabel("Count")
plt.title("Number of Images in Database vs Query")
plt.show()



df = pd.read_csv("/kaggle/input/animal-clef-2025/sample_submission.csv")
df


# checking for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using the device: {device}")


model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name).to(device)
model.eval()


# Image transformation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = feature_extractor.image_mean, std = feature_extractor.image_std)
])


# Function to extract embeddings 
def extract_embeddings(image_path):
    img = Image.open(image_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_transformed)
        embeddings = output.last_hidden_state[:, 0] #CLS token

    return F.normalize(embeddings, p=2, dim=1).squeeze(0) # Normalizing for cosine similarity


database_embeddings = []

for _, row in metadata_df.iterrows():
    if row["split"] == "database":
        img_path = os.path.join(root_dir, row["path"])
        embed = extract_embeddings(img_path)
        # print(len(embed))
        # break

        database_embeddings.append({
            "embedding" : embed,
            "identity" : row["identity"],
            "orientation" : row["orientation"]
        })


len(database_embeddings)


database_embeddings[0]


# identifying similar patterns in query images 

threshold = 0.97
query_results = []
for _, row in metadata_df.iterrows():
    if row["split"] == "query":
        img_path = os.path.join(root_dir, row["path"])
        query_embed = extract_embeddings(img_path)
        query_orientation = row["orientation"]

        if pd.isna(query_orientation):
            filtered_embeddings = database_embeddings
        else:
            filtered_embeddings = [
                e for e in database_embeddings if e["orientation"] == query_orientation
            ]
            if not filtered_embeddings:
                filtered_embeddings = database_embeddings

        if not filtered_embeddings:
            query_results.append([row["image_id"], "new_individual"])
            continue

        embed_matrix = torch.stack([e["embedding"] for e in filtered_embeddings]).to(device)
        img_identities = [e["identity"] for e in filtered_embeddings]

        # compute similarity scores
        similarity_scores = torch.nn.functional.cosine_similarity(query_embed.unsqueeze(0), embed_matrix, dim=1)
        best_indx = torch.argmax(similarity_scores)

        print(f"Image: {row['path']} | Best Similarity: {similarity_scores[best_indx].item():.4f}")

        if similarity_scores[best_indx].item() > threshold:
            query_results.append([row["image_id"], img_identities[best_indx]])
        else:
            query_results.append([row["image_id"], "new_individual"])

            


output_df = pd.DataFrame(query_results, columns=['image_id', 'identity'])
output_df


output_df[output_df["identity"] == "new_individual"]


output_df.to_csv("submission.csv")




