# === Imports ====
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision
import timm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel

sns.set(style='whitegrid')


root_dir ="/kaggle/input/animal-clef-2025"
metadata_path = "/kaggle/input/animal-clef-2025/metadata.csv"


metadata_df = pd.read_csv(metadata_path)
metadata_df


# ==== Counting the no of images v/s identity in database

identity_counts = metadata_df[metadata_df['split'] == 'database']['identity'].value_counts().reset_index()
identity_counts.columns = ['identity', 'num_images']
identity_counts


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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# ==== Loading model and transformations ====
model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
model = model.to(device)
model = model.eval()

transform = transforms.Compose([
    transforms.Resize(size=(384, 384)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# === Function to extract embeddings ===
def extract_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    # Normalize embedding
    embedding = F.normalize(output, p=2, dim=1).squeeze(0)
    return embedding


# === Looping over database and storing embeddings and metadata
database_embeddings = []

for _, row in metadata_df.iterrows():
    if row["split"] == "database":
        img_path = os.path.join(root_dir, row["path"])
        embed = extract_embedding(img_path)
        # print(len(embed))
        # break

        database_embeddings.append({
            "embedding" : embed,
            "identity" : row["identity"],
            "orientation" : row["orientation"]
        })


len(database_embeddings)


# identifying similar patterns in query images 

threshold = 0.68
query_results = []
for _, row in metadata_df.iterrows():
    if row["split"] == "query":
        img_path = os.path.join(root_dir, row["path"])
        query_embed = extract_embedding(img_path)
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


output_df.to_csv("submission.csv")




