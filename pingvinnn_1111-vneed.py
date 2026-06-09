from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
from torch import nn


class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)

    def forward(self, image):
        x = self.model(image)
        return x


def generate_submit(pred_cluster):
    import hashlib
    sub = pd.DataFrame()
    sub['id'] = np.arange(len(pred_cluster))
    sub['target'] = pred_cluster
    hsh = hashlib.sha256(sub.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
    submit_path = f"submit_{hsh}.csv"
    print(f"SUBMIT_NAME: {submit_path}")
    print(sub.head(10))
    sub.to_csv(submit_path, index = None)


X_1 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_1.npz')
X_1 = X_1.f.arr_0
X_2 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_2.npz')
X_2 = X_2.f.arr_0

km = KMeans(32)
X = np.concatenate((X_1.reshape((X_1.shape[0], X_1.shape[1] * X_1.shape[2])), X_2.reshape((X_2.shape[0], X_2.shape[1] * X_2.shape[2]))), 1)
pred_cluster = km.fit_predict(X)

generate_submit(pred_cluster)


import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm


from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

class ImageDataset(Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        if self.transform:
            image = self.transform(image)
        return image

# Assuming X_1 and X_2 are numpy arrays or PIL images
dataset_1 = ImageDataset(X_1, transform=transform)
dataset_2 = ImageDataset(X_2, transform=transform)

dataloader_1 = DataLoader(dataset_1, batch_size=64, shuffle=False)
dataloader_2 = DataLoader(dataset_2, batch_size=64, shuffle=False)

model = EmbNet().eval()

# Ğ˜Ğ·Ğ²Ğ»ĞµÑ‡ĞµĞ½Ğ¸Ğµ Ñ�Ğ¼Ğ±ĞµĞ´Ğ´Ğ¸Ğ½Ğ³Ğ¾Ğ²
def extract_embeddings(dataloader, model):
    embeddings = []
    with torch.no_grad():
        for images in dataloader:
            outputs = model(images)
            embeddings.append(outputs.cpu().numpy())
    return np.vstack(embeddings)

embeddings_1 = extract_embeddings(dataloader_1, model)
embeddings_2 = extract_embeddings(dataloader_2, model)

# Ğ£Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ½Ğ¸Ğµ Ñ�Ğ¼Ğ±ĞµĞ´Ğ´Ğ¸Ğ½Ğ³Ğ¾Ğ²
embeddings = (embeddings_1 + embeddings_2) / 2

# ĞšĞ»Ğ°Ñ�Ñ‚ĞµÑ€Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
kmeans = KMeans(n_clusters=32, random_state=0)
clusters = kmeans.fit_predict(embeddings)

print(clusters)


from collections import defaultdict
import numpy as np

# ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ğ¼Ğ°Ñ�Ñ�Ğ¸Ğ²Ğ¾Ğ² ĞºĞ»Ğ°Ñ�Ñ‚ĞµÑ€Ğ¾Ğ²
clusters_1 = pred_cluster
clusters_2 = clusters

# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€ÑŒ Ğ´Ğ»Ñ� Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ� Ñ�Ğ¾Ğ²Ğ¿Ğ°Ğ´ĞµĞ½Ğ¸Ğ¹
match_dict = defaultdict(lambda: defaultdict(int))

# Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½Ñ�ĞµĞ¼ Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€ÑŒ Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ñ�Ğ¾Ğ²Ğ¿Ğ°Ğ´ĞµĞ½Ğ¸Ğ¹ Ğ¼ĞµĞ¶Ğ´Ñƒ ĞºĞ»Ğ°Ñ�Ñ‚ĞµÑ€Ğ°Ğ¼Ğ¸
for c1, c2 in zip(clusters_1, clusters_2):
    match_dict[c1][c2] += 1

# Ğ�Ğ¿Ñ€ĞµĞ´ĞµĞ»Ñ�ĞµĞ¼ Ğ½Ğ°Ğ¸Ğ»ÑƒÑ‡ÑˆĞµĞµ Ñ�Ğ¾Ğ¾Ñ‚Ğ²ĞµÑ‚Ñ�Ñ‚Ğ²Ğ¸Ğµ Ğ´Ğ»Ñ� ĞºĞ°Ğ¶Ğ´Ğ¾Ğ³Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ° Ğ¸Ğ· clusters_1
final_mapping = {}
for c1, matches in match_dict.items():
    # Ğ�Ğ°Ñ…Ğ¾Ğ´Ğ¸Ğ¼ ĞºĞ»Ğ°Ñ�Ñ� Ğ¸Ğ· clusters_2 Ñ� Ğ¼Ğ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğ¹ Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ¾Ğ¹ Ñ�Ğ¾Ğ²Ğ¿Ğ°Ğ´ĞµĞ½Ğ¸Ñ�
    best_match = max(matches, key=matches.get)
    final_mapping[c1] = best_match

# ĞŸÑ€Ğ¸Ğ²Ğ¾Ğ´Ğ¸Ğ¼ clusters_1 Ğº Ğ²Ğ¸Ğ´Ñƒ clusters_2
unified_clusters_1 = np.array([final_mapping[c] for c in clusters_1])

print("ĞšĞ»Ğ°Ñ�Ñ‚ĞµÑ€Ñ‹ Ğ¿Ğ¾Ñ�Ğ»Ğµ Ğ¿Ñ€Ğ¸Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ� Ğº Ğ¾Ğ´Ğ½Ğ¾Ğ¼Ñƒ Ğ²Ğ¸Ğ´Ñƒ:")
print("Unified Clusters 1:", unified_clusters_1)
print("Clusters 2:", clusters_2)


clusters_1 = clusters_1
clusters_2 = clusters_2
# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ‚ÑŒ Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ñƒ Ğ¿ĞµÑ€ĞµÑ�ĞµÑ‡ĞµĞ½Ğ¸Ñ�
confusion_matrix = pair_confusion_matrix(clusters_1, clusters_2)
confusion_matrix



# ĞœĞ°Ğ¶Ğ¾Ñ€Ğ¸Ñ‚Ğ°Ñ€Ğ½Ğ¾Ğµ Ğ³Ğ¾Ğ»Ğ¾Ñ�Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ
combined_clusters = mode(np.vstack((clusters_2, unified_clusters_1)), axis=0).mode.flatten()

print("Aligned Clusters 2:", unified_clusters_1)
print("Combined Clusters:", combined_clusters)


generate_submit(combined_clusters)




