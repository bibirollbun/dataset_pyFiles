!pip install faiss-gpu


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
import faiss
import numpy as np
import math
import pandas as pd
import os
from tqdm import tqdm


# Paths based on Kaggle directory structure
train_dir = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
dataset_path = "/kaggle/input/imagenet-validation-dataset"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])


# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset_full = datasets.ImageFolder(root=dataset_path, transform=transform)
val_dataset_centroids = torch.utils.data.Subset(val_dataset_full, range(10000))
val_dataset_properties = torch.utils.data.Subset(val_dataset_full, range(50000))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader_centroids = DataLoader(val_dataset_centroids, batch_size=256, shuffle=False, num_workers=4)
val_loader_properties = DataLoader(val_dataset_properties, batch_size=256, shuffle=False, num_workers=4)

print(f"Training samples: {len(train_dataset)}, Validation (Centroids) samples: {len(val_dataset_centroids)}, Validation (Properties) samples: {len(val_dataset_properties)}")


# Load model
feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
feature_extractor.fc = torch.nn.Identity()
feature_extractor.eval().to("cuda")


# Extract training features
print("Extracting training features...")
train_features, train_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(train_loader, desc='Extracting Training Features'):
        images = images.to("cuda")
        features = feature_extractor(images).cpu().numpy()
        train_features.append(features)
        train_labels.append(labels.numpy())
train_features = np.vstack(train_features)
train_labels = np.concatenate(train_labels)


# Extract validation features
val_features, val_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(val_loader_properties, desc='Extracting Validation Features'):
        images = images.to("cuda")
        features = feature_extractor(images).cpu().numpy()
        val_features.append(features)
        val_labels.append(labels.numpy())
val_features = np.vstack(val_features)
val_labels = np.concatenate(val_labels)


from tqdm import tqdm
import faiss
import numpy as np

print("Training FAISS KMeans...")
d = train_features.shape[1]  # Feature dimension
num_clusters = 10000  # Keep your defined number of clusters

# Initialize FAISS KMeans clustering
kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True, gpu=True)
kmeans.train(train_features)  # Train on training features

# Assign clusters to training images
print("Clustering training images...")
_, train_clusters = kmeans.index.search(train_features, 1)
train_clusters = train_clusters.flatten()  # Ensure proper shape

# Assign clusters to validation images
print("Clustering validation images...")
_, val_clusters = kmeans.index.search(val_features, 1)
val_clusters = val_clusters.flatten()  # Ensure proper shape

print("FAISS clustering complete.")

# Debugging: Check cluster assignments
print("Unique train clusters:", np.unique(train_clusters))
print("Unique val clusters:", np.unique(val_clusters))


def compute_C(model, loader, device):
    C_max = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Computing C"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            loss = (predictions != labels).float()
            C_max = max(C_max, loss.max().item())
    return C_max


def compute_epsilon_Zi(val_clusters, val_labels, num_clusters):
    global epsilon_S
    epsilon_Zi = np.zeros(num_clusters)
    for i in tqdm(range(num_clusters), desc="Computing epsilon_Zi"):
        indices = np.where(train_clusters == i)[0]
        if len(indices) > 0:
            cluster_loss = np.mean(train_labels[indices] != np.argmax(np.bincount(train_labels[indices])))
            epsilon_Zi[i] = cluster_loss
        epsilon_S = np.max(epsilon_Zi)
    return epsilon_Zi


def compute_epsiloni_h(val_clusters, val_labels, num_clusters):
    epsiloni_h = np.zeros(num_clusters)
    for i in tqdm(range(num_clusters), desc="Computing epsiloni_h"):
        indices = np.where(train_clusters == i)[0]
        if len(indices) > 0:
            cluster_loss = np.mean(train_labels[indices] != np.argmax(np.bincount(train_labels[indices])))
            epsiloni_h[i] = cluster_loss
    return epsiloni_h


def compute_empirical_loss(train_clusters, train_labels, num_clusters):
    total_loss = 0
    for i in tqdm(range(num_clusters), desc="Computing empirical loss"):
        indices = np.where(train_clusters == i)[0]
        if len(indices) > 0:
            total_loss += np.mean(train_labels[indices] != np.argmax(np.bincount(train_labels[indices]))) * len(indices)
    return total_loss / len(train_labels)


def compute_epsiloni_h_bar(val_clusters, val_labels, num_clusters):
    epsiloni_h_bar = np.zeros(num_clusters)
    for i in tqdm(range(num_clusters), desc="Computing epsiloni_h_bar"):
        indices = np.where(train_clusters == i)[0]
        if len(indices) > 0:
            epsiloni_h_bar[i] = np.mean(np.abs(train_labels[indices] - np.mean(train_labels[indices])))
    return epsiloni_h_bar


def compute_ai_h(val_clusters, val_labels, num_clusters):
    ai_h = np.zeros(num_clusters)
    for i in tqdm(range(num_clusters), desc="Computing ai_h"):
        indices = np.where(train_clusters == i)[0]
        if len(indices) > 0:
            ai_h[i] = np.mean(np.abs(train_labels[indices] - np.mean(train_labels[indices])))
    return ai_h


from collections import Counter
from tqdm import tqdm
import numpy as np
import math
import torch

def run_experiment():

    print("Computing properties...")

    # Compute properties with progress bars
    print("Computing supremum 0-1 loss (C)...")
    C = compute_C(feature_extractor, val_loader_properties, device="cuda")
    
    print("Computing epsilon_Zi...")
    epsilon_Zi = compute_epsilon_Zi(train_clusters, train_labels, num_clusters)
    
    epsilon_S = np.max(epsilon_Zi)

    print("Computing epsiloni_h...")
    epsiloni_h = compute_epsiloni_h(train_clusters, train_labels, num_clusters)
    
    print("Computing empirical loss F_S_h...")
    F_S_h = compute_empirical_loss(train_clusters, train_labels, num_clusters)
    
    print("Computing epsiloni_h_bar...")
    epsiloni_h_bar = compute_epsiloni_h_bar(train_clusters, train_labels, num_clusters)
    
    print("Computing a_i_h...")
    a_i_h = compute_ai_h(train_clusters, train_labels, num_clusters)
    
    # Compute bounds
    delta = 0.01
    T_S = 10000
    n = len(val_dataset_properties)
    cluster_counts = Counter(train_clusters)
    cluster_freqs = np.array([cluster_counts.get(i, 0) / n for i in range(num_clusters)])
    
    print("Computing bound g2...")
    g2 = C * (math.sqrt(2) + 1) * math.sqrt((T_S * math.log(2 * num_clusters / delta)) / n + (2 * C * T_S * math.log(2 * num_clusters / delta)) / n)

    print("Computing bound 6...")
    bound_6 = g2 + F_S_h + np.sum(cluster_freqs * epsiloni_h)
    
    print("Computing bound 2...")
    bound_2 = g2 + F_S_h + epsilon_S
    
    print("Computing bound 7...")
    bound_7 = g2 + F_S_h + np.sum(cluster_freqs * epsiloni_h_bar)
    
    print("Computing bound 8...")
    bound_8 = g2 + np.sum(cluster_freqs * a_i_h)

    # Print results
    print(f"Bound 2: {bound_2}")
    print(f"Bound 6: {bound_6}")
    print(f"Bound 7: {bound_7}")
    print(f"Bound 8: {bound_8}")
    print(f"Supremum 0-1 loss C: {C}")
    print(f"Epsilon_S: {epsilon_S}")
    print(f"Mean epsiloni_h: {np.mean(epsiloni_h)}")
    print(f"g2: {g2}")
    print(f"F_S_h: {F_S_h}")
    print(f"epsiloni_h_bar: {epsiloni_h_bar}")
    print(f"a_i_h: {a_i_h}")

run_experiment()



print("Train cluster distribution:", np.bincount(train_clusters))
print("Validation cluster distribution:", np.bincount(val_clusters))





