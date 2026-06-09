import torchvision.models as models
import torch.nn as nn
import torch

# Load a pretrained ResNet18
resnet = models.resnet18(pretrained=True)

# Replace final classification layer with Identity to get a 512-D embedding
resnet.fc = nn.Identity()

# This is now your embedding encoder
encoder = resnet


import torchvision.models as models
import torch.nn as nn

resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()  # Now outputs 512-D vector
encoder = resnet.eval()
encoder


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import faiss
import numpy as np

# ---------------------------------------------------
# 1) ArcMarginProduct (ArcFace layer)
# ---------------------------------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # Normalize input and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-6)
        
        # Add margin
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # Combine phi for correct class, cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # scale
        return output

# ---------------------------------------------------
# 2) ArcFace Model with ResNet18
# ---------------------------------------------------
class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(ArcFaceModel, self).__init__()
        # Pretrained ResNet
        base = models.resnet18(pretrained=True)
        base.fc = nn.Identity()  # remove final FC
        self.encoder = base
        
        # Optional embedding layer
        self.embedding = nn.Linear(512, embedding_size)
        
        # ArcMargin for classification
        self.arcface = ArcMarginProduct(embedding_size, num_classes)

    def forward(self, x, label=None):
        # Extract features
        x = self.encoder(x)
        # Map to desired embedding dimension
        x = self.embedding(x)
        # Normalize embeddings
        x = F.normalize(x)

        if label is not None:
            # ArcFace classification logits
            logits = self.arcface(x, label)
            return logits
        else:
            # Return embeddings for inference
            return x

# ---------------------------------------------------
# 3) Shopee Dataset Class
# ---------------------------------------------------
class ShopeeDataset(Dataset):
    """
    Reads a CSV file and loads images + labels from the provided folder.
    Here we use 'image_phash' as a proxy for label grouping.
    """
    def __init__(self, csv_file, img_folder, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        
        # Map each unique image_phash to a class index
        label_groups = self.df['image_phash'].unique()
        self.label_to_class = {lg: idx for idx, lg in enumerate(label_groups)}
        
        # Create list of (image_path, class_index)
        self.samples = []
        for _, row in self.df.iterrows():
            img_name = row['image']
            label_group = row['image_phash']
            class_idx = self.label_to_class[label_group]
            img_path = os.path.join(self.img_folder, img_name)
            self.samples.append((img_path, class_idx))
            
        self.num_classes = len(label_groups)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, class_idx = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_idx

# ---------------------------------------------------
# 4) Data Preparation
# ---------------------------------------------------
# Update these paths based on your environment
root_dir = "Shopee - Price Match Guarantee"
train_csv = "/kaggle/input/shopee-product-matching/train.csv"
train_img_folder = "/kaggle/input/shopee-product-matching/train_images"

# Transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create dataset and DataLoader
train_dataset = ShopeeDataset(csv_file=train_csv, 
                              img_folder=train_img_folder,
                              transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# ---------------------------------------------------
# 5) Initialize Model, Loss, Optimizer
# ---------------------------------------------------
num_classes = train_dataset.num_classes
embedding_size = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArcFaceModel(num_classes=num_classes, embedding_size=embedding_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5  # For demonstration

# ---------------------------------------------------
# 6) Training Loop
# ---------------------------------------------------
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}")

# ---------------------------------------------------
# 7) Inference: Extract Embeddings & Build FAISS Index
# ---------------------------------------------------
model.eval()
embeddings_list = []
image_paths = []  # To keep track of which embedding belongs to which image

# Loop over the dataset to extract embeddings for each image
for img_path, _ in train_dataset.samples:
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        continue
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_tensor)  # Returns normalized embedding
    embeddings_list.append(emb.cpu().numpy())
    image_paths.append(img_path)

# Convert the list of embeddings into a NumPy array
gallery_embeddings = np.vstack(embeddings_list).astype('float32')
print("Gallery embeddings shape:", gallery_embeddings.shape)

# Create FAISS index (using L2 distance here)
d = gallery_embeddings.shape[1]  # Dimension of embeddings, e.g., 512
index = faiss.IndexFlatL2(d)
index.add(gallery_embeddings)  # Add all gallery embeddings to the index

print("FAISS index built with {} vectors".format(index.ntotal))

# ---------------------------------------------------
# 8) FAISS Query Example
# ---------------------------------------------------
# For demonstration, take the first image as query



import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# 1) Pick a random index
rand_idx = random.randrange(len(image_paths))
query_img_path = image_paths[rand_idx]

# Helper to get last two chars of the base filename (without extension)
def last_two_chars(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name[-3:]

# 2) Load and embed
query_img = Image.open(query_img_path).convert('RGB')
query_tensor = transform(query_img).unsqueeze(0).to(device)
with torch.no_grad():
    query_embedding = model(query_tensor).cpu().numpy().astype('float32')

# 3) Search FAISS
k = 10
distances, indices = index.search(query_embedding, k)

# 4) Plot
fig, axes = plt.subplots(1, k+1, figsize=(15, 5))

# Query image
lt = last_two_chars(query_img_path)
axes[0].imshow(query_img)
axes[0].set_title(f"Query, ID='{lt}')")
axes[0].axis('off')

# Neighbours
for i, nn_idx in enumerate(indices[0]):
    nn_path = image_paths[nn_idx]
    img = Image.open(nn_path).convert('RGB')
    lt_nn = last_two_chars(nn_path)
    axes[i+1].imshow(img)
    axes[i+1].set_title(f"#{i+1} ID='{lt_nn}'")
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()







