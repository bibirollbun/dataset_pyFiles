import os
import pandas as pd
from PIL import Image
import numpy as np


categories = pd.read_csv(f"/kaggle/input/image-matching-challenge-2024/train/categories.csv")
train_labels = pd.read_csv(f"/kaggle/input/image-matching-challenge-2024/train/train_labels.csv")
sample_submission = pd.read_csv(f"/kaggle/input/image-matching-challenge-2024/sample_submission.csv")



# List all images
image_dir = "/kaggle/input/image-matching-challenge-2024/train/church/images/"
image_files = os.listdir(image_dir)
print(len(image_files))
print("Example image:", image_files[7])



import matplotlib.pyplot as plt
from PIL import Image


def load_image(image_dir, file):
    img = Image.open(f"{image_dir}{file}")
    plt.imshow(img)
    plt.axis('off')  
    plt.show()


image_dir = "/kaggle/input/image-matching-challenge-2024/train/church/images/"
load_image(image_dir, "00001.png")
load_image(image_dir, "00002.png")
load_image(image_dir, "00003.png")
load_image(image_dir, "00004.png")





import os
from PIL import Image
import numpy as np

# Path to the image directory
image_dir = "/kaggle/input/image-matching-challenge-2024/train/church/images"

# List to hold all image matrices
image_matrices = []

image_files = sorted(os.listdir(image_dir))

for file_name in image_files:
    if file_name.endswith(".png"):
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert('RGB').resize((256, 256))
        img_array = np.array(img)
        image_matrices.append(img_array)



image_matrices[0].shape




import cv2

# Example with SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(image_matrices[0], None)
kp2, des2 = sift.detectAndCompute(image_matrices[1], None)

# Brute Force Matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

# Draw matches
matched_img = cv2.drawMatchesKnn(image_matrices[0], kp1, image_matrices[1], kp2, good_matches, None, flags=2)
plt.imshow(matched_img)
plt.show()



N = 20  
subset_images = image_matrices[:N]



import cv2
from tqdm import tqdm
import numpy as np
import itertools

# Initialize SIFT
sift = cv2.SIFT_create()

# Brute-Force Matcher
bf = cv2.BFMatcher()

# Dictionary to store similarity scores
similarity_scores = {}

# Iterate through all unique pairs
for i, j in tqdm(itertools.combinations(range(len(subset_images)), 2), total=(len(subset_images)*(len(subset_images)-1))//2):
    img1 = subset_images[i]
    img2 = subset_images[j]
    
    # Convert to grayscale (SIFT works on single channel)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Detect and compute keypoints/descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        score = 0
    else:
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # Similarity score = number of good matches
        score = len(good_matches)

    similarity_scores[(i, j)] = score



top_matches = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
print(top_matches[:10])



categories


train_labels


print(f"Rotation_matrix {train_labels.loc[0, 'rotation_matrix']}")
print(f"Translation vector {train_labels.loc[0, 'translation_vector']}")


sample_submission.head()


train_labels.head()


train_labels_subset = train_labels.loc[:500, :]


import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import pandas as pd

class ImagePoseDataset(Dataset):
    def __init__(self, dataframe, image_base_path, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): A subset of train_labels.csv.
            image_base_path (str): Base path to the /train folder.
            transform (callable, optional): Optional transform to be applied on a sample image.
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_base_path = image_base_path
        self.transform = transform

        # Optional: prefilter valid rows
        self.valid_rows = []
        for _, row in self.df.iterrows():
            image_path = os.path.join(image_base_path, row['dataset'], "images", row['image_name'])
            try:
                _ = np.array(row['rotation_matrix'].split(';'), dtype=np.float64).reshape(3, 3)
                _ = np.array(row['translation_vector'].split(';'), dtype=np.float64).reshape(3, 1)
                if os.path.exists(image_path):
                    self.valid_rows.append(row)
            except:
                continue

    def __len__(self):
        return len(self.valid_rows)

    def __getitem__(self, idx):
        row = self.valid_rows[idx]
        image_path = os.path.join(self.image_base_path, row['dataset'], "images", row['image_name'])

        # Load image
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [3, H, W] normalized

        # Rotation and translation
        rotation_matrix = torch.tensor(
            np.array(row['rotation_matrix'].split(';'), dtype=np.float32).reshape(3, 3)
        )
        translation_vector = torch.tensor(
            np.array(row['translation_vector'].split(';'), dtype=np.float32).reshape(3, 1)
        )

        return {
            "image": img,
            "rotation_matrix": rotation_matrix,
            "translation_vector": translation_vector,
            "image_name": row['image_name'],
            "scene": row['scene'],
            "dataset": row['dataset']
        }



from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_base_path = "/kaggle/input/image-matching-challenge-2024/train"
dataset = ImagePoseDataset(train_labels_subset, image_base_path, transform=transform)

sample = dataset[0]
print(sample['image'].shape)  
print(sample['rotation_matrix'])
print(sample['translation_vector'])



print(f"Number of samples in the dataset: {len(dataset)}")



sample = dataset[0]
print("Image shape:", sample['image'].shape)  # [C, H, W]
print("Rotation matrix shape:", sample['rotation_matrix'].shape)  # [3, 3]
print("Translation vector shape:", sample['translation_vector'].shape)  # [3, 1]



from torch.utils.data import random_split

# Set lengths
total_len = len(dataset)
train_len = int(0.8 * total_len)
val_len = total_len - train_len

# Random split
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])



import torch.nn as nn
import torch.nn.functional as F

class PoseRegressor(nn.Module):
    def __init__(self):
        super(PoseRegressor, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 112, 112]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 56, 56]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 28, 28]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 64, 1, 1]
        )
        self.fc = nn.Flatten()
        self.fc_rotation = nn.Linear(64, 9)   # 3x3 matrix
        self.fc_translation = nn.Linear(64, 3)  # 3x1 vector

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        rot = self.fc_rotation(x)
        trans = self.fc_translation(x)
        rot = rot.view(-1, 3, 3)
        trans = trans.view(-1, 3, 1)
        return rot, trans



def camera_center_loss(pred_R, pred_T, true_R, true_T):
    # Compute camera centers: C = -R^T @ T
    pred_C = -torch.matmul(pred_R.transpose(1, 2), pred_T)  # [B, 3, 1]
    true_C = -torch.matmul(true_R.transpose(1, 2), true_T)  # [B, 3, 1]

    return F.mse_loss(pred_C, true_C)



from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PoseRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()



from tqdm import tqdm

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for batch in loop: 
        imgs = batch['image'].to(device)
        true_R = batch['rotation_matrix'].to(device)
        true_T = batch['translation_vector'].to(device)
    
        pred_R, pred_T = model(imgs)
    
        loss = camera_center_loss(pred_R, pred_T, true_R, true_T)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        loop.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    print(f"Epoch {epoch+1} | Avg Train Loss: {total_loss/len(train_loader):.4f}")



model.eval()
val_loss = 0
print_limit = 3  

with torch.no_grad():
    val_loop = tqdm(val_loader, desc="Validating", leave=False)

    shown = 0
    for batch in val_loop:
        imgs = batch['image'].to(device)
        true_R = batch['rotation_matrix'].to(device)
        true_T = batch['translation_vector'].to(device)

        pred_R, pred_T = model(imgs)
        loss = camera_center_loss(pred_R, pred_T, true_R, true_T)
        val_loss += loss.item()

        val_loop.set_postfix({"loss": f"{loss.item():.4f}"})

        if shown < print_limit:
            for i in range(min(len(imgs), print_limit - shown)):
                print(f"\nSample {shown + 1}")
                print("Ground Truth Rotation Matrix:\n", true_R[i].cpu().numpy())
                print("Predicted Rotation Matrix:\n", pred_R[i].cpu().numpy())

                print("Ground Truth Translation Vector:\n", true_T[i].cpu().numpy().flatten())
                print("Predicted Translation Vector:\n", pred_T[i].cpu().numpy().flatten())
                
                shown += 1
            if shown >= print_limit:
                break

print(f"\nFinal Validation Loss (Camera Center Based): {val_loss / len(val_loader):.4f}")





