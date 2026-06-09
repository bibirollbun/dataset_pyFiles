import os

input_dir = '/kaggle/input/birdclef-2025/train_audio'
print("Folders in train_audio:", os.listdir(input_dir)[:5])


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def audio_to_melspectrogram(file_path, save_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(2.56, 2.56), dpi=100)
        librosa.display.specshow(S_DB, sr=sr, cmap='magma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"âš ï¸� Error on {file_path}: {e}")

# Limit number of files to avoid long runtime (e.g., for Starter demonstration)
file_list = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.ogg'):
            full_path = os.path.join(root, file)
            file_list.append(full_path)

# Only use first 50 files for demo purposes
file_list = file_list[:50]
print(f"Number of files used for conversion: {len(file_list)}")

output_dir = '/kaggle/working/train_images'
os.makedirs(output_dir, exist_ok=True)

for input_path in tqdm(file_list):
    base_name = os.path.basename(input_path).replace('.ogg', '.png')
    output_path = os.path.join(output_dir, base_name)
    if not os.path.exists(output_path):
        audio_to_melspectrogram(input_path, output_path)


from IPython.display import Image, display
import os

image_folder = '/kaggle/working/train_images'
image_files = os.listdir(image_folder)
sample_image_path = os.path.join(image_folder, image_files[0])  
display(Image(filename=sample_image_path))


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

# Set up a basic transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define a dummy dataset using 10 random images
class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        random.seed(42)
        self.image_files = random.sample(self.image_files, 10)  # select 10 files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor([1.0])  # dummy binary label for demonstration
        return image, label

dataset = SpectrogramDataset('/kaggle/working/train_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load pretrained ResNet18 and modify the output layer
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and loss
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop (3 epochs)
model.train()
for epoch in range(3):
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


import pandas as pd

# Load sample submission format
sub = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')

# Use one image to generate a prediction score (you can replace this with a loop over actual test images)
img_path = os.path.join('/kaggle/working/train_images', os.listdir('/kaggle/working/train_images')[0])
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(img_tensor)
    prob = torch.sigmoid(output).item()

print(f"Predicted probability used for submission: {prob:.4f}")

# Fill all rows and columns with the predicted probability
for col in sub.columns[1:]:
    sub[col] = prob

# Save submission file
sub.to_csv('/kaggle/working/submission.csv', index=False)

