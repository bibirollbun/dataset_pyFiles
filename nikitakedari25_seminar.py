import os
import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


dataset_path = "/kaggle/input/alaska2-image-steganalysis"
cover_path = os.path.join(dataset_path, "Cover")
jmipod_path = os.path.join(dataset_path, "JMiPOD")
juniward_path = os.path.join(dataset_path, "JUNIWARD")
uerd_path = os.path.join(dataset_path, "UERD")
test_path = os.path.join(dataset_path, "Test")


# Get list of images
cover_images = glob.glob(cover_path + "/*.jpg")
jmipod_images = glob.glob(jmipod_path + "/*.jpg")
juniward_images = glob.glob(juniward_path + "/*.jpg")
uerd_images = glob.glob(uerd_path + "/*.jpg")
test_images = glob.glob(test_path + "/*.jpg")


print(f"Total Cover Images: {len(cover_images)}")
print(f"Total JMiPOD Images: {len(jmipod_images)}")
print(f"Total JUNIWARD Images: {len(juniward_images)}")
print(f"Total UERD Images: {len(uerd_images)}")
print(f"Total Test Images: {len(test_images)}")


def show_sample_images(folder_paths, titles, n=3):
    fig, axes = plt.subplots(len(folder_paths), n, figsize=(15, 10))
    for i, path in enumerate(folder_paths):
        sample_imgs = random.sample(glob.glob(os.path.join(path, "*.*")), n)
        for j, img_path in enumerate(sample_imgs):
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(titles[i], fontsize=14)
    plt.tight_layout()
    plt.show()

folder_paths = [cover_path, jmipod_path, juniward_path, uerd_path, test_path]
titles = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD', 'Test']

show_sample_images(folder_paths, titles)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts image to range [0, 1] and channels-first
])


class Alaska2RGBDCTDataset(Dataset):
    def __init__(self, image_folder, label, transform=None):
        self.image_paths = glob.glob(os.path.join(image_folder, "*.*"))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}!")
        self.label = label
        self.transform = transform

    def get_dct_coefficients(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        dct = cv2.dct(np.float32(img))  # 2D DCT
        return torch.tensor(dct).unsqueeze(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        dct_coeffs = self.get_dct_coefficients(img_path)

        return image, dct_coeffs, self.label


def visualize_image_and_dct(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (224, 224))
    dct = cv2.dct(np.float32(img_resized))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img_resized, cmap='gray')
    axs[0].set_title("Grayscale Image")
    axs[0].axis('off')

    axs[1].imshow(np.log(np.abs(dct) + 1), cmap='inferno')  # Log scale for better visibility
    axs[1].set_title("DCT Coefficients")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# Pick a random image from Cover to visualize
sample_img_path = random.choice(glob.glob(os.path.join(cover_path, "*.jpg")))
visualize_image_and_dct(sample_img_path)


import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm


import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 5x5 high-pass filter kernel (as used in SRM filters)
hpf_kernel = torch.tensor([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8,-12,  8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 12.0



def get_noise_residual(image_tensor):
    # image_tensor: [3, H, W] (RGB) → convert to grayscale first
    grayscale = transforms.functional.rgb_to_grayscale(image_tensor, num_output_channels=1)
    grayscale = grayscale.unsqueeze(0)  # Shape: [1, 1, H, W]

    # Apply high-pass filter
    residual = F.conv2d(grayscale, hpf_kernel, padding=2)
    return residual.squeeze(0)  # Shape: [1, H, W]



class Alaska2RGBDCTDataset(Dataset):
    def __init__(self, image_folder, label, transform=None, max_samples=20000):
        self.image_paths = glob.glob(os.path.join(image_folder, "*.*"))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}!")

        random.shuffle(self.image_paths)  # Randomize order
        self.image_paths = self.image_paths[:max_samples]  # Take only 20,000

        self.label = label
        self.transform = transform

    def get_dct_coefficients(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        dct = cv2.dct(np.float32(img))
        return torch.tensor(dct).unsqueeze(0)  # shape: [1, 224, 224]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
    
        dct_coeffs = self.get_dct_coefficients(img_path)
        noise_residual = get_noise_residual(image)  # [1, 224, 224]
    
        return image, dct_coeffs, noise_residual, self.label


# # Visualize RGB + Noise Residual
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(image.permute(1, 2, 0))
# plt.title("RGB Image")
# plt.subplot(1,2,2)
# plt.imshow(noise_residual[0], cmap='gray')
# plt.title("Noise Residual (HPF)")
# plt.show()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])



base_dir = "/kaggle/input/alaska2-image-steganalysis"

cover_path = os.path.join(base_dir, "Cover")
jmipod_path = os.path.join(base_dir, "JMiPOD")
juniward_path = os.path.join(base_dir, "JUNIWARD")
uerd_path = os.path.join(base_dir, "UERD")
test_path = os.path.join(base_dir, "Test")


cover_dataset = Alaska2RGBDCTDataset(cover_path, label=0, transform=transform)
jmipod_dataset = Alaska2RGBDCTDataset(jmipod_path, label=1, transform=transform)
juniward_dataset = Alaska2RGBDCTDataset(juniward_path, label=2, transform=transform)
uerd_dataset = Alaska2RGBDCTDataset(uerd_path, label=3, transform=transform)


batch_size = 32

cover_loader = DataLoader(cover_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
jmipod_loader = DataLoader(jmipod_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
juniward_loader = DataLoader(juniward_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
uerd_loader = DataLoader(uerd_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader =


image_path = cover_dataset[10]


def visualize_image_and_dct(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (224, 224))
    dct = cv2.dct(np.float32(img_resized))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img_resized, cmap='gray')
    axs[0].set_title("Grayscale Image")
    axs[0].axis('off')

    axs[1].imshow(np.log(np.abs(dct) + 1), cmap='inferno')  # Log scale for better visibility
    axs[1].set_title("DCT Coefficients")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# Pick a random image from Cover to visualize
sample_img_path = random.choice(glob.glob(os.path.join(cover_path, "*.jpg")))
visualize_image_and_dct(sample_img_path)


import matplotlib.pyplot as plt

def visualize_sample(dataset, idx=0):
    image, dct, noise, label = dataset[idx]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # RGB Image
    axs[0].imshow(image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    # DCT Coefficients (log scale for visibility)
    axs[1].imshow(torch.log(torch.abs(dct.squeeze()) + 1).cpu(), cmap='inferno')
    axs[1].set_title("DCT Coefficients")
    axs[1].axis("off")

    # Noise Residual
    axs[2].imshow(noise.squeeze().cpu(), cmap='gray')
    axs[2].set_title("Noise Residual")
    axs[2].axis("off")

    plt.suptitle(f"Label: {label}")
    plt.tight_layout()
    plt.show()

# Test it
visualize_sample(jmipod_dataset, idx=5)
visualize_sample(juniward_dataset, idx=5)


import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(CNNFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 224, 224]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 112, 112]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 56, 56]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # [B, 128]



from torchvision.models.vision_transformer import vit_b_16

class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove classification head

    def forward(self, x):
        return self.vit(x)  # Returns [B, 768]




class HybridClassifier(nn.Module):
    def __init__(self, cnn_dim=128, vit_dim=768, num_classes=4):
        super(HybridClassifier, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.vit = ViTFeatureExtractor()

        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.fusion(combined)



model = HybridClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Example training loop:
for epoch in range(5):
    model.train()
    for images, _, _, labels in tqdm(cover_loader):  # Can concat all 4 datasets later
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")



class AlaskaTestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = glob.glob(os.path.join(image_folder, "*.*"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

test_dataset = AlaskaTestDataset(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



from torch.utils.data import ConcatDataset

combined_dataset = ConcatDataset([cover_dataset, jmipod_dataset, juniward_dataset, uerd_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


class BinaryWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, dct, noise, label = self.dataset[idx]
        binary_label = 0 if label == 0 else 1
        return image, binary_label



binary_dataset = BinaryWrapper(combined_dataset)
binary_loader = DataLoader(binary_dataset, batch_size=batch_size, shuffle=True)



model = HybridClassifier(num_classes=2).to(device)



for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(binary_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Epoch [{epoch+1}] Loss: {running_loss/len(binary_loader):.4f} | Acc: {acc:.2f}%")


