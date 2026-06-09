import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


train_path = "/kaggle/input/up-solving-tst-day-1/train/train/"
test_path = "/kaggle/input/up-solving-tst-day-1/test/test/"


class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.filtered_dir = os.path.join(root_dir, 'filtered_images')
        self.original_dir = os.path.join(root_dir, 'real_images')

        self.image_filenames = sorted(os.listdir(self.filtered_dir))
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]

        filtered_path = os.path.join(self.filtered_dir, filename)
        original_path = os.path.join(self.original_dir, filename)

        filtered_img = Image.open(filtered_path).convert('RGB')
        original_img = Image.open(original_path).convert('RGB')

        filtered_img = self.transform(filtered_img)
        original_img = self.transform(original_img)

        return filtered_img, original_img


class FilteredImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.filtered_dir = os.path.join(root_dir, 'filtered_images')
        self.image_filenames = sorted(os.listdir(self.filtered_dir))
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        filtered_path = os.path.join(self.filtered_dir, filename)

        filtered_img = Image.open(filtered_path).convert('RGB')
        filtered_img = self.transform(filtered_img)

        return filtered_img, filename  



train_dataset = PairedImageDataset(train_path)  # path/filtered_images и path/real_images
test_dataset = FilteredImageDataset(test_path)  # path_test/filtered_images


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)  
        )
        
    def forward(self, x):
        return self.conv_net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_psnr(img1, img2):
    # [C, H, W] -> [H, W, C] and to uint8
    img1 = (np.transpose(img1, (1, 2, 0)) * 255).round().astype(np.uint8)
    img2 = (np.transpose(img2, (1, 2, 0)) * 255).round().astype(np.uint8)

    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr


def show_image(i: int, inputs, outputs, targets):
    filtered_img = inputs[i].cpu().detach().numpy()
    output_img = outputs[i].cpu().detach().clamp(0,1).numpy()
    real_img = targets[i].cpu().detach().numpy()

    print(f"Image PSNR {compute_psnr(output_img, real_img)}")
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.title('Filtered (Input)')
    plt.axis('off')
    plt.imshow(np.transpose(filtered_img, (1, 2, 0)))
    
    plt.subplot(1,3,2)
    plt.title('Model Output')
    plt.axis('off')
    plt.imshow(np.transpose(output_img, (1, 2, 0)))
    
    plt.subplot(1,3,3)
    plt.title('Real (Target)')
    plt.axis('off')
    plt.imshow(np.transpose(real_img, (1, 2, 0)))
    
    plt.show()


model = CNN().to(device)

criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-3)


epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets).mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

    show_image(
        2,
        inputs.detach().cpu(),
        outputs.detach().cpu(),
        targets.detach().cpu()
    )


submission_rows = []

model.eval()
for idx in range(len(test_dataset)):
    img_tensor, filename = test_dataset[idx]
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, 128, 128]
    img_id = os.path.splitext(filename)[0]

    with torch.no_grad():
        output = model(img_tensor)  # [1, 3, 128, 128]
        output = output.squeeze(0).clamp(0, 1).cpu().numpy()  # [3, 128, 128]

    output_bgr = output[[2, 1, 0], :, :]
    output_bgr = (output_bgr * 255).round().astype(np.uint8)
    output_flat = output_bgr.transpose(1, 2, 0).flatten()

    submission_rows.append([img_id] + output_flat.tolist())


pixel_columns = [str(i) for i in range(output_flat.size)]
df = pd.DataFrame(submission_rows, columns=["id"] + pixel_columns)

df.to_csv("submission.csv", index=False)

