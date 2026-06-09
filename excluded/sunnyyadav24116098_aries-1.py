import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm


DATASET_PATH = "/kaggle/input/slicee-my-face"
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "images/train")
TRAIN_MASK_DIR = os.path.join(DATASET_PATH, "annotations/train")
TEST_IMG_DIR = os.path.join(DATASET_PATH, "images/test")



IMG_SIZE = (256, 256)  # Resize images
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4  # Learning rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class FaceSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(img_dir))
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            # Load mask if in training mode
            mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)  # Ensure binary mask

            # Apply transformations
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image, mask = transformed["image"], transformed["mask"]
            
            return image, mask
        
        else:
            # For test images, return only image
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, img_name


import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


train_dataset = FaceSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform, train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = FaceSegmentationDataset(TEST_IMG_DIR, transform=test_transform, train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3Plus, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Modify output layer

    def forward(self, x):
        return self.model(x)["out"]

# Initialize model
model = DeepLabV3Plus().to(DEVICE)

# Loss & Optimizer
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

criterion = lambda pred, target: 0.5 * dice_loss(pred, target) + 0.5 * nn.BCEWithLogitsLoss()(pred, target)
optimizer = optim.AdamW(model.parameters(), lr=LR)


for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "deeplabv3_model.pth")
print("Model saved! ✅")



def rle_encode(mask):
    """Convert binary mask to RLE format."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


model.eval()
submission = []

with torch.no_grad():
    for image, img_name in tqdm(test_loader, desc="Generating Predictions"):
        image = image.to(DEVICE)

        # Predict mask
        output = model(image)
        pred_mask = torch.sigmoid(output).cpu().numpy().squeeze()

        # Convert to binary mask (threshold = 0.5)
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

        # Resize back to original dimensions
        original_size = cv2.imread(os.path.join(TEST_IMG_DIR, img_name[0])).shape[:2]
        binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

        # Encode as RLE
        rle_mask = rle_encode(binary_mask)
        submission.append([img_name[0].split(".")[0], rle_mask])

# Save to CSV
submission_df = pd.DataFrame(submission, columns=["id", "predicted"])
submission_df.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv ✅")

