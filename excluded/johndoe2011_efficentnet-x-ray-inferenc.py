# ==============================
#  Inference (Match Sample Submission)
# ==============================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
from PIL import Image
from tqdm import tqdm
import timm
import os

# Config
class Config:
    BASE_PATH = '/kaggle/input/grand-xray-slam-division-a'
    TRAIN_CSV = f'{BASE_PATH}/train1.csv'
    TEST_DIR = f'{BASE_PATH}/test1'
    SAMPLE_SUB = f'{BASE_PATH}/sample_submission_1.csv'
    IMG_SIZE = 128
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True
    NON_BLOCKING = True
    USE_AMP = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
class FastChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use exactly the column name from sample submission
        img_name = row[self.df.columns[0]]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, img_name

# Transform
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model
class FastChestXRayModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=14, pretrained=False):
        super(FastChestXRayModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
        return self.classifier(features)

# ==============================
#  Detect condition labels
# ==============================
train_df = pd.read_csv(Config.TRAIN_CSV)
sample_sub = pd.read_csv(Config.SAMPLE_SUB)

# Use only conditions present in sample_sub (ensures matching order)
condition_labels = [col for col in sample_sub.columns if col != sample_sub.columns[0]]
num_classes = len(condition_labels)
print(f"✅ Submission will follow sample format with {num_classes} labels: {condition_labels}")

# ==============================
#  Inference
# ==============================
test_dataset = FastChestXRayDataset(sample_sub, Config.TEST_DIR, val_transform)
test_loader = DataLoader(
    test_dataset, 
    batch_size=Config.BATCH_SIZE, 
    shuffle=False,
    num_workers=Config.NUM_WORKERS, 
    pin_memory=Config.PIN_MEMORY
)

# Load model
model = FastChestXRayModel('efficientnet_b0', num_classes=num_classes, pretrained=False)
model.load_state_dict(torch.load('/kaggle/input/efficentnet-x-ray-a/best_fast_model.pth', map_location=Config.DEVICE), strict=False)
model = model.to(Config.DEVICE)
model.eval()

# Predict
predictions = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc='Predicting'):
        images = images.to(Config.DEVICE, non_blocking=Config.NON_BLOCKING)
        if Config.USE_AMP:
            with torch.autocast(device_type='cuda'):
                outputs = torch.sigmoid(model(images))
        else:
            outputs = torch.sigmoid(model(images))
        predictions.append(outputs.cpu().numpy())

predictions = np.vstack(predictions)

# ==============================
# Build submission exactly like sample
# ==============================
submission = sample_sub.copy()
for i, condition in enumerate(condition_labels):
    submission[condition] = predictions[:, i]

submission.to_csv('submission.csv', index=False)
print(" Submission saved:", submission.shape)





