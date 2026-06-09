import os
import torch
import timm
import pandas as pd
import numpy as np
import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
BATCH_SIZE = 32

# Dataset (image-only)
class PetTestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.meta_cols = [
            "Subject Focus", "Eyes", "Face", "Near", "Action", "Accessory",
            "Group", "Collage", "Human", "Occlusion", "Info", "Blur"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Id"] + ".jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        metadata = torch.tensor(row[self.meta_cols].values.astype(np.float32))
        return image, metadata, row["Id"]

# Model
class EffNetWithMeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        self.meta = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1536+ 64, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1),
        )

    def forward(self, x_img, x_meta):
        x_img = self.backbone(x_img)
        x_img = torch.flatten(x_img, 1)
        x_meta = self.meta(x_meta)
        x = torch.cat([x_img, x_meta], dim=1)
        return self.classifier(x)


# Transform
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Load model
model = EffNetWithMeta()
model.load_state_dict(torch.load("/kaggle/input/effnet45/pytorch/default/1/hybrid_best_45.pth", map_location=DEVICE))

model.to(DEVICE)
model.eval()

# Load test data
test_df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
test_ds = PetTestDataset(test_df, "/kaggle/input/petfinder-pawpularity-score/test", transform)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Predict
ids, preds = [], []
with torch.no_grad():
    for images, metas, id_batch in test_dl:
        images, metas = images.to(DEVICE), metas.to(DEVICE)
        outputs = model(images, metas).squeeze().cpu().numpy()
        outputs = np.clip(outputs, 0, 100)
        preds.extend(outputs)
        ids.extend(id_batch)

# Save submission
submission = pd.DataFrame({
    "Id": ids,
    "Pawpularity": preds
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved.")

