import sys
sys.path.append("../input/tez-lib/")

import os
import numpy as np
import pandas as pd
import cv2
import torch
import timm
import albumentations
from tez import Tez

import random
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(42)



class CFG:
    image_size = 384
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_dense = 12



class PawpularDataset:
    def __init__(self, image_paths, dense_features=None, targets=None, augmentations=None):
        self.image_paths = image_paths
        self.dense_features = dense_features if dense_features is not None else np.zeros((len(image_paths), 0), dtype=np.float32)
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        features = self.dense_features[idx].astype(np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float)
        }



valid_aug = albumentations.Compose([
    albumentations.LongestMaxSize(CFG.image_size, p=1),
    albumentations.PadIfNeeded(CFG.image_size, CFG.image_size, p=1, border_mode=0),
    albumentations.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], max_pixel_value=255.0, p=1.0),
], p=1.0)



import torch.nn as nn

class PawpularModel(nn.Module):
    def __init__(self, backbone_name="resnet50", n_dense=12, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=3)
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Sequential(
            nn.Linear(feat_dim + n_dense, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, image, features):
        x = self.backbone(image)
        x = torch.cat([x, features], dim=1)
        x = self.dropout(x)
        x = self.out(x)
        return x



test_df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
dense_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

test_features = np.zeros((len(test_df), len(dense_features)), dtype=np.float32)
for i, col in enumerate(dense_features):
    if col in test_df.columns:
        test_features[:, i] = test_df[col].values

test_img_paths = [f"../input/petfinder-pawpularity-score/test/{x}.jpg" for x in test_df.Id.values]
test_ds = PawpularDataset(test_img_paths, dense_features=test_features, augmentations=valid_aug)



from torch.utils.data import DataLoader

n_folds = 5
test_preds = np.zeros(len(test_df), dtype=np.float32)
batch_size = 8
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

for fold in range(n_folds):
    print(f"Processing fold {fold}")
    model_path = f"10158_model_fold{fold}.bin"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path} -- skipping fold")
        continue

    # Load model
    model = PawpularModel(backbone_name="resnet50", n_dense=len(dense_features), pretrained=False)
    tez_model = Tez(model)
    tez_model.load(model_path)  # works because save_weights_only=False in training

    # Predict
    fold_preds = []
    tez_model.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(CFG.device, dtype=torch.float)
            features = batch["features"].to(CFG.device, dtype=torch.float)
            outputs = tez_model.model(images, features)
            fold_preds.extend(outputs.detach().cpu().squeeze().numpy().tolist())

    fold_preds = np.array(fold_preds, dtype=np.float32)
    test_preds += fold_preds / n_folds



test_preds = np.clip(test_preds, 0, 100)
submission = pd.DataFrame({"Id": test_df.Id.values, "Pawpularity": test_preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
display(submission.head())





