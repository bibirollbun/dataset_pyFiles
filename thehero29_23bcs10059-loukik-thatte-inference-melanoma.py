import os, cv2, timm, torch, albumentations
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

# ========= CONFIG =========
class args:
    batch_size = 32
    image_size = 384
    fold = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ↓ change this path to your dataset name containing model_f0.bin
    checkpoint_path = "/kaggle/input/23bcs10050-loukik-thatte-melanoma-training/model_f0.bin"
    test_csv = "/kaggle/input/siim-isic-melanoma-classification/test.csv"
    test_jpeg_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/test"

# ========= AUGMENTATION =========
valid_aug = albumentations.Compose([
    albumentations.LongestMaxSize(args.image_size, p=1),
    albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=0),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0, p=1.0),
], p=1.0)

# ========= DATASET =========
class TestDataset(Dataset):
    def __init__(self, image_paths, dense_features, augmentations):
        self.image_paths = image_paths
        self.dense_features = dense_features.astype(np.float32)
        self.augmentations = augmentations

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.augmentations(image=img)["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        feats = self.dense_features[idx]
        return {
            "image": torch.tensor(img, dtype=torch.float),
            "features": torch.tensor(feats, dtype=torch.float),
        }

# ========= MODEL =========
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=False, in_chans=3)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(1000, 1)
    def forward(self, image, features=None):
        x = self.model(image)
        x = self.dropout(x)
        x = self.out(x)
        return x

# ========= LOAD MODEL =========
model = CustomModel()
state = torch.load(args.checkpoint_path, map_location=args.device)
# remove 'model.' prefix if present
state = {k.replace("model.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.to(args.device)
model.eval()
print(" Model loaded from:", args.checkpoint_path)

# ========= LOAD TEST DATA =========
test_df = pd.read_csv(args.test_csv)
test_img_paths = [os.path.join(args.test_jpeg_dir, f"{x}.jpg") for x in test_df["image_name"]]
dense_features = np.zeros((len(test_df), 1), dtype=np.float32)

test_dataset = TestDataset(test_img_paths, dense_features, valid_aug)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# ========= INFERENCE =========
preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inferencing"):
        imgs = batch["image"].to(args.device)
        feats = batch["features"].to(args.device)
        logits = model(imgs, feats)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        preds.extend(probs.tolist())

# ========= SUBMISSION =========
test_df["target"] = preds
test_df[["image_name", "target"]].to_csv("submission.csv", index=False)
print("Saved submission.csv")


