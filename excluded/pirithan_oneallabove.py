import pandas as pd

LABEL_PATH = "/kaggle/input/solidworks-ai-hackathon/train_labels.csv"  # adjust if name differs

df = pd.read_csv(LABEL_PATH)
print(df.head())
print(df.columns)



print(df.columns)


import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image



!nvidia-smi



import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))



import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())



BASE_PATH = "/kaggle/input/solidworks-ai-hackathon"
TRAIN_DIR = f"{BASE_PATH}/train/train"
TEST_DIR  = f"{BASE_PATH}/test/test"
LABELS    = f"{BASE_PATH}/train_labels.csv"



import os
print(len(os.listdir(TRAIN_DIR)))  # must be ~10000
print(len(os.listdir(TEST_DIR)))   # must be ~2000



import pandas as pd

df = pd.read_csv(LABELS)
print(df.head())
print(df.columns)



from torchvision import transforms

train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class PartsDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        valid = set(os.listdir(img_dir))
        self.df = self.df[self.df.image_name.isin(valid)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row.image_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            [row.bolt, row.locatingpin, row.nut, row.washer],
            dtype=torch.float32
        )
        return image, label



train_dataset = PartsDataset(LABELS, TRAIN_DIR, train_tfms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)



from torch.utils.data import DataLoader

dataset = PartsDataset(LABELS, TRAIN_DIR, train_tfms)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

x, y = next(iter(loader))
print(x.shape)   # [8,3,224,224]
print(y.shape)   # [8,4]



from torchvision import models
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)



criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



for epoch in range(3):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")



model.eval()
results = []

for img_name in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img_name)
    img = Image.open(img_path).convert("RGB")
    img = train_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img).cpu().numpy()[0]

    pred = pred.round().clip(min=0).astype(int)
    results.append([img_name, *pred])



sub = pd.DataFrame(
    results,
    columns=["image_name","bolt","locatingpin","nut","washer"]
)

sub.to_csv("submission.csv", index=False)



%matplotlib inline
import matplotlib.pyplot as plt
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet mean & std
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denormalize(img):
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0,1)

def visualize_prediction(model, dataset, idx):
    model.eval()

    img, gt = dataset[idx]
    img = img.cpu()

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).cpu()[0]

    pred = pred.round().clip(min=0).int()

    img_vis = denormalize(img).permute(1,2,0).numpy()

    # ðŸ”¹ Professional plot
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img_vis)
    ax.axis("off")

    # Clean annotation block
    text = (
        f"Ground Truth\n"
        f"Bolt: {gt[0].int().item()}\n"
        f"Pin: {gt[1].int().item()}\n"
        f"Nut: {gt[2].int().item()}\n"
        f"Washer: {gt[3].int().item()}\n\n"
        f"Prediction\n"
        f"Bolt: {pred[0].item()}\n"
        f"Pin: {pred[1].item()}\n"
        f"Nut: {pred[2].item()}\n"
        f"Washer: {pred[3].item()}"
    )

    ax.text(
        1.02, 0.5, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
    )

    ax.set_title("Model Prediction vs Ground Truth", fontsize=12, pad=10)
    plt.show()

# Execute
visualize_prediction(model, train_dataset, 0)



import numpy as np
import matplotlib.pyplot as plt
import torch

# Collect absolute errors per part
errors = {
    "Bolt": [],
    "Locating Pin": [],
    "Nut": [],
    "Washer": []
}

model.eval()
with torch.no_grad():
    for imgs, gt in loader:
        imgs = imgs.to(device)
        gt = gt.to(device)

        preds = model(imgs).round().clip(min=0)
        diff = (preds - gt).abs().cpu().numpy()

        for i, key in enumerate(errors.keys()):
            errors[key].extend(diff[:, i])

# Compute Mean Absolute Error
mae = {k: np.mean(v) for k, v in errors.items()}

# ---- Plot ----
fig, ax = plt.subplots(figsize=(6,4))

parts = list(mae.keys())
values = list(mae.values())

bars = ax.bar(parts, values)

ax.set_title("Mean Absolute Error per Part", fontsize=12, pad=10)
ax.set_ylabel("Mean Absolute Error (MAE)")
ax.set_xlabel("Mechanical Part")
ax.set_ylim(0, max(values) * 1.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()





