import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

BASE_TRAIN_PATH = "/kaggle/input/binary-biplob-can-you-guess-the-chess-opening/chess_dataset/train_images"
BASE_TEST_PATH = "/kaggle/input/binary-biplob-can-you-guess-the-chess-opening/chess_dataset/test_images"

train_df = pd.read_csv("/kaggle/input/binary-biplob-can-you-guess-the-chess-opening/chess_dataset/train.csv")
label_map = {label: idx for idx, label in enumerate(sorted(train_df.eco_volume.unique()))}
train_df['label'] = train_df.eco_volume.map(label_map)

class ChessDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row.image_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, row.label

class ChessTestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row.image_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(train_df, train_df.label))[0]  
train_data = train_df.iloc[train_idx]
val_data = train_df.iloc[val_idx]

train_loader = DataLoader(ChessDataset(train_data, BASE_TRAIN_PATH, train_tfms), batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(ChessDataset(val_data, BASE_TRAIN_PATH, val_tfms), batch_size=32, shuffle=False, num_workers=2)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(5):  
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch+1}/3")
    for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    print(f"Train Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Validation Accuracy: {correct/total:.4f}")


torch.save(model.state_dict(), "efficientnet_chess.pth")

test_df = pd.read_csv("/kaggle/input/binary-biplob-can-you-guess-the-chess-opening/chess_dataset/test.csv")
test_dataset = ChessTestDataset(test_df, BASE_TEST_PATH, val_tfms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

model.eval()
preds_labels = []
with torch.no_grad():
    for imgs in tqdm(test_loader, desc="Predicting on Test"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        preds_labels.extend(preds.cpu().numpy())

inv_label_map = {v: k for k, v in label_map.items()}
preds_labels = [inv_label_map[p] for p in preds_labels]



submission = pd.DataFrame({
    "image_id": test_df['image_id'],
    "eco_volume": pred_labels
})
submission.to_csv("submission3.csv", index=False)
print("✅ submission3.csv saved!")


