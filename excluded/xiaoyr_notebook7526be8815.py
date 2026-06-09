# âœ… å¯¼å…¥å¿…è¦�åº“
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.model_selection import train_test_split


# âœ… æ£€æŸ¥ GPU
print("GPU å�¯ç”¨:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU å��ç§°:", torch.cuda.get_device_name(0))

# âœ… åŠ è½½æ ‡ç­¾æ–‡ä»¶ï¼Œä¿�ç•™å�Ÿå§‹ç¼–å�·ï¼ˆ0~881ï¼‰
label_df = pd.read_csv("train_labels.csv")
label_df["class"] = label_df["ID"].apply(lambda x: x.split("/")[1])
class_to_idx = label_df.groupby("class")["Label"].min().to_dict()
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = label_df["Label"].max() + 1
print(f"å…±æœ‰ç±»åˆ«: {len(class_to_idx)}ï¼Œè¾“å‡ºç»´åº¦: {num_classes}")

# âœ… æ‰“å�°æ˜ å°„è¡¨
print("ğŸ“‹ ç±»åˆ«ç¼–å�· â†” ä¸­è�¯å��ï¼š")
for idx, name in sorted(idx_to_class.items()):
    print(f"{idx:>3} â†” {name}")



# âœ… è‡ªå®šä¹‰ Dataset
class MedicineDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path)
            if image.mode == 'P':
                image = image.convert("RGBA").convert("RGB")
            else:
                image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return self.__getitem__((idx + 1) % len(self.samples))


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(path)



class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.best_weights = None

    def __call__(self, val_score, model):
        score = val_score if self.mode == "max" else -val_score

        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_weights = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


import numpy as np

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2



# âœ… å›¾åƒ�å¢�å¼º
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# âœ… åŠ è½½å›¾åƒ�è·¯å¾„å¹¶åˆ’åˆ†è®­ç»ƒéªŒè¯�é›†
train_root = 'train'
all_samples = []
for cls in os.listdir(train_root):
    cls_path = os.path.join(train_root, cls)
    if not os.path.isdir(cls_path) or cls not in class_to_idx:
        continue
    for fname in os.listdir(cls_path):
        all_samples.append((os.path.join(cls_path, fname), class_to_idx[cls]))

train_samples, val_samples = train_test_split(
    all_samples, test_size=0.1, stratify=[s[1] for s in all_samples], random_state=42)

train_dataset = MedicineDataset(train_samples, transform=transform)
val_dataset = MedicineDataset(val_samples, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


# âœ… Voting è·¯çº¿è®­ç»ƒä¸‰ä¸ªæ¨¡å�‹ï¼ˆå¤š GPU æ”¯æŒ�ï¼‰
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ä½¿ç”¨è®¾å¤‡: {device}ï¼ŒGPUæ•°é‡�: {torch.cuda.device_count()}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ---------- ConvNeXt ----------
convnext = models.convnext_tiny(weights='DEFAULT')
convnext.classifier = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm((768,), eps=1e-6),
    nn.Linear(768, num_classes)
)
if torch.cuda.device_count() > 1:
    print("âœ… ä½¿ç”¨å¤š GPU è®­ç»ƒ ConvNeXt")
    convnext = nn.DataParallel(convnext)
convnext = convnext.to(device)
optimizer = torch.optim.Adam(convnext.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


early_stopper = EarlyStopping(patience=3, mode="max")
for epoch in range(50):
    convnext.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"ConvNeXt Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
         # âœ… CutMixï¼šå¯¹ä¸€æ‰¹å›¾åƒ�åº”ç”¨ CutMix å¢�å¼º
        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(images.size(0)).to(device)
        target_a = labels
        target_b = labels[rand_index]

        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
       
        optimizer.zero_grad()
        outputs = convnext(images)
        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"[ConvNeXt] Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    convnext.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = convnext(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    acc = correct / len(val_loader.dataset)
    print(f"[ConvNeXt] âœ… éªŒè¯�å‡†ç¡®ç�‡: {acc:.4f}")
    # æ£€æŸ¥æ˜¯å�¦æ—©å�œ
    early_stopper(acc, convnext)
    if early_stopper.early_stop:
        print("â�¹ï¸� æ—©å�œè§¦å�‘ï¼Œå�œæ­¢è®­ç»ƒ")
        break

# æ�¢å¤�æœ€ä¼˜æ¨¡å�‹å�‚æ•°
convnext.load_state_dict(early_stopper.best_weights)

torch.save(convnext.state_dict(), "convnext.pth")
print("âœ… ConvNeXt å·²ä¿�å­˜")


# ---------- ResNet ----------
resnet = models.resnet50(weights='DEFAULT')
resnet.fc = nn.Linear(2048, num_classes)
if torch.cuda.device_count() > 1:
    print("âœ… ä½¿ç”¨å¤š GPU è®­ç»ƒ ResNet")
    resnet = nn.DataParallel(resnet)
resnet = resnet.to(device)
optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4)



early_stopper = EarlyStopping(patience=3, mode="max")
for epoch in range(50):
    resnet.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"ResNet Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        # âœ… CutMixï¼šå¯¹ä¸€æ‰¹å›¾åƒ�åº”ç”¨ CutMix å¢�å¼º
        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(images.size(0)).to(device)
        target_a = labels
        target_b = labels[rand_index]

        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
       
        
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[ResNet] Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    resnet.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    acc = correct / len(val_loader.dataset)
    print(f"[ResNet] âœ… éªŒè¯�å‡†ç¡®ç�‡: {acc:.4f}")

      # æ£€æŸ¥æ˜¯å�¦æ—©å�œ
    early_stopper(acc, resnet)
    if early_stopper.early_stop:
        print("â�¹ï¸� æ—©å�œè§¦å�‘ï¼Œå�œæ­¢è®­ç»ƒ")
        break

# æ�¢å¤�æœ€ä¼˜æ¨¡å�‹å�‚æ•°
resnet.load_state_dict(early_stopper.best_weights)
torch.save(resnet.state_dict(), "resnet.pth")
print("âœ… ResNet50 å·²ä¿�å­˜")

# ---------- EfficientNet ----------
efficientnet = models.efficientnet_b0(weights='DEFAULT')
efficientnet.classifier = nn.Linear(1280, num_classes)
if torch.cuda.device_count() > 1:
    print("âœ… ä½¿ç”¨å¤š GPU è®­ç»ƒ EfficientNet")
    efficientnet = nn.DataParallel(efficientnet)
efficientnet = efficientnet.to(device)
optimizer = torch.optim.Adam(efficientnet.parameters(), lr=1e-4)



early_stopper = EarlyStopping(patience=3, mode="max")
for epoch in range(50):
    efficientnet.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"EffNet Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        # âœ… CutMixï¼šå¯¹ä¸€æ‰¹å›¾åƒ�åº”ç”¨ CutMix å¢�å¼º
        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(images.size(0)).to(device)
        target_a = labels
        target_b = labels[rand_index]

        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))

        
        optimizer.zero_grad()
        outputs = efficientnet(images)
        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[EffNet] Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    efficientnet.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = efficientnet(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    acc = correct / len(val_loader.dataset)
    print(f"[EffNet] âœ… éªŒè¯�å‡†ç¡®ç�‡: {acc:.4f}")
    # æ£€æŸ¥æ˜¯å�¦æ—©å�œ
    early_stopper(acc, efficientnet)
    if early_stopper.early_stop:
        print("â�¹ï¸� æ—©å�œè§¦å�‘ï¼Œå�œæ­¢è®­ç»ƒ")
        break

# æ�¢å¤�æœ€ä¼˜æ¨¡å�‹å�‚æ•°
efficientnet.load_state_dict(early_stopper.best_weights)

torch.save(efficientnet.state_dict(), "efficientnet.pth")
print("âœ… EfficientNet å·²ä¿�å­˜")




# åŠ è½½å·²è®­ç»ƒå¥½çš„ä¸‰ä¸ªæ¨¡å�‹
convnext.load_state_dict(torch.load("convnext.pth"))
resnet.load_state_dict(torch.load("resnet50.pth"))
efficientnet.load_state_dict(torch.load("efficientnet.pth"))

convnext.eval()
resnet.eval()
efficientnet.eval()

test_dataset = TestDataset("test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

results = []
with torch.no_grad():
    for images, filenames in tqdm(test_loader):
        images = images.to(device)

        # æ¯�ä¸ªæ¨¡å�‹é¢„æµ‹
        preds1 = convnext(images)
        preds2 = resnet(images)
        preds3 = efficientnet(images)

        # æ±‚å¹³å�‡æ¦‚ç�‡ï¼Œå†�å�– argmax
        probs = (F.softmax(preds1, dim=1) + F.softmax(preds2, dim=1) + F.softmax(preds3, dim=1)) / 3
        final_preds = torch.argmax(probs, dim=1)

        for fname, pred in zip(filenames, final_preds.cpu().numpy()):
            results.append((fname, pred))

submission = pd.DataFrame(results, columns=["ID", "Label"])
submission.to_csv("submission_voting.csv", index=False)
print("ğŸ“� submission_voting.csv å·²ç”Ÿæˆ�")

