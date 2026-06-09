import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# é…�ç½®è·¯å¾„
CSV_PATH = "/kaggle/input/sartorius-cell-instance-segmentation/train.csv"
SAVE_MASK_DIR = "/kaggle/working/masks"
os.makedirs(SAVE_MASK_DIR, exist_ok=True)

HEIGHT, WIDTH = 520, 704  # å›ºå®šå°ºå¯¸

def rle_decode(mask_rle, shape):
    s = mask_rle.strip().split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape) # è½¬ç½®ä»¥ç¬¦å�ˆå›¾åƒ�æ ¼å¼�

# è¯»å�– CSV
df = pd.read_csv(CSV_PATH)

# å�ˆå¹¶æ¯�å¼ å›¾ç‰‡çš„å¤šä¸ªæ�©è†œ
for image_id, group in tqdm(df.groupby("id")):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for _, row in group.iterrows():
        mask += rle_decode(row["annotation"], (HEIGHT, WIDTH))
    mask = np.clip(mask, 0, 1) * 255
    Image.fromarray(mask).save(f"{SAVE_MASK_DIR}/{image_id}.png")






# ğŸ“� EXP-1: DeepLabv3_ResNet50 Safe Training Script
import os, numpy as np, torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd

# config
IMG_DIR = "/kaggle/input/sartorius-cell-instance-segmentation/train"
MASK_DIR ="/kaggle/working/masks"
SAVE_DIR = "/kaggle/working/exp1_baseline"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
BATCH_SIZE = 4

# model
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.classifier = torch.nn.Sequential(
    torch.nn.Conv2d(2048, 256, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(256, 1, kernel_size=1)
)
model = model.to(DEVICE)
transform = weights.transforms()

# dataset
class CellDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = sorted([f for f in os.listdir(mask_dir) if os.path.exists(os.path.join(image_dir, f))])
        self.transform = transform
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        if self.transform:
            img = self.transform(img)
            mask = transforms.ToTensor()(mask)
        return img, mask

# loss & optimizer
def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds).view(-1)
    targets = targets.view(-1)
    inter = (preds * targets).sum()
    return 1 - (2 * inter + smooth) / (preds.sum() + targets.sum() + smooth)

bce = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# data loader
dataset = CellDataset(IMG_DIR, MASK_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# resume check
start_epoch = 0
metrics_path = f"{SAVE_DIR}/metrics.csv"
train_loss, val_iou, val_f1 = [], [], []
if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path)
    train_loss = df["Loss"].tolist()
    val_iou = df["IoU"].tolist()
    val_f1 = df["F1"].tolist()
    start_epoch = len(df)
    last_ckpt = f"{SAVE_DIR}/model_epoch_{start_epoch}.pth"
    if os.path.exists(last_ckpt):
        model.load_state_dict(torch.load(last_ckpt, map_location=DEVICE))
        print(f" æ�¢å¤�è®­ç»ƒï¼šä»� epoch {start_epoch+1} ç»§ç»­")

# training loop
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)['out']
        loss = bce(preds, masks) + 0.5 * dice_loss(preds, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()

    # evaluation on a single batch
    model.eval()
    with torch.no_grad():
        val_imgs, val_masks = next(iter(loader))
        val_imgs = val_imgs.to(DEVICE)
        output = model(val_imgs)['out']
        pred_bin = (torch.sigmoid(output) > 0.5).cpu().numpy().astype(int).reshape(-1)
        y_true = val_masks.cpu().numpy().astype(int).reshape(-1)
        iou = jaccard_score(y_true, pred_bin)
        f1 = f1_score(y_true, pred_bin)

    avg_loss = total_loss / len(loader)
    train_loss.append(avg_loss)
    val_iou.append(iou)
    val_f1.append(f1)

    # save checkpoint + metrics
    torch.save(model.state_dict(), f"{SAVE_DIR}/model_epoch_{epoch+1}.pth")
    pd.DataFrame({"Loss": train_loss, "IoU": val_iou, "F1": val_f1}).to_csv(metrics_path, index=False)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={avg_loss:.4f} | IoU={iou:.4f} | F1={f1:.4f} | å·²ä¿�å­˜")

# plot curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1); plt.plot(train_loss); plt.title("Training Loss")
plt.subplot(1, 2, 2); plt.plot(val_iou, label="IoU"); plt.plot(val_f1, label="F1"); plt.title("Validation"); plt.legend()
plt.savefig(f"{SAVE_DIR}/curves.png")
print(f"æ‰€æœ‰æ¨¡å�‹ä¸�æ—¥å¿—å·²ä¿�å­˜è‡³ {SAVE_DIR}")


