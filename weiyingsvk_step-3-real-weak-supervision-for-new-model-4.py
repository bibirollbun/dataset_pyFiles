import numpy as np 
import pandas as pd
import os
import cv2

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as nnf


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

NUM_CLASSES = 19
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



import numpy as np
import matplotlib.pyplot as plt

# åŠ è½½ä¸€ä¸ª npz æ�©è†œæ–‡ä»¶
path = "/kaggle/input/hpa2021-p-all-1stx8-2ndx16-imgx16/mask/train/cell/0042017c-bba4-11e8-b2b9-ac1f6b6435d0.npz"
data = np.load(path, allow_pickle=True)
mask = data["arr_0"]  # æˆ– data.files[0]ï¼Œå·²ç»�ç¡®è®¤æ˜¯ 'arr_0'

# æ‰“å�°æ�©è†œä¿¡æ�¯
print("æ�©è†œ shape:", mask.shape)
print("æ�©è†œç±»å�‹:", type(mask))
print("æ�©è†œåƒ�ç´ å�–å€¼ç§�ç±»:", np.unique(mask))

# å�¯è§†åŒ–
plt.imshow(mask, cmap="tab20")
plt.colorbar()
plt.title("Structure Mask Visualization")
plt.show()


import pandas as pd

# è·¯å¾„ï¼ˆä½ å�¯ä»¥æ›¿æ�¢æˆ�å®�é™…çš„è·¯å¾„ï¼‰
bbox_path = "/kaggle/input/hpa2021-p-all-1stx8-2ndx16-imgx16/train_bbox_filtered.csv"
train_path = "/kaggle/input/hpa-single-cell-image-classification/train.csv"

# è¯»å�–ä¸¤ä¸ª CSV
bbox_df = pd.read_csv(bbox_path)
train_df = pd.read_csv(train_path)

# æ˜¾ç¤ºå�˜é‡�å��ç§°å’Œå‰�å‡ è¡Œ
print("========== bbox_filtered.csv ==========")
print("ä½œç”¨ï¼šæ��ä¾›æ¯�ä¸ª image_id ä¸‹æ¯�ä¸ª cell çš„ bounding boxã€�cell_idã€�cell å¤§å°�ç­‰ä¿¡æ�¯")
print("åˆ—å��å¦‚ä¸‹ï¼š")
print(bbox_df.columns.tolist())
print("\nç¤ºä¾‹æ•°æ�®ï¼ˆå‰�5è¡Œï¼‰ï¼š")
print(bbox_df.head())

print("\n========== train.csv ==========")
print("ä½œç”¨ï¼šæ��ä¾›æ¯�å¼ å›¾åƒ�ï¼ˆimage_idï¼‰å¯¹åº”çš„æ•´ä½“æ ‡ç­¾ï¼ˆå›¾åƒ�çº§ï¼‰")
print("åˆ—å��å¦‚ä¸‹ï¼š")
print(train_df.columns.tolist())
print("\nç¤ºä¾‹æ•°æ�®ï¼ˆå‰�5è¡Œï¼‰ï¼š")
print(train_df.head())


import pandas as pd
import numpy as np

# è·¯å¾„
bbox_path = "/kaggle/input/hpa2021-p-all-1stx8-2ndx16-imgx16/train_bbox_filtered.csv"
train_path = "/kaggle/input/hpa-single-cell-image-classification/train.csv"

# è¯»å�–æ•°æ�®
bbox_df = pd.read_csv(bbox_path)
train_df = pd.read_csv(train_path)

# é¢„å¤„ç�†æ ‡ç­¾å­—æ®µ
def parse_label(label_str):
    return [int(x) for x in label_str.split('|') if x != '']

train_df['label_list'] = train_df['Label'].map(parse_label)
train_df = train_df.rename(columns={"ID": "image_id"})

# æ�„å»ºæ˜ å°„ï¼ˆimage_id â†’ multi-hot å�‘é‡�ï¼‰
imageid_to_vector = {}
for _, row in train_df.iterrows():
    vec = [0] * 19
    for i in row['label_list']:
        if 0 <= i < 19:
            vec[i] = 1
    imageid_to_vector[row['image_id']] = vec

# æ·»åŠ  label_vector åˆ—
bbox_df['label_vector'] = bbox_df['image_id'].map(imageid_to_vector)

# ä¿�å­˜ç»“æ�œ
bbox_df.to_csv("/kaggle/working/bbox_with_label_vector.csv", index=False)

print("ä¿�å­˜æˆ�åŠŸï¼�å…± %d ä¸ªç»†èƒ�ï¼Œå­—æ®µå��å¦‚ä¸‹ï¼š" % len(bbox_df))
print(bbox_df.columns.tolist())



bbox_df = pd.read_csv("/kaggle/input/hpa2021-p-all-1stx8-2ndx16-imgx16/train_bbox_filtered.csv")
train_df = pd.read_csv("/kaggle/input/hpa-single-cell-image-classification/train.csv")

# è�·å�– ID é›†å�ˆ
bbox_ids = set(bbox_df["image_id"].unique())
train_ids = set(train_df["ID"].unique())

# è®¡ç®—åŒ¹é…�åº¦
matched_ids = bbox_ids & train_ids

print(f"æ€»å…±æœ‰ {len(bbox_ids)} ä¸ª bbox å›¾åƒ�ID")
print(f"æ€»å…±æœ‰ {len(train_ids)} ä¸ª train å›¾åƒ�ID")
print(f"äº¤é›†å›¾åƒ�IDæ•°: {len(matched_ids)}ï¼ŒåŒ¹é…�æ¯”ä¾‹: {len(matched_ids)/len(bbox_ids)*100:.2f}%")


import numpy as np

path = "/kaggle/input/hpa2021-p-all-1stx8-2ndx16-imgx16/mask/train/cell/0042017c-bba4-11e8-b2b9-ac1f6b6435d0.npz"
data = np.load(path, allow_pickle=True)

# è�·å�–å”¯ä¸€å­—æ®µ arr_0
content = data["arr_0"]

# æ‰“å�°ç»“æ�„ä¿¡æ�¯
print("å­—æ®µç±»å�‹ï¼š", type(content))
print("å…ƒç´ æ•°é‡�ï¼ˆå¦‚æœ‰ï¼‰ï¼š", len(content))

# æ‰“å�°å‰�å‡ ä¸ªå†…å®¹æŸ¥çœ‹ç»“æ�„
if isinstance(content, dict):
    print("dict keys ç¤ºä¾‹ï¼š", list(content.keys())[:3])
elif isinstance(content, list):
    print("list å‰�3é¡¹ï¼š", content[:3])
elif isinstance(content, np.ndarray):
    print("ndarray shapeï¼š", content.shape)
    print("å‰�1ä¸ªå…ƒç´ ç±»å�‹ï¼š", type(content[0]))
    print("å‰�1ä¸ªå…ƒç´ å†…å®¹ï¼ˆç®€ç•¥ï¼‰ï¼š", content[0])
else:
    print("æœªçŸ¥ç±»å�‹ï¼š", content)


from PIL import Image  # ç¡®ä¿�å·²å¯¼å…¥

class WeakSupervisionDataset(Dataset):
    def __init__(self, csv_path, image_dir, mask_dir, transform=None, crop_size=512):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        cell_id = row['cell_id']
        x0, y0, x1, y1 = int(row['x0']), int(row['y0']), int(row['x1']), int(row['y1'])

        # === Step 1: è¯»å�– R/Y/B å›¾åƒ�ï¼Œå¹¶è®°å½•å�Ÿå§‹å°ºå¯¸ ===
        r = cv2.imread(os.path.join(self.image_dir, f"{image_id}_red.png"), cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(os.path.join(self.image_dir, f"{image_id}_yellow.png"), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(os.path.join(self.image_dir, f"{image_id}_blue.png"), cv2.IMREAD_GRAYSCALE)

        original_h, original_w = r.shape[:2]  # âš ï¸� å�Ÿå›¾å°ºå¯¸

        # === Step 2: resize å›¾åƒ�ï¼Œå¹¶æ�„é€ å¤šé€šé�“ ===
        r = cv2.resize(r, (target_w, target_h))
        y = cv2.resize(y, (target_w, target_h))
        b = cv2.resize(b, (target_w, target_h))
        image = np.stack([r, y, b], axis=-1)  # [2048, 2048, 3]

        # === Step 3: è¯»å�–å¹¶ resize æ�©è†œ ===
        npz_path = os.path.join(self.mask_dir, f"{image_id}.npz")
        mask = np.load(npz_path, allow_pickle=True)["arr_0"]
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # === Step 4: âš ï¸� å�Œæ­¥ç¼©æ”¾ bbox å��æ ‡ ===
        scale_x = target_w / original_w
        scale_y = target_h / original_h
        x0 = int(x0 * scale_x)
        x1 = int(x1 * scale_x)
        y0 = int(y0 * scale_y)
        y1 = int(y1 * scale_y)

        # === Step 5: æ�„é€ æ­£æ–¹ bbox å¹¶åŠ  padding ===
        H, W = image.shape[:2]
        bbox_w = x1 - x0
        bbox_h = y1 - y0
        side = max(bbox_w, bbox_h)
        pad = 10

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        new_x0 = max(0, cx - side // 2 - pad)
        new_x1 = min(W, cx + side // 2 + pad)
        new_y0 = max(0, cy - side // 2 - pad)
        new_y1 = min(H, cy + side // 2 + pad)

        # === Step 6: å¼‚å¸¸å¤„ç�† ===
        if new_x1 <= new_x0 or new_y1 <= new_y0:
            dummy = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            crop_img = Image.fromarray(dummy)
            label_vector = eval(row['label_vector'])
            crop_tensor = transforms.ToTensor()(crop_img)
            label_tensor = torch.tensor(label_vector).float()
            return crop_tensor, label_tensor, image_id, cell_id

        # === Step 7: crop å›¾åƒ�å¹¶ resize ===
        crop_img = image[new_y0:new_y1, new_x0:new_x1, :]
        crop_img = cv2.resize(crop_img, (self.crop_size, self.crop_size))
        crop_img = Image.fromarray(crop_img.astype(np.uint8))  # è½¬ä¸º PIL Image

        # === Step 8: å›¾åƒ� transform / å½’ä¸€åŒ– ===
        if self.transform:
            crop_tensor = self.transform(crop_img)
        else:
            crop_np = np.array(crop_img).astype(np.float32) / 255.0
            crop_np = np.transpose(crop_np, (2, 0, 1))
            crop_tensor = torch.tensor(crop_np).float()

        # === Step 9: æ ‡ç­¾å¤„ç�† ===
        label_vector = eval(row['label_vector'])
        label_tensor = torch.tensor(label_vector).float()

        return crop_tensor, label_tensor, image_id, cell_id



from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

# åŠ è½½ CSVï¼ˆå�¯é€‰ï¼‰
df_valid = pd.read_csv('/kaggle/working/bbox_with_label_vector.csv')

import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(256, 256),  # æˆ– 512Ã—512ï¼Œæ ¹æ�®æ˜¾å­˜é€‰
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# âœ… æ­£ç¡®åˆ›å»º datasetï¼ˆä½¿ç”¨å‰�é�¢æ”¹å¥½çš„ç±»ï¼‰
dataset = WeakSupervisionDataset(
    csv_path="/kaggle/working/bbox_with_label_vector.csv",
    image_dir="/kaggle/input/hpa-single-cell-image-classification/train",
    mask_dir="/kaggle/input/hpa2021-p-all-1stx8-2ndx16-imgx16/mask/train/cell",
    transform=transform,
    target_size=2048
)


# âœ… åˆ›å»º dataloader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# âœ… æ£€æŸ¥ç¬¬ä¸€å¼ æ ·æœ¬ï¼ˆå�˜é‡�å��é¡ºåº�æ›´æ–°ï¼‰
image_tensor, mask_tensor, label_tensor, image_id, cell_id = dataset[0]

print(f"å›¾åƒ�å¤§å°�: {image_tensor.shape}")
print(f"æ�©è†œå¤§å°�: {mask_tensor.shape}")
print(f"æ ‡ç­¾å�‘é‡�: {label_tensor}")
print(f"å›¾åƒ� ID: {image_id}, Cell ID: {cell_id}")



import numpy as np
import pandas as pd

df_valid = pd.read_csv('/kaggle/working/bbox_with_label_vector.csv')

# è§£æ��å¹¶æ±‚å’Œ
label_matrix = df_valid['label_vector'].apply(lambda x: np.array(eval(x)))  # CSVä¸­æ˜¯åˆ—è¡¨æ ¼å¼�å­—ç¬¦ä¸²
label_sum = np.sum(np.stack(label_matrix.values), axis=0)

print("æ¯�ä¸ªç±»åˆ«çš„æ ·æœ¬æ•°:", label_sum.astype(int))



import albumentations as A
from albumentations.pytorch import ToTensorV2

# è®¾ç½®è·¯å¾„
csv_path = "/kaggle/working/bbox_with_label_vector.csv"
image_dir = "/kaggle/input/hpa-single-cell-image-classification/train"
mask_dir = "/kaggle/input/hpa2021-p-all-1stx8-2ndx16-imgx16/mask/train/cell"

# ä½¿ç”¨ Albumentations å�šå›¾åƒ�å¢�å¼º + resize
transform = A.Compose([
    A.Resize(256, 256),  # ä½ ä¹Ÿå�¯ä»¥æ”¹ä¸º 512
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# åˆ›å»º dataset å®�ä¾‹ï¼ˆæ³¨æ„� crop_size æ”¹ä¸º target_sizeï¼‰
train_dataset = WeakSupervisionDataset(
    csv_path=csv_path,
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform=transform,
    target_size=2048  # å�Ÿå›¾ resize å°ºå¯¸ï¼Œmask/image éƒ½ä¼šå…ˆ resize åˆ°è¿™ä¸ªå¤§å°�å†� crop
)



import matplotlib.pyplot as plt
import random

def visualize_random_crops(dataset, num_samples=12, rows=3, cols=4, seed=42):
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image_tensor, mask_tensor, label_tensor, image_id, cell_id = dataset[idx]
        
        # å›¾åƒ�è¿˜å�Ÿä¸º numpy
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)  # å�» Normalize

        # mask è½¬ numpyï¼ˆç¡®ä¿�æ˜¯ [H,W]ï¼‰
        mask_np = mask_tensor.squeeze().cpu().numpy()

        # æ˜¾ç¤ºå›¾åƒ� + æ�©è†œ overlay
        ax.imshow(img_np)
        ax.imshow(mask_np, cmap='Reds', alpha=0.4)
        ax.set_title(f"Image: {image_id[-5:]}\nCell: {cell_id}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# âœ… è°ƒç”¨æ–¹æ³•ï¼ˆå�‡è®¾ä½ çš„ dataset å�«å�š train_datasetï¼‰
visualize_random_crops(train_dataset)



import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_full_and_crop(dataset, num_samples=5, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        # è�·å�–æ ·æœ¬æ•°æ�®
        image_tensor, mask_tensor, label_vector, image_id, cell_id = dataset[idx]

        # å›¾åƒ�è¿˜å�Ÿ
        crop_img = image_tensor.permute(1, 2, 0).numpy()  # [H,W,C]
        crop_img = (crop_img * 0.5 + 0.5).clip(0, 1)

        # æ�©è†œè¿˜å�Ÿ
        mask_np = mask_tensor.squeeze().numpy()

        # === Step 1: åŠ è½½å�Ÿå›¾å¹¶ resize åˆ° 2048 ===
        r = cv2.imread(f"/kaggle/input/hpa-single-cell-image-classification/train/{image_id}_red.png", cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(f"/kaggle/input/hpa-single-cell-image-classification/train/{image_id}_yellow.png", cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(f"/kaggle/input/hpa-single-cell-image-classification/train/{image_id}_blue.png", cv2.IMREAD_GRAYSCALE)
        rgb = np.stack([r, y, b], axis=-1)
        rgb = cv2.resize(rgb, (2048, 2048))
        rgb = rgb.astype(np.float32) / 255.0

        # === Step 2: åŠ è½½ mask å¹¶æ��å�– cell çš„åŒºåŸŸ ===
        mask_path = f"/kaggle/input/hpa2021-p-all-1stx8-2ndx16-imgx16/mask/train/cell/{image_id}.npz"
        full_mask = np.load(mask_path, allow_pickle=True)["arr_0"]
        full_mask = cv2.resize(full_mask, (2048, 2048), interpolation=cv2.INTER_NEAREST)
        cell_mask = (full_mask == cell_id).astype(np.uint8)

        # === Step 3: overlay mask åˆ°å�Ÿå›¾ ===
        overlay = rgb.copy()
        overlay[cell_mask == 1] = [1.0, 0.0, 0.0]  # çº¢è‰²åŒºåŸŸé«˜äº®

        # === Step 4: æ˜¾ç¤ºä¸¤å¼ å›¾ ===
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(overlay)
        plt.title(f"Full Image + Mask\n{image_id[-5:]} | Cell {cell_id}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(crop_img)
        plt.title("Cropped Cell Image")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

visualize_full_and_crop(train_dataset)


import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# è¯»å�–å›¾åƒ�çº§æ ‡ç­¾æ•°æ�®
meta_path = '/kaggle/input/hpa-single-cell-image-classification/train.csv'
df_meta = pd.read_csv(meta_path)

# æ‹†åˆ†æ ‡ç­¾åˆ—ä¸º listï¼ˆæ¯�è¡Œæ˜¯ä¸€ä¸ª label listï¼‰
df_meta['Label_list'] = df_meta['Label'].apply(lambda x: list(map(int, x.split('|'))))

# åˆ�å§‹åŒ–æ¯�ç±»è®¡æ•°æ•°ç»„
NUM_CLASSES = 19
label_counts = np.zeros(NUM_CLASSES, dtype=int)

# ç»Ÿè®¡æ¯�ä¸ªç±»åˆ«åœ¨ image-level ä¸Šçš„å‡ºç�°æ¬¡æ•°
for labels in df_meta['Label_list']:
    for cls in labels:
        label_counts[cls] += 1

print("æ¯�ä¸ªç±»åˆ«çš„ image-level å‡ºç�°æ¬¡æ•°:", label_counts)
# å�Ÿå§‹é¢‘ç�‡è®¡ç®—
class_freq = label_counts / label_counts.sum()

# å¯¹æ•°ç¼©æ”¾ï¼Œé�¿å…� weight å·®è·�è¿‡å¤§
pos_weight = np.log((1 - class_freq) / class_freq + 1e-8)

# å¯¹äº� class 18 å’Œ class 0ã€�16 æ‰‹åŠ¨è°ƒæ•´
pos_weight[18] = 0.0
pos_weight[0] = min(pos_weight[0], 3.5)   # é™�ä½�ä¸€ç‚¹æƒ©ç½š
pos_weight[16] = min(pos_weight[16], 3)

# æ‰“å�°æ¯�ä¸ªç±»åˆ«çš„ pos_weight
print("\næ¯�ä¸ªç±»åˆ«çš„ BCEWithLogitsLoss æ�ƒé‡�ï¼ˆpos_weightï¼‰å¦‚ä¸‹ï¼š")
for i in range(NUM_CLASSES):
    print(f"class {i:2d}: pos_weight = {pos_weight[i]:.4f}")


from torchvision import models
import torch
import torch.nn as nn

# æ�„å»ºæ¨¡å�‹
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=None)  # ç­‰ä»·äº�ä¸�åŠ è½½é¢„è®­ç»ƒå�‚æ•°
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# è®¾ç½® class 18 ä¸�å�‚ä¸�è®­ç»ƒ
pos_weight[18] = 0.0

# å®šä¹‰åŠ æ�ƒ BCE æ�Ÿå¤±
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

# ä¼˜åŒ–å™¨
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


from tqdm import tqdm
import torch
import os
import numpy as np

# è®¾ç½®ä¿�å­˜æ¨¡å�‹çš„è·¯å¾„
best_val_loss = float('inf')
best_model_path = "/kaggle/working/best_model_train.pth"

# è®­ç»ƒè½®æ•°ï¼ˆä½ å�¯ä»¥è‡ªå®šä¹‰ï¼‰
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    train_loader_tqdm = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}")
    step = 0  # âœ… åˆ�å§‹åŒ– step

    for images, _, labels, image_ids, cell_ids in train_loader_tqdm:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # å‰�å�‘ä¼ æ’­
        outputs = model(images)
        loss = criterion(outputs, labels)

        # å��å�‘ä¼ æ’­ + ä¼˜åŒ–
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item())

        step += 1
        if step > 4:  # âœ… æœ€å¤šå�ªè®­ç»ƒ 16 ä¸ª batchï¼ˆ32Ã—16=512 ä¸ªæ ·æœ¬ï¼‰
            break

    avg_train_loss = total_loss / step  # æ³¨æ„�é™¤ä»¥ step è€Œä¸�æ˜¯ len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} å®Œæˆ�ï¼Œå¹³å�‡è®­ç»ƒæ�Ÿå¤±: {avg_train_loss:.4f}")

    # éªŒè¯�ï¼ˆå�¯é€‰ï¼‰
    if 'val_loader' in globals():
        model.eval()
        val_loss = 0.0
        val_step = 0
        with torch.no_grad():
            for images, labels, *_ in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_step += 1

        avg_val_loss = val_loss / val_step
        print(f"Validation loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"âœ… æ–°æœ€ä½³æ¨¡å�‹å·²ä¿�å­˜åˆ°: {best_model_path}ï¼ˆval_loss: {best_val_loss:.4f}ï¼‰")

    else:
        # æ²¡æœ‰éªŒè¯�é›†ï¼ŒåŸºäº�è®­ç»ƒ loss ä¿�å­˜
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"âœ… æ–°æœ€ä½³æ¨¡å�‹ï¼ˆåŸºäº�è®­ç»ƒæ�Ÿå¤±ï¼‰å·²ä¿�å­˜åˆ°: {best_model_path}")


import pandas as pd
import torch
from tqdm import tqdm

model.load_state_dict(torch.load("/kaggle/working/best_model_train.pth", map_location=DEVICE))
model.eval()

results = []
threshold = 0.2
NUM_CLASSES = 19
max_steps = 5  # å�ªå¤„ç�†å‰� 30 ä¸ª batch

with torch.no_grad():
    for step, (images, _, labels, image_ids, cell_ids) in enumerate(tqdm(train_loader, desc="Collecting predictions")):
        if step >= max_steps:
            break
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu()

        for i in range(images.size(0)):
            for class_id in range(NUM_CLASSES):
                conf = probs[i, class_id].item()
                if conf > threshold:
                    results.append({
                        "image_id": image_ids[i],
                        "cell_id": int(cell_ids[i]),
                        "class_id": class_id,
                        "confidence": conf,
                        "filename": f"{image_ids[i]}_class{class_id}_cell{cell_ids[i]}.png"
                    })

df_result = pd.DataFrame(results)
df_result.to_csv("/kaggle/working/cell_level_predictions_conf_07_train.csv", index=False)
print("âœ… ä¿�å­˜æˆ�åŠŸï¼Œå…±æœ‰é¢„æµ‹ç±»åˆ«æ•°:", len(df_result))



import pandas as pd

# è¯»å�– CSV æ–‡ä»¶
df = pd.read_csv('/kaggle/working/cell_level_predictions_conf_07_train.csv')

# æ˜¾ç¤ºå‰� 15 è¡Œ
print(df.head(15))



import pandas as pd

df = pd.read_csv('/kaggle/working/cell_level_predictions_conf_07_train.csv')

# å¯¹ image_id åˆ†ç»„ï¼Œå¤„ç�†æ¯�ä¸ª group
def keep_top4_classes(group):
    # è�·å�–è¯¥ image_id çš„å”¯ä¸€ class_id ä¸ªæ•°
    if len(group['class_id'].unique()) > 4:
        # æŒ‰ confidence é™�åº�æ�’åº�ï¼Œä¿�ç•™ top 4 class
        return group.sort_values('confidence', ascending=False).drop_duplicates('class_id').head(4)
    else:
        return group

# æŒ‰ image_id åˆ†ç»„åº”ç”¨ç­›é€‰é€»è¾‘
df_top4 = df.groupby('image_id', group_keys=False).apply(keep_top4_classes).reset_index(drop=True)

# ä¿�å­˜ç»“æ�œ
df_top4.to_csv('/kaggle/working/cell_level_predictions_conf_07_top4_train.csv', index=False)
print("å·²å®Œæˆ�æŒ‰ image_id ä¿�ç•™ top 4 class çš„ç­›é€‰ï¼Œç»“æ�œå·²ä¿�å­˜è‡³:")
print("/kaggle/working/cell_level_predictions_conf_07_top4_train.csv")


import pandas as pd
from collections import defaultdict

# âœ… åŠ è½½é¢„æµ‹ç»“æ�œï¼ˆtop4 ç­›é€‰å��ï¼‰
pred_df = pd.read_csv("/kaggle/working/cell_level_predictions_conf_07_top4_train.csv")

# âœ… åŠ è½½çœŸå®�æ ‡ç­¾ï¼ˆä½ æ¸…æ´—è¿‡çš„ multi-hot label vectorï¼‰
true_df = pd.read_csv("/kaggle/working/bbox_with_label_vector.csv")

# âœ… æ�„å»º pred_dictï¼š[(image_id, cell_id) â†’ set(class_id)]
pred_dict = defaultdict(set)
for _, row in pred_df.iterrows():
    key = (row["image_id"], row["cell_id"])
    pred_dict[key].add(int(row["class_id"]))

# âœ… æ�„å»º true_dictï¼š[(image_id, cell_id) â†’ set(class_id)]
true_dict = defaultdict(set)
for _, row in true_df.iterrows():
    key = (row["image_id"], row["cell_id"])
    label_vec = eval(row["label_vector"])  # è½¬æ�¢ä¸º list
    for i, v in enumerate(label_vec):
        if v == 1:
            true_dict[key].add(i)

# âœ… è�šå�ˆ cell-level é¢„æµ‹å’ŒçœŸå®�
merged = defaultdict(lambda: {"true": set(), "pred": set()})
all_keys = set(pred_dict.keys()) | set(true_dict.keys())
for key in all_keys:
    merged[key]['true'] = true_dict.get(key, set())
    merged[key]['pred'] = pred_dict.get(key, set())

# âœ… è�šå�ˆåˆ° image-levelï¼Œå�ªä¿�ç•™é¢„æµ‹ä¸­å‡ºç�°çš„ image
pred_image_ids = set(pred_df['image_id'].unique())
image_summary = defaultdict(lambda: {"true": set(), "pred": set()})
for (image_id, cell_id), group in merged.items():
    if image_id in pred_image_ids:  # âœ… å�ªç»Ÿè®¡å�šè¿‡é¢„æµ‹çš„å›¾åƒ�
        image_summary[image_id]['true'].update(group['true'])
        image_summary[image_id]['pred'].update(group['pred'])

# âœ… æ‰“å�°å‰�15å¼ å›¾åƒ�çš„åŒ¹é…�ä¿¡æ�¯
print("æ¯�å¼ å›¾åƒ�çš„é¢„æµ‹åŒ¹é…�ç»Ÿè®¡ï¼ˆä»…åŒ…å�«å�šè¿‡é¢„æµ‹çš„å›¾åƒ�ï¼‰:\n")
image_match_total = 0
image_true_total = 0

for i, (image_id, group) in enumerate(image_summary.items()):
    true_set = group['true']
    pred_set = group['pred']
    matched = true_set & pred_set

    if i < 15:
        print(f"Image {i+1}: {image_id}")
        print(f"True class_id(s): {sorted(true_set)}")
        print(f"Pred class_id(s): {sorted(pred_set)}")
        print(f"Matched class_id(s): {sorted(matched)}")
        print(f"Match count: {len(matched)} / {len(true_set)}")
        print("-" * 50)

    image_match_total += len(matched)
    image_true_total += len(true_set)

# âœ… æœ€ç»ˆå‡†ç¡®ç�‡
if image_true_total > 0:
    print(f"\næŒ‰é¢„æµ‹å‡ºç�°çš„ image_id ç»Ÿè®¡çš„æ€»ä½“ Label-match Accuracy: {image_match_total / image_true_total:.4f}")
else:
    print("æ²¡æœ‰æœ‰æ•ˆçš„çœŸå®�æ ‡ç­¾ç”¨äº�åŒ¹é…�ã€‚")



import pandas as pd

# è¯»å�– CSV æ–‡ä»¶
df = pd.read_csv('/kaggle/working/cell_level_predictions_conf_07_top4_train.csv')

# è®¡ç®—å”¯ä¸€ filename æ•°é‡�
num_unique_filenames = df['filename'].nunique()
num_unique_images = df['image_id'].nunique()
print(f"å”¯ä¸€ filename æ•°é‡�ä¸º: {num_unique_filenames}")
print(f"å”¯ä¸€ image_id æ•°é‡�ä¸º: {num_unique_images}")


import pandas as pd

df = pd.read_csv('/kaggle/working/bbox_with_label_vector.csv')

unique_image_ids = df['image_id'].nunique()
print("å”¯ä¸€çš„ image_id æ•°é‡�:", unique_image_ids)



import pandas as pd
import matplotlib.pyplot as plt

# è¯»å�– CSV æ–‡ä»¶
df = pd.read_csv('/kaggle/working/cell_level_predictions_conf_07_top4_train.csv')

# æ¯�ä¸ª image_id ä¸‹çš„å”¯ä¸€ pred_class ç§�ç±»æ•°é‡�
class_counts_per_image = df.groupby('image_id')['class_id'].nunique()

# ç»Ÿè®¡ï¼šæœ‰ N ç§�ç±»çš„ image æœ‰å¤šå°‘å¼ ï¼ˆN=1, 2, ..., 10ï¼‰
count_distribution = class_counts_per_image.value_counts().sort_index()

# ç»˜å›¾
plt.figure(figsize=(8, 5))
count_distribution.plot(kind='bar')
plt.title('Number of Unique Predicted Classes per Image')
plt.xlabel('Number of Unique Predicted Classes')
plt.ylabel('Number of Images')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


import pandas as pd

# è¯»å�– CSV æ–‡ä»¶
df = pd.read_csv('/kaggle/working/cell_level_predictions_conf_07_top4_train.csv')

# æŒ‰ class_id åˆ†ç»„ï¼Œç»Ÿè®¡æ¯�ç±»ä¸­å”¯ä¸€ cell_id çš„æ•°é‡�
cell_count_per_class = df.groupby('class_id')['cell_id'].nunique().sort_index()
# æŒ‰ class_id åˆ†ç»„ï¼Œç»Ÿè®¡æ¯�ç±»å‡ºç�°åœ¨å¤šå°‘å¼ ä¸�å�Œå›¾åƒ�ä¸­ï¼ˆä»¥ filename ä½œä¸º proxyï¼‰
file_count_per_class = df.groupby('class_id')['filename'].nunique().sort_index()

# å±•ç¤ºç»“æ�œ
print("æ¯�ä¸ª class è¢«é¢„æµ‹ä¸ºæ­£çš„ cell æ•°é‡�ï¼š")
print(cell_count_per_class)
print("æ¯�ä¸ª class åœ¨å¤šå°‘å¼ ä¸�å�Œå›¾åƒ�ä¸­è¢« confident åœ°é¢„æµ‹ä¸ºæ­£ï¼š")
print(file_count_per_class)


import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# ========= å�‚æ•°å’Œæ–‡ä»¶è·¯å¾„ =========
pred_csv = "/kaggle/working/cell_level_predictions_conf_07_top4_train.csv"
true_csv = "/kaggle/working/bbox_with_label_vector.csv"

# ========= è¯»å�–æ•°æ�® =========
pred_df = pd.read_csv(pred_csv)
true_df = pd.read_csv(true_csv)

# ========= Step 1: éš�æœºé€‰æ‹©ä¸€ç»„ (image_id, cell_id) =========
sample_row = pred_df.sample(1).iloc[0]
image_id = sample_row["image_id"]
cell_id = sample_row["cell_id"]

# ========= Step 2: è�·å�–é¢„æµ‹ç±»åˆ« =========
cell_preds = pred_df[(pred_df["image_id"] == image_id) & (pred_df["cell_id"] == cell_id)]
pred_classes = sorted(cell_preds["class_id"].unique())

print(f"ğŸ“¸ Image ID: {image_id}")
print(f"ğŸ§« Cell ID: {cell_id}")
print(f"âœ… Predicted class IDs: {pred_classes}")

# ========= Step 3: è�·å�–çœŸå®�æ ‡ç­¾å�‘é‡� =========
true_row = true_df[(true_df["image_id"] == image_id) & (true_df["cell_id"] == cell_id)]
if not true_row.empty:
    label_vector = eval(true_row.iloc[0]["label_vector"])
    true_classes = [i for i, v in enumerate(label_vector) if v == 1]
    print(f"ğŸ�¯ True class IDs: {true_classes}")
else:
    true_classes = []
    label_vector = [0] * 19
    print("âš ï¸� æ— çœŸå®�æ ‡ç­¾")

# ========= Step 4: ä»� dataset ä¸­è�·å�–è¯¥æ ·æœ¬ =========
idx = dataset.df[(dataset.df["image_id"] == image_id) & (dataset.df["cell_id"] == cell_id)].index
if len(idx) == 0:
    print("â�Œ æ— æ³•åœ¨ dataset ä¸­æ‰¾åˆ°æ ·æœ¬")
else:
    img, mask, label_tensor, _, _ = dataset[idx[0]]
    img_np = img[:3].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 0.5 + 0.5).clip(0, 1)

    # ========= Step 5: æ¨¡å�‹æ�¨ç�† =========
    img_tensor = img.unsqueeze(0)[:, :3].to(DEVICE)
    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_probs = torch.sigmoid(pred_logits[0])

    # ========= Step 6: å�¯è§†åŒ– =========
    num_classes = len(pred_classes)
    if num_classes == 0:
        print("â�Œ æ— é¢„æµ‹ç»“æ�œï¼Œè·³è¿‡å�¯è§†åŒ–")
    else:
        plt.figure(figsize=(4 * (num_classes + 1), 4))

        # å�Ÿå›¾
        plt.subplot(1, num_classes + 1, 1)
        plt.imshow(img_np)
        plt.title("Cropped Cell")
        plt.axis("off")

        for idx_plot, class_id in enumerate(pred_classes):
            prob_val = pred_probs[class_id].item()
            plt.subplot(1, num_classes + 1, idx_plot + 2)
            plt.imshow(img_np)
            plt.title(f"class {class_id}\nconf={prob_val:.2f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    # ========= Step 7: åŒ¹é…�è¯„ä¼° =========
    true_vec = label_tensor.cpu().numpy().astype(int)
    pred_vec = np.zeros(19, dtype=int)
    pred_vec[pred_classes] = 1

    intersection = np.logical_and(pred_vec, true_vec).sum()
    union = np.logical_or(pred_vec, true_vec).sum()
    jaccard = intersection / union if union > 0 else 0.0
    cosine = np.dot(pred_vec, true_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(true_vec) + 1e-6)
    accuracy = (pred_vec == true_vec).sum() / 19

    print(f"\nğŸ“Š Jaccard (IoU): {jaccard:.4f}")
    print(f"ğŸ“Š Cosine similarity: {cosine:.4f}")
    print(f"ğŸ“Š Accuracy (full 19-class vector): {accuracy:.4f}")


