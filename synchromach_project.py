# ğŸ“ŒÂ CellÂ 0  â€“  chá»‰ cÃ i thá»© pipeline cáº§n, KHÃ”NG Ä‘á»¥ng torch/numpy
!pip install -q --no-deps \
    iterative-stratification \
    kornia==0.7.2 \
    albumentations==1.4.3 \
    pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg \
    nibabel tqdm


import numpy, scipy, sklearn
print("NumPy :", numpy.__version__)
print("SciPy :", scipy.__version__)
print("sklearn:", sklearn.__version__)
# Ká»³ vá»�ng: 1.26.x  /  1.11.x  /  1.3.x
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
print("iterstrat OK âœ“")



# â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
# ğŸ“Œ Cellâ€¯1 â€“ Config dÃ¹ng dá»¯ liá»‡u /kaggle/input
# â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

import torch, torchvision, nibabel as nib, numpy as np, pandas as pd, random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
from tqdm.notebook import tqdm

# Ä�Æ¯á»œNG DáºªN Gá»�C Ä�Ãƒ Gáº®N (readonly â€“ khÃ´ng tá»‘n quota 20â€¯GB)
ROOT = Path("/kaggle/input/rsna-2022-cervical-spine-fracture-detection")
CSV  = ROOT/"train.csv"
IMG_DIR = ROOT/"train_images"        # chá»©a thÆ° má»¥c UID/*.dcm
MASK_DIR = ROOT/"segmentations"      # Ä‘Ã£ giáº£i nÃ©n sáºµn

# Cáº¤U HÃŒNH GIá»�NG Báº¢N COLAB
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PERCENT    = 0.25      # giá»¯ 30â€¯% study Ä‘á»ƒ train nhanh
VAL_SPLIT  = 0.15
IMG_SIZE   = 256
ROI_SIZE   = 224
BS_SEG     = 8
BS_CLS     = 16
EPOCH_SEG  = 5
EPOCH_CLS  = 10
device_name = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(device_name)

print("Device: ", DEVICE)
print("âœ…  Dataset root:", ROOT)

n_mask = len(list(MASK_DIR.glob("*.nii")))
print("ğŸŸ¢  Sá»‘ file mask:", n_mask)


#Cell 3 â€“ Chá»�n 15 % Study & PhÃ¢n táº§ng Ä‘a nhÃ£n (vá»›i mask luÃ´n vÃ o train)
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

df = pd.read_csv(CSV)
label_cols = [f"C{i}" for i in range(1,8)]
Y = df[label_cols].values
uids = df["StudyInstanceUID"].values

# Táº­p UID Ä‘Ã£ cÃ³ mask
mask_uids = {p.stem for p in MASK_DIR.glob("*.nii")}

# TÃ¡ch 2 nhÃ³m: cÃ³ mask â†´ khÃ´ng cÃ³ mask
uids_with_mask    = [uid for uid in uids if uid in mask_uids]
uids_without_mask = [uid for uid in uids if uid not in mask_uids]
Y_without_mask    = df.loc[df["StudyInstanceUID"].isin(uids_without_mask), label_cols].values

# Láº¥y 15% stratified trÃªn nhÃ³m khÃ´ng cÃ³ mask
sss15 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1-PERCENT, random_state=SEED)
idx_without, _ = next(sss15.split(uids_without_mask, Y_without_mask))
sampled_uids_without = {uids_without_mask[i] for i in idx_without}

# GhÃ©p chung: all mask_uids + sampled non-mask_uids
sub_uids = set(uids_with_mask) | sampled_uids_without
df_sub   = df[df["StudyInstanceUID"].isin(sub_uids)].reset_index(drop=True)
print(f"ğŸ’¾  Selected {len(df_sub)} / {len(df)} study "
      f"({len(uids_with_mask)} mask + {len(sampled_uids_without)} sampled)")

# Rá»“i chia tiáº¿p train/val stratified trÃªn df_sub
sss20 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_idx, val_idx = next(sss20.split(df_sub["StudyInstanceUID"], df_sub[label_cols]))
train_uids = set(df_sub.loc[train_idx, "StudyInstanceUID"])
val_uids   = set(df_sub.loc[val_idx,   "StudyInstanceUID"])
print(f"Train splits: {len(train_uids)} studies (bao gá»“m {len(mask_uids & train_uids)} mask)")

# **ThÃªm**: in ra sá»‘ lÆ°á»£ng validation vÃ  phÃ¢n phá»‘i
print(f"Validation splits: {len(val_uids)} studies (bao gá»“m {len(mask_uids & val_uids)} mask)")

# Náº¿u cáº§n xem thÃªm positives trÃªn validation thÃ¬ in tiáº¿p:
pos_val = df_sub.loc[val_idx, label_cols].sum().to_dict()
print("Positives trÃªn validation:", pos_val)



#Cell X â€“ Táº¡o DataLoader vá»›i WeightedRandomSampler Ä‘á»ƒ cÃ¢n báº±ng multi-label
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import kornia.augmentation as K
aug_cls = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.05, 0.05)),
    K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5])),
    data_keys=["input"]
)
class ClsStudyDS(Dataset):
    def __init__(self, uids, transform=None):
        self.uids = list(uids)
        self.transform = transform
        # df_sub lÃ  DataFrame con Ä‘Ã£ filter á»Ÿ Cell 3, cÃ³ cá»™t StudyInstanceUID vÃ  C1â€“C7
        self.df = df_sub.set_index("StudyInstanceUID")

    def _get_roi_sagittal(self, uid):
        # --- 1) Náº¿u cÃ³ segmentation mask: load .nii.gz ---
        nii_path = MASK_DIR / f"{uid}.nii.gz"
        if nii_path.exists():
            mask = nib.load(str(nii_path)).get_fdata().astype(np.uint8)
            mask = np.transpose(mask, (2,1,0))  # tá»« sagittal -> axial-aligned
        else:
            # --- 2) Náº¿u khÃ´ng, dÃ¹ng U-Net Ä‘á»ƒ predict mask cho toÃ n bá»™ study ---
            study_dir = IMG_DIR / uid
            slices = sorted(study_dir.glob("*.dcm"), key=lambda p: int(p.stem))
            mask = np.zeros((len(slices), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            for iz, p in enumerate(slices):
                img_arr = cv2.resize(load_dcm(p), (IMG_SIZE, IMG_SIZE))
                with torch.no_grad():
                    t = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    pred = unet(t)                     # [1, n_cls, H, W]
                    lab = pred.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                mask[iz] = lab

        # --- 3) TÃ¬m bounding box quanh táº¥t cáº£ label 1â€“7 ---
        pos = np.where((mask >= 1) & (mask <= 7))
        if len(pos[0]) == 0:
            return None

        zmin, zmax = pos[0].min(), pos[0].max()
        ymin, ymax = pos[1].min(), pos[1].max()
        xmin, xmax = pos[2].min(), pos[2].max()

        # --- 4) Táº¡o sagittal MIP view (max projection) ---
        sag = mask[:, :, xmin:xmax+1].max(axis=2)  # káº¿t quáº£ shape [Z, Y]
        sag = (sag > 0).astype(np.float32)

        # --- 5) Resize vá»� ROI_SIZE Ã— ROI_SIZE ---
        sag = cv2.resize(sag, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_NEAREST)
        return sag, zmin, zmax

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        roi_res = self._get_roi_sagittal(uid)

        if roi_res is None:
            # fallback ROI trá»‘ng
            roi = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.float32)
        else:
            roi, _, _ = roi_res

        # --- Chuyá»ƒn numpy ROI [H, W] -> torch.Tensor [C=1, H, W] ---
        img = torch.tensor(roi[np.newaxis, :, :], dtype=torch.float32)

        # --- Náº¿u cÃ³ Kornia transform, thÃªm batch dim trÆ°á»›c khi apply ---
        if self.transform:
            img = img.unsqueeze(0)            # [B=1, C=1, H, W]
            img = self.transform(img)         # Kornia tráº£ vá»� [1,1,H,W]
            img = img.squeeze(0)              # vá»� [C=1, H, W]

        # --- Láº¥y label multi-hot vector length=7 ---
        label = torch.tensor(
            self.df.loc[uid, label_cols].values.astype(np.float32),
            dtype=torch.float32
        )
        return img, label
# 1) Khá»Ÿi táº¡o dataset nhÆ° trÆ°á»›c
train_cls_ds = ClsStudyDS(train_uids, transform=aug_cls)

# 2) TÃ­nh trá»�ng sá»‘ cho má»—i lá»›p (1 / sá»‘ lÆ°á»£ng positive trong train set)
#    vÃ  gÃ¡n weight cho tá»«ng sample báº±ng trung bÃ¬nh weight cá»§a cÃ¡c label=1 cá»§a nÃ³
df_train = df_sub[df_sub["StudyInstanceUID"].isin(train_uids)].reset_index(drop=True)
label_counts = df_train[label_cols].sum().values  # shape (7,)
class_weights = 1.0 / (label_counts + 1e-6)        # trÃ¡nh chia 0

sample_weights = []
for uid in train_uids:
    y = df_train.loc[df_train["StudyInstanceUID"]==uid, label_cols].values.flatten()
    # chá»‰ láº¥y nhá»¯ng class mÃ  sample thá»±c sá»± positive
    pos_idx = np.where(y==1)[0]
    if len(pos_idx)==0:
        # vá»›i sample khÃ´ng positive nÃ o, cho weight = average cá»§a táº¥t cáº£ lá»›p
        sample_weights.append(class_weights.mean())
    else:
        sample_weights.append(class_weights[pos_idx].mean())

# 3) Táº¡o WeightedRandomSampler
sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

# 4) Táº¡o DataLoader dÃ¹ng sampler
train_cls_ld = DataLoader(
    train_cls_ds,
    batch_size=BS_CLS,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

# 5) Táº¡o val loader nhÆ° cÅ©
val_cls_ld = DataLoader(
    ClsStudyDS(val_uids, transform=K.AugmentationSequential(
        K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5])),
        data_keys=["input"]
    )),
    batch_size=BS_CLS,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"âœ”ï¸�  train sampler vá»›i {len(sample_weights)} samples, sum weights={sum(sample_weights):.2f}")



#Cell 4 â€“ Dataset Segmentation 2D (U-Net)
import albumentations as A
import cv2

print(train_uids)

def load_dcm(path):
    import pydicom, numpy as np
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    return img

class SegSliceDS(Dataset):
    def __init__(self, uids, transform=None):
        self.samples=[]
        for uid in tqdm(uids, desc="Build seg-slice list"):
            nii_path = MASK_DIR/f"{uid}.nii"
            if not nii_path.exists():
              print(f"skipping {uid}")
              continue
            print(f"continuing with {uid}")
            mask = nib.load(str(nii_path)).get_fdata().astype(np.uint8)  # sagittal
            mask = np.transpose(mask, (2,1,0))   # â‰ˆ align axial
            for z in range(mask.shape[0]):
                if mask[z].max()==0: continue
                dcm_path = IMG_DIR/uid/f"{z+1}.dcm"
                if not dcm_path.exists(): continue
                self.samples.append((dcm_path, mask[z]))
        self.transform=transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        dcm_path, m = self.samples[idx]
        img = load_dcm(dcm_path)
        m   = cv2.resize(m,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_NEAREST)
        m[m>7] = 0
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        if self.transform:
            aug = self.transform(image=img, mask=m)
            img, m = aug["image"], aug["mask"]
        img = torch.tensor(img).unsqueeze(0).float()       # [1,H,W]
        m   = torch.tensor(m).long()                       # CE Loss Ä‘a lá»›p
        return img, m

aug_seg = A.Compose([A.HorizontalFlip(p=0.5)], additional_targets={'mask':'mask'})

mask_uids = {p.stem for p in MASK_DIR.glob("*.nii")}   # set 900Â UID
train_mask_uids = train_uids & mask_uids
val_mask_uids   = val_uids   & mask_uids
print(f"âœ… train seg UID: {len(train_mask_uids)},  val seg UID: {len(val_mask_uids)}")

# ---- Dataset Segmentation 2D ----
train_seg_ds = SegSliceDS(train_mask_uids, transform=aug_seg)
val_seg_ds   = SegSliceDS(val_mask_uids,   transform=None)

# Náº¿u váº«n =0Â â†’ bá»� qua bÆ°á»›c Uâ€‘Net
if len(train_seg_ds)==0:
    print("âš ï¸�Â KhÃ´ng cÃ³ study cÃ³ maskÂ â†’ bá»� qua CellÂ 5,6 liÃªn quan segmentation")
else:
    train_seg_ld = DataLoader(train_seg_ds, batch_size=BS_SEG, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_seg_ld   = DataLoader(val_seg_ds,   batch_size=BS_SEG, shuffle=False,
                              num_workers=0, pin_memory=True)


#Cell 5 â€“ ResNetUNet
import torch.nn as nn, torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNetUNet(nn.Module):
    def __init__(self, n_cls=8):  # 0 + 1..7
        super().__init__()
        base_model = resnet18(pretrained=True)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.input_conv = nn.Sequential(
            base_model.conv1,  # 64
            base_model.bn1,
            base_model.relu,
        )
        self.pool = base_model.maxpool
        self.enc1 = base_model.layer1  # 64
        self.enc2 = base_model.layer2  # 128
        self.enc3 = base_model.layer3  # 256
        self.enc4 = base_model.layer4  # 512

        def up_blk(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.up3 = up_blk(512 + 256, 256)
        self.up2 = up_blk(256 + 128, 128)
        self.up1 = up_blk(128 + 64, 64)
        self.up0 = up_blk(64 + 64, 32)  # skip from input_conv

        self.final = nn.Conv2d(32, n_cls, kernel_size=1)
    
    def forward(self, x):
        input_size = x.shape[2:]            # Save original H, W for final upsample
    
        x0 = self.input_conv(x)            # [B,64,H/2,W/2]
        x1 = self.pool(x0)                 # [B,64,H/4,W/4]
        x2 = self.enc1(x1)                 # [B,64,H/4,W/4]
        x3 = self.enc2(x2)                 # [B,128,H/8,W/8]
        x4 = self.enc3(x3)                 # [B,256,H/16,W/16]
        x5 = self.enc4(x4)                 # [B,512,H/32,W/32]
    
        u3 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = self.up3(torch.cat([u3, x4], dim=1))
    
        u2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, x3], dim=1))
    
        u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, x2], dim=1))
    
        u0 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u0 = self.up0(torch.cat([u0, x0], dim=1))
    
        out = self.final(u0)              # [B, n_cls, H/2, W/2]
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)  # â¬… Fix here
        return out


unet = ResNetUNet().to(DEVICE)
opt_seg = torch.optim.Adam(unet.parameters(),1e-3)
ce = nn.CrossEntropyLoss()

def run_seg_epoch(loader, training=True):
    unet.train(training)
    tot, n = 0, 0
    for img, m in tqdm(loader, leave=False):
        img, m = img.to(DEVICE), m.to(DEVICE)
        pred = unet(img)
        loss = ce(pred, m)
        if training:
            opt_seg.zero_grad(); loss.backward(); opt_seg.step()
        tot += loss.item() * img.size(0)
        n += img.size(0)
    return tot / n

for ep in range(1, EPOCH_SEG+1):
    tr = run_seg_epoch(train_seg_ld, True)
    # Bá»� qua validation Ä‘á»ƒ train nhanh hÆ¡n 1 tÃ­.
    # vl = run_seg_epoch(val_seg_ld, False)

torch.save(unet.state_dict(), "/kaggle/working/unet_cervical.pth")


#Cell 6 â€“ Táº¡o ROI & Dataset PhÃ¢n loáº¡i (vá»›i load U-Net checkpoint)
import torch
import torch.nn as nn
import numpy as np
import cv2
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset

# â”€â”€ 0) Load U-Net checkpoint trÆ°á»›c khi inference ROI â”€â”€
checkpoint_path = "/kaggle/working/unet_cervical.pth"
if Path(checkpoint_path).exists():
    unet.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    unet.to(DEVICE).eval()
    print(f"âœ… Loaded U-Net weights from {checkpoint_path}")
else:
    print(f"âš ï¸�  KhÃ´ng tÃ¬m tháº¥y checkpoint U-Net táº¡i {checkpoint_path}, dÃ¹ng weights máº·c Ä‘á»‹nh")

# augmentation classification (Kornia)

# Táº¡o Dataset & DataLoader
train_cls_ds = ClsStudyDS(train_uids, transform=aug_cls)
val_cls_ds   = ClsStudyDS(val_uids, transform=K.AugmentationSequential(
    K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5])),
    data_keys=["input"]
))

from torch.utils.data import DataLoader
train_cls_ld = DataLoader(train_cls_ds, batch_size=BS_CLS, shuffle=True,  num_workers=0, pin_memory=True)
val_cls_ld   = DataLoader(val_cls_ds,   batch_size=BS_CLS, shuffle=False, num_workers=0, pin_memory=True)



# === Cell 7 â€“ ResNet-18 + Weighted BCE + Metric fix ===
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm.auto import tqdm

# 1) Build model
resnet = models.resnet18(weights=None)
resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
resnet.fc    = nn.Linear(resnet.fc.in_features, 7)
resnet = resnet.to(DEVICE)

# 2) TÃ­nh pos_weight tá»« df_sub (táº­p con sau Cell 3)
#    pos = sá»‘ samples positive cho má»—i class, neg = tá»•ng â€“ pos
pos = df_sub[label_cols].sum().values
neg = len(df_sub) - pos
pos_weight = torch.tensor(neg/pos, dtype=torch.float32, device=DEVICE)

# 3) Loss + Optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4)

# 4) HÃ m train/val má»—i epoch
def run_cls_epoch(dataloader, train=True):
    resnet.train(train)
    total_loss = 0.0
    correct = 0
    total_labels = 0
    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(dataloader, desc="train" if train else "val"):
            imgs   = imgs.to(DEVICE).float()
            labels = labels.to(DEVICE)
            logits = resnet(imgs)                     # [B,7]
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)

            # --- TÃ­nh metric: dÃ¹ng sigmoid + threshold=0.5 ---
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total_labels += labels.numel()

    avg_loss = total_loss / len(dataloader.dataset)
    acc      = correct / total_labels
    return avg_loss, acc

# 5) VÃ²ng huáº¥n luyá»‡n vá»›i Early Stopping gá»£i Ã½
best_val_loss = float("inf")
patience = 2   # dá»«ng náº¿u Val loss khÃ´ng giáº£m sau 2 epoch
wait = 0

for ep in range(1, EPOCH_CLS+1):
    tr_loss, tr_acc = run_cls_epoch(train_cls_ld, True)
    # Bá»� qua validation Ä‘á»ƒ train nhanh hÆ¡n 1 tÃ­; hardware cá»§a Kaggle khÃ¡ cháº­m.
    # vl_loss, vl_acc = run_cls_epoch(val_cls_ld,   False)
    # print(f"Ep{ep:02d}  TL={tr_loss:.3f}  VL={vl_loss:.3f}  "
    #      f"Tacc={tr_acc:.4f}  Vacc={vl_acc:.4f}")
    print(f"Ep{ep:02d}  TL={tr_loss:.3f}  Tacc={tr_acc:.4f}")

    # Save best
    # if vl_loss < best_val_loss:
    #     best_val_loss = vl_loss
    #     torch.save(resnet.state_dict(), "/kaggle/working/resnet18_best.pth")
    #     wait = 0
    # else:
    #     wait += 1
    #     print(f"  âš ï¸� Val loss â†‘  (wait {wait}/{patience})")
    #     if wait >= patience:
    #         print("ğŸš¨ Early stopping!")
    #         break

    # LÆ°u tá»«ng bÆ°á»›c Ä‘á»ƒ cÃ³ thá»ƒ backup
    torch.save(resnet.state_dict(), "/kaggle/working/resnet18_best.pth")

