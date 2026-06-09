!pip install -q segmentation_models_pytorch pydicom pytorch-lightning


import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import random
import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt


# ================= é…�ç½®å�€ (Configuration) =================
CONFIG = {
    "seed": 42,
    "debug": False,           
    "img_size": 512,         
    "backbone": "resnet50",  
    "batch_size": 8,         
    "lr": 1e-4,
    "epochs": 20,             
    "num_workers": 4,  
    "device": "cuda",
    "dice_w": 0.8,
    "bce_w": 0.2,
}

preprocess_fn = smp.encoders.get_preprocessing_fn(CONFIG["backbone"], pretrained="imagenet")
# ========================================================

# å›ºå®šéš¨æ©Ÿç¨®å­�
pl.seed_everything(CONFIG['seed'])
print(f"ğŸ”§ è¨­å®šè¼‰å…¥å®Œæˆ�ã€‚ç›®å‰�æ¨¡å¼�: {'ğŸ”� å¿«é€Ÿæ¸¬è©¦ (10% Data)' if CONFIG['debug'] else 'ğŸ”¥ å…¨é‡�è¨“ç·´'}")
print(f"ğŸ–¥ï¸� ç›®æ¨™è¨­å‚™: {CONFIG['device']} | è§£æ��åº¦: {CONFIG['img_size']} | Backbone: {CONFIG['backbone']}")


def dicom_to_float32(dcm):
    img = dcm.pixel_array.astype(np.float32)

    # rescaleï¼ˆè‹¥å­˜åœ¨ï¼‰
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # MONOCHROME1 éœ€è¦�å��ç›¸ï¼ˆå¸¸è¦‹æ–¼é†«å­¸å½±åƒ�ï¼‰
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    return img

def medical_intensity_pipeline(img, clip=(1, 99), eps=1e-6):
    lo, hi = np.percentile(img, clip)
    img = np.clip(img, lo, hi)
    img = (img - img.min()) / (img.max() - img.min() + eps)
    return img.astype(np.float32)



class PneumoDataset(Dataset):
    def __init__(self, df, img_map, transform=None, img_size=512):
        self.df = df
        self.img_map = img_map
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['ImageId']
        rle = row[' EncodedPixels'] 
        
        # 1. è®€å�–è·¯å¾‘
        dcm_path = self.img_map.get(img_id)
        if dcm_path is None: 
            # print(f"Missing: {img_id}") # é�¿å…�æ´—ç‰ˆå…ˆè¨»è§£æ�‰
            return torch.zeros((3, self.img_size, self.img_size)), torch.zeros((1, self.img_size, self.img_size))

        # 2. è®€å�– DICOM
        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dicom_to_float32(dcm)                      
            img = medical_intensity_pipeline(img, (1, 99))   #é†«å­¸
        except:
            return torch.zeros((3, self.img_size, self.img_size)), torch.zeros((1, self.img_size, self.img_size))
        
        # 3. è½‰æˆ� Mask
        mask = rle2mask_safe(rle, 1024, 1024)
        
        # 4. Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # ğŸ”¥ ä¿®æ­£ Mask é‚�è¼¯ï¼šå�ªè¦�å¤§æ–¼ 0 å°±æ˜¯ 1.0 (ä¸�è¦�é™¤ä»¥ 255)
        mask = (mask > 0).astype("float32") 
        
        # 5. æ­£è¦�åŒ– (0-1) ä¸¦è½‰æˆ� 3 é€šé�“
        #img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.stack([img]*3, axis=-1).astype('float32')
        img = preprocess_fn(img)
        
        # 6. è½‰ Tensor
        img = torch.tensor(img).permute(2, 0, 1) # (3, H, W)
        mask = torch.tensor(mask).unsqueeze(0)   # (1, H, W)
        
        return img, mask

# ==========================================
# æº–å‚™è³‡æ–™èˆ‡ DataLoader
# ==========================================
DATA_DIR = "/kaggle/input/siim-acr-pneumothorax-segmentation-data"

print(f"ğŸ—‚ï¸� æ­£åœ¨æ�ƒæ��ç¡¬ç¢Ÿå»ºç«‹ç´¢å¼•...")
global_img_map = {}
all_files = glob.glob(os.path.join(DATA_DIR, '**/*.dcm'), recursive=True)
for f in all_files:
    img_id = os.path.splitext(os.path.basename(f))[0]
    global_img_map[img_id] = f
print(f"âœ… ç´¢å¼•å»ºç«‹å®Œæˆ�ï¼�å…±æ‰¾åˆ° {len(global_img_map)} å€‹æª”æ¡ˆã€‚")

TRAIN_RLE_PATH = os.path.join(DATA_DIR, "train-rle.csv")

if os.path.exists(TRAIN_RLE_PATH):
    full_df = pd.read_csv(TRAIN_RLE_PATH)
    
    # å»ºç«‹åˆ†å±¤æ¨™ç±¤ï¼šæœ‰ç—…=1, æ²’ç—…=0
    # æ³¨æ„�ï¼šå�Ÿå§‹è³‡æ–™å�¯èƒ½æœ‰é‡�è¤‡çš„ ImageId (å¤šå€‹ mask)ï¼Œé€™è£¡æˆ‘å€‘é‡�å°�æ¯�å€‹ ImageId è�šå�ˆ
    # ä½†ç‚ºäº†ç°¡åŒ–ä¸¦é…�å�ˆ dataset çµ�æ§‹ (æ¯�å€‹ row ä¸€å€‹ mask)ï¼Œæˆ‘å€‘ç›´æ�¥å°� row é€²è¡Œåˆ†å±¤
    # æ›´å¥½çš„å�šæ³•æ˜¯ group by ImageId å¾Œå†�åˆ†å±¤ï¼Œä½†é€™è£¡å…ˆä»¥ row-based å¯¦ä½œ
    full_df['has_pneumo'] = (full_df[' EncodedPixels'] != ' -1').astype(int)

    # === é—œé�µï¼šå¿«é€Ÿæ¸¬è©¦æ�¡æ¨£ ===
    if CONFIG['debug']:
        print("âœ‚ï¸� å•Ÿå‹•è�°æ˜� Debug æ¨¡å¼� (ä¿�è­‰æœ‰ç—…)...")
        # ... (Debug æ¨¡å¼�çš„æ�¡æ¨£é‚�è¼¯ä¿�æŒ�ä¸�è®Šï¼Œæˆ–å�¯ç•¥é��ä¿®æ”¹)
        positives = full_df[full_df['has_pneumo'] == 1]
        negatives = full_df[full_df['has_pneumo'] == 0]
        n_samples = 200 
        debug_pos = positives.sample(n=n_samples // 2, random_state=CONFIG['seed'], replace=True)
        debug_neg = negatives.sample(n=n_samples // 2, random_state=CONFIG['seed'], replace=True)
        full_df = pd.concat([debug_pos, debug_neg]).sample(frac=1, random_state=CONFIG['seed']).reset_index(drop=True)
        print(f"âœ… Debug è³‡æ–™æº–å‚™å®Œæˆ�ï¼�ç¸½å…± {len(full_df)} ç­†")

    # === ä¿®æ”¹æ ¸å¿ƒï¼šä½¿ç”¨ Stratified K-Fold ===
    # è¨­å®š K=5ï¼Œä¸¦å›ºå®š random_state ä»¥ç¢ºä¿�å�¯é‡�ç�¾
    N_SPLITS = 5
    FOLD_TO_USE = 0  # æˆ‘å€‘ä½¿ç”¨ç¬¬ 0 å€‹ fold ä½œç‚ºé©—è­‰é›†
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CONFIG['seed'])
    
    # æ ¹æ“š 'has_pneumo' æ¬„ä½�é€²è¡Œåˆ†å±¤
    # skf.split å›�å‚³çš„æ˜¯ index
    # æˆ‘å€‘å�ªéœ€è¦�å�–å¾—ç¬¬ FOLD_TO_USE çµ„çš„ train/val index
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df['has_pneumo'])):
        if fold == FOLD_TO_USE:
            train_df = full_df.iloc[train_idx].reset_index(drop=True)
            val_df = full_df.iloc[val_idx].reset_index(drop=True)
            break
            
    print(f"ğŸ”¥ ä½¿ç”¨ Stratified K-Fold (k={N_SPLITS}, fold={FOLD_TO_USE}) åŠƒåˆ†è³‡æ–™")
    print(f"  - è¨“ç·´é›†: {len(train_df)} ç­† (æœ‰ç—…æ¯”ä¾‹: {train_df['has_pneumo'].mean():.2%})")
    print(f"  - é©—è­‰é›†: {len(val_df)} ç­† (æœ‰ç—…æ¯”ä¾‹: {val_df['has_pneumo'].mean():.2%})")

    # å»ºç«‹ Dataset (å‚³å…¥å…¨åŸŸç´¢å¼•)
    train_ds = PneumoDataset(train_df, global_img_map, img_size=CONFIG['img_size'])
    val_ds = PneumoDataset(val_df, global_img_map, img_size=CONFIG['img_size'])
    
    # å»ºç«‹ DataLoader
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    print(f"âœ… DataLoaders æº–å‚™å®Œæˆ�ã€‚")
else:
    print("â�Œ æ‰¾ä¸�åˆ° train-rle.csvï¼Œè«‹ç¢ºèª�è·¯å¾‘")


# === rle2mask ===
def rle2mask_safe(rle, width=1024, height=1024):
    mask = np.zeros(width * height, dtype=np.uint8)
    
    # è™•ç�†ç©ºå€¼æˆ–ç„¡ Mask çš„æƒ…æ³� (-1)
    if str(rle).strip() == "-1" or (isinstance(rle, float) and np.isnan(rle)):
        return mask.reshape((height, width), order='F')
    
    # è§£æ�� RLE å­—ä¸²
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        # ä¿®æ­£é»�ï¼šä½¿ç”¨ç´¯åŠ æ–¹å¼�è¨ˆç®—ä½�ç½®
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape((height, width), order='F')

pos_df = full_df[full_df[" EncodedPixels"] != " -1"].copy()
print("æœ‰ pneumothorax çš„æ¨£æœ¬æ•¸ï¼š", len(pos_df))


class PneumoModel(pl.LightningModule):
    def __init__(self, backbone, lr, dice_w=0.5, bce_w=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.current_threshold = 0.5
        
        # ä½¿ç”¨ SMP å»ºç«‹æ¨¡å�‹
        self.model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

        #dice å’Œ bce loss
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice_w = dice_w
        self.bce_w  = bce_w

    def _loss(self, logits, mask):
        # mask: float32 0/1
        loss_dice = self.dice(logits, mask)
        loss_bce  = self.bce(logits, mask)
        return self.dice_w * loss_dice + self.bce_w * loss_bce

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        
        loss = self._loss(logits, mask)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        
        loss = self._loss(logits, mask)
        
        # è¨ˆç®— IoU æŒ‡æ¨™
        prob = torch.sigmoid(logits)
        
        # ä¿®æ­£ 2 & 3: è®Šæ•¸æ”¹æˆ� mask (å–®æ•¸) ä¸¦ä¸”å¼·åˆ¶è½‰å�‹æˆ� Tensor é˜²æ­¢å ±éŒ¯
        target_mask = (mask>0.5).long()
        target_mask = target_mask.to(prob.device)


        tp, fp, fn, tn = smp.metrics.get_stats(
            prob, 
            target_mask, # é€™è£¡å�Ÿæœ¬å¯« masks æœƒå ±éŒ¯
            mode='binary', 
            threshold=self.current_threshold
        )
        
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", iou_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5) # å»ºè­°æ”¹ç”¨ AdamW
        
        # å»ºç«‹ Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs, # æ ¹æ“šç¸½ epoch æ•¸èª¿æ•´
            eta_min=1e-6 # æœ€å°�å­¸ç¿’ç�‡
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }


# åˆ�å§‹åŒ–æ¨¡å�‹
model = PneumoModel(
    backbone=CONFIG['backbone'], 
    lr=CONFIG['lr'],
    dice_w=CONFIG['dice_w'],
    bce_w=CONFIG['bce_w']
)
csv_logger = CSVLogger("logs", name="pneumo")  

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best")

trainer = pl.Trainer(
    accelerator=CONFIG["device"],
    devices=1,
    max_epochs=CONFIG["epochs"],
    precision="16-mixed",
    logger=csv_logger,           
    callbacks=[early_stop, ckpt],
    enable_progress_bar=False,
    accumulate_grad_batches=8 # è£œå„Ÿ Batch Size
)
# é–‹å§‹è¨“ç·´ 
print("ğŸš€ é–‹å§‹æ¥µé€Ÿè¨“ç·´æ¸¬è©¦...") 
trainer.fit(model, train_loader, val_loader)
print("Best checkpoint:", ckpt.best_model_path)
model.to(CONFIG['device'])
model.eval()


# ================= è·¯å¾‘ =================
#CSV_PATH = "/kaggle/input/siimtesting/logs/pneumo/version_0/metrics.csv" 
CSV_PATH = "/kaggle/working/logs/pneumo/version_0/metrics.csv" 
# ========================================

def plot_training_curves(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"â�Œ æ‰¾ä¸�åˆ°æª”æ¡ˆ: {csv_path}")
        return

    # å�ªä¿�ç•™ epoch-level çš„ log
    df = df[df["epoch"].notna()]

    # æ¯�å€‹ epoch å�–å¹³å�‡ï¼ˆå�ªç®—æ•¸å€¼æ¬„ä½�ï¼‰
    metrics = df.groupby("epoch").mean(numeric_only=True)

    plt.figure(figsize=(12, 5))

    # --- Loss ---
    plt.subplot(1, 2, 1)
    if "train_loss" in metrics.columns:
        plt.plot(metrics.index, metrics["train_loss"], marker=".", label="Train Loss")
    if "val_loss" in metrics.columns:
        plt.plot(metrics.index, metrics["val_loss"], marker=".", label="Val Loss")

    plt.title("Loss Curve (per epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # --- Score ---
    plt.subplot(1, 2, 2)
    score_cols = [c for c in metrics.columns if "dice" in c.lower() or "iou" in c.lower()]
    for col in score_cols:
        plt.plot(metrics.index, metrics[col], marker=".", label=col)

    if score_cols:
        plt.title("Score Curve (per epoch)")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
    else:
        plt.title("No Score Metrics Found")

    plt.tight_layout()
    plt.show()


# åŸ·è¡Œç•«åœ–
plot_training_curves(CSV_PATH)


def dice_iou_from_preds(preds, targets, eps=1e-7):
    """
    preds/targets: torch.Tensor, shape (B,1,H,W), values in {0,1}
    return: dice, iou (micro over batch)
    """
    preds = preds.float()
    targets = targets.float()

    tp = (preds * targets).sum(dim=(0,2,3))
    fp = (preds * (1 - targets)).sum(dim=(0,2,3))
    fn = ((1 - preds) * targets).sum(dim=(0,2,3))

    dice = (2*tp + eps) / (2*tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)

    return dice.item(), iou.item()

@torch.no_grad()
def threshold_sweep(model, val_loader, device, thresholds=None, max_batches=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)  # 0.20~0.80
    model.eval()
    model.to(device)

    results = []
    for thr in thresholds:
        dice_list, iou_list = [], []
        for b, (x, y) in enumerate(val_loader):
            if (max_batches is not None) and (b >= max_batches):
                break

            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()  # (B,1,H,W)

            logits = model(x)
            prob = torch.sigmoid(logits)
            pred = (prob > thr).float()

            d, j = dice_iou_from_preds(pred, y)
            dice_list.append(d)
            iou_list.append(j)

        mean_dice = float(np.mean(dice_list)) if dice_list else 0.0
        mean_iou  = float(np.mean(iou_list)) if iou_list else 0.0
        results.append((thr, mean_dice, mean_iou))
        print(f"thr={thr:>4} | val_dice={mean_dice:.4f} | val_iou={mean_iou:.4f}")

    # ä»¥ Dice ç‚ºä¸»é�¸æœ€ä½³ thresholdï¼ˆä½ ä¹Ÿå�¯ä»¥æ”¹ç”¨ IoUï¼‰
    best = max(results, key=lambda x: x[1])
    best_thr, best_dice, best_iou = best
    print("\nBEST:")
    print(f"best_thr={best_thr} | best_val_dice={best_dice:.4f} | best_val_iou={best_iou:.4f}")
    return best_thr, results

best_thr, sweep_results = threshold_sweep(
    model=model,
    val_loader=val_loader,
    device=CONFIG["device"],
    thresholds=np.round(np.arange(0.3, 0.71, 0.05), 2),
    max_batches=100   # æƒ³å…ˆå¿«é€Ÿæ¸¬å°±å¡« 50 æˆ– 100
)


def remove_small_components(mask, min_area=400):
    """
    mask: 2D np.uint8, values {0,1}
    min_area: pixels threshold
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask)
    for i in range(1, num_labels):  # 0 æ˜¯èƒŒæ™¯
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out


# 1) ç¯©å‡ºæ­£æ¨£æœ¬ï¼ˆEncodedPixels != "-1"ï¼‰
pos_val = val_df[val_df[" EncodedPixels"] != " -1"]
print("Positive val samples:", len(pos_val))

if len(pos_val) == 0:
    raise ValueError("val_df è£¡æ²’æœ‰æ­£æ¨£æœ¬ï¼Œè«‹æª¢æŸ¥åˆ†å‰²æ–¹å¼�æˆ– EncodedPixels æ¸…ç�†")

# 2) æŠ½ 4 å¼µæ­£æ¨£æœ¬
sample_rows = pos_val.sample(n=min(4, len(pos_val)), random_state=42).reset_index(drop=True)

# 3) ç”¨ val_ds é€�å¼µè®€ï¼ˆé�¿å…� val_loader æŠ½åˆ°å…¨è² ï¼‰
id_to_idx = {img_id: i for i, img_id in enumerate(val_ds.df["ImageId"].tolist())}

imgs_list, masks_list = [], []
picked_ids = []

for _, r in sample_rows.iterrows():
    img_id = r["ImageId"]
    if img_id not in id_to_idx:
        continue
    x, y = val_ds[id_to_idx[img_id]]
    imgs_list.append(x)
    masks_list.append(y)
    picked_ids.append(img_id)

imgs  = torch.stack(imgs_list, dim=0)
masks = torch.stack(masks_list, dim=0)

# 4) æ�¨è«–ï¼ˆæ”¯æ�´ Lightning çš„ model æˆ–ç´” torch çš„ netï¼‰
device = CONFIG["device"] if isinstance(CONFIG["device"], str) else str(CONFIG["device"])
imgs = imgs.to(device)

model.eval()
model.to(device)

with torch.no_grad():
    logits = model(imgs)
    probs  = torch.sigmoid(logits).detach().cpu()   # (B,1,H,W)

# threshold
preds = (probs > best_thr).byte()                  # (B,1,H,W) 0/1 tensor on CPU

# connected componentsï¼šé€�å¼µå�š
preds_pp = []
for i in range(preds.shape[0]):
    m = preds[i, 0].numpy()                         # (H,W) numpy
    m = remove_small_components(m, min_area=500)
    preds_pp.append(torch.from_numpy(m).unsqueeze(0))  # (1,H,W)

preds = torch.stack(preds_pp, dim=0).float()       # (B,1,H,W)


imgs  = imgs.detach().cpu()
masks = masks.detach().cpu()

# 5) ç•«åœ–ï¼šInput / GT / Prob / Pred
plt.figure(figsize=(16, 4 * len(picked_ids)))

for i, img_id in enumerate(picked_ids):
    # Input
    plt.subplot(len(picked_ids), 4, i*4 + 1)
    plt.title(f"Input\n{img_id}")
    plt.imshow(imgs[i].permute(1,2,0))
    plt.axis("off")

    # GT
    plt.subplot(len(picked_ids), 4, i*4 + 2)
    plt.title("Ground Truth (mask)")
    plt.imshow(masks[i].squeeze(0), cmap="gray")
    plt.axis("off")

    # Prob (soft)
    plt.subplot(len(picked_ids), 4, i*4 + 3)
    plt.title("Prediction (prob)")
    plt.imshow(probs[i].squeeze(0), cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    # Pred (hard)
    plt.subplot(len(picked_ids), 4, i*4 + 4)
    plt.title("Prediction (thr=0.3)")
    plt.imshow(preds[i].squeeze(0), cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()



# ============ 1) è·¯å¾‘è¨­å®š ============
DATA_DIR = "/kaggle/input/siim-acr-pneumothorax-segmentation"
TEST_DIR = os.path.join(DATA_DIR, "stage_2_images") 
OUT_PATH = "submission.csv"

IMG_SIZE = CONFIG["img_size"]
BACKBONE = CONFIG["backbone"]
DEVICE = CONFIG["device"]
# ============ 2) å½±åƒ�è™•ç�†ï¼ˆèˆ‡ train å°�é½Šï¼‰ ============
preprocess_fn = None
try:
    preprocess_fn = smp.encoders.get_preprocessing_fn(BACKBONE, pretrained="imagenet")
except:
    preprocess_fn = None

def normalize_xray(img, clip=(1, 99), eps=1e-6):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, clip)
    img = np.clip(img, lo, hi)
    img = (img - img.mean()) / (img.std() + eps)
    img = (img - img.min()) / (img.max() - img.min() + eps)
    return img.astype(np.float32)

# ============ 3) RLE encode ============
def mask2rle(mask):
    """
    mask: 2D numpy array, 0/1
    Returns: Relative RLE string
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    # é€™è£¡å�Ÿæœ¬æ˜¯ runs[1::2] -= runs[0::2] (è¨ˆç®—é•·åº¦)
    # ä½†æˆ‘å€‘éœ€è¦�æ›´é€²ä¸€æ­¥è™•ç�† Start ä½�ç½®è½‰ç‚ºç›¸å°�ä½�ç½®
    
    starts = runs[0::2]
    lengths = runs[1::2] - runs[0::2]
    
    # å°‡ Start ä½�ç½®è½‰æ�›ç‚ºç›¸å°�ä½�ç½® (Relative Offset)
    # ç¬¬ä¸€å€‹ start ç¶­æŒ�çµ•å°�ä½�ç½®
    # å¾Œé�¢çš„ start = ç›®å‰�çµ•å°�ä½�ç½® - ä¸Šä¸€æ®µçš„çµ�æ�Ÿä½�ç½®
    
    # å»ºç«‹ä¸€å€‹æ–°é™£åˆ—ä¾†å­˜ç›¸å°� starts
    relative_starts = starts.copy()
    
    # ä¸Šä¸€æ®µçš„çµ�æ�Ÿä½�ç½® = starts[:-1] + lengths[:-1]
    prev_ends = starts[:-1] + lengths[:-1]
    
    # å¾Œé�¢çš„ start æ¸›å�» å‰�ä¸€æ®µçš„ end
    relative_starts[1:] -= prev_ends
    
    # çµ„å�ˆçµ�æ�œ: start length start length ...
    res = []
    for s, l in zip(relative_starts, lengths):
        res.extend([s, l])
        
    return " ".join(str(x) for x in res)

# ============ 4) Test Datasetï¼ˆç©©å�¥ç‰ˆï¼šå›�å‚³ h0, w0 åˆ†é–‹ï¼‰ ============
class PneumoTestDataset(Dataset):
    def __init__(self, dcm_paths, img_size=512):
        self.dcm_paths = dcm_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.dcm_paths)

    def __getitem__(self, idx):
        dcm_path = self.dcm_paths[idx]
        img_id = os.path.splitext(os.path.basename(dcm_path))[0]

        dcm = pydicom.dcmread(dcm_path)
        img = dicom_to_float32(dcm)
        h0, w0 = img.shape

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = medical_intensity_pipeline(img, (1, 99))
        img = np.stack([img]*3, axis=-1).astype(np.float32)

        if preprocess_fn is not None:
            img = preprocess_fn(img)

        x = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) 
        return img_id, x, h0, w0

# ============ 5) DataLoader ============
test_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.dcm")))
print("Test DCM:", len(test_paths))
assert len(test_paths) > 0, "TEST_DIR è£¡æ²’æœ‰ .dcmï¼Œè«‹æª¢æŸ¥è·¯å¾‘"

test_ds = PneumoTestDataset(test_paths, img_size=IMG_SIZE)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

# ============ 7) æ�¨è«– + ç”¢ submissionï¼ˆä¿®æ­£ç‰ˆï¼‰ ============
sub = []
thr = best_thr

model.to(DEVICE)
model.eval()

use_amp = (DEVICE == "cuda")

with torch.no_grad():
    for img_ids, x, h0, w0 in test_loader:
        x = x.to(DEVICE).float()  

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
        else:
            logits = model(x)

        prob = torch.sigmoid(logits).cpu().numpy()  # (B,1,H,W)

        for i in range(prob.shape[0]):
            pred = (prob[i, 0] > thr).astype(np.uint8)  # 512x512
            pred = remove_small_components(pred, min_area=500)

            H = int(h0[i].item())
            W = int(w0[i].item())
            pred_full = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

            rle = "-1" if pred_full.sum() == 0 else mask2rle(pred_full)
            sub.append({"ImageId": img_ids[i], "EncodedPixels": rle})

sub_df = pd.DataFrame(sub)
sub_df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH, "rows =", len(sub_df))
sub_df.head()


