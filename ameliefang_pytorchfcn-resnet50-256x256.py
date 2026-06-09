import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc


# !pip install "numpy<2.0"


!pip install albumentations


!pip install git+https://github.com/qubvel/segmentation_models.pytorch


import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
# from albumentations import *
# import albumentations as A
# import rasterio
# from rasterio.windows import Window
# from albumentations import *
# from albumentations.pytorch import ToTensor
# from albumentations.pytorch import ToTensorV2 as ToTensor
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
# from sklearn.model_selection import KFold
import tifffile as tiff
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, sampler
from tqdm import tqdm_notebook as tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
from torchvision import transforms as T


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    
set_seed()


# def seed_everything(seed=2**3):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
# seed_everything(121)

# fold = 0
# nfolds = 5
reduce = 4
sz = 256

BATCH_SIZE = 16
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
NUM_EPOCHS = 5
SEED = 2020
TH = 0.39

DEVICE = 'cuda' if torch.cuda.is_available() else'cpu'
DATA = '../input/hubmap-kidney-segmentation/test/'
LABELS = '../input/hubmap-kidney-segmentation/train.csv'
MASKS = '../input/hubmap-256x256/masks'
TRAIN = '../input/hubmap-256x256/train'
df_sample = pd.read_csv('../input/hubmap-kidney-segmentation/sample_submission.csv')


def rle_encode_less_memory(img):
    pixels=img.T.flatten()
    pixels[0]=0
    pixels[-1]=0
    runs = np.where(pixels[1:] != pixels[:-1])[0]+2
    runs[1::2]-=runs[::2]
    return ' '.join(str(x) for x in runs)


mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    # è¿™é‡Œçš„ ids å�‚æ•°ï¼šæ˜¯ä½ åœ¨è¿™ä¸ª Dataset é‡Œæƒ³è¦�åŒ…å�«çš„é‚£äº›ç—…äºº/å¤§å›¾ ID çš„åˆ—è¡¨
    def __init__(self, ids, tfms=None):
        self.ids = ids
        self.tfms = tfms
        
        # æ ¸å¿ƒé€»è¾‘ï¼šå�ªåŠ è½½æ–‡ä»¶å��ä¸­åŒ…å�«æŒ‡å®š ID çš„å›¾ç‰‡
        # å�‡è®¾ TRAIN æ˜¯ä½ çš„å›¾ç‰‡æ–‡ä»¶å¤¹è·¯å¾„
        self.fnames = [
            fname for fname in os.listdir(TRAIN) 
            if fname.split('_')[0] in self.ids
        ]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        
        # è¯»å�–å›¾ç‰‡å’Œ Mask (ä¿�æŒ�ä½ å�Ÿæ�¥çš„é€»è¾‘)
        imgs = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, fname)), cv2.COLOR_BGR2RGB)
        masks = cv2.imread(os.path.join(MASKS, fname), cv2.IMREAD_GRAYSCALE)
        
        if self.tfms is not None:
            augmented = self.tfms(image=imgs, mask=masks)
            imgs, masks = augmented['image'], augmented['mask']
            
        # è¿™é‡Œçš„ img2tensor, mean, std åº”è¯¥æ˜¯ä½ åœ¨å¤–éƒ¨å®šä¹‰çš„å…¨å±€å�˜é‡�æˆ–å‡½æ•°ï¼Œä¿�æŒ�ä¸�å�˜
        return img2tensor((imgs / 255.0 - mean) / std), img2tensor(masks)

from albumentations import *

def get_augmentation(p=1.0):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, border_mode=cv2.BORDER_REFLECT),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            # IAAPiecewiseAffine(p=0.3),
            PiecewiseAffine(p=0.3, scale=(0.03, 0.05)),
        ], p=0.3),
        OneOf([
            HueSaturationValue(10, 15, 10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),
        ], p=0.3),
    ], p=p)



import os
import random
import numpy as np
# ç¡®ä¿�å¯¼å…¥äº† Dataset éœ€è¦�çš„åº“ï¼Œå¦‚ torch, cv2, Dataset ç­‰

# --- ç¬¬ä¸€æ­¥ï¼šè�·å�–æ‰€æœ‰å”¯ä¸€çš„ ID ---
# è¿™é‡Œç›´æ�¥æ‰«æ��æ–‡ä»¶å¤¹æ–‡ä»¶å��æ�¥è�·å�– IDï¼Œæ¯”è¯» CSV æ›´ç¨³ï¼Œå› ä¸ºèƒ½ç¡®ä¿�æ–‡ä»¶å­˜åœ¨
all_files = os.listdir(TRAIN)
all_ids = set([f.split('_')[0] for f in all_files]) # æ��å�– ID å¹¶å�»é‡�
unique_ids = list(all_ids)

# --- ç¬¬äºŒæ­¥ï¼šæŒ‰ 7:2:1 åˆ’åˆ† ID ---
random.seed(42) # å›ºå®šéš�æœºç§�å­�
random.shuffle(unique_ids)

n_total = len(unique_ids)
n_train = int(n_total * 0.7)
n_valid = int(n_total * 0.2)

train_ids = unique_ids[:n_train]
valid_ids = unique_ids[n_train : n_train + n_valid]
test_ids  = unique_ids[n_train + n_valid:]

print(f"ID åˆ’åˆ†æƒ…å†µ -> Train: {len(train_ids)}, Valid: {len(valid_ids)}, Test: {len(test_ids)}")

# --- ç¬¬ä¸‰æ­¥ï¼šå®�ä¾‹åŒ–ä¸‰ä¸ª Dataset ---
# æ³¨æ„�ï¼štrain_ds ä¼ å…¥å¢�å¼º (get_augmentation)ï¼Œvalid å’Œ test ä¸�ä¼  (None)

# è®­ç»ƒé›†ï¼šä¼ å…¥ 70% çš„ IDï¼Œå¼€å�¯å¢�å¼º
train_ds = HuBMAPDataset(ids=train_ids, tfms=get_augmentation())

# éªŒè¯�é›†ï¼šä¼ å…¥ 20% çš„ IDï¼Œæ— å¢�å¼º
valid_ds = HuBMAPDataset(ids=valid_ids, tfms=None)

# æµ‹è¯•é›†ï¼šä¼ å…¥ 10% çš„ IDï¼Œæ— å¢�å¼º
test_ds  = HuBMAPDataset(ids=test_ids, tfms=None)

# --- ç¬¬å››æ­¥ï¼šåˆ›å»º DataLoader ---
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"æœ€ç»ˆåˆ‡ç‰‡(Slice)æ•°é‡� -> Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")


len(valid_loader), len(train_loader),len(test_loader)


imgs, masks = next(iter(test_loader))
plt.figure(figsize=(16, 16))
for i, (img, mask) in enumerate(zip(imgs, masks)):
    img = ((img.permute(1, 2, 0)*std + mean) * 255.0).numpy().astype(np.uint8)
    plt.subplot(4, 4, i+1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.imshow(mask.squeeze().numpy(), alpha=0.6)
    plt.axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)
plt.show()


# def get_model():
#     model = torchvision.models.segmentation.fcn_resnet50(True)
#     model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
#     model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
#     return model

def get_model(num_classes=1):
    model = torchvision.models.segmentation.fcn_resnet50(weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1")
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    return model
    


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):

        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc


@torch.no_grad()
def valid_test(model, loader, loss_fn):
    model.eval()
    losses = []
    dices = []
    ious = []

    with torch.no_grad():
        for image, target in loader:
            image = image.to(DEVICE)
            target = target.float().to(DEVICE)

            output = model(image)['out']
            loss = loss_fn(output, target)
            losses.append(loss.item())

            # sigmoid å��å†�ç®—å‡†ç¡®ç�‡
            probs = output.sigmoid()

            # dice score
            y_pred = (probs > 0.5).float()
            intersection = (y_pred * target).sum()
            union = y_pred.sum() + target.sum()
            dice = (2 * intersection + 1e-7) / (union + 1e-7)
            dices.append(dice.item())

            # iou score
            union_iou = y_pred.sum() + target.sum() - intersection
            iou = (intersection + 1e-7) / (union_iou + 1e-7)
            ious.append(iou.item())

    # è¿”å›�ä¸‰ä¸ªå€¼ !!!
    return np.mean(losses), np.mean(dices), np.mean(ious)



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# --------------------------
# Loss functions
# --------------------------
bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()

def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8*bce + 0.2*dice


# --------------------------
# å®�éªŒé…�ç½®
# --------------------------
optimizers = {
    "AdamW": torch.optim.AdamW,
    # "SGD": torch.optim.SGD
}
learning_rates = [1e-4]
#[1e-3, 1e-4, 5e-5]
# learning_rates = [0.1, 0.05, 0.01]

EPOCHES = 5

# è®°å½•ç»“æ�œ
results = {}   # { "AdamW_lr1e-4": { "train":[], "val":[], "dice":[], ... } }


# --------------------------
# â­• ä¸»å¾ªç�¯ï¼šä¸�å�Œ Optimizer Ã— LR
# --------------------------
for opt_name, opt_class in optimizers.items():
    for lr in learning_rates:

        tag = f"{opt_name}_lr{lr}"
        print("\n" + "="*70)
        print(f"â–¶ Running experiment: {tag}")
        print("="*70)

        # è®°å½•
        results[tag] = {"train": [], "val": [], "dice": [], "iou": []}

        # åˆ�å§‹åŒ–æ¨¡å�‹ & ä¼˜åŒ–å™¨
        model = get_model().to(DEVICE)
        optimizer = opt_class(
            model.parameters(),
            lr=lr,
            weight_decay=1e-3 if opt_name=="AdamW" else 0
        )

        best_loss = 999

        # --------------------------
        # ğŸ”¥ è®­ç»ƒå¾ªç�¯
        # --------------------------
        for epoch in range(1, EPOCHES+1):
            model.train()
            train_losses = []
            start = time.time()

            for image, target in train_loader:
                image = image.to(DEVICE)
                target = target.float().to(DEVICE)

                optimizer.zero_grad()

                output = model(image)['out']
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # ğŸ”¥ éªŒè¯�
            vloss, vdice, viou = valid_test(model, valid_loader, loss_fn)

            # ä¿�å­˜
            results[tag]["train"].append(np.mean(train_losses))
            results[tag]["val"].append(vloss)
            results[tag]["dice"].append(vdice)
            results[tag]["iou"].append(viou)

            print(f"Epoch {epoch:02d} | "
                  f"Train {np.mean(train_losses):.4f} | "
                  f"Val {vloss:.4f} | "
                  f"Dice {vdice:.4f} | "
                  f"IoU  {viou:.4f} | "
                  f"{(time.time()-start)/60:.2f} min")

            # ğŸ”¥ ä¿�å­˜æœ€ä½³æ¨¡å�‹
            if vloss < best_loss:
                best_loss = vloss
                torch.save(model.state_dict(), f"best_{tag}.pth")



import openpyxl

def save_results_to_excel(results, filename="training_results.xlsx"):
    wb = openpyxl.Workbook()

    # åˆ é™¤é»˜è®¤Sheet
    default_sheet = wb.active
    wb.remove(default_sheet)

    for tag, metrics in results.items():
        ws = wb.create_sheet(title=tag[:30])  # Excel sheetå��æœ€å¤š31å­—ç¬¦

        # å†™è¡¨å¤´
        ws.append(["Epoch", "Train Loss", "Val Loss", "Dice", "IoU"])

        # å�‡è®¾æ¯�é¡¹éƒ½æ˜¯é•¿åº¦ EPOCHES çš„ list
        E = len(metrics["train"])

        for i in range(E):
            ws.append([
                i+1,
                metrics["train"][i],
                metrics["val"][i],
                metrics["dice"][i],
                metrics["iou"][i]
            ])

    wb.save(filename)
    print(f"âœ” Results saved to {filename}")

save_results_to_excel(results, "optimizer_SGD_lr_comparison.xlsx")




plt.figure(figsize=(16, 4))

# --------------------------
# Loss
# --------------------------
plt.subplot(1, 3, 1)
for tag in results:
    plt.plot(results[tag]["train"], label=f"{tag}_train")
    plt.plot(results[tag]["val"], label=f"{tag}_val", linestyle="--")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# --------------------------
# Dice
# --------------------------
plt.subplot(1, 3, 2)
for tag in results:
    plt.plot(results[tag]["dice"], label=tag)
plt.title("Validation Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.legend()

# --------------------------
# IoU
# --------------------------
plt.subplot(1, 3, 3)
for tag in results:
    plt.plot(results[tag]["iou"], label=tag)
plt.title("Validation IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.legend()

plt.tight_layout()
plt.show()




print(f"\nğŸš€ Start Test Set: best_{tag}.pth ...")

# 1.load model
best_model = get_model().to(DEVICE)

# 2. load best weight
weight_path = f"best_{tag}.pth"
best_model.load_state_dict(torch.load(weight_path))
best_model.eval() 

test_loss, test_dice, test_iou = valid_test(best_model, test_loader, loss_fn)

print("\n" + "="*40)
print(f"ğŸ�‰ (Final Test Results)")
print("="*40)
print(f"Best Config: {tag}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Dice: {test_dice:.4f}")
print(f"Test IoU : {test_iou:.4f}")
print("="*40)


import random
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# ğŸ�¨ å�¯è§†åŒ–å�•å¼ æµ‹è¯•ç»“æ�œ (Auto-Compatible)
# ==========================================
def plot_random_test_result(model, loader, device):
    model.eval()
    
    # 1. ä»� dataset ä¸­éš�æœºæŠ½å�–ä¸€å¼ 
    # loader.dataset æ”¯æŒ�ç›´æ�¥ç´¢å¼•è®¿é—®
    dataset = loader.dataset
    total_idx = len(dataset)
    random_idx = 518
    # random.randint(0, total_idx - 1)
    
    print(f"\nğŸ�¨ æ­£åœ¨ç»˜åˆ¶ç¬¬ [{random_idx}/{total_idx}] å¼ æµ‹è¯•æ ·æœ¬çš„å�¯è§†åŒ–ç»“æ�œ...")
    
    # 2. è�·å�–æ•°æ�® (image: 3xHxW, mask: HxW)
    image, mask = dataset[random_idx]
    
    # å¢�åŠ  Batch ç»´åº¦ (1, 3, H, W) ä»¥é€�å…¥æ¨¡å�‹
    image_input = image.unsqueeze(0).to(device)
    
    # 3. æ¨¡å�‹é¢„æµ‹
    with torch.no_grad():
        output = model(image_input)
        
        # --- å…¼å®¹æ€§å¤„ç�† (è¿™æ˜¯å…³é”®) ---
        # æƒ…å†µA: Torchvisionæ¨¡å�‹ (è¿”å›�å­—å…¸ {'out': ...})
        if isinstance(output, dict) and 'out' in output:
            output = output['out']
        # æƒ…å†µB: TransUNet (å�¯èƒ½è¿”å›�åˆ—è¡¨/å…ƒç»„)
        elif isinstance(output, (tuple, list)):
            output = output[0]
        # æƒ…å†µC: SMP U-Net (ç›´æ�¥è¿”å›� Tensor)
        # ä¸�éœ€è¦�å�šé¢�å¤–å¤„ç�†
        
        prob = output.sigmoid()
        pred = (prob > 0.5).float()

    # 4. æ•°æ�®è½¬æ�¢ (Tensor -> Numpy) ç”¨äº�ç»˜å›¾
    # å�Ÿå›¾å��å½’ä¸€åŒ– (ç®€å�• Min-Max å½’ä¸€åŒ–ï¼Œä¿�è¯�æ˜¾ç¤ºæ­£å¸¸)
    img_np = image.permute(1, 2, 0).numpy()
    img_show = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    mask_np = mask.squeeze().numpy()
    pred_np = pred.squeeze().cpu().numpy()
    
    # 5. ç»˜å›¾ (1è¡Œ3åˆ—)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- å›¾1: å�Ÿå›¾ ---
    axes[0].imshow(img_show)
    axes[0].set_title(f"Image Index: {random_idx}", fontsize=12, color='navy')
    axes[0].axis('off')

    # --- å›¾2: çœŸå€¼ (Ground Truth) ---
    axes[1].imshow(img_show)
    # èƒŒæ™¯é€�æ˜�åŒ–ï¼Œå�ªæ˜¾ç¤ºå‰�æ™¯
    masked_gt = np.ma.masked_where(mask_np == 0, mask_np)
    axes[1].imshow(masked_gt, alpha=0.7, cmap='spring') # äº®ç²‰è‰²è¡¨ç¤ºçœŸå€¼
    axes[1].set_title("Ground Truth (Spring)", fontsize=12)
    axes[1].axis('off')

    # --- å›¾3: é¢„æµ‹ (Prediction) ---
    axes[2].imshow(img_show)
    masked_pred = np.ma.masked_where(pred_np == 0, pred_np)
    
    # é¡ºä¾¿ç®—ä¸€ä¸‹è¿™å¼ å›¾çš„ Dice ç»™ä½ å�‚è€ƒ
    dice_score = (2 * (pred_np * mask_np).sum()) / (pred_np.sum() + mask_np.sum() + 1e-7)
    
    axes[2].imshow(masked_pred, alpha=0.7, cmap='jet') # å½©è™¹è‰²è¡¨ç¤ºé¢„æµ‹
    axes[2].set_title(f"Prediction | Dice: {dice_score:.2f}", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# â–¶ï¸� æ‰§è¡Œç»˜å›¾
# ==========================================
# ç›´æ�¥ä¼ å…¥ best_model å’Œ test_loader å�³å�¯
plot_random_test_result(best_model, test_loader, DEVICE)


@torch.no_grad()
def validation(model, loader, loss_fn):
    model.eval()
    losses = []
    dices = []
    ious = []

    for image, target in loader:
        image = image.to(DEVICE)
        target = target.float().to(DEVICE)

        # -------------------------------------------------
        # â�Œ é”™è¯¯å†™æ³• (é’ˆå¯¹æ—§æ¨¡å�‹): output = model(image)['out']
        # âœ… æ­£ç¡®å†™æ³• (é’ˆå¯¹ SMP ): output = model(image)
        # -------------------------------------------------
        output = model(image)
        
        # ä¸ºäº†å…¼å®¹æ€§ï¼Œä¹Ÿå�¯ä»¥å†™æˆ�è¿™æ ·é˜²å®ˆå�‹ä»£ç �ï¼š
        # if isinstance(output, dict):
        #     output = output['out']

        loss = loss_fn(output, target)
        losses.append(loss.item())

        # sigmoid å��å†�ç®—å‡†ç¡®ç�‡
        probs = output.sigmoid()

        # dice score
        y_pred = (probs > 0.5).float()
        intersection = (y_pred * target).sum()
        union = y_pred.sum() + target.sum()
        dice = (2 * intersection + 1e-7) / (union + 1e-7)
        dices.append(dice.item())

        # iou score
        union_iou = y_pred.sum() + target.sum() - intersection
        iou = (intersection + 1e-7) / (union_iou + 1e-7)
        ious.append(iou.item())

    return np.mean(losses), np.mean(dices), np.mean(ious)


import torch
import torch.nn as nn
import time
import numpy as np
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================
# 1. å‡†å¤‡æ•°æ�® (å¤�ç”¨ä¹‹å‰�çš„ 7:2:1 é€»è¾‘)
# ============================
# å�‡è®¾ train_loader, valid_loader, test_loader å·²ç»�åœ¨ä¸Šä¸€æ­¥åˆ›å»ºå¥½äº†
# å¦‚æ�œæ²¡æœ‰ï¼Œè¯·å…ˆè¿�è¡Œä¸Šä¸€æ­¥ç”Ÿæˆ� Loader çš„ä»£ç �

# ============================
# 2. å®šä¹‰æ¨¡å�‹ (SMP U-Net + Attention)
# ============================
model = smp.Unet(
    encoder_name="mobilenet_v2",    # è½»é‡�çº§ encoderï¼Œé€‚å�ˆå¿«é€Ÿå®�éªŒ
    encoder_weights="imagenet",     # ä½¿ç”¨é¢„è®­ç»ƒæ�ƒé‡�åŠ é€Ÿæ”¶æ•›
    in_channels=3,
    classes=1,
    # --- è¿›é˜¶å�‚æ•°: scSE æ³¨æ„�åŠ›æœºåˆ¶ ---
    decoder_attention_type='scse',
)
model.to(DEVICE)

# ============================
# 3. å®šä¹‰ Loss, ä¼˜åŒ–å™¨, Scheduler
# ============================
# æ··å�ˆ Loss: BCE + Dice
bce_fn = nn.BCEWithLogitsLoss()
dice_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

def loss_fn(y_pred, y_true):
    return 0.5 * bce_fn(y_pred, y_true) + 0.5 * dice_fn(y_pred, y_true)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

# ç›‘æ�§ val_lossï¼Œå¦‚æ�œ 5 ä¸ª epoch ä¸�ä¸‹é™�ï¼Œå­¦ä¹ ç�‡å‡�å�Š
lr_step = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# ============================
# 4. è¾…åŠ©å‡½æ•°: å�•ä¸ª Epoch è®­ç»ƒé€»è¾‘
# ============================
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        
        optimizer.zero_grad()
        output = model(image) # SMP ç›´æ�¥è¿”å›� Tensor
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(loader)

# ============================
# 5. ä¸»è®­ç»ƒå¾ªç�¯
# ============================
EPOCHES = 5
best_dice = 0.0
save_path = "best_model_unet_scse.pth"

# è¡¨å¤´æ ¼å¼�åŒ–
header = r'''
Epoch | Train Loss |  Val Loss  |  Val Dice  |  Val IoU   |  Time (m)
'''
print(header)
# æ ¼å¼�è¯´æ˜�: {:6d}æ•´æ•° | {:10.4f}æµ®ç‚¹æ•° ...
raw_line = '{:6d} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:8.2f}'

for epoch in range(1, EPOCHES + 1):
    start_time = time.time()
    
    # --- è®­ç»ƒ ---
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
    
    # --- éªŒè¯� (å¤�ç”¨ä¹‹å‰�çš„ validation å‡½æ•°) ---
    val_loss, val_dice, val_iou = validation(model, valid_loader, loss_fn) # æ³¨æ„�è¿™é‡Œç”¨éªŒè¯�é›† vloader
    
    # --- æ›´æ–°å­¦ä¹ ç�‡ ---
    # ReduceLROnPlateau éœ€è¦�ä¸€ä¸ªæŒ‡æ ‡ï¼Œè¿™é‡Œæˆ‘ä»¬ç›‘æ�§ val_loss
    lr_step.step(val_loss)

    # --- ä¿�å­˜æœ€ä½³æ¨¡å�‹ ---
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), save_path)
        save_msg = "--> Saved"
    else:
        save_msg = ""

    # --- æ‰“å�°æ—¥å¿— ---
    duration = (time.time() - start_time) / 60
    print(raw_line.format(epoch, train_loss, val_loss, val_dice, val_iou, duration) + save_msg)

print(f"\nè®­ç»ƒç»“æ�Ÿï¼�æœ€ä½³æ¨¡å�‹å·²ä¿�å­˜è‡³: {save_path}, æœ€ä½³ Dice: {best_dice:.4f}")


import torch
import segmentation_models_pytorch as smp
import numpy as np

# ============================
# 1. é‡�æ–°å®šä¹‰æ¨¡å�‹ç»“æ�„
# ============================
# å¿…é¡»ä¸�è®­ç»ƒæ—¶çš„é…�ç½®å®Œå…¨ä¸€è‡´ (encoder, attention ç­‰)
best_model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,           # æµ‹è¯•æ—¶ä¸�éœ€è¦�ä¸‹è½½é¢„è®­ç»ƒæ�ƒé‡�ï¼Œå› ä¸ºæˆ‘ä»¬è¦�åŠ è½½è‡ªå·±çš„
    in_channels=3,
    classes=1,
    decoder_attention_type='scse',  # åˆ«å¿˜äº†è¿™ä¸ªï¼�
)

# ============================
# 2. åŠ è½½ä½ ä¿�å­˜çš„æ�ƒé‡�
# ============================
weight_path = "best_model_unet_scse.pth"
# map_locationç¡®ä¿�å�³ä½¿åœ¨å�ªæœ‰CPUçš„æœºå™¨ä¸Šä¹Ÿèƒ½åŠ è½½
best_model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

# è½¬ç§»åˆ°è®¾å¤‡å¹¶å¼€å�¯è¯„ä¼°æ¨¡å¼�
best_model.to(DEVICE)
best_model.eval() 

print(f"âœ… å·²æˆ�åŠŸåŠ è½½æ¨¡å�‹: {weight_path}")

# ============================
# 3. åœ¨æµ‹è¯•é›†ä¸Šè·‘åˆ†
# ============================
# ä½¿ç”¨ä½ ä¹‹å‰�å®šä¹‰å¥½çš„ validation å‡½æ•° (ç¡®ä¿�æ˜¯å�»æ�‰äº† ['out'] çš„é‚£ä¸ªç‰ˆæœ¬)
test_loss, test_dice, test_iou = validation(best_model, test_loader, loss_fn)

print("\n" + "="*50)
print(f"ğŸ�† æœ€ç»ˆæµ‹è¯•é›†æˆ�ç»© (Final Test Results)")
print("="*50)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Dice : {test_dice:.4f}")
print(f"Test IoU  : {test_iou:.4f}")
print("="*50)


import random
import matplotlib.pyplot as plt
import numpy as np

def plot_random_one(model, dataset, device):
    """
    éš�æœºä»�æ•°æ�®é›†ä¸­æŠ½å�–ä¸€å¼ å¹¶æ˜¾ç¤º
    """
    model.eval()
    
    # 1. éš�æœºç”Ÿæˆ�ä¸€ä¸ªåº�å�· (Index)
    total_idx = len(dataset)
    random_idx = 238
    # random.randint(0, total_idx - 1)
    
    print(f"ğŸ�² æ­£åœ¨éš�æœºæŠ½å�–ç¬¬ [{random_idx}/{total_idx}] å¼ å›¾ç‰‡è¿›è¡Œæµ‹è¯•...")
    
    # 2. è�·å�–å�•å¼ æ•°æ�® (æ³¨æ„�ï¼šè¿™é‡Œæ²¡æœ‰ Batch ç»´åº¦)
    image, mask = dataset[random_idx] 
    # image shape: (3, 256, 256)
    # mask shape:  (1, 256, 256) æˆ– (256, 256)

    # 3. å¢�åŠ  Batch ç»´åº¦: (3, H, W) -> (1, 3, H, W)
    image_input = image.unsqueeze(0).to(device)
    
    # 4. æ¨¡å�‹é¢„æµ‹
    with torch.no_grad():
        output = model(image_input)       # SMP ç›´æ�¥è¿”å›� Tensor
        prob = output.sigmoid()           # è½¬æ¦‚ç�‡
        pred = (prob > 0.5).float()       # è½¬ 0/1

    # 5. æ•°æ�®è½¬æ�¢ç”¨äº�ç”»å›¾ (Tensor -> Numpy)
    # --- å�Ÿå›¾ ---
    img_np = image.permute(1, 2, 0).numpy()
    # ç®€å�•çš„ Min-Max å��å½’ä¸€åŒ–ï¼Œç¡®ä¿�èƒ½çœ‹æ¸…
    img_show = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # --- Mask ---
    mask_np = mask.squeeze().numpy()
    pred_np = pred.squeeze().cpu().numpy()
    
    # 6. å¼€å§‹ç”»å›¾ (1è¡Œ3åˆ—)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- å›¾1: å�Ÿå›¾ ---
    axes[0].imshow(img_show)
    axes[0].set_title(f"Image Index: {random_idx}", fontsize=14, color='blue')
    axes[0].axis('off')

    # --- å›¾2: çœŸå€¼ (Ground Truth) ---
    axes[1].imshow(img_show)
    # éš�è—�èƒŒæ™¯(0)ï¼Œå�ªæ˜¾ç¤ºå‰�æ™¯
    masked_gt = np.ma.masked_where(mask_np == 0, mask_np)
    axes[1].imshow(masked_gt, alpha=0.7, cmap='spring') # spring æ˜¯äº®ç²‰/ç»¿è‰²ï¼Œå¯¹æ¯”åº¦é«˜
    axes[1].set_title("Ground Truth", fontsize=14)
    axes[1].axis('off')

    # --- å›¾3: é¢„æµ‹ (Prediction) ---
    axes[2].imshow(img_show)
    masked_pred = np.ma.masked_where(pred_np == 0, pred_np)
    
    # è®¡ç®—è¿™å¼ å›¾çš„ Dice å�ªæ˜¯ä¸ºäº†å±•ç¤º
    dice_score = (2 * (pred_np * mask_np).sum()) / (pred_np.sum() + mask_np.sum() + 1e-7)
    
    axes[2].imshow(masked_pred, alpha=0.7, cmap='jet') # jet æ˜¯å½©è™¹è‰²ï¼Œå¾ˆæ˜¾çœ¼
    axes[2].set_title(f"Prediction | Dice: {dice_score:.2f}", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# ========================
# è¿�è¡Œå‘½ä»¤
# ========================
# æ³¨æ„�ï¼šè¿™é‡Œä¼ å…¥çš„æ˜¯ test_ds (Dataset)ï¼Œä¸�æ˜¯ test_loader
plot_random_one(best_model, test_ds, DEVICE)


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc


import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import zoom
from albumentations.pytorch import ToTensorV2
from PIL import Image




!pip install albumentations


!wget https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz
#! pip install self-attention-cv
!pip install einops
!pip install ml_collections
tu_path = '../input/transunet/TransUNet-main'
sys.path.append(tu_path)


#æ•´ä¸ªè®­ç»ƒæµ�ç¨‹çš„å¤§è„‘ï¼šæ‰€æœ‰è®­ç»ƒã€�æ¨¡å�‹ã€�ä¼˜åŒ–å™¨ã€�schedulerã€�æ•°æ�®å¢�å¼ºçš„è®¾ç½®éƒ½åœ¨è¿™é‡Œç»Ÿä¸€ç®¡ç�†ã€‚
class CFG:
    data = 256 #512 # æ•°æ�®é›†çš„å°ºå¯¸æˆ– patch å°ºå¯¸ï¼ˆä½ ä½¿ç”¨ 256Ã—256 å›¾åƒ�ï¼‰ã€‚
    debug=False
    apex=False
    print_freq=100
    num_workers=4
    img_size=256 # appropriate input size for encoder 
    # ä½¿ç”¨ä¸€ç§�ä½™å¼¦é€€ç�« + é‡�å�¯çš„å­¦ä¹ ç�‡è°ƒåº¦å™¨ï¼š
    scheduler='CosineAnnealingWarmRestarts' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
   
    epoch=5 # Change epochs
    
    # è®­ç»ƒä½¿ç”¨ Lovasz Lossï¼ˆä¸“é—¨ä¸º IoU ä¼˜åŒ–çš„ lossï¼‰ã€‚
    criterion= 'Lovasz' #'DiceBCELoss' # ['DiceLoss', 'Hausdorff', 'Lovasz']
    base_model='Unet' # ['Unet']
    encoder = 'vit' # ['attention','efficientnet-b5'] or other encoders from smp
    lr=1e-4
    min_lr=1e-6
    batch_size=16
    weight_decay=1e-6
    gradient_accumulation_steps=1
    seed=2021
    n_fold=5
    trn_fold= 0 #[0, 1, 2, 3, 4]
    train=True
    inference=False
    optimizer = 'Adam'
    T_0=10
    
    #RandAugment å�‚æ•°
    N=5 
    M=9
    
    T_max=10
    #factor=0.2
    #patience=4
    #eps=1e-6
    smoothing=1
    in_channels=3
    
    #Vision Transformer çš„ transformer block æ•°é‡�
    vit_blocks=12 #[8, 12]
    
    vit_linear=1024 #1024
    classes=1
    MODEL_NAME = 'R50-ViT-B_16'


#å‡½æ•°é€šè¿‡è®¾ç½® Pythonã€�NumPyã€�PyTorchã€�CUDA å’Œ CuDNN çš„éš�æœºç§�å­�ï¼Œå°�è¯•è®©æ¯�æ¬¡è®­ç»ƒç»“æ�œå°½å�¯èƒ½ä¸€è‡´ï¼ˆå�¯å¤�ç�°ï¼‰
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results 
    torch.backends.cudnn.benchmark = True # from @Iafoss comment

#ä¸ºæ•´ä¸ªè®­ç»ƒç¨‹åº�è®¾å®šç§�å­�
seed_torch(seed=CFG.seed)
print(f"Seeding set to: {CFG.seed}")

# deterministic=True â†’ è¿½æ±‚å�¯å¤�ç�°
# benchmark=True â†’ è¿½æ±‚é€Ÿåº¦ï¼Œå�¯èƒ½å¸¦æ�¥ä¸€äº›éš�æœºæ€§
# ç»“æ�œå¤§å¤šæ•°æ—¶å€™é��å¸¸æ�¥è¿‘ï¼Œä½†ä¸�ä¿�è¯� 100% å®Œå…¨ä¸€æ ·ã€‚
# æ·±åº¦å­¦ä¹ æœ¬èº«æœ‰å™ªå£° â†’ ä½ å�ªéœ€è¦�â€œè¶‹åŠ¿ç¨³å®šâ€�


#å®šä¹‰ 5 å¥—ä¸�å�Œçš„æ•°æ�®å¢�å¼ºï¼Œå°†ä¼šå¾—åˆ°5ç»„æ•°æ�®
# base / weak / strong éƒ½æ˜¯å¯¹å�Ÿå›¾å�šä¸�å�Œç¨‹åº¦çš„ transformation
'''
| mode   | ç”¨é€”     | ç‰¹ç‚¹                            |
| ------ | ------ | ----------------------------- |
| base   | å¸¸ç”¨è®­ç»ƒå¢�å¼º | ç¿»è½¬ã€�æ—‹è½¬ã€�å�˜å½¢ã€�é¢œè‰²å¢�å¼º                 |
| rand   | å¼ºå¢�å¼º    | RandAugment + é¢œè‰² + å‡ ä½•å¢�å¼º       |
| strong | æœ€å¼ºå¢�å¼º   | elastic + noise + distortions |
| weak   | è¾ƒå¼±å¢�å¼º   | å°�å¹…åº¦æ—‹è½¬ + ç¿»è½¬ + resize           |
| valid  | éªŒè¯�     | æ— å¢�å¼ºï¼Œå�ª resize                  |


ä¸ºä»€ä¹ˆè¦�è¿™ä¹ˆå¤šæ¨¡å¼�ï¼Ÿ

å› ä¸ºï¼š

æŸ�äº›æ¨¡å�‹æ—©æœŸé€‚å�ˆ weak

ä¸­æœŸé€‚å�ˆ base

æ��å�‡æ³›åŒ–èƒ½åŠ›é€‚å�ˆ strong / rand

éªŒè¯�å¿…é¡» valid

'''
def get_transform(mode='base'):
    if mode == 'base':
        base_transform = A.Compose([
            A.Resize(CFG.img_size, CFG.img_size, p = 1.0),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(
                shift_limit = 0.0625,
                scale_limit = 0.2,
                rotate_limit = 20,
                p = 0.4,
                border_mode = cv2.BORDER_REFLECT
            ),
            A.OneOf([
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p = 0.1),
                A.PiecewiseAffine(p=0.4)
            ], p=0.3),
            
            A.OneOf([
                A.HueSaturationValue(10,15,10),
                A.CLAHE(clip_limit = 3),
                A.RandomBrightnessContrast(),
            ], p = 0.4),
            ToTensorV2()
        ],p = 1.0)
        
        return base_transform
    
    elif mode == 'rand':
        rand_transform = A.Compose([
            RandAugment(CFG.N, CFG.M),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(CFG.img_size, CFG.img_size, p=1.0),
            A.Normalize(),
            ToTensorV2()
        ])
        return rand_transform
    
    elif mode == 'strong':
        strong_transform = A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.OneOf([
                A.RandomGamma(),
                A.GaussNoise()           
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p=0.2),
                A.PiecewiseAffine(p=0.4),
            ], p=0.5),
            A.OneOf([
                A.HueSaturationValue(10,15,10),
                A.CLAHE(clip_limit=4),
                A.RandomBrightnessContrast(),            
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.Resize(CFG.img_size, CFG.img_size, p=1.0),
            ToTensorV2()
        ])
    
        return strong_transform
    
    elif mode == 'weak':
        weak_transform = A.Compose([
            A.Resize(CFG.img_size, CFG.img_size, p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.4, 
                               border_mode=cv2.BORDER_REFLECT),
            ToTensorV2()
            ], p=1.0)
        
        return weak_transform
    
    elif mode == 'valid':
        val_transform = A.Compose([
                A.Resize(CFG.img_size, CFG.img_size, p=1.0),
                ToTensorV2()
            ], p=1.0)
        return val_transform
    
    else:
        print("Mode Unknown!")
        


# æ ¹æ�® train.csv çš„ idï¼Œä»�ç¡¬ç›˜ä¸Šç­›é€‰å¯¹åº”çš„ patch æ–‡ä»¶å��ï¼Œæ�„å»ºè®­ç»ƒæ•°æ�®é›†ã€‚
# train.csv å†³å®šäº†ã€Œå“ªäº›å�Ÿå›¾æœ‰æ ‡æ³¨ï¼ˆmaskï¼‰ã€�ä»¥å�Šã€Œå“ªäº›å�Ÿå›¾å±�äº�è®­ç»ƒé›†ã€�ã€‚
# æ²¡æœ‰åœ¨ train.csv é‡Œçš„å›¾ï¼Œå°±æ²¡æœ‰ maskï¼Œä¸�èƒ½ç”¨æ�¥è®­ç»ƒã€‚

'''
â‘  çœ‹ train.csv ä¸­å“ªäº›å�Ÿå›¾æœ‰æ ‡ç­¾
â‘¡ ä»� train ç›®å½•ä¸­æ‰¾åˆ°æ‰€æœ‰è¿™äº›å�Ÿå›¾åˆ‡å‡ºæ�¥çš„ patch
â‘¢ ç”¨è¿™äº› patch å’Œå®ƒä»¬çš„ mask æ�¥è®­ç»ƒæ¨¡å�‹

'''
import albumentations as A
# class HuBMAPDataset(Dataset):
#     def __init__(self, main_dir, df, train = True, transform = None):
#         self.ids = df.id.values
#         self.fnames = [fname for fname in os.listdir(train_dir) if fname.split('_')[0] in self.ids]
        
#         self.main_dir = main_dir
#         self.df = df
#         self.train = train
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.fnames)
    
#     def __getitem__(self, idx):
#         fname = self.fnames[idx]
        
#         img = cv2.cvtColor(cv2.imread(os.path.join(main_dir, 'train', fname)), cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(os.path.join(main_dir, 'masks', fname), cv2.IMREAD_GRAYSCALE)
        
#         if self.transform is not None:
#             aug = self.transform(image = img, mask = mask)
#             img, mask = aug['image'], aug['mask']
            
#         img = img.type('torch.FloatTensor')
#         img = img/255
#         mask = mask.type('torch.FloatTensor')
        
#         return img, mask


from pathlib import Path
Path('/kaggle/working/networks').mkdir(parents=True, exist_ok=True)

import requests

if Path("/kaggle/working/networks/vit_seg_modeling.py").is_file():
    print("Already Exist!")
else:
    print("Downloading `vit_seg_modeling.py` ...")
    request = requests.get("https://raw.githubusercontent.com/Beckschen/TransUNet/main/networks/vit_seg_modeling.py")
    with open("/kaggle/working/networks/vit_seg_modeling.py","wb") as f:
        f.write(request.content)
    print("Completed")



import requests

if Path("/kaggle/working/networks/vit_seg_configs.py").is_file():
    print("Already Exist!")
else:
    print("Downloading `vit_seg_configs.py` ...")
    request = requests.get("https://raw.githubusercontent.com/Beckschen/TransUNet/main/networks/vit_seg_configs.py")
    with open("/kaggle/working/networks/vit_seg_configs.py","wb") as f:
        f.write(request.content)
    print("Completed")


import requests

if Path("/kaggle/working/networks/vit_seg_modeling_resnet_skip.py").is_file():
    print("Already Exist!")
else:
    print("Downloading `vit_seg_configs.py` ...")
    request = requests.get("https://raw.githubusercontent.com/Beckschen/TransUNet/main/networks/vit_seg_modeling_resnet_skip.py")
    with open("/kaggle/working/networks/vit_seg_modeling_resnet_skip.py","wb") as f:
        f.write(request.content)
    print("Completed")



#å¯¼å…¥ ViT æ¨¡å�‹ä¸�é…�ç½®å­—å…¸

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

#é€‰æ‹©æ¨¡å�‹é…�ç½®
config_vit = CONFIGS_ViT_seg[CFG.MODEL_NAME]

#ä¿®æ”¹é…�ç½®ï¼ˆé��å¸¸é‡�è¦�ï¼‰

#è®¾ç½®è¾“å‡ºç±»åˆ«æ•°ï¼ˆ0=èƒŒæ™¯ï¼Œ1=è‚¾å°�ç�ƒï¼‰
config_vit.n_classes = 1
# è®¾ç½® skip connections æ•°é‡�
config_vit.n_skip = 3
#è®¾ç½® ViT é¢„è®­ç»ƒæ�ƒé‡�
config_vit.pretrained_path = './R50+ViT-B_16.npz'
# è®¾ç½® transformer dropout
config_vit.transformer.dropout_rate = 0.2
# è®¾ç½® MLP éš�è—�å±‚ç»´åº¦ï¼ˆViT-B/16 çš„é»˜è®¤ mlp_dim å°±æ˜¯ï¼š768*4 = 3072ï¼‰
config_vit.transformer.mlp_dim = 3072
#è®¾ç½® attention heads æ•°é‡�
config_vit.transformer.num_heads = 4
# è®¾ç½® transformer å±‚æ•° ï¼ˆå�Ÿå§‹ ViT-B/16 = 12 layersï¼Œ è¿™é‡Œå‡�å°‘åˆ° 8ï¼Œæ˜¯ä¸ºäº†ï¼š
# é™�ä½�æ˜¾å­˜ï¼Œå‡�å°‘è®­ç»ƒæ—¶é—´ï¼Œä»�ä¿�æŒ�è¾ƒå¥½æ€§èƒ½
config_vit.transformer.num_layers = 8

config_vit


class ViTHuBMAP(nn.Module):
    # æ�„é€ å‡½æ•°ï¼Œè®¾ç½®é»˜è®¤é…�ç½®
    def __init__(self, configs = config_vit):
        super(ViTHuBMAP,self).__init__()
        
        # æ�„å»º TransUNet æ¨¡å�‹ï¼ˆæ ¸å¿ƒï¼�ï¼‰
        self.model = ViT_seg(configs, 
                             img_size = CFG.img_size, 
                             num_classes = CFG.classes)
        # åŠ è½½é¢„è®­ç»ƒæ�ƒé‡�ï¼ˆä¸�éœ€è¦�å†�æ¬¡é¢„è®­ç»ƒï¼‰
        self.model.load_from(weights = np.load(configs.pretrained_path))
        
    # forwardï¼šå®šä¹‰å‰�å�‘ä¼ æ’­ï¼ˆæ¨¡å�‹æ€�ä¹ˆè·‘ï¼‰   
    def forward(self, x):
        img_segs = self.model(x)
        return img_segs


#ä¸“é—¨åˆ›å»ºä¸€ä¸ªæ–‡ä»¶å¤¹ï¼Œç”¨æ�¥å­˜æ”¾æ‰€æœ‰è‡ªå®šä¹‰çš„ loss å‡½æ•°æ–‡ä»¶

Path('/kaggle/working/losses_pytorch').mkdir(parents=True, exist_ok=True)


# æ£€æŸ¥æ˜¯å�¦æœ‰ Hausdorff Lossè¿™ä¸ªæ–‡ä»¶
if Path("/kaggle/working/losses_pytorch/hausdorff.py").is_file():
    print("Already Exist!")
else:
    print("Downloading `hausdorff.py` ...")
    request = requests.get("https://raw.githubusercontent.com/JunMa11/SegLossOdyssey/master/losses_pytorch/hausdorff.py")
    with open("/kaggle/working/losses_pytorch/hausdorff.py","wb") as f:
        f.write(request.content)
    print("Completed")


# æ£€æŸ¥Lovasz Loss
if Path("/kaggle/working/losses_pytorch/lovasz_loss.py").is_file():
    print("Already Exist!")
else:
    print("Downloading `lovasz_loss.py` ...")
    request = requests.get("https://raw.githubusercontent.com/JunMa11/SegLossOdyssey/master/losses_pytorch/lovasz_loss.py")
    with open("/kaggle/working/losses_pytorch/lovasz_loss.py","wb") as f:
        f.write(request.content)
    print("Completed")


# æ£€æŸ¥Focal_loss

if Path("/kaggle/working/losses_pytorch/focal_loss.py").is_file():
    print("Already Exist!")
else:
    print("Downloading `focal_loss.py` ...")
    request = requests.get("https://raw.githubusercontent.com/JunMa11/SegLossOdyssey/master/losses_pytorch/focal_loss.py")
    with open("/kaggle/working/losses_pytorch/focal_loss.py","wb") as f:
        f.write(request.content)
    print("Completed")


from losses_pytorch.hausdorff import HausdorffDTLoss
from losses_pytorch.lovasz_loss import LovaszSoftmax
from losses_pytorch.focal_loss import FocalLoss


#å®šä¹‰è®¡ç®—Diceloss = 1-Dice Score

class DiceLoss(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(DiceLoss, self).__init__()
   
    
    def forward(self, inputs, targets, smooth = CFG.smoothing):
        # #é¦–å…ˆå¯¹æ¨¡å�‹è¾“å‡ºå�š sigmoidï¼ˆSigmoid æŠŠå®ƒå�˜æˆ�æ¦‚ç�‡ï¼ˆ0ï½�1ï¼‰ï¼‰
        inputs = F.sigmoid(inputs)

        #å±•å¹³ä¸º 1D å�‘é‡�
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # è®¡ç®—äº¤é›† intersection
        # ä¹˜ç§¯è¡¨ç¤ºï¼šâ€œé¢„æµ‹ä¸ºæ­£ç±»çš„æ¦‚ç�‡ Ã— Ground truth æ­£ç±»åƒ�ç´ â€� 
        # å…¨éƒ¨ç›¸åŠ å°±æ˜¯é¢„æµ‹ä¸�çœŸå®�çš„â€œé‡�å� é�¢ç§¯â€�ã€‚
        #ä¹Ÿå°±æ˜¯diceå…¬å¼�ä¸­çš„ï¼šâˆ£Pâˆ©Gâˆ£
        intersection = (inputs * targets).sum()

        #è®¡ç®— Dice Score = 2*âˆ£Pâˆ©Gâˆ£/ï¼ˆâˆ£Pï½œ+ï½œGâˆ£ï¼‰
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return dice

# è¿™ä¸ªå®�ç�°å®�é™…ä¸Šè¿”å›�çš„æ˜¯ Dice Scoreï¼ˆè¶Šå¤§è¶Šå¥½ï¼‰ï¼Œä¸�æ˜¯ Lossï¼ˆè¶Šå°�è¶Šå¥½ï¼‰
# Dice Loss = 1 - Dice Scoreã€‚
# Diceè¶Šæ�¥è¿‘1é¢„æµ‹è¶Šå‡†ã€‚


# '''
# ä¸ºä»€ä¹ˆåŠ smoothï¼Ÿ
# 1ï¼šé�¿å…�åˆ†æ¯�ä¸º 0ï¼ˆæœ€é‡�è¦�ï¼‰
# 2.ä½¿ gradient æ›´ç¨³å®šï¼ˆå°¤å…¶æ˜¯å°�ç›®æ ‡ï¼‰
# 3.é�¿å…� Dice åœ¨å‰�æ™¯é�¢ç§¯æ��å°�æ—¶å‡ºç�°æ��ç«¯é«˜/ä½�å€¼
# '''


#DiceBCELoss = Dice Loss + Binary Cross Entropy (BCE)
# '''
# Dice å¼ºåœ¨å“ªé‡Œï¼Ÿ
# â†’ åŒºåŸŸé‡�å� ï¼Œè§£å†³å‰�æ™¯å°�çš„é—®é¢˜

# BCE å¼ºåœ¨å“ªé‡Œï¼Ÿ
# â†’ æ¯�åƒ�ç´ ä¼˜åŒ–ï¼Œæ”¶æ•›å¿«ï¼Œä¸�ä¼šéœ‡è�¡

# è§£å†³ç±»åˆ«ä¸�å¹³è¡¡
# Dice è§£å†³ foreground å°‘çš„é—®é¢˜


# '''

class DiceBCELoss(nn.Module):
    
    # Formula Given Above
    def __init__(self, weight = None, size_average = True):
        super(DiceBCELoss, self).__init__()
        
    def forward(self, inputs, targets, smooth = CFG.smoothing):
        #sigmoid æŠŠ logits è½¬æˆ�æ¦‚ç�‡
        inputs = F.sigmoid(inputs)

        #å±•å¹³æˆ�ä¸€ç»´
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #è®¡ç®— intersectionï¼ˆDice numeratorï¼‰
        intersection = (inputs * targets).mean()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.mean() + targets.mean() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction = 'mean')
        
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE  


class Hausdorff_loss(nn.Module):
    def __init__(self):
        super(Hausdorff_loss, self).__init__()
        
    def forward(self, inputs, targets):
        return HausdorffDTLoss()(inputs, targets)
    
class FocalDLoss(nn.Module):
    def __init__(self):
        super(FocalDLoss, self).__init__()
        
    def forward(self, inputs, targets):
        return FocalLoss()(inputs, targets)
    
    
class Lovasz_loss(nn.Module):
    def __init__(self):
        super(Lovasz_loss, self).__init__()
        
    def forward(self, inputs, targets):
        return LovaszSoftmax()(inputs, targets)


if CFG.criterion == 'DiceBCELoss':
    criterion = DiceBCELoss()
elif CFG.criterion == 'DiceLoss':
    criterion = DiceLoss()
elif CFG.criterion == 'FocalLoss':
    criterion = FocalDLoss()
elif CFG.criterion == 'Hausdorff':
    criterion = Hausdorff_loss()
elif CFG.criterion == 'Lovasz':
    criterion = Lovasz_loss()

# criterion


# def HuBMAPLoss(images, targets, model, device, loss_func = criterion):
#     model.to(device)
#     images = images.to(device)
#     targets = targets.to(device)

#     outputs = model(images)
#     loss = loss_func(outputs, targets)
    
#     return loss, outputs
def HuBMAPLoss(images, targets, model, device, loss_func=criterion):
    # model.to(device) å…¶å®�å»ºè®®æ”¾åœ¨å¾ªç�¯å¤–é�¢å�šä¸€æ¬¡å�³å�¯ï¼Œæ”¾åœ¨è¿™é‡Œä¹Ÿå�¯ä»¥ä½†æ•ˆç�‡ç¨�ä½�
    # model.to(device) 
    
    # -----------------------------------------------------------
    # ğŸ› ï¸� ä¿®å¤� 1: å›¾ç‰‡è½¬æµ®ç‚¹æ•°å¹¶å½’ä¸€åŒ– (è§£å†³ ByteTensor æŠ¥é”™)
    # -----------------------------------------------------------
    # å�Ÿå§‹æ•°æ�®æ˜¯ uint8 (0-255)ï¼Œæ¨¡å�‹éœ€è¦� float32 (0.0-1.0)
    images = images.to(device).float() / 255.0
    
    # -----------------------------------------------------------
    # ğŸ› ï¸� ä¿®å¤� 2: æ ‡ç­¾è½¬æµ®ç‚¹æ•°å¹¶å¢�åŠ é€šé�“ç»´åº¦ (è§£å†³ç»´åº¦ä¸�åŒ¹é…�)
    # -----------------------------------------------------------
    # (Batch, 256, 256) -> (Batch, 1, 256, 256)
    targets = targets.to(device).float().unsqueeze(1)

    # å‰�å�‘ä¼ æ’­
    outputs = model(images)
    
    # -----------------------------------------------------------
    # ğŸ› ï¸� ä¿®å¤� 3: å…¼å®¹æ€§å¤„ç�† (é˜²æ­¢ ViT è¿”å›� tuple)
    # -----------------------------------------------------------
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
        
    # è®¡ç®— Loss
    loss = loss_func(outputs, targets)
    
    return loss, outputs


import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class HuBMAPDataset(Dataset):
    def __init__(self, main_dir, df, train=True, transform=None):
        self.main_dir = main_dir
        self.train = train
        self.transform = transform
        
        # -----------------------------------------------------------
        # ğŸ› ï¸� ä¿®å¤� 1: å…¼å®¹ DataFrame å’Œ List
        # -----------------------------------------------------------
        # å¦‚æ�œä¼ å…¥çš„æ˜¯ DataFrameï¼Œæ��å�– id åˆ—ï¼›å¦‚æ�œä¼ å…¥çš„æ˜¯ list/arrayï¼Œç›´æ�¥ä½¿ç”¨
        if isinstance(df, pd.DataFrame):
            # å�‡è®¾ DataFrame è‚¯å®šæœ‰ä¸€åˆ—å�« 'id' æˆ– 'image_id'
            if 'id' in df.columns:
                self.ids = df['id'].values.astype(str)
            else:
                self.ids = df.iloc[:, 0].values.astype(str) # å�–ç¬¬ä¸€åˆ—ä½œä¸ºID
        else:
            # å¦‚æ�œå·²ç»�æ˜¯åˆ—è¡¨æˆ– numpy array
            self.ids = np.array(df).astype(str)

        # -----------------------------------------------------------
        # ğŸ› ï¸� ä¿®å¤� 2: å®šä¹‰ train_dir å¹¶ä¼˜åŒ–æ–‡ä»¶ç­›é€‰
        # -----------------------------------------------------------
        # ä½ å�Ÿæ�¥çš„ä»£ç �é‡Œç›´æ�¥ç”¨äº† train_dir ä½†æ²¡å®šä¹‰å®ƒ
        self.train_dir = os.path.join(self.main_dir, 'train')
        self.mask_dir = os.path.join(self.main_dir, 'masks')
        
        # è�·å�–æ‰€æœ‰å›¾ç‰‡æ–‡ä»¶å��
        # æ³¨æ„�ï¼šä¸ºäº†åŠ å¿«é€Ÿåº¦ï¼Œè¿™é‡Œå»ºè®®æŠŠ ids è½¬ä¸º set è¿›è¡ŒæŸ¥æ‰¾
        id_set = set(self.ids)
        
        # é��å�† train æ–‡ä»¶å¤¹ï¼Œå�ªä¿�ç•™é‚£äº› ID åœ¨æˆ‘ä»¬åˆ—è¡¨é‡Œçš„å›¾ç‰‡
        # æ–‡ä»¶å��æ ¼å¼�é€šå¸¸æ˜¯: "id_slice.png" -> split('_')[0] å¾—åˆ° id
        self.fnames = [
            fname for fname in os.listdir(self.train_dir) 
            if fname.split('_')[0] in id_set
        ]

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        
        # è¯»å�–å›¾ç‰‡
        img_path = os.path.join(self.train_dir, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # è¯»å�– Mask
        mask_path = os.path.join(self.mask_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # åº”ç”¨æ•°æ�®å¢�å¼º (Albumentations)
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
            
        # -----------------------------------------------------------
        # ğŸ› ï¸� ä¿®å¤� 3: æ ‡å‡†åŒ– Torch Tensor è½¬æ�¢å†™æ³•
        # -----------------------------------------------------------
        # å¦‚æ�œ transform é‡Œå·²ç»�æœ‰ ToTensorV2ï¼Œè¿™é‡Œå°±ä¸�éœ€è¦�æ‰‹åŠ¨è½¬ tensor äº†
        # å¦‚æ�œ transform å�ªæœ‰å‡ ä½•å�˜æ�¢ï¼Œè¿™é‡Œéœ€è¦�æ‰‹åŠ¨è½¬ï¼š
        
        # ç¡®ä¿�æ˜¯ Tensor æ ¼å¼� (C, H, W)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            img = img / 255.0
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
            # Mask é€šå¸¸ä¸�éœ€è¦�é™¤ä»¥ 255ï¼Œå› ä¸ºå®ƒåº”è¯¥æ˜¯ 0/1 æˆ–è€…æ˜¯ç±»åˆ«ç´¢å¼•
            # å�ªæœ‰å½“ mask æ˜¯ 0-255 çš„ç�°åº¦å›¾ä¸”ä»£è¡¨æ¦‚ç�‡æ—¶æ‰�é™¤ä»¥ 255
            # å�‡è®¾ä½ çš„ mask æ˜¯ 0å’Œ255 (äºŒå€¼åŒ–)ï¼Œè¿™é‡Œå½’ä¸€åŒ–åˆ° 0-1
            mask = mask / 255.0 
            # å¢�åŠ  channel ç»´åº¦ (H, W) -> (1, H, W)
            mask = mask.unsqueeze(0) 
            
        return img, mask


main_dir = '../input/hubmap-256x256/'
train_dir = '../input/hubmap-256x256//train/'
masks_dir = '../input/hubmap-256x256//masks/'
directory_list = os.listdir('../input/hubmap-256x256/train')
train_df = pd.read_csv('../input/hubmap-kidney-segmentation/train.csv')
    
directory_list = [fnames.split('_')[0] for fnames in directory_list]
dir_df = pd.DataFrame(directory_list, columns=['id'])
dir_df
# #å‰�é�¢å·²ç»�åˆ’åˆ†è¿‡
# train_ds = HuBMAPDataset(main_dir, train_ids, train = True, transform = get_transform('base'))
# valid_ds = HuBMAPDataset(main_dir, valid_ids, train = True, transform = get_transform('valid'))
# test_ds = HuBMAPDataset(main_dir, test_ids, train = True, transform = get_transform('valid'))

# len(train_ds), len(valid_ds),len(test_ds)  


import os
import random
import numpy as np
# ç¡®ä¿�å¯¼å…¥äº† Dataset éœ€è¦�çš„åº“ï¼Œå¦‚ torch, cv2, Dataset ç­‰

# --- ç¬¬ä¸€æ­¥ï¼šè�·å�–æ‰€æœ‰å”¯ä¸€çš„ ID ---
# è¿™é‡Œç›´æ�¥æ‰«æ��æ–‡ä»¶å¤¹æ–‡ä»¶å��æ�¥è�·å�– IDï¼Œæ¯”è¯» CSV æ›´ç¨³ï¼Œå› ä¸ºèƒ½ç¡®ä¿�æ–‡ä»¶å­˜åœ¨
all_files = os.listdir(train_dir)
all_ids = set([f.split('_')[0] for f in all_files]) # æ��å�– ID å¹¶å�»é‡�
unique_ids = list(all_ids)

# --- ç¬¬äºŒæ­¥ï¼šæŒ‰ 7:2:1 åˆ’åˆ† ID ---
random.seed(42) # å›ºå®šéš�æœºç§�å­�
random.shuffle(unique_ids)

n_total = len(unique_ids)
n_train = int(n_total * 0.7)
n_valid = int(n_total * 0.2)

train_ids = unique_ids[:n_train]
valid_ids = unique_ids[n_train : n_train + n_valid]
test_ids  = unique_ids[n_train + n_valid:]

print(f"ID åˆ’åˆ†æƒ…å†µ -> Train: {len(train_ids)}, Valid: {len(valid_ids)}, Test: {len(test_ids)}")



train_ds = HuBMAPDataset(main_dir, train_ids, train = True, transform = get_transform('base'))
valid_ds = HuBMAPDataset(main_dir, valid_ids, train = True, transform = get_transform('valid'))
test_ds = HuBMAPDataset(main_dir, test_ids, train = False)

train_loader = DataLoader(train_ds, batch_size = CFG.batch_size, pin_memory = True, shuffle = True, num_workers=CFG.num_workers)
valid_loader = DataLoader(valid_ds, batch_size = CFG.batch_size, pin_memory = True, shuffle = False, num_workers= CFG.num_workers)
test_loader = DataLoader(test_ds, batch_size = CFG.batch_size, pin_memory = True, shuffle = False, num_workers= CFG.num_workers)

print(f"æœ€ç»ˆåˆ‡ç‰‡(Slice)æ•°é‡� -> Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")
print(f"loader -> Train: {len(train_loader)}, Valid: {len(valid_loader)}, Test: {len(test_loader)}")   


def train_one_epoch(epoch, model, device, optimizer, scheduler, trainloader):
    model.train()
    t = time.time()
    total_loss = 0 
    
    for step, (images, targets) in enumerate(trainloader):
        loss, outputs = HuBMAPLoss(images, targets, model, device)
        loss.backward()
        if ((step+1)%4==0 or (step+1)==len(trainloader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss = loss.detach().item()
        total_loss += loss
        
        if ((step+1)%10==0 or (step+1)==len(trainloader)):
            print(
                    f'epoch {epoch} train step {step+1}/{len(trainloader)}, ' + \
                    f'loss: {total_loss/len(trainloader):.4f}, ' + \
                    f'time: {(time.time() - t):.4f}', end= '\r' if (step + 1) != len(trainloader) else '\n'
                )

            
        
def valid_one_epoch(epoch, model, device, optimizer, scheduler, validloader):
    model.eval()
    t = time.time()
    total_loss = 0
    
    for step, (images, targets) in enumerate(validloader):
        loss, outputs = HuBMAPLoss(images, targets, model, device)
        loss = loss.detach().item()
        total_loss += loss
        
        if ((step+1)%4==0 or (step+1)==len(validloader)):
            scheduler.step(total_loss/len(validloader))
        
        if ((step+1)%10==0 or (step+1)==len(validloader)):
            print(
                    f'**epoch {epoch} trainz step {step+1}/{len(validloader)}, ' + \
                    f'loss: {total_loss/len(validloader):.4f}, ' + \
                    f'time: {(time.time() - t):.4f}', end= '\r' if (step + 1) != len(validloader) else '\n'
                )
    

        





device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTHuBMAP().to(device)
optimizer = Adam(model.parameters(), lr = CFG.lr, weight_decay= CFG.weight_decay, amsgrad = False)

# scheduler setting
if CFG.scheduler == 'CosineAnnealingWarmRestarts':
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
elif CFG.scheduler == 'ReduceLROnPlateau':
    scheduler = ReduceLROnPlateauReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
elif CFG.scheduler == 'CosineAnnealingLR':
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)


# print(f'Training Loop...')
# # for fold, (tr_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
# #     if fold != CFG.trn_fold: # Train only one fold
# #         continue 

# #     trainloader, validloader = prepare_train_valid_dataloader(dir_df, [fold])

# for epoch in range(CFG.epoch):
#     train_one_epoch(epoch, model, device, optimizer, scheduler, train_loader)
#     with torch.no_grad():
#         valid_one_epoch(epoch, model, device, optimizer, scheduler, valid_loader)
        
#         #torch.save(model.state_dict(),f'FOLD-{fold}-EPOCH-{epoch}-model.pth')
        
# torch.save(model.state_dict(),f'bestvit-model.pth')


def valid_epoch(epoch, model, device, optimizer, scheduler, validloader):
    model.eval()
    t = time.time()
    total_loss = 0
    total_dice = 0 # 1. æ–°å¢�ï¼šè®°å½•æ€» Dice
    
    for step, (images, targets) in enumerate(validloader):
        loss, outputs = HuBMAPLoss(images, targets, model, device)
        loss = loss.detach().item()
        total_loss += loss
        
        # 2. æ–°å¢�ï¼šè®¡ç®— Dice æŒ‡æ ‡ (è¿™å¯¹åˆ†å‰²ä»»åŠ¡å¾ˆé‡�è¦�)
        # ---------------------------------------------------
        # å‡†å¤‡ targets (åŠ ç»´åº¦ä»¥åŒ¹é…�è¾“å‡º)
        targets_dice = targets.to(device).float().unsqueeze(1)
        
        # è®¡ç®—é¢„æµ‹ (Sigmoid -> äºŒå€¼åŒ–)
        prob = outputs.sigmoid()
        pred = (prob > 0.5).float()
        
        # è®¡ç®—äº¤å¹¶æ¯”
        intersection = (pred * targets_dice).sum()
        union = pred.sum() + targets_dice.sum()
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        total_dice += dice.item()
        # ---------------------------------------------------
        
        # 3. ä¿®æ­£ï¼šæ‰“å�°è¿›åº¦
        if ((step+1)%10==0 or (step+1)==len(validloader)):
            # æ³¨æ„�ï¼šè¿™é‡Œçš„ loss åº”è¯¥æ˜¯ total_loss / (step+1) æ‰�æ˜¯å½“å‰�çš„å¹³å�‡å€¼
            # å�Ÿä»£ç �é™¤ä»¥ len(validloader) ä¼šå¯¼è‡´åˆšå¼€å§‹æ˜¾ç¤ºçš„ loss é��å¸¸å°�
            print(
                f'**epoch {epoch} valid step {step+1}/{len(validloader)}, ' + \
                f'loss: {total_loss/(step+1):.4f}, ' + \
                f'dice: {total_dice/(step+1):.4f}, ' + \
                f'time: {(time.time() - t):.4f}', end= '\r' if (step + 1) != len(validloader) else '\n'
            )
            
    # è®¡ç®—æ•´ä¸ª Epoch çš„å¹³å�‡ Loss
    avg_loss = total_loss / len(validloader)
    
    # 4. ä¿®æ­£ï¼šScheduler æ›´æ–°ç§»åˆ°å¾ªç�¯å¤– (Epoch ç»“æ�Ÿæ—¶æ›´æ–°)
    if scheduler is not None:
        # å¦‚æ�œæ˜¯ ReduceLROnPlateauï¼Œéœ€è¦�ä¼ å…¥ loss
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)
        else:
            # å…¶ä»– scheduler (å¦‚ CosineAnnealing) é€šå¸¸ä¸�éœ€è¦�ä¼ å�‚æˆ–å�ªä¼  epoch
            # å¦‚æ�œä½ çš„ scheduler ä¸�éœ€è¦�åœ¨è¿™é‡Œæ›´æ–°ï¼Œå�¯ä»¥æ³¨é‡Šæ�‰
            pass 

    # 5. å…³é”®ï¼šå¿…é¡»è¿”å›� lossï¼Œä¾›å¤–éƒ¨ä¿�å­˜æ¨¡å�‹ä½¿ç”¨
    return avg_loss

# åˆ�å§‹åŒ–æœ€å°� loss
best_valid_loss = float('inf')

for epoch in range(CFG.epoch):
    train_one_epoch(epoch, model, device, optimizer, scheduler, train_loader)
    
    with torch.no_grad():
        # âœ… ç�°åœ¨è¿™é‡Œæœ‰è¿”å›�å€¼äº†
        val_loss = valid_epoch(epoch, model, device, optimizer, scheduler, valid_loader)
        
    # âœ… å�¯ä»¥æ¯”è¾ƒå¹¶ä¿�å­˜äº†
    if val_loss < best_valid_loss:
        print(f"ğŸ”¥ Loss Improved: {best_valid_loss:.4f} -> {val_loss:.4f}")
        best_valid_loss = val_loss
        torch.save(model.state_dict(), 'bestvit-model-b16.pth')


from IPython.display import FileLink

# å°† 'vit-model.pth' æ›¿æ�¢æˆ�ä½ å®�é™…ä¿�å­˜çš„æ–‡ä»¶å��
FileLink('bestvit-model-b16.pth')


#Step 1: é‡�æ–°å®�ä¾‹åŒ–æ¨¡å�‹ç»“æ�„ (å¿…é¡»å…ˆé€ ä¸€ä¸ªç©ºçš„â€œå£³â€�)
# è¿™é‡Œçš„ config_vit å¿…é¡»å’Œä½ è®­ç»ƒæ—¶ç”¨çš„é…�ç½®å®Œå…¨ä¸€è‡´
model = ViTHuBMAP(config_vit) 

# Step 2: åŠ è½½è®­ç»ƒå¥½çš„æ�ƒé‡� (æŠŠè®­ç»ƒå¥½çš„â€œè‚‰â€�è£…è¿›å�»)
# 'vit-model.pth' å¿…é¡»æ˜¯ä½ ä¹‹å‰� torch.save() æ—¶ä¿�å­˜çš„æ–‡ä»¶å��
# map_location=DEVICE ç¡®ä¿�æ�ƒé‡�è¢«åŠ è½½åˆ°æ­£ç¡®çš„è®¾å¤‡(CPU/GPU)ä¸Š
# weight_path = '/kaggle/input/vit/pytorch/default/1/FOLD-0-model.pth'  # <--- è¯·ç¡®è®¤ä½ çš„æ–‡ä»¶å��æ˜¯è¿™ä¸ªï¼Œè¿˜æ˜¯ FOLD-0-model.pthï¼Ÿ
weight_path = '/kaggle/working/bestvit-model-b16.pth'
model.load_state_dict(torch.load(weight_path, map_location=device))

# Step 3: æŠŠå®Œæ•´æ¨¡å�‹ç§»åˆ° GPU (å¦‚æ�œè¿˜æ²¡ç§»çš„è¯�)
model.to(device)
model.eval() # æ��å…¶é‡�è¦�ï¼�å…³é—­ Dropout å’Œ BatchNormal çš„è®­ç»ƒè¡Œä¸º

print(f"âœ… æ¨¡å�‹å·²åŠ è½½å®Œæ¯•: {weight_path}")


import torch
import time
import numpy as np
DEVICE = device
@torch.no_grad()
def inference_test(model, test_loader, device):
    """
    åœ¨æµ‹è¯•é›†ä¸Šè¯„ä¼°æ¨¡å�‹è¡¨ç�°
    è¿”å›�: å¹³å�‡ Loss, å¹³å�‡ Dice, å¹³å�‡ IoU
    """
    model.eval()
    t = time.time()
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    # æ‰“å�°å¼€å§‹æ��ç¤º
    print(f"ğŸš€ Start Testing on {len(test_loader.dataset)} images...")

    for step, (images, targets) in enumerate(test_loader):
        # 1. è®¡ç®— Loss (å¤�ç”¨ä½ ç�°æœ‰çš„ HuBMAPLoss)
        # æ³¨æ„�: HuBMAPLoss å†…éƒ¨å·²ç»�åŒ…å�«äº† images.float()/255.0 å’Œ targets.unsqueeze(1) çš„å¤„ç�†
        loss, outputs = HuBMAPLoss(images, targets, model, device)
        
        # è®°å½• Loss
        loss_val = loss.detach().item()
        total_loss += loss_val
        
        # 2. è®¡ç®— Dice å’Œ IoU æŒ‡æ ‡ (æ ¸å¿ƒè¯„ä¼°éƒ¨åˆ†)
        # éœ€è¦�æ‰‹åŠ¨å¤„ç�† targets ä»¥åŒ¹é…� Dice è®¡ç®—çš„ç»´åº¦
        targets_dice = targets.to(device).float().unsqueeze(1)
        
        prob = outputs.sigmoid()      # è½¬ä¸º 0-1 æ¦‚ç�‡
        pred = (prob > 0.5).float()   # äºŒå€¼åŒ–é¢„æµ‹ (0 æˆ– 1)
        
        intersection = (pred * targets_dice).sum()
        union = pred.sum() + targets_dice.sum()
        
        # Dice Score
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        total_dice += dice.item()
        
        # IoU Score
        iou = (intersection + 1e-7) / (union - intersection + 1e-7)
        total_iou += iou.item()

        # 3. æ‰“å�°è¿›åº¦ (ç±»ä¼¼ä½ çš„ valid å‡½æ•°)
        if ((step + 1) % 10 == 0) or ((step + 1) == len(test_loader)):
            print(
                f'Test Step {step+1}/{len(test_loader)} | '
                f'Loss: {total_loss / (step+1):.4f} | '
                f'Dice: {total_dice / (step+1):.4f} | '
                f'IoU:  {total_iou  / (step+1):.4f} | '
                f'Time: {(time.time() - t):.2f}s', 
                end='\r' if (step + 1) != len(test_loader) else '\n'
            )
            
    # è®¡ç®—æœ€ç»ˆå¹³å�‡åˆ†
    avg_loss = total_loss / len(test_loader)
    avg_dice = total_dice / len(test_loader)
    avg_iou  = total_iou  / len(test_loader)
    
    print("\n" + "="*40)
    print(f"ğŸ�† Final Test Results")
    print("="*40)
    print(f"Avg Loss : {avg_loss:.4f}")
    print(f"Avg Dice : {avg_dice:.4f}")
    print(f"Avg IoU  : {avg_iou:.4f}")
    print("="*40)
    
    return avg_loss, avg_dice, avg_iou

test_loss, test_dice, test_iou = inference_test(model, test_loader, DEVICE)


# Metrices
import torch
import numpy as np

def dice_coefficient(y_true, y_pred):
    smooth = 1
    inputs = y_true.view(-1)
    targets = y_pred.view(-1)
        
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
    return dice

def iou(y_true, y_pred):
    smooth = 1
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def accuracy(y_true, y_pred):
    correct = torch.sum((y_pred == y_true).float())
    total = y_true.numel()
    acc = correct / total
    return acc

def precision(y_true, y_pred):
    smooth = 1
    true_positive = torch.sum(y_true * y_pred)
    predicted_positive = torch.sum(y_pred)
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision

def recall(y_true, y_pred):
    smooth = 1
    true_positive = torch.sum(y_true * y_pred)
    actual_positive = torch.sum(y_true)
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall



def calculate_metrices(validloader, model, device):
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_prec = 0.0
    total_rec = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for img, mask in validloader:
            img = img.to(device).float()/255.0
            mask = mask.to(device)
            pred_mask = model(img).cpu()
        
            pred_new_mask = (pred_mask.squeeze() > 0.5).float()
            new_tensor = torch.ones_like(pred_new_mask) - pred_new_mask

            batch_dice = 0.0
            batch_iou_score = 0.0
            batch_acc = 0.0
            batch_prec = 0.0
            batch_rec = 0.0
            
            batch_size = img.size(0)
            
            for i in range(batch_size):
                           
                batch_dice += dice_coefficient(mask[i].cpu(),  new_tensor[i].squeeze())
                batch_iou_score += iou(mask[i].cpu(), new_tensor[i].squeeze())
                batch_acc += accuracy(mask[i].cpu(), new_tensor[i].squeeze())
                batch_prec += precision(mask[i].cpu(),new_tensor[i].squeeze())
                batch_rec += recall(mask[i].cpu(), new_tensor[i].squeeze())

            
            batch_dice = batch_dice/batch_size
            batch_iou_score = batch_iou_score/batch_size
            batch_acc = batch_acc/batch_size
            batch_prec = batch_prec/batch_size
            batch_rec = batch_rec/batch_size
            

            total_dice += batch_dice
            total_iou += batch_iou_score
            total_acc += batch_acc
            total_prec += batch_prec
            total_rec += batch_rec
            
            
            num_batches += 1
    
        mean_dice = total_dice / num_batches
        mean_iou = total_iou / num_batches
        mean_acc = total_acc / num_batches
        mean_prec = total_prec / num_batches
        mean_rec = total_rec / num_batches
        
        
        data = {
            'Mean Dice' : mean_dice,
            'IoU' : mean_iou,
            'Accuracy' : mean_acc,
            'Precision' : mean_prec,
            'Recall' : mean_rec,
        }
        
        df = pd.DataFrame.from_dict(data, orient='index').T
        
        csv_file_path = '/kaggle/working/metrices-bestvit-b16.csv'
        df.to_csv(csv_file_path, index = True)
        
# Assuming 'model' and 'device' are defined elsewhere
calculate_metrices(test_loader, model, device)


import matplotlib.pyplot as plt
import numpy as np

def plot_result(validloader, n_sample=2):
    # 1. è�·å�–ä¸€ä¸ª Batch çš„æ•°æ�®
    img, mask = next(iter(validloader))
    
    # 2. å‡†å¤‡æ¨¡å�‹è¾“å…¥ (ä¿®å¤�ä¹‹å‰�é�‡åˆ°çš„ ByteTensor é”™è¯¯)
    # img æ˜¯ uint8 (0-255), éœ€è¦�è½¬æˆ� float (0-1) å–‚ç»™æ¨¡å�‹
    img_input = img.to(device).float() / 255.0
    
    # 3. åŠ è½½æ�ƒé‡�å¹¶æ�¨ç�†
    # model.load_state_dict(...) # å»ºè®®åœ¨å‡½æ•°å¤–åŠ è½½å¥½æ¨¡å�‹ï¼Œä¸�è¦�åœ¨å¾ªç�¯é‡Œå��å¤�åŠ è½½æ–‡ä»¶ï¼Œå¤ªæ…¢äº†
    model.eval()
    
    with torch.no_grad():
        output = model(img_input)
        if isinstance(output, (tuple, list)):
            output = output[0]
            
        # åŠ ä¸Š sigmoid è½¬æˆ�æ¦‚ç�‡ (0~1)ï¼Œæ–¹ä¾¿å�¯è§†åŒ–
        pred_mask = output.sigmoid().cpu()

    # 4. ç»˜å›¾
    plt.figure(figsize=(15, 10))
    
    # ç¡®ä¿� n_sample ä¸�è¶…è¿‡ batch size
    n_sample = min(n_sample, img.shape[0])
    
    for i in range(n_sample):
        # --- (1) å�Ÿå›¾ ---
        plt.subplot(n_sample, 3, 3*i+1)
        # img æ˜¯ (C, H, W)ï¼Œéœ€è¦�è½¬ç½®ä¸º (H, W, C) æ‰�èƒ½ç”»å›¾
        # å¦‚æ�œ img æ˜¯ floatï¼Œimshow æœŸæœ› 0-1ï¼›å¦‚æ�œ uint8ï¼ŒæœŸæœ› 0-255ã€‚è¿™é‡Œ img æ˜¯å�Ÿå§‹ uint8
        plt.imshow(np.transpose(img[i].numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.title("Image")

        # --- (2) é¢„æµ‹ Mask ---
        plt.subplot(n_sample, 3, 3*i+2)
        # pred_mask[i] æ˜¯ (1, H, W) -> squeeze -> (H, W)
        plt.imshow(pred_mask[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title("Predicted Probability")
        
        # --- (3) çœŸå®� Mask ---
        plt.subplot(n_sample, 3, 3*i+3)
        # ğŸ› ï¸� å…³é”®ä¿®å¤�: mask[i] æ˜¯ (1, H, W)ï¼Œå¿…é¡» squeeze æ�‰é‚£ä¸ª 1 å�˜æˆ� (H, W)
        plt.imshow(mask[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title("Ground Truth")
    
    plt.tight_layout()
    plt.savefig('single_batch_result.png')
    plt.show()

plot_result(test_loader)


dice_metric = []
iou_metric = []
acc_metric = []
prec_metric = []
recall_metric = []

import matplotlib.pyplot as plt

def plot_result(validloader, model, device, batch_idx, n_sample=4):
    model.eval()
    with torch.no_grad():
        for idx, (img, mask) in enumerate(validloader):
            if idx == batch_idx:
                break
                
        # img = img.to(device)
        img = img.to(device).float() / 255.0
        mask = mask.to(device)
        pred_mask = model(img).cpu()
        
        pred_new_mask = (pred_mask.squeeze() > 0.5).float()
        new_tensor = torch.ones_like(pred_new_mask) - pred_new_mask

        N = n_sample // 2
        plt.figure(figsize=(15, 8))
        for i in range(n_sample):
            plt.subplot(N, 4, 2*i+1)
            plt.imshow(mask[i].cpu().squeeze(), cmap = 'gray')
            plt.axis('off')
            plt.title('Mask')

            plt.subplot(N, 4, 2*i+2)
            plt.imshow(new_tensor[i])
            plt.axis('off')
            plt.title('Predicted Mask')

            plt.tight_layout()
            plt.savefig('/kaggle/working/another_batch.png')
#             print(mask[i].shape)  256 * 256
#             print(new_tensor[i].squeeze().shape)  #1 * 256 * 256
            
            # Compute evaluation metrics
            dice = dice_coefficient(mask[i].cpu(),  new_tensor[i].squeeze())
            iou_score = iou(mask[i].cpu(), new_tensor[i].squeeze())
            acc = accuracy(mask[i].cpu(), new_tensor[i].squeeze())
            prec = precision(mask[i].cpu(),new_tensor[i].squeeze())
            rec = recall(mask[i].cpu(), new_tensor[i].squeeze())
            

            dice_metric.append(dice)
            iou_metric.append(iou_score)
            acc_metric.append(acc)
            prec_metric.append(prec)
            recall_metric.append(rec)
            


# Assuming 'model' and 'device' are defined elsewhere
with torch.no_grad():
    plot_result(test_loader, model, device, 1)


