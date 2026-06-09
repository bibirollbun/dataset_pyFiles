# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -U \
  albumentations==2.0.8 \
  opencv-python-headless==4.12.0.88 \
  tifffile==2025.5.24 \
  torchmetrics==1.4.0 \
  numpy==2.2.4 \
  scikit-learn==1.7.1 \
  scipy==1.16.0 \
  transformers==4.55.3


# === ONE-CELL PIPELINE: DeepLabV3-ResNet50 (no U-Net) for vessel segmentation ===

!pip install numpy==1.23.5 scipy==1.14.1 opencv-python-headless==4.10.0.84 scikit-learn==1.2.2 albumentations==1.3.0 tifffile

import os, random
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchmetrics.functional import dice as tm_dice

# ------------------------------- CONFIG -------------------------------
DATA = Path("/kaggle/input/blood-vessel-segmentation")
TRAIN = DATA/"train"
TEST = DATA/"test"
RLE_CSV = DATA/"train_rles.csv"
OUTDIR = Path("./outputs"); OUTDIR.mkdir(exist_ok=True)

CFG = dict(
    img_size=512,
    batch_size=4,
    num_workers=2,
    epochs=16,
    lr=1e-3,
    weight_decay=1e-5,
    seed=42,
    mask_threshold=0.35,
    any_threshold=0.5,
    tta=True,
    use_mixed_precision=True,
)

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

# ------------------------------- UTILITIES -------------------------------
def rle_encode(mask):
    m = mask.flatten(order='F')
    m = np.concatenate([[0], m, [0]])
    runs = np.where(m[1:] != m[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))

def rle_decode(rle, shape):
    if not isinstance(rle, str) or rle.strip()=="":
        return np.zeros(shape, dtype=np.uint8)
    s = np.asarray([int(x) for x in rle.split()], dtype=int)
    starts, lengths = s[0::2]-1, s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends): img[lo:hi] = 1
    return img.reshape(shape, order='F')

def read_tiff(path):
    img = cv2.imread(str(path), -1)
    if img is None: img = tiff.imread(str(path))
    if img.ndim==3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# ------------------------------- DATA DISCOVERY -------------------------------
def find_train_items(root):
    items = []
    for ds in sorted(os.listdir(root)):
        dpath = Path(root)/ds
        if not (dpath/"images").exists(): continue
        for ip in sorted(dpath.glob("images/*.tif")):
            sid = f"{ds}_{Path(ip).stem}"
            lpath = dpath/"labels"/Path(ip).name
            if lpath.exists(): items.append((ip,str(lpath),sid))
            else: items.append((ip,None,sid))
    return items

pairs = find_train_items(TRAIN)
print("Train slices:", len(pairs))

rle_map = {}
if RLE_CSV.exists():
    df_rle = pd.read_csv(RLE_CSV)
    rle_map = dict(zip(df_rle.id.values, df_rle.rle.fillna("")))

# ------------------------------- DATASET -------------------------------
class KidneySegDataset(Dataset):
    def __init__(self, items, augment=True, size=512):
        self.items = items
        self.size = size
        if augment:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.RandomRotate90(p=0.5), A.Flip(p=0.5),
                A.Affine(scale=(0.9,1.1), rotate=(-10,10), shear=(-8,8), translate_percent=(-0.05,0.05), p=0.5),
                A.ElasticTransform(p=0.2, alpha=50, sigma=7, alpha_affine=10),
                A.RandomBrightnessContrast(p=0.35), A.CLAHE(clip_limit=2.0, p=0.3),
                A.GaussianBlur(blur_limit=(3,5), p=0.2),
                A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.2),
                A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()
            ])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ipath, lpath, sid = self.items[i]
        
        img = read_tiff(ipath)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if lpath is not None and Path(lpath).exists():
            mask = read_tiff(lpath)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = rle_decode(rle_map.get(sid, ""), img.shape)
        
        if img.ndim == 2:
            img = img[..., None]
        
        aug = self.tf(image=img, mask=mask)
        x = aug["image"].float()
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        
        y = torch.as_tensor(aug["mask"], dtype=torch.float).unsqueeze(0)  # 1 x H x W
        
        return x, y, sid

        



# ------------------------------- LOAD DATA -------------------------------
random.shuffle(pairs)
split=int(0.9*len(pairs))
train_ds=KidneySegDataset(pairs[:split],augment=True,size=CFG["img_size"])
valid_ds=KidneySegDataset(pairs[split:],augment=False,size=CFG["img_size"])

train_loader=DataLoader(train_ds,batch_size=CFG["batch_size"],shuffle=True,num_workers=CFG["num_workers"],pin_memory=True,drop_last=True)
valid_loader=DataLoader(valid_ds,batch_size=CFG["batch_size"]*2,shuffle=False,num_workers=CFG["num_workers"],pin_memory=True)

# ------------------------------- MODEL -------------------------------
weights = DeepLabV3_ResNet50_Weights.DEFAULT
net = deeplabv3_resnet50(weights=weights, aux_loss=True)
in_ch = net.classifier[-1].in_channels
net.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)
net.aux_classifier=None

class AnyVesselHead(nn.Module):
    def __init__(self, in_ch=2048):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, 1)
    def forward(self, feat):
        return self.fc(self.pool(feat).flatten(1))

aux_head = AnyVesselHead().cuda()
net = net.cuda()



# Grab both layer4 features and the 'out' output for the classifier
from torchvision.models._utils import IntermediateLayerGetter

# Keep original backbone for DeepLab classifier
backbone_orig = net.backbone

# Create a separate IntermediateLayerGetter to grab layer4 features for aux head
return_layers = {"layer4": "layer4_out"}
backbone_for_aux = IntermediateLayerGetter(backbone_orig, return_layers=return_layers)

def forward_with_feats(images):
    """
    Returns segmentation logits and layer4 features for aux head.
    """
    # Layer4 features for aux head
    layer4_feats = backbone_for_aux(images)["layer4_out"]  # 2048 channels

    # Full DeepLab segmentation (original backbone + classifier)
    mask_logits = net(images)["out"]  # 1 channel segmentation

    return mask_logits, layer4_feats


# ------------------------------- LOSS & OPTIMIZER -------------------------------
bce = nn.BCEWithLogitsLoss()
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2*(probs*targets).sum(dim=(2,3)) + eps
    den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps
    return 1 - (num/den).mean()

def loss_fn(mask_logits, mask_true, any_logits):
    l_mask = 0.5*bce(mask_logits, mask_true) + 0.5*dice_loss(mask_logits, mask_true)
    any_true = (mask_true.sum(dim=(2,3))>0).float()
    l_any = bce(any_logits, any_true)
    return l_mask + 0.2*l_any, l_mask.detach(), l_any.detach()

optimizer = torch.optim.AdamW(list(net.parameters()) + list(aux_head.parameters()), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scaler = GradScaler(enabled=CFG["use_mixed_precision"])

# ------------------------------- TRAIN / VALID -------------------------------
def evaluate():
    net.eval(); aux_head.eval()
    dices=[]
    with torch.no_grad():
        for x,y,_ in valid_loader:
            x,y = x.cuda(), y.cuda()
            with autocast(enabled=CFG["use_mixed_precision"]):
                mask_logits, feats = forward_with_feats(x)
            p=(torch.sigmoid(mask_logits)>CFG["mask_threshold"]).float()
            for i in range(p.size(0)):
                dices.append(tm_dice(p[i,0].to(torch.bool), y[i,0].to(torch.bool)).item())
    return float(np.mean(dices)) if len(dices) else 0.0

best_dice=0.0
for epoch in range(1,CFG["epochs"]+1):
    net.train(); aux_head.train()
    pbar=tqdm(train_loader,desc=f"Epoch {epoch}")
    run_loss=0.0
    for x,y,_ in pbar:
        x,y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=CFG["use_mixed_precision"]):
            mask_logits, feats = forward_with_feats(x)
            any_logits = aux_head(feats)
            loss,lm,la = loss_fn(mask_logits, y, any_logits)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        run_loss = 0.98*run_loss + 0.02*loss.item() if run_loss>0 else loss.item()
        pbar.set_postfix(loss=f"{run_loss:.4f}")
    val_dice = evaluate()
    print(f"val dice: {val_dice:.4f}")
    if val_dice>best_dice:
        best_dice=val_dice
        torch.save({"net":net.state_dict(),"aux":aux_head.state_dict()}, OUTDIR/"best_deeplab50.pt")
        print("Saved best.")

# ------------------------------- INFERENCE -------------------------------
def preprocess_for_infer(img, size):
    tf = A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
    out=tf(image=img[...,None])
    x=out["image"].float().permute(2,0,1).repeat(3,1,1)
    return x, out

@torch.no_grad()
def predict_batch(imgs3):
    imgs3=imgs3.cuda()
    with autocast(enabled=CFG["use_mixed_precision"]):
        mlog, feats = forward_with_feats(imgs3)
        probs=torch.sigmoid(mlog)
        anyp=torch.sigmoid(aux_head(feats)).squeeze(1)
        if CFG["tta"]:
            mlog2, feats2 = forward_with_feats(torch.flip(imgs3,dims=[-1]))
            pr2=torch.sigmoid(mlog2); pr2=torch.flip(pr2,dims=[-1])
            ap2=torch.sigmoid(aux_head(feats2)).squeeze(1)
            mlog3, feats3 = forward_with_feats(torch.flip(imgs3,dims=[-2]))
            pr3=torch.sigmoid(mlog3); pr3=torch.flip(pr3,dims=[-2])
            ap3=torch.sigmoid(aux_head(feats3)).squeeze(1)
            probs=(probs+pr2+pr3)/3.0
            anyp=(anyp+ap2+ap3)/3.0
    return probs[:,0].cpu().numpy(), anyp.cpu().numpy()


#from glob import glob

# Ensure your model and aux_head are loaded with the best weights if not already in memory:
# checkpoint = torch.load("./outputs/best_deeplab50.pt", map_location='cuda')
# net.load_state_dict(checkpoint["net"])
# aux_head.load_state_dict(checkpoint["aux"])
# net.eval(); aux_head.eval()
from pathlib import Path
DATA = Path("/kaggle/input/blood-vessel-segmentation")
TEST = DATA/"test"

from glob import glob
ids, rles, any_flags, any_probs = [], [], [], []
test_paths = sorted(glob(str(TEST / "*/*.tif")))

for path in tqdm(test_paths, desc="Inference"):
    img = read_tiff(path)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = img[..., None]
    x, _ = preprocess_for_infer(img, CFG["img_size"])
    x = x.unsqueeze(0)
    probs, anyp = predict_batch(x)
    mask = (probs[0] > CFG["mask_threshold"]).astype(np.uint8)
    rle = rle_encode(mask)
    img_id = Path(path).stem
    ids.append(img_id)
    rles.append(rle)
    any_flags.append(int(anyp[0] > CFG["any_threshold"]))
    any_probs.append(float(anyp[0]))

sub = pd.DataFrame({"id": ids, "rle": rles})
sub.to_csv("submission.csv", index=False)
aux = pd.DataFrame({"id": ids, "has_vessel": any_flags, "prob_any_vessel": any_probs})
aux.to_csv("slice_posneg.csv", index=False)
print(f"Saved submission.csv ({len(sub)} rows) and slice_posneg.csv")


