# ================================================================
# Grand X-Ray Multilabel — ConvNeXt Base @512 with DataParallel
# ================================================================
import os, math, random, warnings, cv2
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torchvision.transforms as T
import timm
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
SEED = 42
IMG_SIZE = 512
BATCH_SIZE = 8  # safe for T4; try 16 if enough VRAM
EPOCHS = 7
WARMUP_EPOCHS = 1
BASE_LR = 2e-5
HEAD_LR = 8e-5
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.999
FOCAL_GAMMA = 2.0
NUM_WORKERS = 4

TRAIN_CSV = "/kaggle/input/grand-xray-slam-division-a/train1.csv"
TRAIN_DIR = "/kaggle/input/grand-xray-slam-division-a/train1"
TEST_DIR  = "/kaggle/input/grand-xray-slam-division-a/test1"

LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

SAVE_PATH = "convnext_best.pth"  # where to save model weights

# -------------------------
# Repro
# -------------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Dataset
# -------------------------
class XRayDataset(Dataset):
    def __init__(self, df, image_dir, img_size=IMG_SIZE, is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.is_train = is_train
        self.img_size = img_size
        if is_train:
            self.tf = T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        else:
            self.tf = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, row['Image_name'])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tf(img)
        if self.is_train:
            y = torch.tensor(row[LABEL_COLS].values.astype(np.float32), dtype=torch.float32)
            return img, y
        else:
            return img, row['Image_name']

# -------------------------
# Focal Loss
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = (1 - pt)**self.gamma * bce
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

# -------------------------
# EMA
# -------------------------
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

# -------------------------
# Data prep
# -------------------------
df = pd.read_csv(TRAIN_CSV)
df[LABEL_COLS] = df[LABEL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

if 'No Finding' in LABEL_COLS:
    others = [c for c in LABEL_COLS if c != 'No Finding']
    df['No Finding'] = (df[others].sum(axis=1) == 0).astype(int)

df['sum_labels'] = df[LABEL_COLS].sum(axis=1)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED,
                                   stratify=np.clip(df['sum_labels'], 0, 5))
train_df = train_df.drop(columns=['sum_labels']).reset_index(drop=True)
val_df = val_df.drop(columns=['sum_labels']).reset_index(drop=True)

pos_counts = train_df[LABEL_COLS].sum()
neg_counts = len(train_df) - pos_counts
alpha = torch.tensor((neg_counts/(pos_counts+1e-6)).values, dtype=torch.float32)

train_ds = XRayDataset(train_df, TRAIN_DIR, is_train=True)
val_ds = XRayDataset(val_df, TRAIN_DIR, is_train=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# -------------------------
# Model + training helpers
# -------------------------
def make_model():
    return timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True, num_classes=len(LABEL_COLS))

def param_groups(m):
    module = m.module if isinstance(m, nn.DataParallel) else m
    head = module.get_classifier()
    head_ids = set(map(id, head.parameters()))
    base_params = [p for p in module.parameters() if id(p) not in head_ids]
    return [
        {"params": base_params, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": head.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
    ]

criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)

def evaluate(model):
    model.eval()
    logits_all, targets_all = [], []
    with torch.no_grad():
        for imgs, y in val_loader:
            imgs, y = imgs.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
                out = model(imgs)
            logits_all.append(out.detach().cpu().numpy())
            targets_all.append(y.detach().cpu().numpy())
    logits_all = np.concatenate(logits_all,0)
    targets_all = np.concatenate(targets_all,0)
    probs = 1/(1+np.exp(-logits_all))
    auc = roc_auc_score(targets_all, probs, average="macro")
    return auc

# -------------------------
# Train ConvNeXt Base (with DataParallel)
# -------------------------
model = make_model().to(device)

# ✅ wrap with DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"✅ Using {torch.cuda.device_count()} GPUs for DataParallel")
    model = nn.DataParallel(model)

optimizer = optim.AdamW(param_groups(model))
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda e: (e+1)/WARMUP_EPOCHS if e<WARMUP_EPOCHS
    else 0.5*(1+math.cos(math.pi*(e-WARMUP_EPOCHS)/max(1,EPOCHS-WARMUP_EPOCHS)))
)
scaler = torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
ema = EMA(model)
best_auc=-1

for epoch in range(EPOCHS):
    model.train()
    running=0
    pbar=tqdm(train_loader,desc=f"ConvNeXt Epoch {epoch+1}/{EPOCHS}")
    for imgs,y in pbar:
        imgs,y=imgs.to(device),y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
            out=model(imgs)
            loss=criterion(out,y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP_NORM)
        scaler.step(optimizer); scaler.update()
        ema.update(model)
        running+=loss.item()
        pbar.set_postfix(loss=running/max(1,pbar.n))
    scheduler.step()
    ema.apply_to(model)
    auc=evaluate(model)
    print(f"ConvNeXt Epoch {epoch+1}: AUC={auc:.4f}")
    # ✅ save correctly with DataParallel
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    if auc>best_auc:
        best_auc=auc
        torch.save(state_dict, SAVE_PATH)
print(f"Best AUC for ConvNeXt: {best_auc:.4f}")

# -------------------------
# Inference with TTA
# -------------------------
class TestDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df=df.reset_index(drop=True)
        self.image_dir=image_dir
        self.tf=T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self):return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        path=os.path.join(self.image_dir,row['Image_name'])
        img=cv2.imread(path,cv2.IMREAD_COLOR)
        if img is None:
            img=np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=self.tf(img)
        return img,row['Image_name']

# Load best weights for inference
model_infer = make_model().to(device)
if torch.cuda.device_count() > 1:
    model_infer = nn.DataParallel(model_infer)

state_dict = torch.load(SAVE_PATH,map_location=device)
if isinstance(model_infer, nn.DataParallel):
    model_infer.module.load_state_dict(state_dict)
else:
    model_infer.load_state_dict(state_dict)
model_infer.eval()

test_names=sorted(os.listdir(TEST_DIR))
test_df=pd.DataFrame({'Image_name':test_names})
test_ds=TestDataset(test_df,TEST_DIR)
test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)

@torch.no_grad()
def predict_tta(model,loader):
    preds_all=[];names_all=[]
    for imgs,names in tqdm(loader,desc="Predicting ConvNeXt"):
        imgs=imgs.to(device)
        logits1=model(imgs)
        imgs_flipped=torch.flip(imgs,dims=[3])
        logits2=model(imgs_flipped)
        logits=0.5*(logits1+logits2)
        probs=torch.sigmoid(logits).cpu().numpy()
        preds_all.append(probs)
        names_all.extend(names)
    return np.concatenate(preds_all,axis=0),names_all

preds,names=predict_tta(model_infer,test_loader)
sub=pd.DataFrame(preds,columns=LABEL_COLS)
sub.insert(0,"Image_name",names)
sub.to_csv("submission.csv",index=False)
print("✅ Created submission.csv")
print(sub.head())





