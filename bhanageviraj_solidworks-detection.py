import os, glob, random, warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


try:
    import timm
    print("timm version:", timm.__version__)
except:
    raise ImportError("Please install timm")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


class Config:
    SEED = 42
    IMG_SIZE = 384
    BATCH_SIZE = 16
    EPOCHS = 10
    N_FOLDS = 5
    LR = 3e-4
    WD = 1e-4
    model_name = "tf_efficientnetv2_s"
    num_workers = 2

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(Config.SEED)



DATA_DIR = "/kaggle/input/solidworks-ai-hackathon"
train_csv_path = os.path.join(DATA_DIR, "train_labels.csv")

train_candidates = [
    os.path.join(DATA_DIR, "train", "train"),
    os.path.join(DATA_DIR, "train")
]
test_candidates = [
    os.path.join(DATA_DIR, "test", "test"),
    os.path.join(DATA_DIR, "test")
]

TRAIN_IMG_DIR = next(p for p in train_candidates if os.path.isdir(p))
TEST_IMG_DIR  = next(p for p in test_candidates if os.path.isdir(p))

print("Train images:", len(os.listdir(TRAIN_IMG_DIR)))
print("Test images :", len(os.listdir(TEST_IMG_DIR)))



df = pd.read_csv(train_csv_path)
cols = ["bolt", "locatingpin", "nut", "washer"]

df[cols] = df[cols].astype(int)

print(df.head())
print("\nClass ranges:")

ncls = {c: df[c].max() + 1 for c in cols}
print(ncls)



train_tfms = T.Compose([
    T.Resize(Config.IMG_SIZE),
    T.CenterCrop(Config.IMG_SIZE),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.RandomRotation(15),
    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

valid_tfms = T.Compose([
    T.Resize(Config.IMG_SIZE),
    T.CenterCrop(Config.IMG_SIZE),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])




class PartsDataset(Dataset):
    def __init__(self, df, img_dir, tfm, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm = tfm
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.loc[idx, "image_name"]
        path = os.path.join(self.img_dir, name)

        assert os.path.exists(path), f"Missing image: {name}"

        img = Image.open(path).convert("RGB")
        img = self.tfm(img)

        if self.is_test:
            return img, name

        label = torch.tensor(self.df.loc[idx, cols].values.astype(np.int64),dtype=torch.long)
        return img, label



class MultiHeadCounter(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.backbone = timm.create_model(
            Config.model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        in_f = self.backbone.num_features

        self.heads = nn.ModuleDict({
            k: nn.Linear(in_f, ncls[k]) for k in cols
        })

    def forward(self, x):
        f = self.backbone(x)
        return {k: h(f) for k, h in self.heads.items()}




def train_one_epoch(model, loader, opt, scaler):
    model.train()
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
            out = model(x)
            loss = sum(
                F.cross_entropy(out[c], y[:, i])
                for i, c in enumerate(cols)
            )

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * x.size(0)

    return total / len(loader.dataset)

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total, correct = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)

        loss = sum(
            F.cross_entropy(out[c], y[:, i])
            for i, c in enumerate(cols)
        )

        preds = torch.stack(
            [out[c].argmax(1) for c in cols], dim=1
        )

        correct += (preds == y).all(1).sum().item()
        total += loss.item() * x.size(0)

    acc = correct / len(loader.dataset)
    return total / len(loader.dataset), acc



kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
saved_models = []

for fold, (tr, va) in enumerate(kf.split(df)):
    print(f"\n===== Fold {fold+1} =====")

    ds_tr = PartsDataset(df.iloc[tr], TRAIN_IMG_DIR, train_tfms)
    ds_va = PartsDataset(df.iloc[va], TRAIN_IMG_DIR, valid_tfms)

    dl_tr = DataLoader(ds_tr, Config.BATCH_SIZE, True, num_workers=2)
    dl_va = DataLoader(ds_va, Config.BATCH_SIZE, False, num_workers=2)

    model = MultiHeadCounter(ncls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), Config.LR, weight_decay=Config.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, Config.EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE=="cuda"))

    best = 0
    path = f"fold_{fold}.pt"
    saved_models.append(path)

    for ep in range(Config.EPOCHS):
        tr_l = train_one_epoch(model, dl_tr, opt, scaler)
        va_l, va_a = validate(model, dl_va)
        sch.step()

        print(f"Ep {ep+1}: Tr {tr_l:.4f} | Va {va_l:.4f} | Acc {va_a:.4f}")

        if va_a > best:
            best = va_a
            torch.save(model.state_dict(), path)
            print("Saved best")

    del model
    torch.cuda.empty_cache()



test_files = sorted(glob.glob(TEST_IMG_DIR + "/*.png"))
df_test = pd.DataFrame({"image_name": [os.path.basename(x) for x in test_files]})

ds_test = PartsDataset(df_test, TEST_IMG_DIR, valid_tfms, is_test=True)
dl_test = DataLoader(ds_test, Config.BATCH_SIZE, False)

models = []
for p in saved_models:
    m = MultiHeadCounter(ncls).to(DEVICE)
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    m.eval()
    models.append(m)

rows = []

with torch.no_grad():
    for x, names in dl_test:
        x = x.to(DEVICE)
        xf = torch.flip(x, [3])

        probs = {c: torch.zeros(x.size(0), ncls[c]).to(DEVICE) for c in cols}

        for m in models:
            o1, o2 = m(x), m(xf)
            for c in cols:
                probs[c] += F.softmax(o1[c],1) + F.softmax(o2[c],1)

        for i, n in enumerate(names):
            rows.append([
                n,
                *(probs[c][i].argmax().item() for c in cols)
            ])



sub = pd.DataFrame(rows, columns=["image_name"] + cols)
sub.to_csv("submission.csv", index=False)

sub.head()


