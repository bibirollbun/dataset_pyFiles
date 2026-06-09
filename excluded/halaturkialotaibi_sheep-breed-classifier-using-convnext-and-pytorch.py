# Settings
CFG = {
    "root_dir": "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images",
    "img_size": 224,                 
    "batch_size": 4,                 
    "epochs": 5,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "model_name": "convnext_tiny_in22k",  
    "seed": 42,
    "amp": True,
    "accum": 4,                      
}

# import libraries
import os, gc, random, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import torch, timm, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def seed_everything(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything(CFG["seed"])



# Data
train_df = pd.read_csv(Path(CFG["root_dir"]) / "train_labels.csv")
img_dir = Path(CFG["root_dir"]) / "train"
actual = {f.name for f in img_dir.iterdir() if f.suffix.lower() in [".jpg",".jpeg",".png"]}
train_df = train_df[train_df.filename.isin(actual)].reset_index(drop=True)

labels = sorted(train_df.label.unique())
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
train_df["encoded"] = train_df.label.map(label2id)
n_classes = len(labels)

tr_tfms = T.Compose([
    T.RandomResizedCrop(CFG["img_size"], scale=(0.8,1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tfms = T.Compose([
    T.Resize((CFG["img_size"], CFG["img_size"])),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class SheepDS(Dataset):
    def __init__(self, df, root, tfms): self.df=df; self.root=root; self.tfms=tfms
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row=self.df.iloc[i]
        img = Image.open(self.root/row.filename).convert("RGB")
        return self.tfms(img), torch.tensor(row.encoded).long()

val_pct=0.1
val_df=train_df.sample(frac=val_pct, random_state=CFG["seed"]); trn_df=train_df.drop(val_df.index)
trn_dl = DataLoader(SheepDS(trn_df,img_dir,tr_tfms), batch_size=CFG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(SheepDS(val_df,img_dir,val_tfms), batch_size=CFG["batch_size"]*2, shuffle=False, num_workers=2, pin_memory=True)



# The model
model = timm.create_model(CFG["model_name"], pretrained=True, num_classes=n_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scaler = torch.cuda.amp.GradScaler(enabled=CFG["amp"])


# Training
for epoch in range(CFG["epochs"]):
    model.train(); loss_sum=tot=correct=0
    opt.zero_grad(set_to_none=True)
    for step,(x,y) in enumerate(tqdm(trn_dl, desc=f"Ep {epoch+1}")):
        x,y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=CFG["amp"]):
            out = model(x); loss = criterion(out,y)/CFG["accum"]
        scaler.scale(loss).backward()
        if (step+1)%CFG["accum"]==0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        loss_sum+=loss.item()*CFG["accum"]; tot+=x.size(0); correct+=(out.argmax(1)==y).sum().item()
    print(f"  Train acc {correct/tot:.4f}")


# Verification
model.eval()
vloss, vtot, vcorr = 0, 0, 0

with torch.no_grad():
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=CFG["amp"]):
            out = model(x)
            l   = criterion(out, y)

        vloss += l.item() * x.size(0)
        vtot  += x.size(0)
        vcorr += (out.argmax(1) == y).sum().item()

print(f"  Val   acc {vcorr / vtot:.4f}")

torch.cuda.empty_cache()
gc.collect()




# Prepare test folder and images
test_dir = Path(CFG["root_dir"]) / "test"
test_files = sorted([f.name for f in test_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

# Create a DataFrame with a dummy encoded column for SheepDS compatibility
test_df = pd.DataFrame({
    "filename": test_files,
    "encoded": [0] * len(test_files)  # ← is necessary because __getitem__ requires it.
})

# Create a DataLoader
test_dl = DataLoader(
    SheepDS(test_df, test_dir, val_tfms),
    batch_size=CFG["batch_size"] * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Implement the prediction
model.eval()
preds = []

with torch.no_grad():
    for x, _ in tqdm(test_dl):
        x = x.to(device)
        with torch.cuda.amp.autocast(enabled=CFG["amp"]):
            out = model(x)
        preds.extend(out.argmax(1).cpu().numpy())

# Convert predictions to lineage names using id2label
pred_labels = [id2label[p] for p in preds]




# Create the submission.csv file
submission = pd.DataFrame({
    "filename": test_files,
    "label": pred_labels
})

submission.to_csv("submission.csv", index=False)
print("✅ Saved submission.csv with", len(submission), "rows.")


