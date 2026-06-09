
CFG = {
    "seed": 42,
    "strategy": "full_ft",        # "full_ft" | "lp_unfreeze"
    "model_name": "convnext_tiny",# "convnext_tiny" | "resnet18"
    # <<< Change the path below to your best pre-trained weights (*.pt) from the large dataset >>>
    "ckpt_path": "/kaggle/input/inatweight/convnext_tiny_inat_pretrain_best.pt",

    # Data Path (Kaggle's official Plant Seedlings dataset)
    "data_root": "/kaggle/input/plant-seedlings-classification",
    "val_ratio": 0.1,

    # Training Configuration
    "img_size": 224,
    "batch_size": 128,
    "num_workers": 2,
    "epochs_fullft": 30,          # Number of epochs for full fine-tuning
    "epochs_lp": 5,               # Number of epochs for linear probing (backbone frozen)
    "epochs_unfreeze": 15,        # Number of epochs for gradual unfreezing
    "lr_head": 1e-3,              # Learning rate for the classification head
    "lr_backbone": 3e-4,          # Learning rate for the backbone (used in full_ft or unfreeze phase)
    "weight_decay": 1e-2,
    "label_smoothing": 0.1,
    "mixed_precision": True,

    "out_root": "/kaggle/working/seedlings_transfer",
    "run_name": None,             # If None, automatically named based on strategy/model
}


import os, math, random, time, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

import torchvision
from torchvision import transforms as T, models

# ---------------------------
# Basic Utilities
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(p: Path):
    """Create directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)

def build_seedlings_split(root: Path, val_ratio=0.1, seed=42):
    """
    Creates a stratified train/validation split from the Plant Seedlings dataset directory.
    """
    train_dir = root/"train"
    assert train_dir.exists(), f"Plant Seedlings data not found: {train_dir}"
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    c2i = {c:i for i,c in enumerate(classes)}
    items=[]
    exts={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    for c in classes:
        for p in (train_dir/c).glob("*.*"):
            if p.suffix.lower() in exts:
                items.append((p, c2i[c]))
    y = np.array([b for _,b in items]); idx = np.arange(len(items))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr, va = next(sss.split(idx,y))
    return classes, [items[i] for i in tr], [items[i] for i in va]

class DS(Dataset):
    """Custom Dataset for loading images."""
    def __init__(self, items, tfm): self.items, self.tfm = items, tfm
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return x, y, str(p)

def get_tfms(size):
    """
    Returns training and validation transforms.
    """
    train = T.Compose([
        T.Resize(int(size*1.14)),
        T.RandomResizedCrop(size, scale=(0.85,1.0), ratio=(3/4,4/3)),
        T.RandomHorizontalFlip(0.5),
        T.RandAugment(2, 9),
        T.ToTensor(),
        T.RandomErasing(p=0.1),
        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    valid = T.Compose([
        T.Resize(int(size*1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    return train, valid


def accuracy(out, tgt):
    """Calculates accuracy."""
    with torch.no_grad():
        return (out.argmax(1)==tgt).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, num_classes):
    """
    Evaluates the model on a given dataloader.
    Returns loss, accuracy, F1-score, confusion matrix, and predictions.
    """
    model.eval()
    all_pred=[]; all_true=[]; loss_sum=acc_sum=n=0
    crit = nn.CrossEntropyLoss()
    for x,y,_ in loader:
        x=x.to(device); y=y.to(device)
        with torch.amp.autocast("cuda", enabled=CFG["mixed_precision"]):
            lo = model(x); ls = crit(lo,y)
        bs=x.size(0)
        loss_sum += ls.item()*bs
        acc_sum  += accuracy(lo,y)*bs
        n += bs
        all_pred.append(lo.argmax(1).cpu().numpy())
        all_true.append(y.cpu().numpy())
    pred = np.concatenate(all_pred); true = np.concatenate(all_true)
    f1 = f1_score(true, pred, average="macro")
    cm = confusion_matrix(true, pred, labels=list(range(num_classes)))
    return loss_sum/n, acc_sum/n, f1, cm, true, pred

def save_cm_and_report(cm, y_true, y_pred, class_names, out_dir: Path, epoch: int):
    """
    Saves the confusion matrix and a per-class classification report to CSV files.
    """
    ensure_dir(out_dir)
    df_cm = pd.DataFrame(cm, columns=class_names)
    df_cm.insert(0, "true\\pred", class_names)
    df_cm.to_csv(out_dir / f"confusion_matrix_epoch{epoch:03d}.csv", index=False)

    rep = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                 target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.reset_index().rename(columns={"index":"class"}).to_csv(
        out_dir / f"per_class_report_epoch{epoch:03d}.csv", index=False
    )

def save_curve(curves, out_csv: Path):
    """Saves the training history (loss, accuracy, F1) to a CSV file."""
    pd.DataFrame(curves).to_csv(out_csv, index=False)


def build_model(num_classes, name="convnext_tiny"):
    """
    Builds the specified model with a new classification head.
    """
    n = name.lower()
    if n == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    elif n == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported model_name: {name}")

def load_backbone_from_ckpt(model, ckpt_path, model_name):
    """
    Loads only the backbone weights, discarding the old classification head.
    The checkpoint can be a state_dict or a dictionary containing a 'state_dict' key.
    """
    assert Path(ckpt_path).exists(), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)  # Compatible with both state_dict formats

    new_sd = model.state_dict()
    drop_keys = []
    if model_name == "convnext_tiny":
        drop_prefixes = ["classifier."]
    else:  # resnet18
        drop_prefixes = ["fc."]

    # Remove the old classification head from the checkpoint state_dict
    for k in list(sd.keys()):
        if any(k.startswith(p) for p in drop_prefixes):
            drop_keys.append(k)
            sd.pop(k)

    # Load only keys that match in name and shape
    matched = {k: v for k, v in sd.items() if (k in new_sd and new_sd[k].shape == v.shape)}
    missing = [k for k in new_sd.keys() if k not in matched]
    print(f"[ckpt] load backbone: matched={len(matched)} | missing(new head etc.)={len(missing)} | dropped_old_head={len(drop_keys)}")

    new_state = model.state_dict()
    new_state.update(matched)
    model.load_state_dict(new_state)
    return model


def set_backbone_trainable(model, model_name, trainable: bool):
    """
    Sets the entire backbone to be trainable or frozen.
    Also handles the train/eval mode of normalization layers.
    """
    if model_name=="convnext_tiny":
        backbone = [model.features]  # ConvNeXt's backbone is the 'features' module
    else:
        backbone = [nn.Sequential(model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4)]
    
    for m in backbone:
        for p in m.parameters(): p.requires_grad = trainable
        # For BN/LN: set to eval() when frozen, and train() when trainable
        for mm in m.modules():
            if isinstance(mm, (nn.BatchNorm2d, nn.LayerNorm)):
                mm.eval() if not trainable else mm.train()

def unfreeze_last_stages(model, model_name, stages=1):
    """
    Unfreezes the last N stages of the backbone. Useful for the gradual unfreezing strategy.
    convnext_tiny: The 'features' module contains sequential blocks.
    resnet18: Has 4 main layers (layer1 to layer4).
    """
    set_backbone_trainable(model, model_name, trainable=False)  # Freeze everything first
    if model_name=="convnext_tiny":
        blocks = model.features  # Sequential module
        to_unfreeze = list(range(len(blocks)-stages, len(blocks)))
        for i in to_unfreeze:
            for p in blocks[i].parameters(): p.requires_grad=True
            for mm in blocks[i].modules():
                if isinstance(mm, (nn.BatchNorm2d, nn.LayerNorm)): mm.train()
    else: # resnet18
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        for l in layers[-stages:]:
            for p in l.parameters(): p.requires_grad=True
            for mm in l.modules():
                if isinstance(mm, nn.BatchNorm2d): mm.train()


def make_optimizer(model, model_name, lr_backbone, lr_head, wd=1e-2, full=False):
    """
    Creates an AdamW optimizer. Can set a smaller learning rate for the backbone.
    """
    if model_name=="convnext_tiny":
        head_params = list(model.classifier.parameters())
        bb_params   = list(model.features.parameters())
    else: # resnet18
        head_params = list(model.fc.parameters())
        bb_params   = [p for n,p in model.named_parameters() if not n.startswith("fc.")]
    
    params=[]
    if full:
        # Discriminative LR: use a smaller LR for the backbone
        params=[{"params": bb_params, "lr": lr_backbone},
                {"params": head_params, "lr": lr_head}]
    else: # Only train the head
        params=[{"params": head_params, "lr": lr_head}]
        
    return torch.optim.AdamW(params, weight_decay=wd)

def train_one_phase(model, train_loader, valid_loader, epochs, opt, crit, curves, out_met_dir, class_names, start_epoch=1):
    """
    Runs the main training and validation loop for a specified number of epochs.
    """
    scaler = torch.amp.GradScaler("cuda", enabled=CFG["mixed_precision"])
    best_f1=-1; best_ep=-1; best_path = out_met_dir.parent / f"{run_name}_best.pt"
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,epochs))
    
    for e in range(start_epoch, start_epoch+epochs):
        model.train(); t0=time.time()
        loss_sum=acc_sum=n=0
        for x,y,_ in train_loader:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=CFG["mixed_precision"]):
                lo = model(x)
                ls = crit(lo,y)
            scaler.scale(ls).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            
            loss_sum += ls.item()*x.size(0)
            acc_sum  += accuracy(lo,y)*x.size(0)
            n += x.size(0)
        sched.step()

        tr_loss, tr_acc = loss_sum/n, acc_sum/n
        va_loss, va_acc, va_f1, cm, y_true, y_pred = evaluate(model, valid_loader, num_classes=len(class_names))
        print(f"[{run_name}] Epoch {e:02d}/{start_epoch+epochs-1:02d} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | time={int(time.time()-t0)}s")

        curves["epoch"].append(e)
        curves["train_loss"].append(tr_loss)
        curves["train_acc"].append(tr_acc)
        curves["val_acc"].append(va_acc)
        curves["val_f1"].append(va_f1)
        save_curve(curves, out_met_dir/"train_curve.csv")
        save_cm_and_report(cm, y_true, y_pred, class_names, out_met_dir, e)

        if va_f1 > best_f1:
            best_f1, best_ep = va_f1, e
            torch.save({"state_dict": model.state_dict(), "classes": class_names}, best_path)
            
    return best_f1, best_ep, best_path


# ---------------------------
# Main Execution
# ---------------------------
set_seed(CFG["seed"])

# --- Data Loading ---
root = Path(CFG["data_root"])
classes, tr_items, va_items = build_seedlings_split(root, CFG["val_ratio"], CFG["seed"])
num_classes = len(classes)
train_tfm, valid_tfm = get_tfms(CFG["img_size"])
train_ds, valid_ds = DS(tr_items, train_tfm), DS(va_items, valid_tfm)
train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

# --- Naming and Output Directory ---
if CFG["run_name"] is None:
    run_name = f"seedlings_{CFG['model_name']}_{CFG['strategy']}"
else:
    run_name = CFG["run_name"]
out_dir = Path(CFG["out_root"])/run_name
met_dir = out_dir/"metrics"
ensure_dir(met_dir)

# --- Model Creation and Weight Loading ---
# Build a new model with a head for the Seedlings dataset, then load the pre-trained backbone.
model = build_model(num_classes, CFG["model_name"]).to(device)
model = load_backbone_from_ckpt(model, CFG["ckpt_path"], CFG["model_name"])

# --- Log Resource Info ---
with open(met_dir/"resource.json", "w") as f:
    json.dump({
        "strategy": CFG["strategy"],
        "model": CFG["model_name"],
        "img_size": CFG["img_size"],
        "batch_size": CFG["batch_size"],
        "params": sum(p.numel() for p in model.parameters())
    }, f)

# --- Training Execution based on Strategy ---
curves = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
best_f1 = -1
best_ep = -1

if CFG["strategy"] == "full_ft":
    # Full fine-tuning: make backbone trainable; use discriminative LR
    print("--- Starting Full Fine-Tuning ---")
    set_backbone_trainable(model, CFG["model_name"], trainable=True)
    opt = make_optimizer(model, CFG["model_name"], CFG["lr_backbone"], CFG["lr_head"],
                         wd=CFG["weight_decay"], full=True)
    best_f1, best_ep, best_path = train_one_phase(
        model, train_loader, valid_loader, CFG["epochs_fullft"], opt, crit, curves, met_dir, classes
    )

elif CFG["strategy"] == "lp_unfreeze":
    # STEP 1: Linear Probing - freeze backbone, train only the head
    print("--- Starting Phase 1: Linear Probing ---")
    set_backbone_trainable(model, CFG["model_name"], trainable=False)
    opt = make_optimizer(model, CFG["model_name"], CFG["lr_backbone"], CFG["lr_head"],
                         wd=CFG["weight_decay"], full=False)
    best_f1, best_ep, best_path = train_one_phase(
        model, train_loader, valid_loader, CFG["epochs_lp"], opt, crit, curves, met_dir, classes, start_epoch=1
    )

    # STEP 2: Gradual Unfreezing - unfreeze the last N stages and train with discriminative LR
    print("\n--- Starting Phase 2: Gradual Unfreezing ---")
    unfreeze_last_stages(model, CFG["model_name"], stages=1 if CFG["model_name"]=="convnext_tiny" else 2)
    opt = make_optimizer(model, CFG["model_name"], CFG["lr_backbone"], CFG["lr_head"],
                         wd=CFG["weight_decay"], full=True)
    b2_f1, b2_ep, best_path_2 = train_one_phase(
        model, train_loader, valid_loader, CFG["epochs_unfreeze"], opt, crit, curves, met_dir, classes,
        start_epoch=(CFG["epochs_lp"]+1)
    )
    if b2_f1 > best_f1:
        best_f1, best_ep = b2_f1, b2_ep

else:
    raise ValueError("CFG['strategy'] must be 'full_ft' or 'lp_unfreeze'")

print(f"\nDone. Best macro-F1={best_f1:.4f} @ epoch {best_ep}.")
print("Outputs ->", out_dir)

