# Core
import os, gc, math, warnings, json, random
from pathlib import Path

# Data / math
import numpy as np
import pandas as pd

# Metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# PyTorch
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# HF Transformers
from transformers import (
    ViTFeatureExtractor, ViTForImageClassification,
    TrainingArguments, Trainer
)

# Albumentations
import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2

# PyTorch-Lightning
import pytorch_lightning as pl

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


IMG_SIZE = 224
N_CLASSES = 7
FOLDS = 5
SEED  = 42

EFF_EPOCHS = 15     # EfficientNet-V2-S
PL_EPOCHS  = 4      # ViT + Lightning
HF_EPOCHS  = 30     # ViT + HF Trainer

BATCH_EFF = 32
BATCH_VIT = 16      # ViT needs more VRAM
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)


DATA_ROOT = Path("/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR  = DATA_ROOT / "test"
TRAIN_CSV = DATA_ROOT / "train_labels.csv"

df = pd.read_csv(TRAIN_CSV)
label2id = {lbl:i for i,lbl in enumerate(sorted(df.label.unique()))}
id2label = {i:l for l,i in label2id.items()}
df["label_idx"] = df["label"].map(label2id)
df.head()


def get_train_aug():
    return A.Compose([
        A.Resize(256,256),
        A.RandomResizedCrop((IMG_SIZE,IMG_SIZE), scale=(0.8,1.0), ratio=(0.75,1.33)),
        A.HorizontalFlip(p=.5),
        A.Affine(rotate=(-15,15), shear=(-5,5), p=.4),
        A.ColorJitter(.2,.2,.2,.1,p=.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=.5),
        A.Normalize(), ToTensorV2(),
    ])

def get_val_aug():
    return A.Compose([
        A.Resize(256,256),
        A.CenterCrop(IMG_SIZE,IMG_SIZE),
        A.Normalize(), ToTensorV2(),
    ])


def get_train_aug_hf():
    return A.Compose([
        A.RandomResizedCrop((IMG_SIZE,IMG_SIZE), scale=(0.8,1.0)),
        A.HorizontalFlip(p=.5),
        A.Affine(rotate=(-15,15), shear=(-5,5), p=.4),
        A.ColorJitter(.2,.2,.2,.1,p=.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=.5)
    ])

def get_val_aug_hf():
    return A.Compose([
        A.Resize(IMG_SIZE,IMG_SIZE),
        A.CenterCrop(IMG_SIZE,IMG_SIZE),
    ])


class SheepDS(Dataset):
    "Torchvision/EfficientNet + Lightning ViT (tensor, already normalized)"
    def __init__(self, df, root, split="train"):
        self.df   = df.reset_index(drop=True)
        self.root = root
        self.aug  = get_train_aug() if split=="train" else get_val_aug()
        self.has_y= "label_idx" in self.df

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = cv2.cvtColor(cv2.imread(str(self.root/row.filename)), cv2.COLOR_BGR2RGB)
        img  = self.aug(image=img)["image"]        # CÃ—HÃ—W tensor
        return (img, row.label_idx) if self.has_y else (img, row.filename)


class SheepHFDataset(Dataset):
    "Albumentations (uint8) â†’ HF ViT extractor â†’ dict"
    def __init__(self, df, root, train=True):
        self.df   = df.reset_index(drop=True)
        self.root = root
        self.train= train
        self.aug  = get_train_aug_hf() if train else get_val_aug_hf()
        self.extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(str(self.root/row.filename)), cv2.COLOR_BGR2RGB)
        img = self.aug(image=img)["image"]         # still uint8 HWC
        batch = self.extractor(img, return_tensors="pt")
        item  = {k:v.squeeze(0) for k,v in batch.items()}
        item["labels"] = torch.tensor(row.label_idx if self.train else 0)
        return item


def save_npz(tag, oof, test, oof_labels, test_files):
    np.save(f"{tag}_oof.npy",  oof)
    np.save(f"{tag}_test.npy", test)
    np.save("oof_labels.npy",  oof_labels)
    # save filenames as unicode â†’ no pickle
    np.save("test_filenames.npy",
            np.asarray(test_files, dtype="U"))
    print(f"Saved â†’ {tag}_oof.npy / {tag}_test.npy")

def logits_to_labels(logits):
    return [id2label[i] for i in logits.argmax(1)]


def run_effnet():
    model = models.efficientnet_v2_s(weights="DEFAULT")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, N_CLASSES)
    model.to(device)

    oof  = np.zeros((len(df), N_CLASSES))
    test = np.zeros((len(os.listdir(TEST_DIR)), N_CLASSES))
    test_df = pd.DataFrame({"filename": sorted(os.listdir(TEST_DIR))})

    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold,(tr,va) in enumerate(skf.split(df, df.label_idx),1):
        print(f"\nğŸ”¹ EfficientNet fold {fold}/{FOLDS}")
        dl_tr = DataLoader(SheepDS(df.iloc[tr], TRAIN_DIR,"train"),
                           batch_size=BATCH_EFF, shuffle=True, num_workers=2)
        dl_va = DataLoader(SheepDS(df.iloc[va], TRAIN_DIR,"val"),
                           batch_size=BATCH_EFF, shuffle=False,num_workers=2)

        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2,factor=.5)
        best, best_logits = 0., None

        for epoch in range(1, EFF_EPOCHS+1):
            model.train()
            for x,y in dl_tr:
                x,y = x.to(device), y.to(device)
                opt.zero_grad(); nn.CrossEntropyLoss()(model(x),y).backward(); opt.step()

            # â”€â”€ validation
            model.eval(); preds, gts = [], []
            with torch.no_grad():
                for x,y in dl_va:
                    preds.append(torch.softmax(model(x.to(device)),1).cpu().numpy())
                    gts.extend(y.numpy())
            preds = np.concatenate(preds)
            f1 = f1_score(gts, preds.argmax(1), average="macro")
            sched.step(1-f1)
            if f1>best: best, best_logits = f1, preds
            print(f"  epoch {epoch:02}/{EFF_EPOCHS}  F1={f1:.4f}", end="\r")

        oof[va] = best_logits

        # â”€â”€ test
        model.eval(); fold_test=[]
        dl_test = DataLoader(SheepDS(test_df, TEST_DIR,"val"),
                             batch_size=BATCH_EFF, shuffle=False, num_workers=2)
        with torch.no_grad():
            for x,_ in dl_test:
                fold_test.append(torch.softmax(model(x.to(device)),1).cpu().numpy())
        test += np.concatenate(fold_test)/FOLDS

    save_npz("efficientnet", oof, test, df.label_idx.values, test_df.filename.values)
run_effnet()


class LitViT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=N_CLASSES, id2label=id2label, label2id=label2id)

    def forward(self,x): return self.model(pixel_values=x).logits
    def training_step(self,batch, _): x,y=batch; loss=nn.CrossEntropyLoss()(self(x),y);self.log("loss",loss)
    def configure_optimizers(self):   return torch.optim.Adam(self.parameters(), lr=2e-5)

def run_vit_pl():
    oof  = np.zeros((len(df), N_CLASSES))
    test = np.zeros((len(os.listdir(TEST_DIR)), N_CLASSES))
    test_df = pd.DataFrame({"filename": sorted(os.listdir(TEST_DIR))})

    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold,(tr,va) in enumerate(skf.split(df, df.label_idx),1):
        print(f"\nğŸ”¹ ViT-PL fold {fold}/{FOLDS}")
        model = LitViT()

        dl_tr = DataLoader(SheepDS(df.iloc[tr], TRAIN_DIR,"train"),
                           batch_size=BATCH_VIT, shuffle=True, num_workers=2)
        dl_va = DataLoader(SheepDS(df.iloc[va], TRAIN_DIR,"val"),
                           batch_size=BATCH_VIT, shuffle=False,num_workers=2)

        trainer = pl.Trainer(max_epochs=PL_EPOCHS, accelerator="gpu", devices=1,
                             precision="16-mixed", logger=False, enable_progress_bar=True)
        trainer.fit(model, dl_tr, dl_va)

        # â”€â”€ inference (model already on GPU)
        model.eval()
        preds=[]
        with torch.no_grad():
            for x,_ in dl_va:
                preds.append(torch.softmax(model(x),1).cpu().numpy())
        oof[va] = np.concatenate(preds)

        fold_test=[]
        dl_test=DataLoader(SheepDS(test_df, TEST_DIR,"val"),
                           batch_size=BATCH_VIT, shuffle=False,num_workers=2)
        with torch.no_grad():
            for x,_ in dl_test:
                fold_test.append(torch.softmax(model(x),1).cpu().numpy())
        test += np.concatenate(fold_test)/FOLDS
        del model; torch.cuda.empty_cache()

    save_npz("vitPL", oof, test, df.label_idx.values, test_df.filename.values)
run_vit_pl()


# ğŸ”§ PATCH: ViT-HF with focal-loss â€“ argument-safe for any ğŸ¤— Transformers â‰¥2.0
def run_vit_hf():
    """
    The other two models already work; this cell is the only one that
    failed because the local `transformers` build does NOT accept the
    newer TrainingArguments keyword `evaluation_strategy` (and maybe
    `save_strategy`, `disable_tqdm`, â€¦).

    The helper `_safe_TA` keeps only the kwargs that the installed
    version actually supports, so you never crash on TypeError again.
    """
    def _safe_TA(**kwargs):
        """filter out TrainingArguments kwargs that aren't supported"""
        valid = set(TrainingArguments.__init__.__code__.co_varnames)
        clean = {k: v for k, v in kwargs.items() if k in valid}
        return TrainingArguments(**clean)

    # ------------------------------------------------------------------
    oof  = np.zeros((len(df), N_CLASSES))
    test = np.zeros((len(os.listdir(TEST_DIR)), N_CLASSES))
    test_df = pd.DataFrame({"filename": sorted(os.listdir(TEST_DIR))})

    # class-balanced focal loss -------------------------------------------------
    class_w = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=df.label_idx),
        dtype=torch.float, device=device)

    def focal(logits, labels, Î³=1.5):
        ce = nn.CrossEntropyLoss(reduction="none")(logits, labels)
        pt = torch.exp(-ce)
        return ((1-pt)**Î³ * ce * class_w[labels]).mean()

    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **_):
            labels = inputs.pop("labels")
            out    = model(**inputs)
            loss   = focal(out.logits, labels)
            return (loss, out) if return_outputs else loss
    # --------------------------------------------------------------------------

    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(df, df.label_idx), 1):
        print(f"\nğŸ”¹ ViT-HF fold {fold}/{FOLDS}")

        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=N_CLASSES, id2label=id2label, label2id=label2id
        ).to(device)

        args = _safe_TA(
            output_dir      = f"vit_hf_fold{fold}",
            num_train_epochs= HF_EPOCHS,
            per_device_train_batch_size = BATCH_VIT,
            per_device_eval_batch_size  = BATCH_VIT,
            learning_rate   = 5e-5,
            fp16            = True,
            seed            = SEED,
            logging_steps   = 100,
            report_to       = [],          # silence TB / wandb
            dataloader_num_workers = 0,
        )

        trainer = FocalTrainer(
            model=model, args=args,
            train_dataset=SheepHFDataset(df.iloc[tr], TRAIN_DIR, train=True),
            eval_dataset =SheepHFDataset(df.iloc[va], TRAIN_DIR, train=False)
        )
        trainer.train()

        # â”€â”€ OOF logits
        val_logits = trainer.predict(
            SheepHFDataset(df.iloc[va], TRAIN_DIR, train=False)
        ).predictions
        oof[va] = torch.softmax(torch.tensor(val_logits), 1).cpu().numpy()

        # â”€â”€ Test logits
        tst_logits = trainer.predict(
            SheepHFDataset(test_df.assign(label_idx=0), TEST_DIR, train=False)
        ).predictions
        test += torch.softmax(torch.tensor(tst_logits), 1).cpu().numpy() / FOLDS

        del model, trainer; gc.collect(); torch.cuda.empty_cache()

    save_npz("vitHF", oof, test, df.label_idx.values, test_df.filename.values)

run_vit_hf()


# load arrays back (safety)
oof_eff = np.load("efficientnet_oof.npy")
oof_pl  = np.load("vitPL_oof.npy")
oof_hf  = np.load("vitHF_oof.npy")
y_true  = np.load("oof_labels.npy")

test_eff = np.load("efficientnet_test.npy")
test_pl  = np.load("vitPL_test.npy")
test_hf  = np.load("vitHF_test.npy")
test_files = np.load("test_filenames.npy")

test_logits = {"EffNet": test_eff, "ViT-PL": test_pl, "ViT-HF": test_hf}

for tag, oof in [("EffNet", oof_eff), ("ViT-PL", oof_pl), ("ViT-HF", oof_hf)]:
    f1 = f1_score(y_true, oof.argmax(1), average="macro")
    print(f"{tag:8s}  OOF F1 = {f1:.4f}")

    sub = pd.DataFrame({
        "filename": test_files,
        "label"   : logits_to_labels(test_logits[tag])   # â†� simple & safe
    })
    sub.to_csv(f"submission_{tag.lower().replace('-','_')}.csv", index=False)
    print(f"  â†³ wrote submission_{tag.lower().replace('-','_')}.csv")


def search_blend(oofs, names, grid=np.linspace(0,1,11)):
    best_f1, best_w = -1, None
    for a in grid:
        for b in grid:
            c = 1-a-b
            if c < 0: continue
            blend = a*oofs[0] + b*oofs[1] + c*oofs[2]
            f1 = f1_score(y_true, blend.argmax(1), average="macro")
            if f1 > best_f1: best_f1, best_w = f1, (a,b,c)
    print(f"Best { names }  F1={best_f1:.4f}  weights={best_w}")
    return best_w, best_f1


# 1ï¸�âƒ£  tidy mapping of tag â†’ (oof_logits , test_logits)
logits = {
    "EffNet" : (oof_eff,   test_eff),
    "ViT-PL" : (oof_pl,    test_pl),
    "ViT-HF" : (oof_hf,    test_hf),
}

# 2ï¸�âƒ£  pair-wise blends
pairs = [("EffNet", "ViT-PL"),
         ("EffNet", "ViT-HF"),
         ("ViT-PL", "ViT-HF")]

for A, B in pairs:
    o1, _ = logits[A]
    o2, _ = logits[B]

    best_w, best_f1 = search_blend([o1, o2, np.zeros_like(o1)],
                                   f"{A}+{B}",
                                   grid=np.linspace(0, 1, 21))
    a, b, _ = best_w
    _, t1 = logits[A]
    _, t2 = logits[B]
    test_blend = a * t1 + b * t2

    sub = pd.DataFrame({
        "filename": test_files,
        "label"   : logits_to_labels(test_blend)
    })
    sub.to_csv(f"submission_{A.lower()[:4]}_{B.lower()[:4]}.csv", index=False)
    print(f"  â†³ wrote submission_{A.lower()[:4]}_{B.lower()[:4]}.csv")

# 3ï¸�âƒ£  triple blend
best_w, _ = search_blend([logits["EffNet"][0],
                          logits["ViT-PL"][0],
                          logits["ViT-HF"][0]],
                         "EffNet+ViT-PL+ViT-HF",
                         grid=np.linspace(0, 1, 11))

a, b, c = best_w
test_triple = (a * logits["EffNet"][1] +
               b * logits["ViT-PL"][1] +
               c * logits["ViT-HF"][1])

sub = pd.DataFrame({
    "filename": test_files,
    "label"   : logits_to_labels(test_triple)
})
sub.to_csv("submission_blended.csv", index=False)
print("  â†³ wrote submission_blended.csv")


print("\nNotebook finished - all models trained, F1 scores printed, and 7 submissions written:")
print("  â€¢ 3 single-model csvs")
print("  â€¢ 3 pairwise-blend csvs")
print("  â€¢ 1 triple-blend csv  (submission_blended.csv)")




