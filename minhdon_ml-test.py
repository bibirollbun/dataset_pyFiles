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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import json

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2PreTrainedModel
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch, gc

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold


MODEL_NAME="microsoft/deberta-v3-large"
TRAIN_PATH="/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH="/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
TOKENIZER="/kaggle/input/deberta-v3-small-tokenizer/pytorch/default/1"
MODEL="/kaggle/input/deberta-v3-small-model/pytorch/default/1"
MODE = "regression"  # "regression" hoáº·c "classification"

MAX_LEN = 1024
OVERLAP = 152
MIN_CHUNK_RATIO = 0.3  # bá»� Ä‘oáº¡n náº¿u < 0.3 * MAX_LEN
BATCH_SIZE = 8
TEST_SIZE = 0.2
EPOCHS = 5
LR = 2e-5
NUM_CLASSES=6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = DebertaV2Tokenizer.from_pretrained(TOKENIZER)


sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)


class CustomAES2Model_regression(nn.Module):
    def __init__(self, model_name, model_dir=MODEL, dropout_rate=0.2, freeze_layers=False):
        super().__init__()
        # Backbone tá»« pretrained model
        # Náº¿u model Ä‘Ã£ Ä‘Æ°á»£c lÆ°u trong folder model_dir thÃ¬ load tá»« Ä‘Ã³
        if os.path.exists(model_dir) and any(os.scandir(model_dir)):
            print(f"ğŸ”¹ Loading backbone from local directory: {model_dir}")
            self.backbone = AutoModel.from_pretrained(model_dir)
        else:
            print(f"ğŸ”¸ Downloading backbone from HuggingFace Hub: {model_name}")
            self.backbone = AutoModel.from_pretrained(model_name)
            # LÆ°u láº¡i backbone Ä‘á»ƒ dÃ¹ng sau
            os.makedirs(model_dir, exist_ok=True)
            self.backbone.save_pretrained(model_dir)

        hidden_size = self.backbone.config.hidden_size

        # Regression head cho 1 output duy nháº¥t (score 1â€“6)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )

        # TÃ¹y chá»�n freeze backbone náº¿u muá»‘n fine-tune cháº­m
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # token [CLS]
        pooled_output = self.dropout(pooled_output)
        preds = self.regressor(pooled_output)
        return preds.squeeze(-1)  # [batch_size]


class CustomAES2Model_classification(nn.Module):
    def __init__(self, model_name, num_classes=6, model_dir=MODEL, dropout_rate=0.2, freeze_layers=False):
        super().__init__()
        if os.path.exists(model_dir) and any(os.scandir(model_dir)):
            print(f"ğŸ”¹ Loading backbone from local directory: {model_dir}")
            self.backbone = AutoModel.from_pretrained(model_dir)
        else:
            print(f"ğŸ”¸ Downloading backbone from HuggingFace Hub: {model_name}")
            self.backbone = AutoModel.from_pretrained(model_name)
            os.makedirs(model_dir, exist_ok=True)
            self.backbone.save_pretrained(model_dir)

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head: tá»« hidden_size â†’ hidden_size/2 â†’ num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits  # shape [batch_size, num_classes]



# model = CustomAES2Model_classification(MODEL_NAME, freeze_layers=False)
# model.to(DEVICE)


df = pd.read_csv(TRAIN_PATH)
num_labels = df["score"].nunique()
print("Total samples: ",len(df))
print("Number of labels:", num_labels)
print("Sample: ")
print(df.head(5))


from torch.utils.data import Dataset
import torch

class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, text_col="full_text", max_length=512, overlap=128,
                 min_chunk_ratio=0.3, mode="classification", num_classes=6, label_col="score"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio
        self.mode = mode
        self.num_classes = num_classes
        self.label_col = label_col
        self.samples = []
        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            text = row[text_col]
            # NhÃ£n gá»‘c (vÃ­ dá»¥ 1-6)
            orig_label = row[self.label_col] if self.label_col in df.columns else None
            if orig_label is None:
                raise ValueError(f"Label column '{self.label_col}' not in dataframe")
            # Map nhÃ£n vá»� lá»›p integer 0..(num_classes-1)
            class_label = int(orig_label) - 1
            if class_label < 0 or class_label >= self.num_classes:
                raise ValueError(f"Label {orig_label} is out of bounds for num_classes={self.num_classes}")
            tokens = tokenizer.encode(text, add_special_tokens=False)
            n = len(tokens)
            stride = max_length - overlap - 2  # Account for [CLS] and [SEP]
            if n <= max_length - 2:
                self.samples.append((tokens, class_label, essay_id))
            else:
                for start in range(0, n, stride):
                    end = min(start + max_length - 2, n)
                    chunk = tokens[start:end]
                    if len(chunk) < (max_length - 2) * min_chunk_ratio:
                        continue
                    self.samples.append((chunk, class_label, essay_id))
                    if end == n:
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, class_label, essay_id = self.samples[idx]
        # Directly build input_ids with specials and pad
        input_ids = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "essay_id": essay_id,
            "labels": torch.tensor(class_label, dtype=torch.long)
        }
        return item


# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=LR,
#     betas=(0.9, 0.999),
#     weight_decay=5e-5
# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=1000,   # má»—i 2 epoch restart 1 láº§n
#     T_mult=2,
#     eta_min=5e-8
# )


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import wandb  # cáº§n import

def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    grad_accum_steps: int = 5,
    wandb_run = None,            # thÃªm tham sá»‘ run cá»§a W&B
    log_every_n_batches: int = 50,  # má»—i n batches log má»™t láº§n
    epoch: int = 0,
    global_step: int = 0,
):
    
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # TÃ­nh logits vÃ  loss bÃ¬nh thÆ°á»�ng
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, labels)

        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # Log vá»›i W&B má»—i n batches
        if wandb_run is not None and (global_step + 1) % log_every_n_batches == 0:
            wandb_run.log({
                "train_batch_loss": loss.item(),
                "train_batch_step": step + 1
            }, step=global_step)

        if step % 50 == 0:
            torch.cuda.empty_cache()

        global_step += 1

    avg_loss = total_loss / len(dataloader)

    try:
        qwk = cohen_kappa_score(all_preds, all_labels, weights="quadratic")
    except Exception as e:
        qwk = None
        print("Could not compute QWK:", e)

    metrics = {
        "train_loss": avg_loss,
        "train_qwk": qwk,
    }

    # Log epochâ€‘level metric vÃ o W&B
    if wandb_run is not None:
        wandb_run.log(metrics)

    return metrics, global_step



import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import wandb

def quadratic_weighted_kappa(y_true_int: np.ndarray, y_pred_int: np.ndarray,
                             min_label: int = 0, max_label: int = None) -> float:
    if max_label is None:
        max_label = max(int(y_true_int.max()), int(y_pred_int.max()))
    if (y_true_int < min_label).any() or (y_true_int > max_label).any():
        raise ValueError(f"y_true_int contains values outside [{min_label}, {max_label}]")
    if (y_pred_int < min_label).any() or (y_pred_int > max_label).any():
        raise ValueError(f"y_pred_int contains values outside [{min_label}, {max_label}]")
    return cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")


@torch.no_grad()
def eval_epoch(model, dataloader, device,
               num_classes: int = 6, label_offset: int = 1,
               wandb_run=None, log_every_n_batches: int = 50,
               batch_limit: int = None, epoch: int = 0):
    """
    model: classification model tráº£ logits size [batch_size, num_classes]
    dataloader: yield batches cÃ³ "input_ids", "attention_mask", "labels" (integer lá»›p)
    device: torch device
    num_classes: sá»‘ lá»›p (vÃ­ dá»¥ 6)
    label_offset: náº¿u lá»›p 0â†’ Ä‘iá»ƒm 1, thÃ¬ label_offset=1 Ä‘á»ƒ map ra Ä‘iá»ƒm (náº¿u báº¡n muá»‘n dÃ¹ng Ä‘iá»ƒm ngoÃ i QWK)
    wandb_run: má»™t run cá»§a wandb Ä‘á»ƒ log náº¿u khÃ´ng None
    log_every_n_batches: log má»—i n batch
    batch_limit: náº¿u khÃ´ng None, chá»‰ cháº¡y thá»­ batch_limit batches rá»“i dá»«ng
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    for step, batch in enumerate(progress_bar):
        if batch_limit is not None and step >= batch_limit:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        labels = batch["labels"].to(device).long()

        # TÃ­nh logits vÃ  loss bÃ¬nh thÆ°á»�ng, khÃ´ng AMP
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        if wandb_run is not None and (step + 1) % log_every_n_batches == 0:
            wandb_run.log({
                "eval_batch_loss": loss.item(),
                "eval_batch_step": step + 1
            })

    if len(dataloader) > 0:
        avg_loss = total_loss / (batch_limit if batch_limit is not None else len(dataloader))
    else:
        avg_loss = float('nan')

    np_preds = np.array(all_preds, dtype=int)
    np_labels = np.array(all_labels, dtype=int)

    try:
        qwk = quadratic_weighted_kappa(np_labels, np_preds,
                                        min_label=0, max_label=num_classes-1)
    except ValueError as ve:
        print("ValueError in QWK computation:", ve)
        qwk = None

    accuracy = (np_preds == np_labels).mean() if np_labels.size > 0 else float('nan')

    metrics = {
        "val_loss": avg_loss,
        "val_qwk": qwk,
        "val_accuracy": accuracy
    }

    if wandb_run is not None:
        wandb_run.log(metrics)

    return metrics



NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["score"])):
    print(f"\n===== Fold {fold + 1}/{NUM_FOLDS} =====")
    if fold + 1 < 4:
        continue
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df   = df.loc[val_idx].reset_index(drop=True)
    print("  Train samples:", len(train_df))
    print("  Val   samples:", len(val_df))
    
    torch.cuda.empty_cache()
    gc.collect()

    wandb.login(key="a7df3964ffcf4c24262d52ad7ea0ee7a362b8fbc")

    # initialize a new wandb run for this fold
    run = wandb.init(
        project="ML_test",
        name=f"fold_{fold+1}",
        group="kfold_experiment",
        config={
            "fold": fold + 1,
            "num_folds": NUM_FOLDS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LEN,
            "overlap": OVERLAP,
            "min_chunk_ratio": MIN_CHUNK_RATIO,
            "num_classes": NUM_CLASSES,
            "model_name": MODEL_NAME,
            "learning_rate": LR,
            "epochs": EPOCHS
        }
    )
    config = run.config

    model = CustomAES2Model_classification(MODEL_NAME, freeze_layers=False)
    model.to(DEVICE)
    
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=5e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,   # má»—i 2 epoch restart 1 láº§n
        T_mult=2,
        eta_min=5e-8
    )

    # --- Datasets & Loaders ---
    train_dataset = EssayDataset(
        train_df, tokenizer,
        max_length=MAX_LEN,
        overlap=OVERLAP,
        min_chunk_ratio=MIN_CHUNK_RATIO,
        mode="classification",
        num_classes=NUM_CLASSES,
        label_col="score",
    )
    val_dataset = EssayDataset(
        val_df, tokenizer,
        max_length=MAX_LEN,
        overlap=OVERLAP,
        min_chunk_ratio=MIN_CHUNK_RATIO,
        mode="classification",
        num_classes=NUM_CLASSES,
        label_col="score"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    best_qwk = -1.0
    history = {
        "train_loss": [], "train_qwk": [],
        "val_loss": [],   "val_qwk": [],
        "lr": []
    }
    global_step = 0

    # --- Training loop ---
    for epoch in range(EPOCHS):
        print(f"\nğŸŸ¢ Epoch {epoch + 1}/{EPOCHS}")

        gc.collect()
        torch.cuda.empty_cache()
        global_step_new = 0

        # Train
        train_metrics, global_step_new = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            grad_accum_steps=1,
            wandb_run=run,
            global_step=global_step,
            epoch=epoch
        )
        train_loss = train_metrics["train_loss"]
        train_qwk  = train_metrics["train_qwk"]

        # Val
        val_metrics = eval_epoch(
            model=model,
            dataloader=val_loader,
            device=DEVICE,
            num_classes=NUM_CLASSES,
            label_offset=1,
            wandb_run=run,
            epoch=epoch
        )
        val_loss = val_metrics["val_loss"]
        val_qwk  = val_metrics["val_qwk"]

        global_step = global_step_new

        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        history["train_loss"].append(train_loss)
        history["train_qwk"].append(train_qwk)
        history["val_loss"].append(val_loss)
        history["val_qwk"].append(val_qwk)
        history["lr"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f} | Train QWK: {train_qwk:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f} | LR: {current_lr:.2e}")

        # Save best checkpoint foldâ€‘specific
        if val_qwk is not None and val_qwk > best_qwk:
            best_qwk = val_qwk
            ckpt_path = f"checkpoint_fold{fold+1}_epoch{epoch+1}.pth"
            torch.save({
                'fold': fold+1,
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_qwk': val_qwk,
            }, ckpt_path)
            print(f"âœ”ï¸�  Saved best checkpoint for fold {fold+1} at epoch {epoch+1}")

    # --- Save training history for this fold ---
    hist_path = f"history_fold{fold+1}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"âœ… Fold {fold+1} complete! History saved to {hist_path}")

    # ğŸ§¹ --- Dá»ŒN Sáº CH GPU VÃ€ Bá»˜ NHá»š SAU Má»–I FOLD ---
    print(f"\nğŸ§¹ Clearing GPU memory after fold {fold+1} ...")
    del model, optimizer, scheduler, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    run.finish()  # âœ… káº¿t thÃºc W&B run cá»§a fold hiá»‡n táº¡i





































