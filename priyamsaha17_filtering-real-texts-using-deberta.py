import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup


import os, re
from pathlib import Path
import pandas as pd

TRAIN_DIR = Path("/kaggle/input/fake-or-real-the-impostor-hunt/data/train")   
TRAIN_CSV = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"     

def find_folder_for_id(base_dir: Path, id_value):
    """Locate the subfolder corresponding to the given id_value."""
    id_str = str(id_value).zfill(4)
    candidates = {
        f"article_{id_str}"
    }
    for cand in candidates:
        p = base_dir / cand
        if p.exists() and p.is_dir():
            return p

    # Fallbacks
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.endswith(f"_{id_str}") or p.name.startswith(f"{id_str}_"):
            return p
        m = re.search(r"(\d+)", p.name)
        if m and int(m.group(1)) == int(id_value):
            return p
        if id_str in p.name:
            return p
    return None

def safe_read_file(path: Path):
    """Read text with fallbacks and ignore errors."""
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""

def read_pair_from_folder(folder: Path):
    """Read file_1 and file_2 text contents."""
    candidates_1 = ["file_1.txt"]
    candidates_2 = ["file_2.txt"]

    f1 = f2 = None
    for name in candidates_1:
        if (folder / name).exists():
            f1 = folder / name
            break
    for name in candidates_2:
        if (folder / name).exists():
            f2 = folder / name
            break

    if f1 is None or f2 is None:
        text_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.txt', ''}])
        if len(text_files) >= 2:
            f1 = f1 or text_files[0]
            f2 = f2 or text_files[1]

    return safe_read_file(f1), safe_read_file(f2)

# -------- Build dataframe --------
meta = pd.read_csv(TRAIN_CSV)
rows = []
missing = []

for _, r in meta.iterrows():
    csv_id = r['id']
    real_text_id = int(r['real_text_id'])  # 1 or 2
    folder_path = find_folder_for_id(TRAIN_DIR, csv_id)

    if folder_path is None:
        missing.append(csv_id)
        rows.append({
            'id': csv_id,
            'folder_name': None,
            'text1': "",
            'text2': "",
            'real_text_id': real_text_id
        })
    else:
        t1, t2 = read_pair_from_folder(folder_path)
        rows.append({
            'id': csv_id,
            'folder_name': folder_path.name,
            'text1': t1,
            'text2': t2,
            'real_text_id': real_text_id
        })

train_df = pd.DataFrame(rows)

print("Total rows:", len(train_df))
if missing:
    print(f"⚠️ WARNING: {len(missing)} CSV ids had no matching folder. Example missing ids: {missing[:10]}")
else:
    print("✅ All CSV ids matched to a folder.")

train_df.head(10)


train_df.shape


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
max_len = 512
BATCH_SIZE = 1
LR = 2e-5
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
model.to(DEVICE)


import torch
from torch.utils.data import Dataset

class PairwiseDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text1, text2, label = row["text1"], row["text2"], row["real_text_id"]

        # Tokenize separately
        enc_a = self.tokenizer(
            text1,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc_b = self.tokenizer(
            text2,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Convert label {1,2} → target {+1,-1} for MarginRankingLoss
        target = 1 if label == 1 else -1

        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "target": torch.tensor(target, dtype=torch.float),  # for loss
            "label": torch.tensor(label, dtype=torch.long)      # for accuracy
        }

dataset = PairwiseDataset(train_df, tokenizer, max_len=max_len)


import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# Config
# -------------------------
MODEL_NAME = "microsoft/deberta-v3-base"   # use base to avoid OOM
MAX_LEN = 768
VAL_SPLIT = 0.1
BATCH_SIZE = 1
LR = 2e-5
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "best_deberta_pairwise.pt"
ACCUM_STEPS = 2  # gradient accumulation steps

# -------------------------
# Tokenizer & Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1  # scalar score for ranking
).to(DEVICE)

# -------------------------
# Train/Validation Split
# -------------------------
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Optimizer, Scheduler, Loss
# -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda")  # new API
criterion = nn.MarginRankingLoss(margin=0.0)  # Pairwise ranking loss

early_stop_patience = 3
best_val_loss = float("inf")
no_improve_epochs = 0
best_model_wts = copy.deepcopy(model.state_dict())

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    running_train_loss, correct_train, total_train = 0.0, 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for i, batch in enumerate(progress_bar):
        input_ids_a = batch["input_ids_a"].to(DEVICE)
        attention_mask_a = batch["attention_mask_a"].to(DEVICE)
        input_ids_b = batch["input_ids_b"].to(DEVICE)
        attention_mask_b = batch["attention_mask_b"].to(DEVICE)
        target = batch["target"].to(DEVICE)   # +1 or -1
        label = batch["label"].to(DEVICE)     # 1 or 2

        with torch.amp.autocast("cuda"):
            score_a = model(input_ids=input_ids_a,
                            attention_mask=attention_mask_a).logits.view(-1)
            score_b = model(input_ids=input_ids_b,
                            attention_mask=attention_mask_b).logits.view(-1)

            loss = criterion(score_a, score_b, target) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_train_loss += loss.item() * ACCUM_STEPS

        # --- Accuracy ---
        pred = torch.where(score_a > score_b, 1, 2)  # pick the higher score
        correct_train += (pred == label).sum().item()
        total_train += label.size(0)

        progress_bar.set_postfix({
            "loss": running_train_loss / (i + 1),
            "acc": 100.0 * correct_train / total_train
        })

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_acc = correct_train / total_train
    train_accs.append(train_acc)

    # --- Validation ---
    model.eval()
    running_val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(DEVICE)
            attention_mask_a = batch["attention_mask_a"].to(DEVICE)
            input_ids_b = batch["input_ids_b"].to(DEVICE)
            attention_mask_b = batch["attention_mask_b"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            with torch.amp.autocast("cuda"):
                score_a = model(input_ids=input_ids_a,
                                attention_mask=attention_mask_a).logits.view(-1)
                score_b = model(input_ids=input_ids_b,
                                attention_mask=attention_mask_b).logits.view(-1)

                loss = criterion(score_a, score_b, target)
                running_val_loss += loss.item()

            # --- Accuracy ---
            pred = torch.where(score_a > score_b, 1, 2)
            correct_val += (pred == label).sum().item()
            total_val += label.size(0)

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_acc = correct_val / total_val
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

    # --- Early stopping on loss ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        no_improve_epochs = 0
        print(f"✅ New best model saved (Val Loss={best_val_loss:.4f})")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_patience:
            print(f"⚠️ Early stopping triggered at epoch {epoch+1}")
            break

# Load best model weights
model.load_state_dict(best_model_wts)

# -------------------------
# Plot Loss & Accuracy Curves
# -------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, len(train_losses)+1), train_losses, marker="o", label="Train Loss")
plt.plot(range(1, len(val_losses)+1), val_losses, marker="o", label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, len(train_accs)+1), train_accs, marker="o", label="Train Acc")
plt.plot(range(1, len(val_accs)+1), val_accs, marker="o", label="Validation Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve"); plt.legend()
plt.show()

print(f"✅ Best model saved at {BEST_MODEL_PATH} with Val Loss={best_val_loss:.4f}")


import os, re
from pathlib import Path
import pandas as pd

TEST_DIR = Path("/kaggle/input/fake-or-real-the-impostor-hunt/data/test")  # test folders
TEST_CSV = None  # no test CSV, we'll just list folders

def find_folder_for_id(base_dir: Path, id_value):
    """Locate the subfolder corresponding to the given id_value."""
    id_str = str(id_value).zfill(4)
    candidates = {f"article_{id_str}"}
    for cand in candidates:
        p = base_dir / cand
        if p.exists() and p.is_dir():
            return p

    # Fallbacks
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.endswith(f"_{id_str}") or p.name.startswith(f"{id_str}_"):
            return p
        m = re.search(r"(\d+)", p.name)
        if m and int(m.group(1)) == int(id_value):
            return p
        if id_str in p.name:
            return p
    return None

def safe_read_file(path: Path):
    """Read text with fallbacks and ignore errors."""
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""

def read_pair_from_folder(folder: Path):
    """Read file_1 and file_2 text contents."""
    candidates_1 = ["file_1.txt"]
    candidates_2 = ["file_2.txt"]

    f1 = f2 = None
    for name in candidates_1:
        if (folder / name).exists():
            f1 = folder / name
            break
    for name in candidates_2:
        if (folder / name).exists():
            f2 = folder / name
            break

    if f1 is None or f2 is None:
        text_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.txt', ''}])
        if len(text_files) >= 2:
            f1 = f1 or text_files[0]
            f2 = f2 or text_files[1]

    return safe_read_file(f1), safe_read_file(f2)

# -------- Build dataframe for test --------
rows = []
for folder_path in sorted(TEST_DIR.iterdir()):
    if not folder_path.is_dir():
        continue
    csv_id = re.search(r"(\d+)", folder_path.name)
    if csv_id:
        csv_id = int(csv_id.group(1))
    else:
        csv_id = folder_path.name  # fallback to folder name

    t1, t2 = read_pair_from_folder(folder_path)
    rows.append({
        'id': csv_id,
        'folder_name': folder_path.name,
        'text1': t1,
        'text2': t2
    })

test_df = pd.DataFrame(rows)
print("Total test rows:", len(test_df))
test_df.head(10)


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# -------------------------
# Test Dataset
# -------------------------
class PairwiseTestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):  # match training
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text1, text2 = row["text1"], row["text2"]

        enc_a = self.tokenizer(
            text1,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc_b = self.tokenizer(
            text2,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "id": row["id"]
        }

# -------------------------
# Test Loader
# -------------------------
test_dataset = PairwiseTestDataset(test_df, tokenizer, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -------------------------
# Generate Predictions
# -------------------------
model.eval()
preds, ids = [], []

for batch in tqdm(test_loader, desc="Predicting"):
    input_ids_a = batch["input_ids_a"].to(DEVICE)
    attention_mask_a = batch["attention_mask_a"].to(DEVICE)
    input_ids_b = batch["input_ids_b"].to(DEVICE)
    attention_mask_b = batch["attention_mask_b"].to(DEVICE)
    sample_id = batch["id"].item()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        score_a = model(input_ids=input_ids_a,
                        attention_mask=attention_mask_a).logits.squeeze().item()
        score_b = model(input_ids=input_ids_b,
                        attention_mask=attention_mask_b).logits.squeeze().item()

    # Higher score → more "real"
    pred_label = 1 if score_a > score_b else 2
    preds.append(pred_label)
    ids.append(sample_id)

# -------------------------
# Prepare Submission
# -------------------------
submission_df = pd.DataFrame({
    "id": ids,
    "real_text_id": preds
})

submission_df.to_csv("submission.csv", index=False)
print("✅ Submission file created: submission.csv")
submission_df.head(10)

