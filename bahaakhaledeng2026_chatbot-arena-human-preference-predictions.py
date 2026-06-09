!pip install transformers peft accelerate -U --no-index --find-links /kaggle/input/lmsys-wheel-files
!pip install bitsandbytes -U --no-index --find-links /kaggle/input/bitsandbytes-0-43-2-py3-none-manylinux-2-24-x86-64


import os, time, random
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import (
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.cuda.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor



# -------------------------
# Configuration
# -------------------------
@dataclass
class CFG:
    # Data paths (Kaggle)
    train_csv = "/kaggle/input/lmsys-chatbot-arena/train.csv"
    test_csv = "/kaggle/input/lmsys-chatbot-arena/test.csv"
    sample_csv = "/kaggle/input/lmsys-chatbot-arena/sample_submission.csv"
    # Local model dirs (update if necessary)
    gemma_dir = "/kaggle/input/gemma-2-9b-it-bnb-4bit-unsloth/transformers/default/1"
    # If you already have a LoRA adapter to initialize, set lora_dir; otherwise leave None
    lora_dir = None
    output_dir = "./lora_saved"  # where to save trained LoRA adapter

    # training hyperparams
    max_length = 2048
    train_batch_size = 2              # per device
    eval_batch_size = 4
    gradient_accumulation_steps = 8   # effective batch = train_batch_size * gradient_accumulation_steps
    num_train_epochs = 2
    learning_rate = 2e-4
    weight_decay = 0.0
    warmup_steps = 100
    seed = 42

    # LoRA config
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    # augmentation & inference
    augment_prob = 0.5    # prob to swap A/B in training
    tta_swaps = True
    tta_repeats = 1

cfg = CFG()



# reproducibility
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

# -------------------------
# Utilities & text processing
# -------------------------
def ensure_local_model_path(path: str):
    # Return path if exists; if not, attempt to use as-is (from_pretrained will raise helpful error)
    if os.path.exists(path):
        return path
    # try kagglehub fallback if available
    try:
        import kagglehub
        # Example: the user may prefer to call kagglehub to fetch models; we leave path as-is and let fallback occur later
        return path
    except Exception:
        return path

cfg.gemma_dir = ensure_local_model_path(cfg.gemma_dir)
if cfg.lora_dir:
    cfg.lora_dir = ensure_local_model_path(cfg.lora_dir)

def process_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    stripped_str = text.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return ' '.join([s for s in sentences if s is not None and len(s.strip()) > 0])

def tokenize_pairwise(tokenizer, prompt_list, resp_a_list, resp_b_list, max_length=cfg.max_length, spread_max_length=False):
    prom_pref = ["<prompt>: " + p for p in prompt_list]
    r_a_pref = ["\n\n<response_a>: " + r_a for r_a in resp_a_list]
    r_b_pref = ["\n\n<response_b>: " + r_b for r_b in resp_b_list]
    if spread_max_length:
        p_ids = tokenizer(prom_pref, max_length=max_length*2//10, truncation=True, padding=False).input_ids
        a_ids = tokenizer(r_a_pref, max_length=max_length*4//10, truncation=True, padding=False).input_ids
        b_ids = tokenizer(r_b_pref, max_length=max_length*4//10, truncation=True, padding=False).input_ids
        input_ids = [p + a + b for p, a, b in zip(p_ids, a_ids, b_ids)]
        attention_mask = [[1] * len(x) for x in input_ids]
    else:
        texts = [p + a + b for p, a, b in zip(prom_pref, r_a_pref, r_b_pref)]
        tokenized = tokenizer(texts, max_length=max_length, truncation=True, padding=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
    return input_ids, attention_mask



# -------------------------
# Dataset with train-time augmentation (swap)
# -------------------------
class PreferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: GemmaTokenizerFast, is_train=True, augment_prob=0.0):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.augment_prob = augment_prob if is_train else 0.0

        # clean text
        self.df["prompt"] = self.df["prompt"].apply(process_text)
        self.df["response_a"] = self.df["response_a"].apply(process_text)
        self.df["response_b"] = self.df["response_b"].apply(process_text)

        if is_train:
            votes = np.vstack([self.df.winner_model_a.values,
                               self.df.winner_model_b.values,
                               self.df.winner_tie.values]).T
            self.labels = votes.argmax(axis=1).astype(np.int64)
        else:
            self.labels = None

        # Pre-tokenize original & swapped to speed up
        self.orig_input_ids, self.orig_attention_mask = tokenize_pairwise(
            self.tokenizer, self.df["prompt"].tolist(), self.df["response_a"].tolist(), self.df["response_b"].tolist(), max_length=cfg.max_length
        )
        self.swap_input_ids, self.swap_attention_mask = tokenize_pairwise(
            self.tokenizer, self.df["prompt"].tolist(), self.df["response_b"].tolist(), self.df["response_a"].tolist(), max_length=cfg.max_length
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        do_swap = False
        if self.is_train and random.random() < self.augment_prob:
            do_swap = True

        if do_swap:
            input_ids = torch.tensor(self.swap_input_ids[idx], dtype=torch.long)
            attention_mask = torch.tensor(self.swap_attention_mask[idx], dtype=torch.long)
            if self.labels is not None:
                lab = int(self.labels[idx])
                # flip labels 0 <-> 1, tie (2) remains
                if lab == 0:
                    lab = 1
                elif lab == 1:
                    lab = 0
                label = torch.tensor(lab, dtype=torch.long)
            else:
                label = None
        else:
            input_ids = torch.tensor(self.orig_input_ids[idx], dtype=torch.long)
            attention_mask = torch.tensor(self.orig_attention_mask[idx], dtype=torch.long)
            label = torch.tensor(self.labels[idx], dtype=torch.long) if self.labels is not None else None

        item = {"input_ids": input_ids, "attention_mask": attention_mask}
        if label is not None:
            item["labels"] = label
        return item

def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    batch_t = pad_without_fast_tokenizer_warning(tokenizer, {"input_ids": input_ids, "attention_mask": attention_mask}, padding="longest", pad_to_multiple_of=None, return_tensors="pt")
    if "labels" in batch[0]:
        labels = torch.stack([b["labels"] for b in batch])
        batch_t["labels"] = labels
    return batch_t



# -------------------------
# Load dataframes
# -------------------------
train_df = pd.read_csv(cfg.train_csv)
test_df = pd.read_csv(cfg.test_csv)

# small validation split
train_split_df, valid_df = train_test_split(train_df, test_size=0.05, random_state=cfg.seed)

# -------------------------
# Safe tokenizer load (local_files_only)
# -------------------------
try:
    tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir, local_files_only=True, trust_remote_code=True)
    print(f"[OK] Tokenizer loaded from {cfg.gemma_dir}")
except Exception as e:
    # try fallback via kagglehub if available (best-effort)
    print(f"[WARN] Could not load tokenizer directly from {cfg.gemma_dir}: {e}")
    try:
        import kagglehub
        print("[INFO] Attempting kagglehub.model_download fallback...")
        cfg.gemma_dir = kagglehub.model_download("seguride/gemma-2-9b-it-bnb-4bit-unsloth/transformers/default")
        tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir, local_files_only=True, trust_remote_code=True)
        print(f"[OK] Tokenizer loaded via kagglehub from {cfg.gemma_dir}")
    except Exception as e2:
        raise RuntimeError("Failed to load tokenizer from local path and kagglehub fallback. Ensure the model directory contains tokenizer files.") from e2

tokenizer.add_eos_token = True
tokenizer.padding_side = "right"


# -------------------------
# Datasets & loaders
# -------------------------
train_ds = PreferenceDataset(train_split_df, tokenizer, is_train=True, augment_prob=cfg.augment_prob)
valid_ds = PreferenceDataset(valid_df, tokenizer, is_train=True, augment_prob=0.0)

train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=cfg.train_batch_size, collate_fn=collate_fn)
valid_loader = DataLoader(valid_ds, sampler=SequentialSampler(valid_ds), batch_size=cfg.eval_batch_size, collate_fn=collate_fn)

# -------------------------
# BitsAndBytes config (4-bit)
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# -------------------------
# Load base model (pin to single GPU for training)
# -------------------------
# Choose device for training (pin entire model to this device to avoid cross-device label problems)
train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training device: {train_device}")

try:
    # Force model to be placed on single device to avoid cross-device tensor errors during loss computation
    model = Gemma2ForSequenceClassification.from_pretrained(
        cfg.gemma_dir,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=bnb_config,
        num_labels=3,
        use_cache=False,
        device_map={"": train_device},  # pin everything to train_device
    )
    print("[OK] Base model loaded and pinned to train_device.")
except Exception as e:
    print("[ERROR] Failed to load base model pinned to train_device:", e)
    raise

# Attach LoRA
lora_config = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # adjust if needed
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, lora_config)

# If user provided a lora_dir, try loading it (optional)
if cfg.lora_dir and os.path.exists(cfg.lora_dir):
    print("[INFO] Loading initial LoRA adapter from cfg.lora_dir")
    model = PeftModel.from_pretrained(model, cfg.lora_dir)

# Put model in train_device (should already be pinned)
model = model.to(train_device)
model.print_trainable_parameters()



# -------------------------
# Optimizer, scheduler, scaler
# -------------------------
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
effective_batch_size = cfg.train_batch_size * cfg.gradient_accumulation_steps
num_update_steps_per_epoch = max(1, len(train_loader) // cfg.gradient_accumulation_steps)
total_training_steps = num_update_steps_per_epoch * cfg.num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=max(1, total_training_steps))

# AMP scaler (no device argument needed)
scaler = GradScaler()

# -------------------------
# Training loop (mixed precision, gradient accumulation)
# -------------------------
device = train_device
model.train()
best_val_loss = float("inf")
global_step = 0

print(f"[TRAIN] epochs={cfg.num_train_epochs}, steps_per_epoch={len(train_loader)}, total_steps~{total_training_steps}")
for epoch in range(cfg.num_train_epochs):
    running_loss = 0.0
    model.train()
    for step, batch in enumerate(train_loader):
        # move batch to train_device
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.amp.autocast(device_type="cpu"):
            outputs = model(**batch)
            loss = outputs.loss / cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()
        running_loss += loss.item() * cfg.gradient_accumulation_steps

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

    # Validation
    model.eval()
    val_losses = []
    preds = []
    labels_all = []
    with torch.no_grad():
        for vb in valid_loader:
            vb = {k: v.to(device) for k, v in vb.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.amp.autocast(device_type="cpu"):
                outputs = model(**vb)
            logits = outputs.logits.detach().cpu().numpy()
            lab = vb["labels"].detach().cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            val_losses.append(log_loss(lab, probs, labels=[0, 1, 2]))
            preds.extend(probs.argmax(axis=1).tolist())
            labels_all.extend(lab.tolist())

    mean_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float("nan")
    val_acc = accuracy_score(labels_all, preds) if len(labels_all) > 0 else 0.0
    val_f1 = f1_score(labels_all, preds, average="macro") if len(labels_all) > 0 else 0.0
    print(f"[EPOCH {epoch+1}] train_loss={running_loss:.4f} val_loss={mean_val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

    # Save best LoRA
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        os.makedirs(cfg.output_dir, exist_ok=True)
        model.save_pretrained(cfg.output_dir)
        print(f"[SAVE] Saved LoRA adapter to {cfg.output_dir}")



# -------------------------
# Prepare inference models (two copies pinned to cuda:0 and cuda:1 if available)
# -------------------------
print("[INFO] Preparing inference copies (base + saved LoRA adapter)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def load_inference_copy(device_index):
    dev = torch.device(f"cuda:{device_index}") if torch.cuda.device_count() > device_index else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Gemma2ForSequenceClassification.from_pretrained(
        cfg.gemma_dir,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=bnb_config,
        num_labels=3,
        use_cache=False,
        device_map={"": dev},
    )
    # Load saved LoRA adapter (saved in cfg.output_dir)
    if os.path.exists(cfg.output_dir):
        base = PeftModel.from_pretrained(base, cfg.output_dir)
    base.eval()
    return base, dev

if torch.cuda.device_count() >= 2:
    model0, device0 = load_inference_copy(0)
    model1, device1 = load_inference_copy(1)
else:
    model0, device0 = load_inference_copy(0)
    model1, device1 = model0, device0

# -------------------------
# Tokenize test set for inference and TTA swap
# -------------------------
test_df["prompt"] = test_df["prompt"].apply(process_text)
test_df["response_a"] = test_df["response_a"].apply(process_text)
test_df["response_b"] = test_df["response_b"].apply(process_text)

test_input_ids, test_attn = tokenize_pairwise(tokenizer, test_df["prompt"].tolist(), test_df["response_a"].tolist(), test_df["response_b"].tolist(), max_length=cfg.max_length)
test_df_inf = pd.DataFrame({"id": test_df["id"].values})
test_df_inf["input_ids"] = test_input_ids
test_df_inf["attention_mask"] = test_attn
test_df_inf["length"] = test_df_inf["input_ids"].apply(len)

if cfg.tta_swaps:
    swap_input_ids, swap_attn = tokenize_pairwise(tokenizer, test_df["prompt"].tolist(), test_df["response_b"].tolist(), test_df["response_a"].tolist(), max_length=cfg.max_length)
    swap_df_inf = pd.DataFrame({"id": test_df["id"].values})
    swap_df_inf["input_ids"] = swap_input_ids
    swap_df_inf["attention_mask"] = swap_attn
    swap_df_inf["length"] = swap_df_inf["input_ids"].apply(len)
else:
    swap_df_inf = None

# -------------------------
# Inference functions
# -------------------------
@torch.no_grad()
@torch.amp.autocast(device_type='cuda')
def inference_on_df(df, model, device, batch_size=cfg.eval_batch_size):
    a_win, b_win, tie = [], [], []
    for start_idx in range(0, len(df), batch_size):
        tmp = df.iloc[start_idx:start_idx+batch_size]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(tokenizer, {"input_ids": input_ids, "attention_mask": attention_mask}, padding="longest", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        proba = outputs.logits.softmax(-1).cpu().numpy()
        for p in proba:
            a_win.append(float(p[0])); b_win.append(float(p[1])); tie.append(float(p[2]))
    out = df.copy()
    out["winner_model_a"] = a_win
    out["winner_model_b"] = b_win
    out["winner_tie"] = tie
    return out

def parallel_inference_on_df(df):
    df_sorted = df.sort_values("length", ascending=False)
    sub_1 = df_sorted.iloc[0::2].copy()
    sub_2 = df_sorted.iloc[1::2].copy()
    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(inference_on_df, (sub_1, sub_2), (model0, model1), (device0, device1)))
    merged = pd.concat(results, axis=0).sort_index()
    return merged

# -------------------------
# Run inference and TTA averaging
# -------------------------
print("[INF] Running original predictions...")
orig_preds = parallel_inference_on_df(test_df_inf)

if cfg.tta_swaps and swap_df_inf is not None:
    print("[INF] Running swapped predictions for TTA...")
    swap_preds = parallel_inference_on_df(swap_df_inf)
    orig_preds = orig_preds.sort_index()
    swap_preds = swap_preds.sort_index()
    avg_a = (orig_preds["winner_model_a"].values + swap_preds["winner_model_b"].values) / 2.0
    avg_b = (orig_preds["winner_model_b"].values + swap_preds["winner_model_a"].values) / 2.0
    avg_tie = (orig_preds["winner_tie"].values + swap_preds["winner_tie"].values) / 2.0
    final_df = pd.DataFrame({"id": orig_preds["id"].values, "winner_model_a": avg_a, "winner_model_b": avg_b, "winner_tie": avg_tie})
else:
    final_df = orig_preds[["id", "winner_model_a", "winner_model_b", "winner_tie"]].copy()

# Normalize probabilities
probs = final_df[["winner_model_a", "winner_model_b", "winner_tie"]].values
probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
probs = probs / probs.sum(axis=1, keepdims=True)
final_df[["winner_model_a", "winner_model_b", "winner_tie"]] = probs

final_df.to_csv("submission.csv", index=False)
print("[DONE] submission.csv saved.")


