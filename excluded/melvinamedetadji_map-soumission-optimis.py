import os, math, gc, json, shutil, numpy as np, pandas as pd, torch, transformers
import torch.nn as nn
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments
)

# ========== Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def set_seed(sd=42):
    import random
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    os.environ["PYTHONHASHSEED"] = str(sd)

set_seed(42)

# ========== GESTION DISQUE MAXIMALE ==========
def cleanup_disk(verbose=False):
    dirs_to_clean = [
        "/kaggle/temp/hf", "/kaggle/temp/hfds", "/kaggle/temp/out",
        "/kaggle/temp/tb", "/root/.cache", "/tmp/pip*", "/tmp/tmpfs*"
    ]
    
    for d in dirs_to_clean:
        try:
            if "*" in d:
                os.system(f"rm -rf {d} 2>/dev/null")
            elif Path(d).exists():
                shutil.rmtree(d, ignore_errors=True)
        except:
            pass
    
    for p in ["/kaggle/temp/hf", "/kaggle/temp/out"]:
        os.makedirs(p, exist_ok=True)
    
    gc.collect()
    torch.cuda.empty_cache()
    os.system("find /kaggle -name '*.pyc' -delete 2>/dev/null")
    os.system("find /tmp -name '*.pyc' -delete 2>/dev/null")
    
    # Nettoyage extra agressif
    os.system("rm -rf /kaggle/temp/* 2>/dev/null")
    os.system("rm -rf /tmp/* 2>/dev/null")
    
    for p in ["/kaggle/temp/hf", "/kaggle/temp/out"]:
        os.makedirs(p, exist_ok=True)

def check_disk_space():
    import subprocess
    try:
        result = subprocess.run(['df', '-h', '/kaggle'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            print("Espace disque:")
            print(lines[1].split()[3], "disponible")
    except:
        pass

print("\n" + "="*70)
print("NETTOYAGE INITIAL AGRESSIF")
print("="*70)
cleanup_disk(verbose=True)
check_disk_space()

os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/temp/hf"
os.environ["TORCH_HOME"] = "/kaggle/temp/hf"
os.environ["TMPDIR"] = "/kaggle/temp"

# ========== DonnÃ©es ==========
BASE = Path("/kaggle/input/map-charting-student-math-misunderstandings")
TRAIN_PATH = BASE/"train.csv" if (BASE/"train.csv").exists() else Path("train.csv")
TEST_PATH = BASE/"test.csv" if (BASE/"test.csv").exists() else Path("test.csv")
SUB_PATH = BASE/"sample_submission.csv" if (BASE/"sample_submission.csv").exists() else Path("sample_submission.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SUB_PATH)
print("\nTrain:", train.shape, "Test:", test.shape)

train["Misconception"] = train["Misconception"].fillna("NA")
train["target"] = train["Category"].astype(str) + ":" + train["Misconception"].astype(str)

def format_text_B(row):
    return (f"Student Explanation: {row['StudentExplanation']}\n"
            f"Question: {row['QuestionText']}\n"
            f"MC Answer: {row['MC_Answer']}")

train["text"] = train.apply(format_text_B, axis=1)
test["text"] = test.apply(format_text_B, axis=1)

canonical_labels = sorted(train["target"].unique())
label2id = {l:i for i,l in enumerate(canonical_labels)}
id2label = {i:l for l,i in label2id.items()}
n_classes = len(canonical_labels)
print("n_classes:", n_classes)

# ========== Configurations ModÃ¨les ==========
MODELS = [
    {
        "name": "deberta-v3-xsmall",
        "path": "/kaggle/input/map-prefit-v2-deberta-xsmall/transformers/default/1",
        "T": 1.7376,  
    },
    {
        "name": "roberta-large",
        "path": "/kaggle/input/map-prefit-v2-roberta-large/transformers/default/1",
        "T": 2.7455,
    },
    {
        "name": "electra-large",
        "path": "/kaggle/input/map-prefit-v2-electra-large/transformers/default/1",
        "T": 2.4693,
    },
]

print("\nVÃ©rification des modÃ¨les:")
for m in MODELS:
    exists = Path(m["path"]).exists()
    status = "âœ“" if exists else "âœ—"
    print(f"  {status} {m['name']}: {m['path']}")
    if not exists:
        raise FileNotFoundError(f"ModÃ¨le introuvable: {m['path']}")
        
FINITION_EPOCHS = 3
MAX_LEN = 320
OPTION_NAME = "A - 3 epochs + MAX_LEN=320"
LR = 1.5e-5
GRAD_CLIP = 1.0

print("\n" + "="*70)
print(f"PHASE 2: OPTION {OPTION_NAME}")
print("="*70)
print(f"FINITION_EPOCHS: {FINITION_EPOCHS}")
print(f"MAX_LEN: {MAX_LEN}")
print(f"LR: {LR}")
print(f"GRAD_CLIP: {GRAD_CLIP}")
print("="*70)

# ========== Dataset ==========
class TextClsDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, label2id=None, has_label=True, text_col="text"):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.has_label = has_label
        self.label2id = label2id or {}
        self.text_col = text_col
    
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(row[self.text_col], truncation=True, max_length=self.max_len, padding=False)
        if self.has_label:
            enc["labels"] = self.label2id[str(row["target"])]
        return enc

# ========== TrainingArguments ==========
def build_training_args(output_dir, steps_per_epoch, batch_size=16, grad_accum=2, epochs=3):
    warmup_ratio = 0.15
    warmup_steps = int(max(1, steps_per_epoch) * epochs * warmup_ratio)
    bf16 = torch.cuda.is_bf16_supported()
    fp16 = (not bf16) and torch.cuda.is_available()
    
    try:
        return TrainingArguments(
            output_dir=output_dir,
            do_train=True, 
            do_eval=False,
            learning_rate=LR,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size*2,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            max_grad_norm=GRAD_CLIP,
            lr_scheduler_type="cosine",
            label_smoothing_factor=0.1,
            logging_steps=300,
            logging_dir=None,
            report_to="none",
            bf16=bf16, 
            fp16=fp16,
            save_strategy="no",
            evaluation_strategy="no",
            save_total_limit=0,
            load_best_model_at_end=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
    except TypeError:
        return TrainingArguments(
            output_dir=output_dir, do_train=True, do_eval=False,
            learning_rate=LR, num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size*2,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            logging_steps=300, report_to="none",
        )

# ========== Utils ==========
def softmax_np(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def reindex_to_canonical(model_id2label, canonical_labels):
    idx_by_label = {v: int(k) for k,v in model_id2label.items()}
    reindex = []
    missing = []
    for lbl in canonical_labels:
        if lbl in idx_by_label:
            reindex.append(idx_by_label[lbl])
        else:
            missing.append(lbl)
    if missing:
        raise ValueError(f"Labels manquants: {missing[:5]} ...")
    return np.array(reindex, dtype=np.int64)

def bs_for_model(name, max_len):
    """Batch size adaptÃ© RÃ‰DUIT pour Ã©conomiser disque/mÃ©moire"""
    n = name.lower()
    
    if max_len >= 352:
        # Batch sizes TRÃˆS rÃ©duits pour MAX_LEN=352
        if "roberta-large" in n: return 4
        if "electra-large" in n: return 6
        if "xsmall" in n: return 12
        return 8
    elif max_len >= 320:
        # Batch sizes rÃ©duits pour MAX_LEN=320
        if "roberta-large" in n: return 5
        if "electra-large" in n: return 7
        if "xsmall" in n: return 14
        return 10
    else:
        # Batch sizes standard pour MAX_LEN=256
        if "roberta-large" in n: return 6
        if "electra-large" in n: return 8
        if "xsmall" in n: return 16
        return 12

# ========== Run Model avec nettoyage ultra-agressif ==========
def run_one_model(cfg, train_df, test_df, max_len=320, model_idx=0):
    name = cfg["name"]
    path = cfg["path"]
    T = float(cfg.get("T", 1.0))
    
    print(f"\n{'='*70}")
    print(f"MODÃˆLE [{model_idx+1}/3]: {name}")
    print(f"Temperature: {T:.4f} | MAX_LEN: {max_len} | EPOCHS: {FINITION_EPOCHS}")
    print(f"{'='*70}")
    
    # Nettoyage AGRESSIF avant
    print("[CLEANUP] Nettoyage agressif prÃ©-modÃ¨le...")
    cleanup_disk(verbose=False)
    check_disk_space()
    
    set_seed(42)
    
    print("[LOAD] Tokenizer...")
    tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    data_collator = DataCollatorWithPadding(tokenizer=tok)
    
    print("[LOAD] Model...")
    config = AutoConfig.from_pretrained(
        path, num_labels=n_classes, id2label=id2label, label2id=label2id,
        cache_dir="/kaggle/temp/hf"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        path, config=config, ignore_mismatched_sizes=True,
        local_files_only=True, cache_dir="/kaggle/temp/hf"
    ).to(device)
    
    print("[DATA] PrÃ©paration datasets...")
    tr_ds = TextClsDataset(train_df, tok, max_len, label2id=label2id, has_label=True)
    te_ds = TextClsDataset(test_df, tok, max_len, label2id=None, has_label=False)
    
    bsz = bs_for_model(name, max_len)
    print(f"[INFO] Batch size: {bsz} (rÃ©duit pour Ã©conomiser)")
    
    eff_devices = max(1, torch.cuda.device_count())
    steps_per_epoch = math.ceil(len(tr_ds) / (bsz * eff_devices) / 2)
    
    out_dir = f"/kaggle/temp/out/{name}"
    args = build_training_args(
        output_dir=out_dir, steps_per_epoch=steps_per_epoch,
        batch_size=bsz, grad_accum=2, epochs=FINITION_EPOCHS
    )
    
    trainer = Trainer(
        model=model, args=args, train_dataset=tr_ds,
        tokenizer=tok, data_collator=data_collator
    )
    
    print(f"[TRAIN] {FINITION_EPOCHS} epochs...")
    import time
    train_start = time.time()
    trainer.train()
    train_time = (time.time() - train_start) / 60
    print(f"  âœ“ Training: {train_time:.1f} min")
    
    print("[PREDICT] GÃ©nÃ©ration logits...")
    pred_start = time.time()
    logits = trainer.predict(te_ds).predictions
    pred_time = (time.time() - pred_start)
    print(f"  âœ“ PrÃ©diction: {pred_time:.1f} sec")
    
    reindex = reindex_to_canonical(config.id2label, canonical_labels)
    logits = logits[:, reindex] / max(T, 1e-3)
    
    logits_path = f"/kaggle/working/logits_{model_idx}.npy"
    np.save(logits_path, logits)
    print(f"[SAVE] {logits_path}")
    
    # Nettoyage ULTRA-AGRESSIF aprÃ¨s
    print("[CLEANUP] Nettoyage ultra-agressif post-modÃ¨le...")
    del trainer, model, tok, tr_ds, te_ds, data_collator, config, args, logits
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        shutil.rmtree(out_dir, ignore_errors=True)
    except:
        pass
    
    cleanup_disk(verbose=False)
    
    # Supprimer aussi les caches cachÃ©s
    os.system("rm -rf ~/.cache/* 2>/dev/null")
    os.system("rm -rf /tmp/* 2>/dev/null")
    
    print(f"âœ“ {name} terminÃ© ({train_time:.1f}min)")
    check_disk_space()
    
    return logits_path

# ========== EXÃ‰CUTION ==========
print("\n" + "="*70)
print("DÃ‰BUT ENTRAÃ�NEMENT")
print("="*70)

import time
total_start = time.time()
logits_paths = []

for i, cfg in enumerate(MODELS):
    try:
        logits_path = run_one_model(cfg, train, test, max_len=MAX_LEN, model_idx=i)
        logits_paths.append(logits_path)
        
        elapsed = (time.time() - total_start) / 60
        remaining = (elapsed / (i + 1)) * (len(MODELS) - i - 1)
        print(f"\n[PROGRESS] {i+1}/{len(MODELS)} | Elapsed: {elapsed:.1f}min | Restant: ~{remaining:.1f}min")
        
    except Exception as e:
        print(f"\nâš ï¸�  ERREUR sur {cfg['name']}: {e}")
        print("Continuons avec les autres modÃ¨les...")
        continue

if len(logits_paths) == 0:
    raise RuntimeError("Aucun modÃ¨le n'a pu Ãªtre entraÃ®nÃ©!")

# ========== ENSEMBLE ==========
print("\n" + "="*70)
print("ENSEMBLE")
print("="*70)

print(f"[LOAD] Chargement {len(logits_paths)} modÃ¨les...")
all_logits = [np.load(path) for path in logits_paths]

mean_logits = np.mean(all_logits, axis=0)
probs = softmax_np(mean_logits, axis=1)

top3_idx = np.argsort(-probs, axis=1)[:, :3]
idx2label = {i: lbl for i, lbl in enumerate(canonical_labels)}
top3_labels = np.vectorize(idx2label.get)(top3_idx)
pred_strings = [" ".join(row) for row in top3_labels]

submission = sample.copy()
submission["Category:Misconception"] = pred_strings

print("\n" + "="*70)
print("STATISTIQUES")
print("="*70)
conf_max = probs.max(axis=1)
print(f"Confidence moyenne: {conf_max.mean():.4f}")
print(f"Confidence mÃ©diane: {np.median(conf_max):.4f}")
entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)
print(f"Entropie moyenne: {entropy.mean():.4f}")

total_time = (time.time() - total_start) / 60
print(f"\nTemps total: {total_time:.1f} min")

submission.to_csv("submission.csv", index=False)
cleanup_disk(verbose=False)

print("\n" + "="*70)
print("âœ… TERMINÃ‰")
print("="*70)
print(f"Option testÃ©e: {OPTION_NAME}")
print(f"ModÃ¨les rÃ©ussis: {len(logits_paths)}/3")
print("\nğŸ“„ submission.csv sauvegardÃ©")
print("="*70)

