# === MAP: Charting Student Math Misunderstandings — Inference (LoRA) : FIRST CELL (clean imports) ===
# Fixed paths

LORA_ADAPTER_DIR = "/kaggle/input/deepseek-math70-lora/deepseek-math-classifier"
BASE_MODEL_DIR   = "/kaggle/input/deepseek-math/pytorch/deepseek-math-7b-instruct/1"

DATA_DIR = "/kaggle/input/map-charting-student-math-misunderstandings"
OUT_DIR  = "/kaggle/working"

import os
os.environ['TRANSFORMERS_NO_TORCHVISION'] = '1'

import json, pickle, warnings
import numpy as np
import pandas as pd
import torch

from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

print("LoRA adapter dir:", LORA_ADAPTER_DIR)
print("Base model dir  :", BASE_MODEL_DIR)
print("Data dir        :", DATA_DIR)
print("Out dir         :", OUT_DIR)

# Memory tweak
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True

# Silence sklearn warning
warnings.filterwarnings(
    "ignore",
    message=r"Trying to unpickle estimator MultiLabelBinarizer",
    category=UserWarning,
    module="sklearn.base",
)

# ---------------- Label mapping ----------------
mlb_path = os.path.join(LORA_ADAPTER_DIR, "mlb.pkl")
if not os.path.exists(mlb_path):
    raise FileNotFoundError(f"{mlb_path} not found.")
with open(mlb_path, "rb") as f:
    mlb = pickle.load(f)
num_labels = len(mlb.classes_)
print("Num classes:", num_labels)

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
tokenizer.padding_side = "right"
pad_id = tokenizer.pad_token_id
print("PAD token id:", pad_id)

# ---------------- Load Base Model ----------------
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    )
    print("Base model loaded in 4-bit.")
except Exception as e:
    print("4-bit load failed:", e)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=num_labels,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

'''
# ---------------- Load Base Model (8-bit) ----------------
try:
    # Use LLM.int8 for better accuracy vs 4-bit while still saving memory
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,                 # <<< switch to 8-bit
        llm_int8_threshold=6.0,            # default; safe for most models
        llm_int8_has_fp16_weight=False,    # keep weights int8 on GPU
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",                 # let HF place the modules
        low_cpu_mem_usage=True,
        local_files_only=True,
        # torch_dtype left to model default; logits will be cast to float later
    )
    print("Base model loaded in 8-bit (LLM.int8).")
except Exception as e:
    print("8-bit load failed:", e)
    # Fallback: non-quantized load (may OOM on small GPUs)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=num_labels,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
'''
base_model.config.pad_token_id = pad_id
base_model.config.problem_type = "single_label_classification"

try:
    base_model.resize_token_embeddings(len(tokenizer))
except Exception as e:
    print("Skip resize_token_embeddings:", e)





from pathlib import Path
from peft import PeftModel

adapter_dir = Path(LORA_ADAPTER_DIR).resolve()
assert (adapter_dir / "adapter_config.json").exists(), f"Missing adapter_config.json in {adapter_dir}"
# one of these should exist:
assert (adapter_dir / "adapter_model.safetensors").exists() or (adapter_dir / "adapter_model.bin").exists(), \
       f"Missing adapter weights in {adapter_dir}"

# Force local load to avoid HF Hub validation
model = PeftModel.from_pretrained(
    base_model,
    str(adapter_dir),
    is_trainable=False,
    local_files_only=True,   # <- key line
)
print("✅ Loaded LoRA adapter from:", adapter_dir)












'''
# ---------------- Attach LoRA adapter ----------------
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_DIR,
    device_map="auto",
    local_files_only=True,
)
model.config.pad_token_id = pad_id
model.eval()
'''














# ---------------- Load classifier head ----------------
head_path = os.path.join(LORA_ADAPTER_DIR, "classifier_head.pt")
if os.path.exists(head_path):
    base_core = getattr(model, "base_model", None) or model.get_base_model()
    head_module = getattr(base_core, "score", None) or getattr(model, "score", None)
    if head_module is not None:
        state = torch.load(head_path, map_location="cpu")
        # map keys if needed
        if not all(k in state for k in ("weight", "bias")):
            mapped = {}
            for k, v in state.items():
                nk = k.replace("classifier.", "")
                mapped[nk] = v
            state = mapped
        missing, unexpected = head_module.load_state_dict(state, strict=False)
        print(f"✅ Loaded classifier head from {head_path}")
        print("   missing:", missing, "unexpected:", unexpected)
        print("   head shape:", tuple(head_module.weight.shape))
    else:
        print("⚠️ Could not locate classification head module.")
else:
    print("⚠️ classifier_head.bin not found — model will use random head.")

# ---------------- Move model to CUDA if needed ----------------
if not hasattr(model, "hf_device_map") and device.type == "cuda":
    try:
        model.to(device)
    except RuntimeError as oom:
        print("Keeping on CPU to avoid OOM:", oom)

print("✅ Model ready for inference.")

# ---------------- Inference Dataset ----------------
from torch.utils.data import Dataset, DataLoader

class InferenceDS(Dataset):
    def __init__(self, df, tokenizer, max_length=1024):
        self.df = df.copy()
        for col in ["QuestionText", "MC_Answer", "StudentExplanation"]:
            if col not in self.df.columns:
                self.df[col] = ""
        self.texts = [
            f"Question: {q}\nAnswer: {a}\nExplanation: {e}"
            for q, a, e in zip(
                self.df["QuestionText"].fillna(""),
                self.df["MC_Answer"].fillna(""),
                self.df["StudentExplanation"].fillna(""),
            )
        ]
        self.enc = tokenizer(self.texts, padding=False, truncation=True, max_length=max_length)

    def __len__(self): return len(self.df)
    def __getitem__(self, i): return {k: torch.tensor(v[i]) for k, v in self.enc.items()}

def collate_fn(batch):
    return tokenizer.pad(batch, padding=True, return_tensors="pt")




#https://drive.google.com/drive/folders/1qiFMuHSaSRPK17qpqElXWKhir6VIjoc0?usp=sharing

#!gdown --id 1qiFMuHSaSRPK17qpqElXWKhir6VIjoc0


# === MAP: Inference (LoRA) — CELL 1: load + helpers ===
# Fixed paths
'''
LORA_ADAPTER_DIR = "/kaggle/input/deepseek-math70-lora/deepseek-math-classifier"
BASE_MODEL_DIR   = "/kaggle/input/deepseek-math/pytorch/deepseek-math-7b-instruct/1"

DATA_DIR = "/kaggle/input/map-charting-student-math-misunderstandings"
OUT_DIR  = "/kaggle/working"

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import torch

from peft import PeftModel
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
)

# Perf & warnings
os.environ['TRANSFORMERS_NO_TORCHVISION'] = '1'
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
warnings.filterwarnings("ignore", message=r"Trying to unpickle estimator MultiLabelBinarizer", category=UserWarning, module="sklearn.base")

# ---------------- Label mapping ----------------
mlb_path = os.path.join(LORA_ADAPTER_DIR, "mlb.pkl")
with open(mlb_path, "rb") as f: mlb = pickle.load(f)
num_labels = len(mlb.classes_)
print("Num classes:", num_labels)

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
tokenizer.padding_side = "right"
pad_id = tokenizer.pad_token_id
print("PAD token id:", pad_id)

# ---------------- Load Base Model (8-bit) ----------------
'''
'''
try:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print("Base model loaded in 8-bit (LLM.int8).")
except Exception as e:
    print("8-bit load failed:", e)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=num_labels,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
'''
'''
# ---------------- Load Base Model (4-bit NF4) ----------------
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print("Base model loaded in 4-bit (NF4).")
except Exception as e:
    print("4-bit load failed:", e)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=num_labels,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )




base_model.config.pad_token_id = pad_id
base_model.config.problem_type = "single_label_classification"
try: base_model.resize_token_embeddings(len(tokenizer))
except Exception: pass

# ---------------- Attach LoRA adapter ----------------
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR, device_map="auto", local_files_only=True)
model.config.pad_token_id = pad_id
model.eval()

# ---------------- Load classifier head ----------------
head_path = os.path.join(LORA_ADAPTER_DIR, "classifier_head.bin")
if os.path.exists(head_path):
    base_core = getattr(model, "base_model", None) or model.get_base_model()
    head_module = getattr(base_core, "score", None) or getattr(model, "score", None)
    if head_module is not None:
        state = torch.load(head_path, map_location="cpu")
        if not all(k in state for k in ("weight", "bias")):
            state = {k.replace("classifier.", ""): v for k, v in state.items()}
        head_module.load_state_dict(state, strict=False)
        print("✅ Loaded classifier head.")
else:
    print("⚠️ classifier_head.bin not found — using random head.")

# ---------------- Device map helper ----------------
def get_runtime_device(m, fallback=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if hasattr(m, "hf_device_map") and isinstance(m.hf_device_map, dict) and m.hf_device_map:
        for v in m.hf_device_map.values():
            s = str(v)
            if s.startswith("cuda"): return torch.device(s)
            if s.isdigit(): return torch.device(f"cuda:{s}")
    try: return next(m.parameters()).device
    except StopIteration: return fallback

target_device = get_runtime_device(model)
print("Runtime device:", target_device)

# ---------------- Prompt variants (non-instruction, minimal edits) ----------------
# Keep SAME order & meaning; tiny variations only (punctuation/separators) to stay close to training.
def variant_a(q, a, e):
    return f"Question: {q}\nAnswer: {a}\nExplanation: {e}"


def variant_b(q, a, e):
    return f"Question: {q}\n\nAnswer: {a}\n\nExplanation: {e}"       # blank lines

def variant_c(q, a, e):
    return f"Question: {q}\nAnswer: {a}\nExplanation: {e}\n"         # trailing newline

def variant_d(q, a, e):
    return f"Q: {q}\nAnswer: {a}\nExplanation: {e}"                  # short 'Q:' only


#PROMPT_BANK = [variant_a, variant_b, variant_c, variant_d]  # start with 3–4; you can prune/expand

PROMPT_BANK = [variant_a]  # start with 3–4; you can prune/expand

# ---------------- Tokenize a dataframe with a variant ----------------
def encode_with_variant(df, tokenizer, v_fn, max_length=1024):
    texts = [
        v_fn(
            (q or "").strip(),
            (a or "").strip(),
            (e or "").strip()
        )
        for q, a, e in zip(
            df.get("QuestionText", pd.Series([""]*len(df))).fillna(""),
            df.get("MC_Answer", pd.Series([""]*len(df))).fillna(""),
            df.get("StudentExplanation", pd.Series([""]*len(df))).fillna(""),
        )
    ]
    return tokenizer(texts, padding=False, truncation=True, max_length=1024)

print("✅ Setup complete.")
'''


model.eval()
with torch.no_grad():
    ids = tokenizer("test", return_tensors="pt").to(next(model.parameters()).device)
    out = model(**ids)
    print(out.logits.shape)  # لازم تكون [1, 65]
    print(torch.isfinite(out.logits).all())  # True

probs = torch.softmax(out.logits, dim=-1)
print(probs.shape)   # [1, 65]
print("Total No. of Props",probs.sum(dim=-1))  # لازم تكون ≈ 1


# === MAP: Run inference on test.csv & build submission (robust device handling) ===

from transformers import DataCollatorWithPadding
import torch, os, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from IPython.display import display

TEST_PATH   = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUB_PATH    = os.path.join(OUT_DIR, "submission.csv")

print("Reading:", TEST_PATH)
test_df = pd.read_csv(TEST_PATH)

# Detect ID column from test (fallback to row_id if none)
id_col = None
for c in ["row_id", "id", "Id", "example_id"]:
    if c in test_df.columns:
        id_col = c
        break
if id_col is None:
    test_df["row_id"] = np.arange(len(test_df))
    id_col = "row_id"
print("ID column:", id_col)

# Fast collator (quiet fast-tokenizer warning)
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)

# Dataset/Loader
batch_size = 16 if torch.cuda.is_available() else 2  # slightly larger for faster inference
ds = InferenceDS(test_df, tokenizer, max_length=1024)
loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# --- robust device selection when using device_map="auto" ---
def get_runtime_device(m, fallback=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if hasattr(m, "hf_device_map") and isinstance(m.hf_device_map, dict) and len(m.hf_device_map) > 0:
        devs = set()
        for v in m.hf_device_map.values():
            v = str(v)
            if v.startswith("cuda"): devs.add(v)
            elif v.isdigit(): devs.add(f"cuda:{v}")
            elif v == "cpu": devs.add("cpu")
        cuda_devs = [d for d in devs if d.startswith("cuda")]
        return torch.device(cuda_devs[0]) if cuda_devs else torch.device("cpu")
    try:
        return next(m.parameters()).device
    except StopIteration:
        return fallback

target_device = get_runtime_device(model)
print("Runtime device for inputs:", target_device)

all_top3, all_probs = [], []

# --- Inference ---
model.eval()
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(target_device) for k, v in batch.items()}

        # Safe autocast for BF16 inference
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(**batch).logits.float()
            probs_t = torch.softmax(logits, dim=-1).to(torch.float32)

        probs = probs_t.cpu().numpy()
        top3_idx = np.argsort(-probs, axis=1)[:, :3]
        all_top3.extend(top3_idx.tolist())
        all_probs.extend(probs.tolist())

# --- Map indices -> class labels ---
top3_labels = [[mlb.classes_[j] for j in row] for row in all_top3]
print("Sample top-3:", top3_labels[0] if len(top3_labels) else None)

# --- Build submission from sample format ---
print("Reading:", SAMPLE_PATH)
sample_sub = pd.read_csv(SAMPLE_PATH)
print("Sample submission columns:", list(sample_sub.columns))

sub = sample_sub.copy()
non_id_cols = [c for c in sub.columns if c != id_col]

if len(non_id_cols) == 1:
    col = non_id_cols[0]
    sub[id_col] = test_df[id_col].values
    sub[col] = [" ".join(lbls) for lbls in top3_labels]

elif len(non_id_cols) == 3:
    c1, c2, c3 = non_id_cols
    sub[id_col] = test_df[id_col].values
    sub[c1] = [x[0] for x in top3_labels]
    sub[c2] = [x[1] for x in top3_labels]
    sub[c3] = [x[2] for x in top3_labels]

elif set(non_id_cols) == set(mlb.classes_):
    sub = pd.DataFrame({id_col: test_df[id_col].values})
    probs_np = np.array(all_probs, dtype=np.float32)
    for i, cls in enumerate(mlb.classes_):
        sub[cls] = probs_np[:, i]

else:
    print("⚠️ Unrecognized sample format. Writing generic [id, predictions].")
    sub = pd.DataFrame({
        id_col: test_df[id_col].values,
        "predictions": [" ".join(lbls) for lbls in top3_labels],
    })

sub.to_csv(SUB_PATH, index=False)
print(f"✅ submission.csv written to: {SUB_PATH}")
display(sub.head())



# === MAP: Inference Ensemble + MC Dropout + Temperature — CELL 2 ===
'''
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from IPython.display import display

TEST_PATH   = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUB_PATH    = os.path.join(OUT_DIR, "submission.csv")

print("Reading:", TEST_PATH)
test_df = pd.read_csv(TEST_PATH)

# Detect ID column
id_col = next((c for c in ["row_id","id","Id","example_id"] if c in test_df.columns), None)
if id_col is None:
    test_df["row_id"] = np.arange(len(test_df))
    id_col = "row_id"
print("ID column:", id_col)

# Collator / batch
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)
batch_size = 16 if torch.cuda.is_available() else 2

# --- MC Dropout control ---
def enable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.train()

N_SAMPLES   = 3     # try 5–10
TEMPERATURE = 1.1   # tune 0.95–1.2 on your val set

# --- Run ensemble ---
logits_sum = None
for v_fn in PROMPT_BANK:
    enc = encode_with_variant(test_df, tokenizer, v_fn, max_length=1024)

    # Build a minimal dataset from tokenized dict
    class EncodedDS(torch.utils.data.Dataset):
        def __init__(self, enc_dict): self.enc = enc_dict
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self, i): return {k: torch.tensor(v[i]) for k, v in self.enc.items()}

    ds_enc = EncodedDS(enc)
    loader = DataLoader(ds_enc, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    # MC Dropout averaging
    model.eval()
    enable_dropout(model)
    cur_accum = None
    with torch.no_grad():
        for _ in range(N_SAMPLES):
            cur_logits = []
            for batch in loader:
                batch = {k: v.to(target_device) for k, v in batch.items()}
                # Use autocast for speed; logits -> float32 for stability
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    out = model(**batch).logits.float()
                cur_logits.append(out.cpu())
            cur_logits = torch.cat(cur_logits, dim=0)
            cur_accum = cur_logits if cur_accum is None else (cur_accum + cur_logits)
    cur_accum = cur_accum / N_SAMPLES

    logits_sum = cur_accum if logits_sum is None else (logits_sum + cur_accum)

# Average over variants, apply temperature, softmax
logits_avg = logits_sum / len(PROMPT_BANK)
probs = torch.softmax(logits_avg / TEMPERATURE, dim=-1).numpy()

# --- Top-3 extraction ---
top3_idx = np.argsort(-probs, axis=1)[:, :3]
top3_labels = [[mlb.classes_[j] for j in row] for row in top3_idx]
print("Sample top-3:", top3_labels[0] if len(top3_labels) else None)

# --- Build submission ---
sample_sub = pd.read_csv(SAMPLE_PATH)
non_id_cols = [c for c in sample_sub.columns if c != id_col]
sub = sample_sub.copy()

if len(non_id_cols) == 1:
    sub[id_col] = test_df[id_col].values
    sub[non_id_cols[0]] = [" ".join(lbls) for lbls in top3_labels]
elif len(non_id_cols) == 3:
    c1, c2, c3 = non_id_cols
    sub[id_col] = test_df[id_col].values
    sub[c1], sub[c2], sub[c3] = zip(*top3_labels)
elif set(non_id_cols) == set(mlb.classes_):
    sub = pd.DataFrame({id_col: test_df[id_col].values})
    for i, cls in enumerate(mlb.classes_):
        sub[cls] = probs[:, i].astype(np.float32)
else:
    sub = pd.DataFrame({id_col: test_df[id_col].values, "predictions": [" ".join(lbls) for lbls in top3_labels]})

sub.to_csv(SUB_PATH, index=False)
print(f"✅ submission.csv written to: {SUB_PATH}")
display(sub.head())
'''




