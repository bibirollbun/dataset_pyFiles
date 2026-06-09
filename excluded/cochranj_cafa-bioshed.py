#!/usr/bin/env python3
# ============================================================
# CAFA-5 — ESM2-650M Embedder (Streaming + Shards + Hotfix)
# Outputs:
#   /kaggle/working/emb_cache/train_<id>.npz
#   /kaggle/working/emb_cache/test_<id>.npz
#
# Run multiple times with:
#   MODE=train / test
#   SHARDS=<N>, SHARD_ID=0..N-1
#
# Then bundle emb_cache as dataset for Notebook B.
# ============================================================

import os
os.environ.setdefault("TRANSFORMERS_SKIP_CHAT_TEMPLATE_LOAD", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# ----------------------
# Chat-template hotfix
# ----------------------
def _install_chat_template_hotfix():
    try:
        import transformers
        from transformers.utils import hub as _tf_hub
        orig = getattr(_tf_hub, "list_repo_templates", None)
        if callable(orig):
            def _safe(*a, **k):
                try:
                    return orig(*a, **k)
                except Exception:
                    return []
            _tf_hub.list_repo_templates = _safe
            print("[Hotfix] transformers.utils.hub.list_repo_templates patched.")
    except Exception as e:
        print(f"[Hotfix] hub patch skipped: {e}")
    try:
        from transformers import tokenization_utils_base as _tk
        orig2 = getattr(_tk, "list_repo_templates", None)
        if callable(orig2):
            def _safe2(*a, **k):
                try:
                    return orig2(*a, **k)
                except Exception:
                    return []
            _tk.list_repo_templates = _safe2
            print("[Hotfix] tokenization_utils_base.list_repo_templates patched.")
    except Exception as e:
        print(f"[Hotfix] tokenization patch skipped: {e}")

_install_chat_template_hotfix()
print("Chat-template hotfix installed.\n")

import gzip
from pathlib import Path
from glob import glob
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# ----------------------
# Config
# ----------------------
INPUT_DIR = "/kaggle/input/cafa-5-protein-function-prediction"
WORK_DIR  = "/kaggle/working"
CACHE_DIR = f"{WORK_DIR}/emb_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Which part to embed this run:
MODE = os.environ.get("MODE", "train")  # "train", "test", or "both"

# Sharding: strongly recommend SHARDS>=16 for 650M on P100
SHARDS   = int(os.environ.get("SHARDS", "32"))
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))

CFG = dict(
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
    esm2_name="facebook/esm2_t33_650M_UR50D",
    seq_stride=1022,
    seq_pool="mean",
    batch_size_embed=1,      # keep 1 for safety with 650M
    cache_fp16=True,
)

device = torch.device(CFG["device"])
USE_CUDA = torch.cuda.is_available()

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(CFG["seed"])

def _glob_one(patterns):
    for pat in patterns:
        hits = sorted(glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None

TRAIN_FASTA = _glob_one([
    f"{INPUT_DIR}/Train/train_sequences.fasta",
    f"{INPUT_DIR}/Train/train_sequences.fa",
    f"{INPUT_DIR}/**/*train*sequence*.fa*",
])
TEST_FASTA  = _glob_one([
    f"{INPUT_DIR}/Test/testsuperset.fasta",
    f"{INPUT_DIR}/Test/test_superset.fasta",
    f"{INPUT_DIR}/Test/targets.fasta",
    f"{INPUT_DIR}/**/*test*super*set*.fa*",
    f"{INPUT_DIR}/**/*target*.fa*",
])

print(f"Device: {device} | MODE={MODE} | SHARD {SHARD_ID}/{SHARDS-1}")
print("TRAIN_FASTA:", TRAIN_FASTA)
print("TEST_FASTA :", TEST_FASTA)

assert TRAIN_FASTA or TEST_FASTA, "No FASTA files found."

# ----------------------
# Dtype + AMP
# ----------------------
def _choose_dtype():
    if not USE_CUDA:
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

WEIGHT_DTYPE = _choose_dtype()
AMP_ENABLED = USE_CUDA

def amp_ctx():
    if AMP_ENABLED:
        return torch.amp.autocast(device_type="cuda", dtype=WEIGHT_DTYPE)
    return nullcontext()

# ----------------------
# Load ESM2-650M
# ----------------------
print(f"Loading backbone: {CFG['esm2_name']} with dtype={WEIGHT_DTYPE}")
esm_tokenizer = AutoTokenizer.from_pretrained(
    CFG["esm2_name"],
    use_fast=True
)
esm_model = AutoModel.from_pretrained(
    CFG["esm2_name"],
    torch_dtype=WEIGHT_DTYPE,
    low_cpu_mem_usage=True
).to(device)
esm_model.eval()

esm_dim = getattr(getattr(esm_model, "config", None), "hidden_size", 1280)
print("esm_dim:", esm_dim)

# ----------------------
# FASTA streaming (no big dict)
# ----------------------
def fasta_records(path):
    """
    Stream (id, seq) from FASTA without keeping all in RAM.
    """
    if not path:
        return
    is_gz = str(path).endswith(".gz")
    fh = gzip.open(path, "rt") if is_gz else open(path, "r")
    ident = None
    buf = []
    with fh as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                # flush previous
                if ident is not None and buf:
                    seq = "".join(buf).strip()
                    if seq:
                        yield ident, seq
                # new id
                raw = line[1:].strip().split()[0]
                if "|" in raw:
                    parts = raw.split("|")
                    if len(parts) >= 2:
                        raw = parts[1]
                ident = raw
                buf = []
            else:
                buf.append(line.strip())
        # last record
        if ident is not None and buf:
            seq = "".join(buf).strip()
            if seq:
                yield ident, seq

def in_shard(tid: str) -> bool:
    if SHARDS <= 1:
        return True
    h = (hash(tid) % SHARDS + SHARDS) % SHARDS
    return h == SHARD_ID

# ----------------------
# Embedding helpers
# ----------------------
def _chunk_ids(ids, max_len=1022, stride=1022):
    s = 0
    out = []
    while s < len(ids):
        e = min(s + max_len, len(ids))
        out.append(ids[s:e])
        if e == len(ids):
            break
        s = max(0, e - stride)
    return out

_EMBED_BS = [max(1, CFG["batch_size_embed"])]

@torch.inference_mode()
def embed_sequence(seq: str):
    toks = esm_tokenizer(seq, add_special_tokens=False)["input_ids"]
    if len(toks) == 0:
        return np.zeros((esm_dim,), np.float32)

    chunks = _chunk_ids(toks, max_len=1022, stride=CFG["seq_stride"])
    outs = []
    i = 0

    while i < len(chunks):
        bs = _EMBED_BS[0]
        batch = chunks[i : i + bs]
        built = [esm_tokenizer.build_inputs_with_special_tokens(x) for x in batch]
        L = max(len(x) for x in built)

        input_ids = torch.zeros((len(built), L), dtype=torch.long, device=device)
        mask      = torch.zeros((len(built), L), dtype=torch.long, device=device)
        for bi, x in enumerate(built):
            l = len(x)
            input_ids[bi, :l] = torch.as_tensor(x, device=device)
            mask[bi, :l] = 1

        try:
            with amp_ctx():
                out = esm_model(input_ids=input_ids, attention_mask=mask).last_hidden_state
                pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            outs.append(pooled.detach().float().cpu().numpy())
            del out, pooled, input_ids, mask
            if USE_CUDA:
                torch.cuda.empty_cache()
            i += bs
        except torch.cuda.OutOfMemoryError:
            if USE_CUDA:
                torch.cuda.empty_cache()
            _EMBED_BS[0] = max(1, bs // 2)
            print(f"[embed_sequence] OOM → reducing batch_size_embed to {_EMBED_BS[0]}")
            if _EMBED_BS[0] == bs:
                # stuck; return safe zero to avoid killing job
                return np.zeros((esm_dim,), np.float32)

    if not outs:
        return np.zeros((esm_dim,), np.float32)
    return np.vstack(outs).mean(0)

def _save_npz(path, v_seq):
    if v_seq is None:
        return
    if CFG["cache_fp16"]:
        v_seq = v_seq.astype(np.float16)
    np.savez_compressed(path, v_seq=v_seq)

# ----------------------
# Main embedding loops
# ----------------------
def run_split(name, fasta_path):
    if not fasta_path:
        return
    print(f"\n=== Embedding {name} split on shard {SHARD_ID}/{SHARDS-1} ===")
    # We don't know count without scanning; just stream with tqdm without total.
    for tid, seq in tqdm(fasta_records(fasta_path), desc=f"{name} (stream)"):
        if not tid or not seq:
            continue
        if not in_shard(tid):
            continue
        path = Path(CACHE_DIR) / f"{name}_{tid}.npz"
        if path.exists():
            continue
        try:
            v_seq = embed_sequence(seq)
        except Exception as e:
            print(f"[WARN] {name} {tid} failed: {e}")
            continue
        _save_npz(path, v_seq=v_seq)

if MODE in ("train", "both"):
    run_split("train", TRAIN_FASTA)

if MODE in ("test", "both"):
    run_split("test", TEST_FASTA)

print("\n✅ Done for this shard/mode. Collect emb_cache from /kaggle/working.")

