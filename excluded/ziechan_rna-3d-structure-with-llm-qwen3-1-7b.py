!pip install --no-index --find-links=/kaggle/input/vllm-offline-install blake3 msgspec py-cpuinfo tqdm requests transformers


!pip install --no-index --find-links=/kaggle/input/vllm-offline-install vllm


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
import re
import time
import random
import warnings
from collections import Counter
import polars as pl
import pandas as pd, numpy as np, json, math, re, gc, time, tqdm, torch, os, textwrap

import torch
import vllm
from vllm import LLM, SamplingParams

# import kaggle_evaluation.aimo_2_inference_server

warnings.simplefilter('ignore')
print('PyTorch version:', torch.__version__)
print('vLLM:', vllm.__version__)


# ------------------------------------------------------------------
#   Imports & checkpoint paths
# ------------------------------------------------------------------


CKPT   = "/kaggle/input/qwen-3-rna-grope-250/transformers/default/1"   # <-- your .safetensors or HF repo
TEST_CSV = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
OUT_CSV  = "/kaggle/working/submission.csv"

BATCH_SIZE    = 8      # prompts per vLLM batch
TEMPERATURE   = 0.9     # sampling settings for diversity
TOP_P         = 0.9
MAX_NEW_TOK   = 4096

# ------------------------------------------------------------------
#   vLLM engine
# ------------------------------------------------------------------
llm = LLM(
    model=CKPT, 
    tensor_parallel_size=2,
    max_model_len = 8192,
    dtype='float16',

)
sampling = SamplingParams(
    n             = 1,        # we call .generate() 5×
    temperature   = TEMPERATURE,
    top_p         = TOP_P,
    max_tokens    = MAX_NEW_TOK,
    # stop=['### Input:','### Response:'],  # never cross into next prompt
)



CHUNK = 100     # whatever you used in training

INSTR = (
    "You are given an RNA primary sequence of length **{L}**. "
    "This slice covers positions **{s}-{e}**.\n\n"
    "Return **exactly {n} lines**, one per nucleotide in the slice, "
    "with 5 TAB‑separated columns: global_index res x y z.\n\n"
    "Print nothing else."
)

ALPACA_TMPL = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

def spaced(seq:str)->str: return " ".join(seq)

# regex: capture last 3 floats on each line (handles Unicode minus)
_FLOAT3 = re.compile(
    r"(?:\S+\s+\S+\s+)?" r"(-?\d+(?:\.\d+)?)\s+" r"(-?\d+(?:\.\d+)?)\s+" r"(-?\d+(?:\.\d+)?)"
)

def rows_to_xyz(block:str)->np.ndarray:
    block = block.replace("\u2212","-").replace("\\t","\t")
    rows  = [list(map(float,t)) for t in _FLOAT3.findall(block)]
    return np.asarray(rows,dtype=np.float32) if rows else np.empty((0,3))




test_df  = pd.read_csv(TEST_CSV).rename(columns={"sequence":"seq"})
records  = []

for tid, grp in test_df.groupby("target_id", sort=False):
    seq, desc = grp["seq"].iloc[0], grp["description"].iloc[0]
    L = len(seq)
    for s in range(1, L+1, CHUNK):
        e   = min(s+CHUNK-1, L)
        sub = seq[s-1:e]
        prompt = ALPACA_TMPL.format(
            instruction = INSTR.format(L=L, s=s, e=e, n=len(sub)),
            input = textwrap.dedent(f"""\
                target_id: {tid}
                full_sequence: {spaced(seq)}
                slice_start: {s}
                slice_end: {e}
                slice_seq: {spaced(sub)}
                full_length: {L}
                description: {desc}""")
        )
        records.append({
            "prompt":      prompt,
            "target_id":   tid,
            "slice_start": s,
            "n":           len(sub),
            "slice_seq":   sub,          # ← keep the plain 5′‑to‑3′ letters
        })
print("Total slices:", len(records))


# ================================================================
# 4   Generate 5 predictions per slice
# ================================================================
def vllm_generate(batch): return [o.outputs[0].text for o in llm.generate(batch, sampling)]

all_xyz = {k: [] for k in range(5)}
for k in range(5):
    for i in tqdm.trange(0, len(records), BATCH_SIZE, desc=f"pass{k+1}/5"):
        batch_prompts = [r["prompt"] for r in records[i:i+BATCH_SIZE]]
        outs = vllm_generate(batch_prompts)
        all_xyz[k].extend(outs)
    gc.collect()




# all_xyz[4]


SAMPLE_CSV = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"


# ================================================================
# 5   Assemble submission rows
# ================================================================

sub_rows = []
for idx, rec in enumerate(records):
    n     = rec["n"]
    tid   = rec["target_id"]
    off   = rec["slice_start"]
    seq   = rec["slice_seq"]          # ← fetch stored slice letters

    # pad / truncate xyz blocks exactly as before
    blocks = []
    for k in range(5):
        arr = rows_to_xyz(all_xyz[k][idx])
        m   = len(arr)
        if   m == n: blocks.append(arr)
        elif m > n: blocks.append(arr[:n])
        else:
            pad = np.zeros((n,3), dtype=np.float32); pad[:m] = arr
            blocks.append(pad)

    # one row per nucleotide
    for r in range(n):
        row = {
            "ID":      f"{tid}_{off + r}",
            "resname": seq[r],        # ← true A/C/G/U letter
            "resid":   r + 1,
        }
        for k, arr in enumerate(blocks, 1):
            x, y, z = arr[r]
            row[f"x_{k}"], row[f"y_{k}"], row[f"z_{k}"] = x, y, z
        sub_rows.append(row)


submission = pd.DataFrame(sub_rows)
submission = submission[pd.read_csv(SAMPLE_CSV, nrows=0).columns]  # align col order
submission.to_csv(OUT_CSV, index=False)
print("✓ submission.csv written with shape:", submission.shape)




