import torch, torchvision
import transformers, peft

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)


%%writefile train_qwen3_14b.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print("PID:", os.getpid(), "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
assert torch.cuda.is_available()
print("Visible device count:", torch.cuda.device_count())
torch.cuda.set_device(0)            # 0 == the ONLY visible GPU in this process
print("Using:", torch.cuda.get_device_name(0))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never hit the Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")     # Transformers offline
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")      # Datasets offline
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1") # no telemetry

import math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from typing import List, Tuple

from unsloth.chat_templates import CHAT_TEMPLATES

# =========================
# Merged CONSTANTS
# =========================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH",
    "/kaggle/input/qwen3-14b-unsloth-bnb-4bit/transformers/default/1"
)
DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/input/jigsaw-agile-community-rules/")

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."

# Inference knobs
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", 4))
WRITE_SUBMISSION = os.environ.get("WRITE_SUBMISSION", "1") == "1"  # write submission.csv

# =========================
# Merged UTILS
# =========================
def get_dataframe_to_train(data_path: str) -> pd.DataFrame:
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=1001).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])
    return dataframe


# =========================
# Unsloth training helpers
# =========================
def load_dataframe() -> pd.DataFrame:
    df = get_dataframe_to_train(DATA_PATH)  # body, rule, rule_violation
    if "completion" not in df.columns:
        df = df.copy()
        df["completion"] = df["rule_violation"].map(
            {1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER}
        )
    return df[["body", "rule", "completion"]]

def make_conversations_dataset(df: pd.DataFrame) -> Dataset:
    # Each sample: system + user + assistant(Yes/No)
    convos = []
    for body, rule, comp in zip(df["body"], df["rule"], df["completion"]):
        convos.append([
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
            {"role": "assistant", "content": str(comp)},
        ])
    return Dataset.from_dict({"conversations": convos})

def build_text_dataset(tokenizer, conv_dataset: Dataset):
    # Convert conversations -> single 'text' string via chat template.
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )[:-11]
            for conv in convos
        ]
        return {"text": texts}
    return conv_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=conv_dataset.column_names,
    )

# =========================
# Inference helpers (Unsloth, in-memory)
# =========================
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]
def _first_token_ids(tok, txt_or_texts) -> List[int]:
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_texts] if isinstance(txt_or_texts, str) else list(txt_or_texts)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def _encode_batch(tok, bodies, rules, device, max_len=512):
    """
    Manual left truncation + left padding + explicit position_ids.
    Returns tensors on `device` with keys: input_ids, attention_mask, position_ids.
    """
    pad_id = tok.pad_token_id
    if pad_id is None:
        # Qwen templates usually set pad_token = eos_token already in your code,
        # but keep a safe fallback:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # 1) Tokenize each sample via chat template
    seqs = []
    for body, rule in zip(bodies, rules):
        msgs = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
        ]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )

        # Left truncation: keep the last `max_len` tokens
        if len(ids) > max_len:
            ids = ids[-max_len:]

        seqs.append(torch.tensor(ids, dtype=torch.long))

    # 2) Left padding to a uniform length (no longer than max_len)
    B = len(seqs)
    T = min(max_len, max(len(x) for x in seqs)) if seqs else 1

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)

    for i, ids in enumerate(seqs):
        L = min(T, len(ids))
        # Left pad: write actual ids into the *rightmost* part
        input_ids[i, T - L : T] = ids[-L:]
        attention_mask[i, T - L : T] = 1

    # 3) position_ids: first non-pad gets 0; pads use 0 (safe default)
    #    (Equivalent to: pos = cumsum(mask) - 1, then masked_fill(pad, 0))
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    position_ids = position_ids.to(dtype=torch.long)

    # 4) Ship to device
    batch = {
        "input_ids":      input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "position_ids":   position_ids.to(device, non_blocking=True),
    }
    return batch


@torch.inference_mode()
def run_inference_unsloth_generate(
    model, tokenizer, data_path=DATA_PATH, batch_size=INFER_BATCH_SIZE,
    max_len=512, write_csv=WRITE_SUBMISSION, sort_by_length=True,
):
    tok = tokenizer

    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)
    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    # NOTE: this probably should be 'test.csv'; your current code reads 'train.csv'
    test_df = pd.read_csv(f"{data_path}/test.csv")
    bodies  = list(test_df["body"])
    rules   = list(test_df["rule"])
    rowids  = list(test_df["row_id"])

    N = len(bodies)
    if sort_by_length:
        approx_lens = [
            (len(tok.encode(b, add_special_tokens=False)) +
             len(tok.encode(r, add_special_tokens=False)))
            for b, r in zip(bodies, rules)
        ]
        sorted_idx = sorted(range(N), key=lambda i: min(approx_lens[i], max_len))
    else:
        sorted_idx = list(range(N))

    FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    probs_yes = [None] * N

    for i in range(0, N, batch_size):
        batch_indices = sorted_idx[i:i+batch_size]
        bb = [bodies[j] for j in batch_indices]
        rr = [rules[j]  for j in batch_indices]

        enc = _encode_batch(tok, bb, rr, device=device, max_len=max_len)

        out = model(**enc, use_cache=True)            # <— single forward pass
        step_scores = out.logits[:, -1, :]
        sel = step_scores[:, tgt_ids]
        logp = torch.log_softmax(sel.to(torch.float32), dim=-1)

        y_logp = (torch.logsumexp(logp[:, yes_idx], dim=-1)
                  if yes_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        n_logp = (torch.logsumexp(logp[:,  no_idx], dim=-1)
                  if no_idx  else torch.full((sel.size(0),), -1e9, device=sel.device))
        p_yes  = torch.softmax(torch.stack([y_logp, n_logp], dim=-1), dim=-1)[:, 0]

        for k, j in enumerate(batch_indices):
            probs_yes[j] = float(p_yes[k].item())

    # Build per-rule ranked scores in [0, 1]
    df_scores = pd.DataFrame({
        "row_id": rowids,
        "rule":   rules,
        "prob":   probs_yes,
    })

    grp   = df_scores.groupby("rule")
    rank  = grp["prob"].rank(method="average", ascending=True)   # low prob -> 1, high prob -> n
    n     = grp["prob"].transform("size")
    score = (rank - 1.0) / np.maximum(n - 1.0, 1.0)              # low -> 0.0, high -> 1.0
    df_scores["rule_violation"] = score

    out_df = df_scores[["row_id", "rule_violation"]].sort_values("row_id").reset_index(drop=True)
    if write_csv:
        out_df.to_csv("submission1.csv", index=False)
    return out_df


# =========================
# Main
# =========================
def main():
    # ---- Config knobs ----
    quant_mode = os.environ.get("QUANT_MODE", "4bit").lower()   # "4bit" or "8bit"
    load_in_4bit = (quant_mode == "4bit")
    dtype = None  # auto (fp16 on T4/V100, bf16 on Ampere+)

    # ---- Load base + LoRA via Unsloth ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_PATH,
        max_seq_length = 512,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
        load_in_8bit   = (quant_mode == "8bit"),
        local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1001,
        use_rslora=False,
        loftq_config=None,
    )

    # ---- Tokenizer ----
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # ---- Data ----
    df = load_dataframe()
    conv_dataset = make_conversations_dataset(df)
    train_dataset = build_text_dataset(tokenizer, conv_dataset)

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=0,
            logging_steps=10,
            optim="adamw_8bit",
            seed=1001,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=2,
        ),
    )

    # Only compute loss over assistant spans (our "Yes"/"No")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user",
        response_part = "<think>\n\n</think>\n\n",
    )

    trainer.train()

    # ---- Inference uses the SAME limit & left settings ----
    submission_df = run_inference_unsloth_generate(model, tokenizer, max_len=512)
    print(submission_df.head(10))
    print("Wrote submission.csv")

if __name__ == "__main__":
    main()


%%writefile train_phi4.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print("PID:", os.getpid(), "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
assert torch.cuda.is_available()
print("Visible device count:", torch.cuda.device_count())
torch.cuda.set_device(0)            # 0 == the ONLY visible GPU in this process
print("Using:", torch.cuda.get_device_name(0))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never hit the Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")     # Transformers offline
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")      # Datasets offline
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1") # no telemetry

import math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from typing import List, Tuple

from unsloth.chat_templates import CHAT_TEMPLATES

# =========================
# Merged CONSTANTS
# =========================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH",
    "/kaggle/input/phi-4-unsloth-bnb-4bit/transformers/default/1"
)
DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/input/jigsaw-agile-community-rules/")

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."

# Inference knobs
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", 4))
WRITE_SUBMISSION = os.environ.get("WRITE_SUBMISSION", "1") == "1"  # write submission.csv

# =========================
# Merged UTILS
# =========================
def get_dataframe_to_train(data_path: str) -> pd.DataFrame:
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=1002).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])
    return dataframe


# =========================
# Unsloth training helpers
# =========================
def load_dataframe() -> pd.DataFrame:
    df = get_dataframe_to_train(DATA_PATH)  # body, rule, rule_violation
    if "completion" not in df.columns:
        df = df.copy()
        df["completion"] = df["rule_violation"].map(
            {1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER}
        )
    return df[["body", "rule", "completion"]]

def make_conversations_dataset(df: pd.DataFrame) -> Dataset:
    # Each sample: system + user + assistant(Yes/No)
    convos = []
    for body, rule, comp in zip(df["body"], df["rule"], df["completion"]):
        convos.append([
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
            {"role": "assistant", "content": str(comp)},
        ])
    return Dataset.from_dict({"conversations": convos})

def build_text_dataset(tokenizer, conv_dataset: Dataset):
    # Convert conversations -> single 'text' string via chat template.
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )[:-10]
            for conv in convos
        ]
        return {"text": texts}
    return conv_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=conv_dataset.column_names,
    )

# =========================
# Inference helpers (Unsloth, in-memory)
# =========================
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]
def _first_token_ids(tok, txt_or_texts) -> List[int]:
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_texts] if isinstance(txt_or_texts, str) else list(txt_or_texts)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def _encode_batch(tok, bodies, rules, device, max_len=512):
    """
    Manual left truncation + left padding + explicit position_ids.
    Returns tensors on `device` with keys: input_ids, attention_mask, position_ids.
    """
    pad_id = tok.pad_token_id
    if pad_id is None:
        # Qwen templates usually set pad_token = eos_token already in your code,
        # but keep a safe fallback:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # 1) Tokenize each sample via chat template
    seqs = []
    for body, rule in zip(bodies, rules):
        msgs = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
        ]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Left truncation: keep the last `max_len` tokens
        if len(ids) > max_len:
            ids = ids[-max_len:]

        seqs.append(torch.tensor(ids, dtype=torch.long))

    # 2) Left padding to a uniform length (no longer than max_len)
    B = len(seqs)
    T = min(max_len, max(len(x) for x in seqs)) if seqs else 1

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)

    for i, ids in enumerate(seqs):
        L = min(T, len(ids))
        # Left pad: write actual ids into the *rightmost* part
        input_ids[i, T - L : T] = ids[-L:]
        attention_mask[i, T - L : T] = 1

    # 3) position_ids: first non-pad gets 0; pads use 0 (safe default)
    #    (Equivalent to: pos = cumsum(mask) - 1, then masked_fill(pad, 0))
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    position_ids = position_ids.to(dtype=torch.long)

    # 4) Ship to device
    batch = {
        "input_ids":      input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "position_ids":   position_ids.to(device, non_blocking=True),
    }
    return batch


@torch.inference_mode()
def run_inference_unsloth_generate(
    model, tokenizer, data_path=DATA_PATH, batch_size=INFER_BATCH_SIZE,
    max_len=512, write_csv=WRITE_SUBMISSION, sort_by_length=True,
):
    tok = tokenizer

    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)
    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    # NOTE: this probably should be 'test.csv'; your current code reads 'train.csv'
    test_df = pd.read_csv(f"{data_path}/test.csv")
    bodies  = list(test_df["body"])
    rules   = list(test_df["rule"])
    rowids  = list(test_df["row_id"])

    N = len(bodies)
    if sort_by_length:
        approx_lens = [
            (len(tok.encode(b, add_special_tokens=False)) +
             len(tok.encode(r, add_special_tokens=False)))
            for b, r in zip(bodies, rules)
        ]
        sorted_idx = sorted(range(N), key=lambda i: min(approx_lens[i], max_len))
    else:
        sorted_idx = list(range(N))

    FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    probs_yes = [None] * N

    for i in range(0, N, batch_size):
        batch_indices = sorted_idx[i:i+batch_size]
        bb = [bodies[j] for j in batch_indices]
        rr = [rules[j]  for j in batch_indices]

        enc = _encode_batch(tok, bb, rr, device=device, max_len=max_len)

        out = model(**enc, use_cache=True)            # <— single forward pass
        step_scores = out.logits[:, -1, :]
        sel = step_scores[:, tgt_ids]
        logp = torch.log_softmax(sel.to(torch.float32), dim=-1)

        y_logp = (torch.logsumexp(logp[:, yes_idx], dim=-1)
                  if yes_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        n_logp = (torch.logsumexp(logp[:,  no_idx], dim=-1)
                  if no_idx  else torch.full((sel.size(0),), -1e9, device=sel.device))
        p_yes  = torch.softmax(torch.stack([y_logp, n_logp], dim=-1), dim=-1)[:, 0]

        for k, j in enumerate(batch_indices):
            probs_yes[j] = float(p_yes[k].item())

    # Build per-rule ranked scores in [0, 1]
    df_scores = pd.DataFrame({
        "row_id": rowids,
        "rule":   rules,
        "prob":   probs_yes,
    })

    grp   = df_scores.groupby("rule")
    rank  = grp["prob"].rank(method="average", ascending=True)   # low prob -> 1, high prob -> n
    n     = grp["prob"].transform("size")
    score = (rank - 1.0) / np.maximum(n - 1.0, 1.0)              # low -> 0.0, high -> 1.0
    df_scores["rule_violation"] = score

    out_df = df_scores[["row_id", "rule_violation"]].sort_values("row_id").reset_index(drop=True)
    if write_csv:
        out_df.to_csv("submission4.csv", index=False)
    return out_df


# =========================
# Main
# =========================
def main():
    # ---- Config knobs ----
    quant_mode = os.environ.get("QUANT_MODE", "4bit").lower()   # "4bit" or "8bit"
    load_in_4bit = (quant_mode == "4bit")
    dtype = None  # auto (fp16 on T4/V100, bf16 on Ampere+)

    # ---- Load base + LoRA via Unsloth ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_PATH,
        max_seq_length = 512,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
        load_in_8bit   = (quant_mode == "8bit"),
        local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1002,
        use_rslora=False,
        loftq_config=None,
    )

    # ---- Tokenizer ----
    tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # ---- Data ----
    df = load_dataframe()
    conv_dataset = make_conversations_dataset(df)
    train_dataset = build_text_dataset(tokenizer, conv_dataset)

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=0,
            logging_steps=10,
            optim="adamw_8bit",
            seed=1002,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=2,
        ),
    )

    # Only compute loss over assistant spans (our "Yes"/"No")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user<|im_sep|>",
        response_part="<|im_start|>assistant<|im_sep|>",
    )

    trainer.train()

    # ---- Inference uses the SAME limit & left settings ----
    submission_df = run_inference_unsloth_generate(model, tokenizer, max_len=512)
    print(submission_df.head(10))
    print("Wrote submission.csv")

if __name__ == "__main__":
    main()


%%writefile train_qwen2.5_14b.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print("PID:", os.getpid(), "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
assert torch.cuda.is_available()
print("Visible device count:", torch.cuda.device_count())
torch.cuda.set_device(0)            # 0 == the ONLY visible GPU in this process
print("Using:", torch.cuda.get_device_name(0))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never hit the Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")     # Transformers offline
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")      # Datasets offline
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1") # no telemetry

import math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from typing import List, Tuple

from unsloth.chat_templates import CHAT_TEMPLATES

# =========================
# Merged CONSTANTS
# =========================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH",
    "/kaggle/input/qwen2.5-14b-instruct-bnb-4bit/transformers/default/1"
)
DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/input/jigsaw-agile-community-rules/")

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."

# Inference knobs
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", 4))
WRITE_SUBMISSION = os.environ.get("WRITE_SUBMISSION", "1") == "1"  # write submission.csv

# =========================
# Merged UTILS
# =========================
def get_dataframe_to_train(data_path: str) -> pd.DataFrame:
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=1004).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])
    return dataframe


# =========================
# Unsloth training helpers
# =========================
def load_dataframe() -> pd.DataFrame:
    df = get_dataframe_to_train(DATA_PATH)  # body, rule, rule_violation
    if "completion" not in df.columns:
        df = df.copy()
        df["completion"] = df["rule_violation"].map(
            {1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER}
        )
    return df[["body", "rule", "completion"]]

def make_conversations_dataset(df: pd.DataFrame) -> Dataset:
    # Each sample: system + user + assistant(Yes/No)
    convos = []
    for body, rule, comp in zip(df["body"], df["rule"], df["completion"]):
        convos.append([
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
            {"role": "assistant", "content": str(comp)},
        ])
    return Dataset.from_dict({"conversations": convos})

def build_text_dataset(tokenizer, conv_dataset: Dataset):
    # Convert conversations -> single 'text' string via chat template.
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )[:-11]
            for conv in convos
        ]
        return {"text": texts}
    return conv_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=conv_dataset.column_names,
    )

# =========================
# Inference helpers (Unsloth, in-memory)
# =========================
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]
def _first_token_ids(tok, txt_or_texts) -> List[int]:
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_texts] if isinstance(txt_or_texts, str) else list(txt_or_texts)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def _encode_batch(tok, bodies, rules, device, max_len=512):
    """
    Manual left truncation + left padding + explicit position_ids.
    Returns tensors on `device` with keys: input_ids, attention_mask, position_ids.
    """
    pad_id = tok.pad_token_id
    if pad_id is None:
        # Qwen templates usually set pad_token = eos_token already in your code,
        # but keep a safe fallback:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # 1) Tokenize each sample via chat template
    seqs = []
    for body, rule in zip(bodies, rules):
        msgs = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
        ]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Left truncation: keep the last `max_len` tokens
        if len(ids) > max_len:
            ids = ids[-max_len:]

        seqs.append(torch.tensor(ids, dtype=torch.long))

    # 2) Left padding to a uniform length (no longer than max_len)
    B = len(seqs)
    T = min(max_len, max(len(x) for x in seqs)) if seqs else 1

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)

    for i, ids in enumerate(seqs):
        L = min(T, len(ids))
        # Left pad: write actual ids into the *rightmost* part
        input_ids[i, T - L : T] = ids[-L:]
        attention_mask[i, T - L : T] = 1

    # 3) position_ids: first non-pad gets 0; pads use 0 (safe default)
    #    (Equivalent to: pos = cumsum(mask) - 1, then masked_fill(pad, 0))
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    position_ids = position_ids.to(dtype=torch.long)

    # 4) Ship to device
    batch = {
        "input_ids":      input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "position_ids":   position_ids.to(device, non_blocking=True),
    }
    return batch


@torch.inference_mode()
def run_inference_unsloth_generate(
    model, tokenizer, data_path=DATA_PATH, batch_size=INFER_BATCH_SIZE,
    max_len=512, write_csv=WRITE_SUBMISSION, sort_by_length=True,
):
    tok = tokenizer

    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)
    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    # NOTE: this probably should be 'test.csv'; your current code reads 'train.csv'
    test_df = pd.read_csv(f"{data_path}/test.csv")
    bodies  = list(test_df["body"])
    rules   = list(test_df["rule"])
    rowids  = list(test_df["row_id"])

    N = len(bodies)
    if sort_by_length:
        approx_lens = [
            (len(tok.encode(b, add_special_tokens=False)) +
             len(tok.encode(r, add_special_tokens=False)))
            for b, r in zip(bodies, rules)
        ]
        sorted_idx = sorted(range(N), key=lambda i: min(approx_lens[i], max_len))
    else:
        sorted_idx = list(range(N))

    FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    probs_yes = [None] * N

    for i in range(0, N, batch_size):
        batch_indices = sorted_idx[i:i+batch_size]
        bb = [bodies[j] for j in batch_indices]
        rr = [rules[j]  for j in batch_indices]

        enc = _encode_batch(tok, bb, rr, device=device, max_len=max_len)

        out = model(**enc, use_cache=True)            # <— single forward pass
        step_scores = out.logits[:, -1, :]
        sel = step_scores[:, tgt_ids]
        logp = torch.log_softmax(sel.to(torch.float32), dim=-1)

        y_logp = (torch.logsumexp(logp[:, yes_idx], dim=-1)
                  if yes_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        n_logp = (torch.logsumexp(logp[:,  no_idx], dim=-1)
                  if no_idx  else torch.full((sel.size(0),), -1e9, device=sel.device))
        p_yes  = torch.softmax(torch.stack([y_logp, n_logp], dim=-1), dim=-1)[:, 0]

        for k, j in enumerate(batch_indices):
            probs_yes[j] = float(p_yes[k].item())

    # Build per-rule ranked scores in [0, 1]
    df_scores = pd.DataFrame({
        "row_id": rowids,
        "rule":   rules,
        "prob":   probs_yes,
    })

    grp   = df_scores.groupby("rule")
    rank  = grp["prob"].rank(method="average", ascending=True)   # low prob -> 1, high prob -> n
    n     = grp["prob"].transform("size")
    score = (rank - 1.0) / np.maximum(n - 1.0, 1.0)              # low -> 0.0, high -> 1.0
    df_scores["rule_violation"] = score

    out_df = df_scores[["row_id", "rule_violation"]].sort_values("row_id").reset_index(drop=True)
    if write_csv:
        out_df.to_csv("submission2.csv", index=False)
    return out_df


# =========================
# Main
# =========================
def main():
    # ---- Config knobs ----
    quant_mode = os.environ.get("QUANT_MODE", "4bit").lower()   # "4bit" or "8bit"
    load_in_4bit = (quant_mode == "4bit")
    dtype = None  # auto (fp16 on T4/V100, bf16 on Ampere+)

    # ---- Load base + LoRA via Unsloth ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_PATH,
        max_seq_length = 512,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
        load_in_8bit   = (quant_mode == "8bit"),
        local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1004,
        use_rslora=False,
        loftq_config=None,
    )

    # ---- Tokenizer ----
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # ---- Data ----
    df = load_dataframe()
    conv_dataset = make_conversations_dataset(df)
    train_dataset = build_text_dataset(tokenizer, conv_dataset)

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=0,
            logging_steps=10,
            optim="adamw_8bit",
            seed=1004,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=2,
        ),
    )

    # Only compute loss over assistant spans (our "Yes"/"No")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    trainer.train()

    # ---- Inference uses the SAME limit & left settings ----
    submission_df = run_inference_unsloth_generate(model, tokenizer, max_len=512)
    print(submission_df.head(10))
    print("Wrote submission.csv")

if __name__ == "__main__":
    main()


%%writefile train_qwen3_8b.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print("PID:", os.getpid(), "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
assert torch.cuda.is_available()
print("Visible device count:", torch.cuda.device_count())
torch.cuda.set_device(0)            # 0 == the ONLY visible GPU in this process
print("Using:", torch.cuda.get_device_name(0))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never hit the Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")     # Transformers offline
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")      # Datasets offline
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1") # no telemetry

import math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from typing import List, Tuple

from unsloth.chat_templates import CHAT_TEMPLATES

# =========================
# Merged CONSTANTS
# =========================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH",
    "/kaggle/input/qwen3-8b-unsloth-bnb-4bit/transformers/default/1"
)
DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/input/jigsaw-agile-community-rules/")

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."

# Inference knobs
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", 8))
WRITE_SUBMISSION = os.environ.get("WRITE_SUBMISSION", "1") == "1"  # write submission.csv

# =========================
# Merged UTILS
# =========================
def get_dataframe_to_train(data_path: str) -> pd.DataFrame:
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=1009).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])
    return dataframe


# =========================
# Unsloth training helpers
# =========================
def load_dataframe() -> pd.DataFrame:
    df = get_dataframe_to_train(DATA_PATH)  # body, rule, rule_violation
    if "completion" not in df.columns:
        df = df.copy()
        df["completion"] = df["rule_violation"].map(
            {1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER}
        )
    return df[["body", "rule", "completion"]]

def make_conversations_dataset(df: pd.DataFrame) -> Dataset:
    # Each sample: system + user + assistant(Yes/No)
    convos = []
    for body, rule, comp in zip(df["body"], df["rule"], df["completion"]):
        convos.append([
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
            {"role": "assistant", "content": str(comp)},
        ])
    return Dataset.from_dict({"conversations": convos})

def build_text_dataset(tokenizer, conv_dataset: Dataset):
    # Convert conversations -> single 'text' string via chat template.
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )[:-11]
            for conv in convos
        ]
        return {"text": texts}
    return conv_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=conv_dataset.column_names,
    )

# =========================
# Inference helpers (Unsloth, in-memory)
# =========================
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]
def _first_token_ids(tok, txt_or_texts) -> List[int]:
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_texts] if isinstance(txt_or_texts, str) else list(txt_or_texts)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def _encode_batch(tok, bodies, rules, device, max_len=512):
    """
    Manual left truncation + left padding + explicit position_ids.
    Returns tensors on `device` with keys: input_ids, attention_mask, position_ids.
    """
    pad_id = tok.pad_token_id
    if pad_id is None:
        # Qwen templates usually set pad_token = eos_token already in your code,
        # but keep a safe fallback:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # 1) Tokenize each sample via chat template
    seqs = []
    for body, rule in zip(bodies, rules):
        msgs = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
        ]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )

        # Left truncation: keep the last `max_len` tokens
        if len(ids) > max_len:
            ids = ids[-max_len:]

        seqs.append(torch.tensor(ids, dtype=torch.long))

    # 2) Left padding to a uniform length (no longer than max_len)
    B = len(seqs)
    T = min(max_len, max(len(x) for x in seqs)) if seqs else 1

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)

    for i, ids in enumerate(seqs):
        L = min(T, len(ids))
        # Left pad: write actual ids into the *rightmost* part
        input_ids[i, T - L : T] = ids[-L:]
        attention_mask[i, T - L : T] = 1

    # 3) position_ids: first non-pad gets 0; pads use 0 (safe default)
    #    (Equivalent to: pos = cumsum(mask) - 1, then masked_fill(pad, 0))
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    position_ids = position_ids.to(dtype=torch.long)

    # 4) Ship to device
    batch = {
        "input_ids":      input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "position_ids":   position_ids.to(device, non_blocking=True),
    }
    return batch


@torch.inference_mode()
def run_inference_unsloth_generate(
    model, tokenizer, data_path=DATA_PATH, batch_size=INFER_BATCH_SIZE,
    max_len=512, write_csv=WRITE_SUBMISSION, sort_by_length=True,
):
    tok = tokenizer

    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)
    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    # NOTE: this probably should be 'test.csv'; your current code reads 'train.csv'
    test_df = pd.read_csv(f"{data_path}/test.csv")
    bodies  = list(test_df["body"])
    rules   = list(test_df["rule"])
    rowids  = list(test_df["row_id"])

    N = len(bodies)
    if sort_by_length:
        approx_lens = [
            (len(tok.encode(b, add_special_tokens=False)) +
             len(tok.encode(r, add_special_tokens=False)))
            for b, r in zip(bodies, rules)
        ]
        sorted_idx = sorted(range(N), key=lambda i: min(approx_lens[i], max_len))
    else:
        sorted_idx = list(range(N))

    FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    probs_yes = [None] * N

    for i in range(0, N, batch_size):
        batch_indices = sorted_idx[i:i+batch_size]
        bb = [bodies[j] for j in batch_indices]
        rr = [rules[j]  for j in batch_indices]

        enc = _encode_batch(tok, bb, rr, device=device, max_len=max_len)

        out = model(**enc, use_cache=True)            # <— single forward pass
        step_scores = out.logits[:, -1, :]
        sel = step_scores[:, tgt_ids]
        logp = torch.log_softmax(sel.to(torch.float32), dim=-1)

        y_logp = (torch.logsumexp(logp[:, yes_idx], dim=-1)
                  if yes_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        n_logp = (torch.logsumexp(logp[:,  no_idx], dim=-1)
                  if no_idx  else torch.full((sel.size(0),), -1e9, device=sel.device))
        p_yes  = torch.softmax(torch.stack([y_logp, n_logp], dim=-1), dim=-1)[:, 0]

        for k, j in enumerate(batch_indices):
            probs_yes[j] = float(p_yes[k].item())

    # Build per-rule ranked scores in [0, 1]
    df_scores = pd.DataFrame({
        "row_id": rowids,
        "rule":   rules,
        "prob":   probs_yes,
    })

    grp   = df_scores.groupby("rule")
    rank  = grp["prob"].rank(method="average", ascending=True)   # low prob -> 1, high prob -> n
    n     = grp["prob"].transform("size")
    score = (rank - 1.0) / np.maximum(n - 1.0, 1.0)              # low -> 0.0, high -> 1.0
    df_scores["rule_violation"] = score

    out_df = df_scores[["row_id", "rule_violation"]].sort_values("row_id").reset_index(drop=True)
    if write_csv:
        out_df.to_csv("submission3.csv", index=False)
    return out_df


# =========================
# Main
# =========================
def main():
    # ---- Config knobs ----
    quant_mode = os.environ.get("QUANT_MODE", "4bit").lower()   # "4bit" or "8bit"
    load_in_4bit = (quant_mode == "4bit")
    dtype = None  # auto (fp16 on T4/V100, bf16 on Ampere+)

    # ---- Load base + LoRA via Unsloth ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_PATH,
        max_seq_length = 512,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
        load_in_8bit   = (quant_mode == "8bit"),
        local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1009,
        use_rslora=False,
        loftq_config=None,
    )

    # ---- Tokenizer ----
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # ---- Data ----
    df = load_dataframe()
    conv_dataset = make_conversations_dataset(df)
    train_dataset = build_text_dataset(tokenizer, conv_dataset)

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=0,
            logging_steps=10,
            optim="adamw_8bit",
            seed=1009,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=2,
        ),
    )

    # Only compute loss over assistant spans (our "Yes"/"No")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user",
        response_part = "<think>\n\n</think>\n\n",
    )

    trainer.train()

    # ---- Inference uses the SAME limit & left settings ----
    submission_df = run_inference_unsloth_generate(model, tokenizer, max_len=512)
    print(submission_df.head(10))
    print("Wrote submission.csv")

if __name__ == "__main__":
    main()


%%writefile train_qwen3_4b.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print("PID:", os.getpid(), "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
assert torch.cuda.is_available()
print("Visible device count:", torch.cuda.device_count())
torch.cuda.set_device(0)            # 0 == the ONLY visible GPU in this process
print("Using:", torch.cuda.get_device_name(0))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never hit the Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")     # Transformers offline
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")      # Datasets offline
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1") # no telemetry

import math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from typing import List, Tuple

from unsloth.chat_templates import CHAT_TEMPLATES

# =========================
# Merged CONSTANTS
# =========================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH",
    "/kaggle/input/qwen3-4b-instruct-2507/transformers/default/1"
)
DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/input/jigsaw-agile-community-rules/")

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."

# Inference knobs
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", 16))
WRITE_SUBMISSION = os.environ.get("WRITE_SUBMISSION", "1") == "1"  # write submission.csv

# =========================
# Merged UTILS
# =========================
def get_dataframe_to_train(data_path: str) -> pd.DataFrame:
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=1010).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])
    return dataframe


# =========================
# Unsloth training helpers
# =========================
def load_dataframe() -> pd.DataFrame:
    df = get_dataframe_to_train(DATA_PATH)  # body, rule, rule_violation
    if "completion" not in df.columns:
        df = df.copy()
        df["completion"] = df["rule_violation"].map(
            {1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER}
        )
    return df[["body", "rule", "completion"]]

def make_conversations_dataset(df: pd.DataFrame) -> Dataset:
    # Each sample: system + user + assistant(Yes/No)
    convos = []
    for body, rule, comp in zip(df["body"], df["rule"], df["completion"]):
        convos.append([
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
            {"role": "assistant", "content": str(comp)},
        ])
    return Dataset.from_dict({"conversations": convos})

def build_text_dataset(tokenizer, conv_dataset: Dataset):
    # Convert conversations -> single 'text' string via chat template.
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )[:-11]
            for conv in convos
        ]
        return {"text": texts}
    return conv_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=conv_dataset.column_names,
    )

# =========================
# Inference helpers (Unsloth, in-memory)
# =========================
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]
def _first_token_ids(tok, txt_or_texts) -> List[int]:
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_texts] if isinstance(txt_or_texts, str) else list(txt_or_texts)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def _encode_batch(tok, bodies, rules, device, max_len=512):
    """
    Manual left truncation + left padding + explicit position_ids.
    Returns tensors on `device` with keys: input_ids, attention_mask, position_ids.
    """
    pad_id = tok.pad_token_id
    if pad_id is None:
        # Qwen templates usually set pad_token = eos_token already in your code,
        # but keep a safe fallback:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # 1) Tokenize each sample via chat template
    seqs = []
    for body, rule in zip(bodies, rules):
        msgs = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
        ]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Left truncation: keep the last `max_len` tokens
        if len(ids) > max_len:
            ids = ids[-max_len:]

        seqs.append(torch.tensor(ids, dtype=torch.long))

    # 2) Left padding to a uniform length (no longer than max_len)
    B = len(seqs)
    T = min(max_len, max(len(x) for x in seqs)) if seqs else 1

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)

    for i, ids in enumerate(seqs):
        L = min(T, len(ids))
        # Left pad: write actual ids into the *rightmost* part
        input_ids[i, T - L : T] = ids[-L:]
        attention_mask[i, T - L : T] = 1

    # 3) position_ids: first non-pad gets 0; pads use 0 (safe default)
    #    (Equivalent to: pos = cumsum(mask) - 1, then masked_fill(pad, 0))
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    position_ids = position_ids.to(dtype=torch.long)

    # 4) Ship to device
    batch = {
        "input_ids":      input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "position_ids":   position_ids.to(device, non_blocking=True),
    }
    return batch


@torch.inference_mode()
def run_inference_unsloth_generate(
    model, tokenizer, data_path=DATA_PATH, batch_size=INFER_BATCH_SIZE,
    max_len=512, write_csv=WRITE_SUBMISSION, sort_by_length=True,
):
    tok = tokenizer

    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)
    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    # NOTE: this probably should be 'test.csv'; your current code reads 'train.csv'
    test_df = pd.read_csv(f"{data_path}/test.csv")
    bodies  = list(test_df["body"])
    rules   = list(test_df["rule"])
    rowids  = list(test_df["row_id"])

    N = len(bodies)
    if sort_by_length:
        approx_lens = [
            (len(tok.encode(b, add_special_tokens=False)) +
             len(tok.encode(r, add_special_tokens=False)))
            for b, r in zip(bodies, rules)
        ]
        sorted_idx = sorted(range(N), key=lambda i: min(approx_lens[i], max_len))
    else:
        sorted_idx = list(range(N))

    FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    probs_yes = [None] * N

    for i in range(0, N, batch_size):
        batch_indices = sorted_idx[i:i+batch_size]
        bb = [bodies[j] for j in batch_indices]
        rr = [rules[j]  for j in batch_indices]

        enc = _encode_batch(tok, bb, rr, device=device, max_len=max_len)

        out = model(**enc, use_cache=True)            # <— single forward pass
        step_scores = out.logits[:, -1, :]
        sel = step_scores[:, tgt_ids]
        logp = torch.log_softmax(sel.to(torch.float32), dim=-1)

        y_logp = (torch.logsumexp(logp[:, yes_idx], dim=-1)
                  if yes_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        n_logp = (torch.logsumexp(logp[:,  no_idx], dim=-1)
                  if no_idx  else torch.full((sel.size(0),), -1e9, device=sel.device))
        p_yes  = torch.softmax(torch.stack([y_logp, n_logp], dim=-1), dim=-1)[:, 0]

        for k, j in enumerate(batch_indices):
            probs_yes[j] = float(p_yes[k].item())

    # Build per-rule ranked scores in [0, 1]
    df_scores = pd.DataFrame({
        "row_id": rowids,
        "rule":   rules,
        "prob":   probs_yes,
    })

    grp   = df_scores.groupby("rule")
    rank  = grp["prob"].rank(method="average", ascending=True)   # low prob -> 1, high prob -> n
    n     = grp["prob"].transform("size")
    score = (rank - 1.0) / np.maximum(n - 1.0, 1.0)              # low -> 0.0, high -> 1.0
    df_scores["rule_violation"] = score

    out_df = df_scores[["row_id", "rule_violation"]].sort_values("row_id").reset_index(drop=True)
    if write_csv:
        out_df.to_csv("submission5.csv", index=False)
    return out_df


# =========================
# Main
# =========================
def main():
    # ---- Config knobs ----
    quant_mode = os.environ.get("QUANT_MODE", "4bit").lower()   # "4bit" or "8bit"
    load_in_4bit = (quant_mode == "4bit")
    dtype = None  # auto (fp16 on T4/V100, bf16 on Ampere+)

    # ---- Load base + LoRA via Unsloth ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_PATH,
        max_seq_length = 512,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
        load_in_8bit   = (quant_mode == "8bit"),
        local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1010,
        use_rslora=False,
        loftq_config=None,
    )

    # ---- Tokenizer ----
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # ---- Data ----
    df = load_dataframe()
    conv_dataset = make_conversations_dataset(df)
    train_dataset = build_text_dataset(tokenizer, conv_dataset)

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=0,
            logging_steps=10,
            optim="adamw_8bit",
            seed=1010,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=2,
        ),
    )

    # Only compute loss over assistant spans (our "Yes"/"No")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user",
        response_part = "<|im_start|>assistant\n",
    )

    trainer.train()

    # ---- Inference uses the SAME limit & left settings ----
    submission_df = run_inference_unsloth_generate(model, tokenizer, max_len=512)
    print(submission_df.head(10))
    print("Wrote submission.csv")

if __name__ == "__main__":
    main()


%%writefile train_llama3_8b.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print("PID:", os.getpid(), "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
assert torch.cuda.is_available()
print("Visible device count:", torch.cuda.device_count())
torch.cuda.set_device(0)            # 0 == the ONLY visible GPU in this process
print("Using:", torch.cuda.get_device_name(0))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never hit the Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")     # Transformers offline
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")      # Datasets offline
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1") # no telemetry

import math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from typing import List, Tuple

from unsloth.chat_templates import CHAT_TEMPLATES

# =========================
# Merged CONSTANTS
# =========================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH",
    "/kaggle/input/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit/transformers/default/1"
)
DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/input/jigsaw-agile-community-rules/")

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."

# Inference knobs
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", 10))
WRITE_SUBMISSION = os.environ.get("WRITE_SUBMISSION", "1") == "1"  # write submission.csv

# =========================
# Merged UTILS
# =========================
def get_dataframe_to_train(data_path: str) -> pd.DataFrame:
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=1003).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])
    return dataframe


# =========================
# Unsloth training helpers
# =========================
def load_dataframe() -> pd.DataFrame:
    df = get_dataframe_to_train(DATA_PATH)  # body, rule, rule_violation
    if "completion" not in df.columns:
        df = df.copy()
        df["completion"] = df["rule_violation"].map(
            {1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER}
        )
    return df[["body", "rule", "completion"]]

def make_conversations_dataset(df: pd.DataFrame) -> Dataset:
    # Each sample: system + user + assistant(Yes/No)
    convos = []
    for body, rule, comp in zip(df["body"], df["rule"], df["completion"]):
        convos.append([
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
            {"role": "assistant", "content": str(comp)},
        ])
    return Dataset.from_dict({"conversations": convos})

def build_text_dataset(tokenizer, conv_dataset: Dataset):
    # Convert conversations -> single 'text' string via chat template.
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )[:-10]
            for conv in convos
        ]
        return {"text": texts}
    return conv_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=conv_dataset.column_names,
    )

# =========================
# Inference helpers (Unsloth, in-memory)
# =========================
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]
def _first_token_ids(tok, txt_or_texts) -> List[int]:
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_texts] if isinstance(txt_or_texts, str) else list(txt_or_texts)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def _encode_batch(tok, bodies, rules, device, max_len=512):
    """
    Manual left truncation + left padding + explicit position_ids.
    Returns tensors on `device` with keys: input_ids, attention_mask, position_ids.
    """
    pad_id = tok.pad_token_id
    if pad_id is None:
        # Qwen templates usually set pad_token = eos_token already in your code,
        # but keep a safe fallback:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # 1) Tokenize each sample via chat template
    seqs = []
    for body, rule in zip(bodies, rules):
        msgs = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
        ]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Left truncation: keep the last `max_len` tokens
        if len(ids) > max_len:
            ids = ids[-max_len:]

        seqs.append(torch.tensor(ids, dtype=torch.long))

    # 2) Left padding to a uniform length (no longer than max_len)
    B = len(seqs)
    T = min(max_len, max(len(x) for x in seqs)) if seqs else 1

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)

    for i, ids in enumerate(seqs):
        L = min(T, len(ids))
        # Left pad: write actual ids into the *rightmost* part
        input_ids[i, T - L : T] = ids[-L:]
        attention_mask[i, T - L : T] = 1

    # 3) position_ids: first non-pad gets 0; pads use 0 (safe default)
    #    (Equivalent to: pos = cumsum(mask) - 1, then masked_fill(pad, 0))
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    position_ids = position_ids.to(dtype=torch.long)

    # 4) Ship to device
    batch = {
        "input_ids":      input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "position_ids":   position_ids.to(device, non_blocking=True),
    }
    return batch


@torch.inference_mode()
def run_inference_unsloth_generate(
    model, tokenizer, data_path=DATA_PATH, batch_size=INFER_BATCH_SIZE,
    max_len=512, write_csv=WRITE_SUBMISSION, sort_by_length=True,
):
    tok = tokenizer

    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)
    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    # NOTE: this probably should be 'test.csv'; your current code reads 'train.csv'
    test_df = pd.read_csv(f"{data_path}/test.csv")
    bodies  = list(test_df["body"])
    rules   = list(test_df["rule"])
    rowids  = list(test_df["row_id"])

    N = len(bodies)
    if sort_by_length:
        approx_lens = [
            (len(tok.encode(b, add_special_tokens=False)) +
             len(tok.encode(r, add_special_tokens=False)))
            for b, r in zip(bodies, rules)
        ]
        sorted_idx = sorted(range(N), key=lambda i: min(approx_lens[i], max_len))
    else:
        sorted_idx = list(range(N))

    FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    probs_yes = [None] * N

    for i in range(0, N, batch_size):
        batch_indices = sorted_idx[i:i+batch_size]
        bb = [bodies[j] for j in batch_indices]
        rr = [rules[j]  for j in batch_indices]

        enc = _encode_batch(tok, bb, rr, device=device, max_len=max_len)

        out = model(**enc, use_cache=True)            # <— single forward pass
        step_scores = out.logits[:, -1, :]
        sel = step_scores[:, tgt_ids]
        logp = torch.log_softmax(sel.to(torch.float32), dim=-1)

        y_logp = (torch.logsumexp(logp[:, yes_idx], dim=-1)
                  if yes_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        n_logp = (torch.logsumexp(logp[:,  no_idx], dim=-1)
                  if no_idx  else torch.full((sel.size(0),), -1e9, device=sel.device))
        p_yes  = torch.softmax(torch.stack([y_logp, n_logp], dim=-1), dim=-1)[:, 0]

        for k, j in enumerate(batch_indices):
            probs_yes[j] = float(p_yes[k].item())

    # Build per-rule ranked scores in [0, 1]
    df_scores = pd.DataFrame({
        "row_id": rowids,
        "rule":   rules,
        "prob":   probs_yes,
    })

    grp   = df_scores.groupby("rule")
    rank  = grp["prob"].rank(method="average", ascending=True)   # low prob -> 1, high prob -> n
    n     = grp["prob"].transform("size")
    score = (rank - 1.0) / np.maximum(n - 1.0, 1.0)              # low -> 0.0, high -> 1.0
    df_scores["rule_violation"] = score

    out_df = df_scores[["row_id", "rule_violation"]].sort_values("row_id").reset_index(drop=True)
    if write_csv:
        out_df.to_csv("submission4.csv", index=False)
    return out_df


# =========================
# Main
# =========================
def main():
    # ---- Config knobs ----
    quant_mode = os.environ.get("QUANT_MODE", "4bit").lower()   # "4bit" or "8bit"
    load_in_4bit = (quant_mode == "4bit")
    dtype = None  # auto (fp16 on T4/V100, bf16 on Ampere+)

    # ---- Load base + LoRA via Unsloth ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_PATH,
        max_seq_length = 512,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
        load_in_8bit   = (quant_mode == "8bit"),
        local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1003,
        use_rslora=False,
        loftq_config=None,
    )

    # ---- Tokenizer ----
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # ---- Data ----
    df = load_dataframe()
    conv_dataset = make_conversations_dataset(df)
    train_dataset = build_text_dataset(tokenizer, conv_dataset)

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=0,
            logging_steps=10,
            optim="adamw_8bit",
            seed=1003,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=2,
        ),
    )

    # Only compute loss over assistant spans (our "Yes"/"No")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer.train()

    # ---- Inference uses the SAME limit & left settings ----
    submission_df = run_inference_unsloth_generate(model, tokenizer, max_len=512)
    print(submission_df.head(10))
    print("Wrote submission.csv")

if __name__ == "__main__":
    main()


%%writefile train_llama3_3b.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print("PID:", os.getpid(), "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
assert torch.cuda.is_available()
print("Visible device count:", torch.cuda.device_count())
torch.cuda.set_device(0)            # 0 == the ONLY visible GPU in this process
print("Using:", torch.cuda.get_device_name(0))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never hit the Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")     # Transformers offline
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")      # Datasets offline
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1") # no telemetry

import math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from typing import List, Tuple

from unsloth.chat_templates import CHAT_TEMPLATES

# =========================
# Merged CONSTANTS
# =========================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH",
    "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
)
DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/input/jigsaw-agile-community-rules/")

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."

# Inference knobs
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", 18))
WRITE_SUBMISSION = os.environ.get("WRITE_SUBMISSION", "1") == "1"  # write submission.csv

# =========================
# Merged UTILS
# =========================
def get_dataframe_to_train(data_path: str) -> pd.DataFrame:
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=5003).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])
    return dataframe


# =========================
# Unsloth training helpers
# =========================
def load_dataframe() -> pd.DataFrame:
    df = get_dataframe_to_train(DATA_PATH)  # body, rule, rule_violation
    if "completion" not in df.columns:
        df = df.copy()
        df["completion"] = df["rule_violation"].map(
            {1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER}
        )
    return df[["body", "rule", "completion"]]

def make_conversations_dataset(df: pd.DataFrame) -> Dataset:
    # Each sample: system + user + assistant(Yes/No)
    convos = []
    for body, rule, comp in zip(df["body"], df["rule"], df["completion"]):
        convos.append([
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
            {"role": "assistant", "content": str(comp)},
        ])
    return Dataset.from_dict({"conversations": convos})

def build_text_dataset(tokenizer, conv_dataset: Dataset):
    # Convert conversations -> single 'text' string via chat template.
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )[:-10]
            for conv in convos
        ]
        return {"text": texts}
    return conv_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=conv_dataset.column_names,
    )

# =========================
# Inference helpers (Unsloth, in-memory)
# =========================
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]
def _first_token_ids(tok, txt_or_texts) -> List[int]:
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_texts] if isinstance(txt_or_texts, str) else list(txt_or_texts)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def _encode_batch(tok, bodies, rules, device, max_len=512):
    """
    Manual left truncation + left padding + explicit position_ids.
    Returns tensors on `device` with keys: input_ids, attention_mask, position_ids.
    """
    pad_id = tok.pad_token_id
    if pad_id is None:
        # Qwen templates usually set pad_token = eos_token already in your code,
        # but keep a safe fallback:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # 1) Tokenize each sample via chat template
    seqs = []
    for body, rule in zip(bodies, rules):
        msgs = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
        ]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Left truncation: keep the last `max_len` tokens
        if len(ids) > max_len:
            ids = ids[-max_len:]

        seqs.append(torch.tensor(ids, dtype=torch.long))

    # 2) Left padding to a uniform length (no longer than max_len)
    B = len(seqs)
    T = min(max_len, max(len(x) for x in seqs)) if seqs else 1

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)

    for i, ids in enumerate(seqs):
        L = min(T, len(ids))
        # Left pad: write actual ids into the *rightmost* part
        input_ids[i, T - L : T] = ids[-L:]
        attention_mask[i, T - L : T] = 1

    # 3) position_ids: first non-pad gets 0; pads use 0 (safe default)
    #    (Equivalent to: pos = cumsum(mask) - 1, then masked_fill(pad, 0))
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    position_ids = position_ids.to(dtype=torch.long)

    # 4) Ship to device
    batch = {
        "input_ids":      input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "position_ids":   position_ids.to(device, non_blocking=True),
    }
    return batch


@torch.inference_mode()
def run_inference_unsloth_generate(
    model, tokenizer, data_path=DATA_PATH, batch_size=INFER_BATCH_SIZE,
    max_len=512, write_csv=WRITE_SUBMISSION, sort_by_length=True,
):
    tok = tokenizer

    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)
    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    # NOTE: this probably should be 'test.csv'; your current code reads 'train.csv'
    test_df = pd.read_csv(f"{data_path}/test.csv")
    bodies  = list(test_df["body"])
    rules   = list(test_df["rule"])
    rowids  = list(test_df["row_id"])

    N = len(bodies)
    if sort_by_length:
        approx_lens = [
            (len(tok.encode(b, add_special_tokens=False)) +
             len(tok.encode(r, add_special_tokens=False)))
            for b, r in zip(bodies, rules)
        ]
        sorted_idx = sorted(range(N), key=lambda i: min(approx_lens[i], max_len))
    else:
        sorted_idx = list(range(N))

    FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    probs_yes = [None] * N

    for i in range(0, N, batch_size):
        batch_indices = sorted_idx[i:i+batch_size]
        bb = [bodies[j] for j in batch_indices]
        rr = [rules[j]  for j in batch_indices]

        enc = _encode_batch(tok, bb, rr, device=device, max_len=max_len)

        out = model(**enc, use_cache=True)            # <— single forward pass
        step_scores = out.logits[:, -1, :]
        sel = step_scores[:, tgt_ids]
        logp = torch.log_softmax(sel.to(torch.float32), dim=-1)

        y_logp = (torch.logsumexp(logp[:, yes_idx], dim=-1)
                  if yes_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        n_logp = (torch.logsumexp(logp[:,  no_idx], dim=-1)
                  if no_idx  else torch.full((sel.size(0),), -1e9, device=sel.device))
        p_yes  = torch.softmax(torch.stack([y_logp, n_logp], dim=-1), dim=-1)[:, 0]

        for k, j in enumerate(batch_indices):
            probs_yes[j] = float(p_yes[k].item())

    # Build per-rule ranked scores in [0, 1]
    df_scores = pd.DataFrame({
        "row_id": rowids,
        "rule":   rules,
        "prob":   probs_yes,
    })

    grp   = df_scores.groupby("rule")
    rank  = grp["prob"].rank(method="average", ascending=True)   # low prob -> 1, high prob -> n
    n     = grp["prob"].transform("size")
    score = (rank - 1.0) / np.maximum(n - 1.0, 1.0)              # low -> 0.0, high -> 1.0
    df_scores["rule_violation"] = score

    out_df = df_scores[["row_id", "rule_violation"]].sort_values("row_id").reset_index(drop=True)
    if write_csv:
        out_df.to_csv("submission6.csv", index=False)
    return out_df


# =========================
# Main
# =========================
def main():
    # ---- Config knobs ----
    quant_mode = os.environ.get("QUANT_MODE", "4bit").lower()   # "4bit" or "8bit"
    load_in_4bit = (quant_mode == "4bit")
    dtype = None  # auto (fp16 on T4/V100, bf16 on Ampere+)

    # ---- Load base + LoRA via Unsloth ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_PATH,
        max_seq_length = 512,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
        load_in_8bit   = (quant_mode == "8bit"),
        local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=5003,
        use_rslora=False,
        loftq_config=None,
    )

    # ---- Tokenizer ----
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # ---- Data ----
    df = load_dataframe()
    conv_dataset = make_conversations_dataset(df)
    train_dataset = build_text_dataset(tokenizer, conv_dataset)

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=0,
            logging_steps=10,
            optim="adamw_8bit",
            seed=5003,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=2,
        ),
    )

    # Only compute loss over assistant spans (our "Yes"/"No")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer.train()

    # ---- Inference uses the SAME limit & left settings ----
    submission_df = run_inference_unsloth_generate(model, tokenizer, max_len=512)
    print(submission_df.head(10))
    print("Wrote submission.csv")

if __name__ == "__main__":
    main()


%%writefile train_ettin_400m.py
import os, math, time, argparse

# ----------------------
# Constants
# ----------------------
BASE_MODEL_PATH = "/kaggle/input/berts/transformers/default/1/ettin-encoder-400m"
OUTPUT_DIR = "output_deberta/"   # only used if you choose to save
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."

# ----------------------
# Imports
# ----------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)

# Speed-friendly defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))     # train batch size
EPOCHS     = int(os.environ.get("EPOCHS", "1"))
LR         = float(os.environ.get("LR", "2e-5"))
WD         = float(os.environ.get("WD", "0.01"))
WARMUP     = float(os.environ.get("WARMUP_RATIO", "0.00"))  # fraction of total steps
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "2"))
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1"

# Inference knobs
INFER_MAX_LENGTH = int(os.environ.get("INFER_MAX_LENGTH", str(MAX_LENGTH)))
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", "32"))

# ----------------------
# Utils
# ----------------------
def build_prompt(row):
    # Kept for parity; not used by this pipeline
    return f"""
{BASE_PROMPT}

Comment: {row["body"]}

rule: {row["rule"]}
---
{COMPLETE_PHRASE}"""

def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # upsample target block (test examples) by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples once more (2x extra copies -> appears 3x total)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    dataframe = dataframe.sample(frac=1.0, random_state=3001).reset_index(drop=True)
    dataframe = dataframe.drop(columns=["source"])
    return dataframe

def build_classification_dataframe(df, tok_sep_token="</s>"):
    """
    Build a dataframe for classification using text = rule + [SEP] + comment.
    """
    df = df.copy()
    sep = tok_sep_token if tok_sep_token else "</s>"
    df["text"] = df["rule"].astype(str) + sep + df["body"].astype(str)
    cols = ["text"]
    if "rule_violation" in df:
        cols.append("rule_violation")
    return df[cols]

# ----------------------
# Dataset & Collate
# ----------------------
class TxtClsDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        if self.labels is None:
            return {"text": self.texts[i]}
        return {"text": self.texts[i], "label": float(self.labels[i])}

def collate_fn_builder(tokenizer, max_length):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "label" in batch[0]:
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32).unsqueeze(-1)
            out["labels"] = labels
        return out
    return _fn

# ----------------------
# Training (single-GPU) -> returns model & tokenizer so you can infer in-memory
# ----------------------
def train_single_gpu(no_save: bool = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    raw_df = get_dataframe_to_train(DATA_PATH)

    # Tokenizer first, to know SEP token for concatenation
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=False)
    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"

    df = build_classification_dataframe(raw_df, tok_sep_token=sep_token)

    train_ds = TxtClsDataset(df["text"], df["rule_violation"])

    collate_fn = collate_fn_builder(tokenizer, MAX_LENGTH)
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Model
    cfg = AutoConfig.from_pretrained(BASE_MODEL_PATH, num_labels=1, problem_type=None)  # we'll set loss manually
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH, config=cfg)
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    model.to(device)

    # Optimizer / Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = max(1, EPOCHS * len(loader) // max(1, GRAD_ACCUM))
    warmup_steps = int(WARMUP * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss
    bce = torch.nn.BCEWithLogitsLoss()

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.train()

    global_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        running = 0.0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)  # shape (B,1)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits  # (B,1)
                loss = bce(logits, labels)

            loss = loss / max(1, GRAD_ACCUM)
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += loss.item() * max(1, GRAD_ACCUM)

            if global_step > 0 and global_step % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[epoch {epoch+1}] step {global_step}/{total_steps} | lr={lr_now:.6e} | loss={running/10.0:.4f}")
                running = 0.0

        print(f"Epoch {epoch+1} done in {(time.time()-t0)/60:.1f} min")

    if not no_save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Saved model + tokenizer to {OUTPUT_DIR}")
    else:
        print("Skipping save (in-memory use only).")

    # Return in-memory artifacts for immediate inference
    return model, tokenizer

# ----------------------
# Inference (single-GPU)
# - If model/tokenizer are provided, uses them in-memory (no disk I/O).
# - Otherwise, loads from OUTPUT_DIR.
# ----------------------
def _bucket_by_length(tok, texts, max_length=512):
    lens = tok(texts, return_length=True, truncation=True, max_length=max_length)
    order = sorted(range(len(texts)), key=lambda k: lens["length"][k], reverse=True)
    return order

def infer_single_gpu(model: AutoModelForSequenceClassification = None,
                     tokenizer: AutoTokenizer = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model is None or tokenizer is None:
        # fall back to disk
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True)
        cfg = AutoConfig.from_pretrained(OUTPUT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR, config=cfg)
        print(f"Loaded model + tokenizer from {OUTPUT_DIR}")

    model.eval()
    model.to(device)

    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"

    # Load test
    df = pd.read_csv(f"{DATA_PATH}/test.csv")

    # --- robust column handling ---
    if "body" in df.columns:
        text_col = "body"
    elif "comment" in df.columns:
        text_col = "comment"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise KeyError(f"Could not find a text column. Available columns: {list(df.columns)}")

    if "rule" not in df.columns:
        raise KeyError("Expected 'rule' column in test.csv but did not find it.")

    # Build the classification dataframe using rule + [SEP] + body/comment
    tmp = df.rename(columns={text_col: "body"})  # utils expects 'body'
    cls_df = build_classification_dataframe(tmp, tok_sep_token=sep_token)

    texts = list(cls_df["text"])
    rows  = list(df["row_id"]) if "row_id" in df.columns else list(range(len(df)))
    rules = list(df["rule"])

    order = _bucket_by_length(tokenizer, texts, max_length=INFER_MAX_LENGTH)
    texts = [texts[i] for i in order]
    rows  = [rows[i]  for i in order]
    rules = [rules[i] for i in order]

    out_ids, out_probs, out_rules = [], [], []

    for i in range(0, len(texts), INFER_BATCH_SIZE):
        batch_texts = texts[i:i+INFER_BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=INFER_MAX_LENGTH,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(**enc).logits  # (B,1)
            probs = torch.sigmoid(logits).squeeze(-1)  # (B,)

        out_probs.extend(probs.float().cpu().tolist())
        out_ids.extend(rows[i:i+INFER_BATCH_SIZE])
        out_rules.extend(rules[i:i+INFER_BATCH_SIZE])

    out_df = pd.DataFrame({"row_id": out_ids, "rule": out_rules, "rule_violation": out_probs})

    # Per-rule ranks scaled to [0,1]
    r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
    n = out_df.groupby("rule")["rule_violation"].transform("size")
    denom = (n - 1).where(n > 1, 1)
    out_df["rule_violation"] = (r - 1) / denom

    out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission7.csv", index=False)
    print("Wrote submission.csv")

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-GPU DeBERTa train + inference (in-memory capable)")
    p.add_argument("--do_train", action="store_true", help="Run training")
    p.add_argument("--do_infer", action="store_true", help="Run inference (loads from disk if no in-memory model)")
    p.add_argument("--train_then_infer", action="store_true",
                   help="Run training then inference (passes the trained model in-memory; no disk I/O unless you omit --no_save).")
    p.add_argument("--no_save", action="store_true", help="Do not save model/tokenizer after training.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Default behavior: train_then_infer with no-save (most efficient path)
    if not any([args.do_train, args.do_infer, args.train_then_infer]):
        args.train_then_infer = True
        args.no_save = True

    if args.do_train and not args.do_infer and not args.train_then_infer:
        train_single_gpu(no_save=args.no_save)

    elif args.do_infer and not args.do_train and not args.train_then_infer:
        # inference only -> will load from OUTPUT_DIR
        infer_single_gpu(model=None, tokenizer=None)

    elif args.train_then_infer:
        model, tok = train_single_gpu(no_save=args.no_save)
        # Pass in-memory model + tokenizer (no save/load needed)
        infer_single_gpu(model=model, tokenizer=tok)


%%writefile train.py
import os, sys, subprocess, threading, io

def tee_stream(stream, fileobj, prefix=""):
    for line in iter(stream.readline, b""):
        try:
            decoded = line.decode("utf-8", "replace")
        except Exception:
            decoded = line.decode(errors="replace")
        msg = f"{prefix} {decoded}" if prefix else decoded
        print(msg, end="")
        fileobj.write(msg)
        fileobj.flush()
    stream.close()

def launch(script: str, cuda_id: str, log_path: str, name: str):
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = cuda_id

    p = subprocess.Popen(
        [sys.executable, "-u", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    f = open(log_path, "w", buffering=1)  # line-buffered file
    t = threading.Thread(target=tee_stream, args=(p.stdout, f, f"[{name}]"), daemon=True)
    t.start()
    return {"p": p, "f": f, "t": t, "name": name, "log": log_path}

def run_sequence(sequence_name: str, cuda_id: str, jobs, results_sink: dict):
    """
    Run a list of (script, log_path, display_name) sequentially on the same CUDA device.
    Store per-job exit codes in results_sink[sequence_name].
    """
    print(f"\n=== Starting sequence {sequence_name} on CUDA:{cuda_id} ===")
    seq_results = []
    for script, log_path, disp in jobs:
        print(f"[{sequence_name}] Launching {disp}: {script} (CUDA:{cuda_id}), log: {log_path}")
        j = launch(script, cuda_id, log_path, f"{sequence_name}:{disp}")
        rc = j["p"].wait()
        j["t"].join()
        j["f"].close()
        print(f"[{sequence_name}] Finished {disp} with exit code {rc}")
        seq_results.append({"name": j["name"], "rc": rc, "log": j["log"]})
    results_sink[sequence_name] = seq_results
    print(f"=== Sequence {sequence_name} complete ===\n")

def tail(path, n=60):
    with open(path, "rb") as f:
        f.seek(0, io.SEEK_END)
        size = f.tell()
        block, data = 2048, b""
        while size > 0 and data.count(b"\n") <= n:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
        return data.decode("utf-8", "replace").splitlines()[-n:]

def main():
    # Define per-GPU sequential job lists
    gpu0_jobs = [
        ("train_qwen3_14b.py", "gpu0_qwen3_14b.log", "qwen3-14b"),
        #("train_phi4.py", "gpu0_phi4.log", "phi4"),
        ("train_qwen3_8b.py", "gpu0_qwen3_8b.log", "qwen3-8b"),
        ("train_qwen3_4b.py", "gpu0_qwen3_4b.log", "qwen3-4b"),
        #("train_llama3_3b.py", "gpu0_llama3_3b.log", "llama3-3b"),
    ]
    gpu1_jobs = [
        ("train_qwen2.5_14b.py", "gpu0_qwen2_14b.log", "qwen2-14b"),
        #("train_qwen3_8b.py", "gpu0_qwen3_8b.log", "qwen3-8b"),
        #("train_phi4.py", "gpu0_phi4.log", "phi4"),
        #("train_qwen3_4b.py", "gpu0_qwen3_4b.log", "qwen3-4b"),
        ("train_llama3_8b.py", "gpu0_llama3_8b.log", "llama3-8b"),
        #("train_llama3_3b.py", "gpu0_llama3_3b.log", "llama3-3b"),
        ("train_ettin_400m.py", "gpu0_ettin_400m.log", "ettin-400m"),
    ]

    print("Launching training sequences:")
    print(" - GPU0: qwen3-14b")
    print(" - GPU1: qwen2.5-14b")

    results = {}
    t0 = threading.Thread(target=run_sequence, args=("GPU0", "0", gpu0_jobs, results), daemon=True)
    t1 = threading.Thread(target=run_sequence, args=("GPU1", "1", gpu1_jobs, results), daemon=True)
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    # Summarize exit codes
    print("\n=== Exit codes ===")
    for seq_name in ("GPU0", "GPU1"):
        for item in results.get(seq_name, []):
            print(f"{item['name']}: {item['rc']}")

    # Quick tails for convenience
    for seq_name in ("GPU0", "GPU1"):
        for item in results.get(seq_name, []):
            print(f"\n----- {item['name']} tail ({item['log']}) -----")
            try:
                print("\n".join(tail(item["log"], 80)))
            except FileNotFoundError:
                print("(log file not found)")

if __name__ == "__main__":
    main()


!python -u train.py


import pandas as pd
df1 = pd.read_csv("submission1.csv")
df2 = pd.read_csv("submission2.csv")
df3 = pd.read_csv("submission3.csv")
df4 = pd.read_csv("submission4.csv")
df5 = pd.read_csv("submission5.csv")
#df6 = pd.read_csv("submission6.csv")
df7 = pd.read_csv("submission7.csv")
df1["rule_violation"] = 0.2*df1["rule_violation"] + 0.2*df2["rule_violation"] + 0.2*df3["rule_violation"] + 0.2*df4["rule_violation"] + 0.2*df5["rule_violation"] + 0.1*df7["rule_violation"]
df1.to_csv("submission.csv", index=False)
print(df1.head(10))


%%writefile constants.py
BASE_MODEL_PATH = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
LORA_PATH = "output/"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."


%%writefile utils.py
import pandas as pd
from datasets import Dataset
from constants import POSITIVE_ANSWER, NEGATIVE_ANSWER, COMPLETE_PHRASE, BASE_PROMPT


def build_prompt(row):
    return f"""
{BASE_PROMPT}

Comment: {row["body"]}

rule: {row["rule"]}
---
{COMPLETE_PHRASE}"""


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # >>> Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=2002).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])

    return dataframe


def build_dataset(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    columns = ["prompt"]
    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: POSITIVE_ANSWER,
                0: NEGATIVE_ANSWER,
            }
        )
        columns.append("completion")

    dataframe = dataframe[columns]
    dataset = Dataset.from_pandas(dataframe)
    return dataset


%%writefile train_fp16.py
import os, math, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import torch
import torch.distributed as dist
from datetime import timedelta
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils import get_dataframe_to_train, build_dataset
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH

# Speed-friendly defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
# QUANT_MODE: "4bit" (default) or "8bit"
QUANT_MODE = os.environ.get("QUANT_MODE", "4bit").lower()
# For 4-bit only:
BNB_4BIT_TYPE = os.environ.get("BNB_4BIT_TYPE", "nf4")          # "nf4" or "fp4"
BNB_4BIT_DQ   = os.environ.get("BNB_4BIT_DQ", "1") == "1"       # double quant: 1/0
# --------------------------------------

def setup_ddp():
    if "RANK" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            timeout=timedelta(minutes=30)
        )
    else:
        world, rank, local_rank = 1, 0, 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    if torch.cuda.is_available() and world > torch.cuda.device_count():
        raise RuntimeError(
            f"WORLD_SIZE={world} but only {torch.cuda.device_count()} visible CUDA device(s). "
            "Match --nproc_per_node to the number of GPUs or set CUDA_VISIBLE_DEVICES accordingly."
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return world, rank, local_rank, device

class CompletionOnlyDataset(Dataset):
    def __init__(self, tokenizer, dataframe, max_length=256, add_eos_to_completion=True):
        self.tok = tokenizer
        self.max_length = max_length
        self.samples = []
        prompts = list(dataframe["prompt"])
        completions = list(dataframe["completion"])

        eos = self.tok.eos_token or ""
        for p, c in zip(prompts, completions):
            p_ids = self.tok.encode(p, add_special_tokens=False)
            c_text = c + (eos if add_eos_to_completion else "")
            c_ids = self.tok.encode(c_text, add_special_tokens=False)
            input_ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + c_ids
            if len(input_ids) > self.max_length:
                excess = len(input_ids) - self.max_length
                input_ids = input_ids[excess:]
                labels = labels[excess:]
            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def left_pad_collate(batch, pad_id, max_length=None):
    max_len = max(len(x["input_ids"]) for x in batch)
    if max_length is not None:
        max_len = min(max_len, max_length)
    input_ids, labels, attn = [], [], []
    for ex in batch:
        ids = ex["input_ids"][-max_len:]
        labs = ex["labels"][-max_len:]
        pad_len = max_len - len(ids)
        input_ids.append([pad_id]*pad_len + ids)
        labels.append([-100]*pad_len + labs)
        attn.append([0]*pad_len + [1]*len(ids))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }

def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def cosine_with_warmup_lambda(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def main():
    world, rank, local_rank, device = setup_ddp()

    # Data
    df = get_dataframe_to_train(DATA_PATH)
    train_ds_hf = build_dataset(df)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ---- Quantized base (bitsandbytes), fp16 compute only ----
    if QUANT_MODE == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        if rank == 0: print("Using bitsandbytes: 8-bit (LLM.int8)")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_TYPE,
            bnb_4bit_use_double_quant=BNB_4BIT_DQ,
            bnb_4bit_compute_dtype=torch.float16,  # force fp16
        )
        if rank == 0: print(f"Using bitsandbytes: 4-bit ({BNB_4BIT_TYPE}, double_quant={BNB_4BIT_DQ})")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )

    # Disable cache during training & prefer SDPA if available
    if hasattr(base, "config"):
        base.config.use_cache = False
        try:
            base.config._attn_implementation = "sdpa"
        except Exception:
            pass
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tok.pad_token_id

    # Prepare for k-bit LoRA training (keeps fp16 compute)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    if rank == 0:
        tr, tot = count_trainable_params(model)
        print(f"Trainable params: {tr:,} / {tot:,} ({100*tr/tot:.4f}%)")

    max_length = 256
    torch_ds = CompletionOnlyDataset(tok, train_ds_hf.to_pandas(), max_length=max_length)
    if world > 1:
        sampler = DistributedSampler(torch_ds, num_replicas=world, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    per_device_batch_size = 8
    loader = DataLoader(
        torch_ds,
        batch_size=per_device_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: left_pad_collate(b, pad_id=tok.pad_token_id, max_length=max_length),
        drop_last=False,
    )

    # Optimizer & scheduler (trainable params only)
    lr, wd = 1.5e-4, 0.01
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=wd,
            fused=(torch.cuda.is_available() and torch.__version__ >= "2.0")
        )
    except TypeError:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    num_epochs = 1
    steps_per_epoch = len(loader)
    total_steps = num_epochs * steps_per_epoch

    warmup_steps = 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: cosine_with_warmup_lambda(s, warmup_steps, total_steps)
    )

    # fp16 autocast (T4-safe; no bf16)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    model.train()

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False, broadcast_buffers=False
        )

    global_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = 0.0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            running += loss.item()

            scaler.unscale_(optimizer)
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            if rank == 0 and global_step % 10 == 0:
                avg = running / 10.0
                print(f"[epoch {epoch+1}] step {global_step}/{total_steps} | "
                      f"lr={scheduler.get_last_lr()[0]:.6e} | loss={avg:.4f}")
                running = 0.0

        if rank == 0:
            print(f"Epoch {epoch+1} done in {(time.time()-t0)/60:.1f} min")

    if rank == 0:
        os.makedirs(LORA_PATH, exist_ok=True)
        to_save = model.module if hasattr(model, "module") else model
        to_save.save_pretrained(LORA_PATH)
        tok.save_pretrained(LORA_PATH)
        print(f"Saved LoRA + tokenizer to {LORA_PATH}")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()


#!QUANT_MODE=8bit CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --master_addr=127.0.0.1 --master_port=29513 --nproc_per_node=2 train_fp16.py


%%writefile inference_fp16_ddp.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, torch.distributed as dist
import pandas as pd
from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from utils import build_dataset
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH, POSITIVE_ANSWER, NEGATIVE_ANSWER

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
QUANT_MODE = os.environ.get("QUANT_MODE", "4bit").lower()
BNB_4BIT_TYPE = os.environ.get("BNB_4BIT_TYPE", "nf4")
BNB_4BIT_DQ   = os.environ.get("BNB_4BIT_DQ", "1") == "1"
# --------------------------------------

# Variants to aggregate (same spirit as train.py)
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]

def _first_token_ids(tok, txt_or_list):
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_list] if isinstance(txt_or_list, str) else list(txt_or_list)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def bucket_by_length(tok, prompts, max_length=512):
    lens = tok(prompts, return_length=True, truncation=True, max_length=max_length)
    order = sorted(range(len(prompts)), key=lambda k: lens["length"][k], reverse=True)
    return order

def main():
    # DDP-style sharding
    if "RANK" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    else:
        world, rank, local_rank = 1, 0, 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Quantized base (bnb), fp16 compute only
    if QUANT_MODE == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        if rank == 0: print("Using bitsandbytes: 8-bit (LLM.int8)")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_TYPE,
            bnb_4bit_use_double_quant=BNB_4BIT_DQ,
            bnb_4bit_compute_dtype=torch.float16,  # force fp16
        )
        if rank == 0: print(f"Using bitsandbytes: 4-bit ({BNB_4BIT_TYPE}, double_quant={BNB_4BIT_DQ})")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )

    # Load LoRA *without* merging (keep quantized)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    # Prefer SDPA if available
    try:
        model.config._attn_implementation = "sdpa"
    except Exception:
        pass

    # --- Aggregate first-token IDs for Yes/No (matches train.py approach) ---
    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)

    # Fallback to the constants if variants somehow mapped to nothing
    if not yes_ids:
        yes_ids = _first_token_ids(tok, POSITIVE_ANSWER)
    if not no_ids:
        no_ids = _first_token_ids(tok, NEGATIVE_ANSWER)

    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    df = pd.read_csv(f"{DATA_PATH}/test.csv")
    ds = build_dataset(df)
    prompts = list(ds["prompt"])
    rows    = list(df["row_id"])
    rules   = list(df["rule"])

    idx = list(range(len(prompts)))
    shard_idx     = idx[rank::world]
    shard_prompts = [prompts[i] for i in shard_idx]
    shard_rows    = [rows[i]    for i in shard_idx]
    shard_rules   = [rules[i]   for i in shard_idx]

    order = bucket_by_length(tok, shard_prompts, max_length=512)
    shard_prompts = [shard_prompts[i] for i in order]
    shard_rows    = [shard_rows[i]    for i in order]
    shard_rules   = [shard_rules[i]   for i in order]

    out_ids, out_probs, out_rules = [], [], []
    bs = 16
    max_len = 512

    for i in range(0, len(shard_prompts), bs):
        batch_prompts = shard_prompts[i:i+bs]
        enc = tok(
            batch_prompts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        # fp16 autocast (T4-safe; no bf16)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(**enc).logits[:, -1, :]

            # Select only the Yes/No first-token columns and compute in float32 for stability
            sel = logits[:, tgt_ids].to(torch.float32)
            logp = torch.log_softmax(sel, dim=-1)

        # Aggregate log-probs across all "Yes" and all "No" variants
        if yes_idx:
            ylp = torch.logsumexp(logp[:, yes_idx], dim=-1)
        else:
            ylp = torch.full((logp.size(0),), -1e9, device=logp.device)

        if no_idx:
            nlp = torch.logsumexp(logp[:, no_idx], dim=-1)
        else:
            nlp = torch.full((logp.size(0),), -1e9, device=logp.device)

        prob_yes = torch.softmax(torch.stack([ylp, nlp], dim=-1), dim=-1)[:, 0].float().cpu().tolist()

        out_probs.extend(prob_yes)
        out_ids.extend(shard_rows[i:i+bs])
        out_rules.extend(shard_rules[i:i+bs])

    if world > 1:
        gi, gp, gr = [None]*world, [None]*world, [None]*world
        dist.all_gather_object(gi, out_ids)
        dist.all_gather_object(gp, out_probs)
        dist.all_gather_object(gr, out_rules)
        if rank == 0:
            all_ids   = [x for L in gi for x in L]
            all_p     = [x for L in gp for x in L]
            all_rules = [x for L in gr for x in L]

            # Per-rule ranks scaled to [0,1]
            out_df = pd.DataFrame({"row_id": all_ids, "rule": all_rules, "rule_violation": all_p})
            r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
            n = out_df.groupby("rule")["rule_violation"].transform("size")
            denom = (n - 1).where(n > 1, 1)  # avoid /0 for singletons
            out_df["rule_violation"] = (r - 1) / denom

            out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission5.csv", index=False)
        dist.destroy_process_group()
    else:
        out_df = pd.DataFrame({"row_id": out_ids, "rule": out_rules, "rule_violation": out_probs})
        r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
        n = out_df.groupby("rule")["rule_violation"].transform("size")
        denom = (n - 1).where(n > 1, 1)
        out_df["rule_violation"] = (r - 1) / denom
        out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission5.csv", index=False)

if __name__ == "__main__":
    main()


#!QUANT_MODE=8bit CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --master_addr=127.0.0.1 --master_port=29514 --nproc_per_node=2 inference_fp16_ddp.py
#!head submission5.csv


%%writefile constants.py
BASE_MODEL_PATH = "/kaggle/input/phi-4-mini-instruct/transformers/default/1"
LORA_PATH = "output/"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."


%%writefile utils.py
import pandas as pd
from datasets import Dataset
from constants import POSITIVE_ANSWER, NEGATIVE_ANSWER, COMPLETE_PHRASE, BASE_PROMPT


def build_prompt(row):
    return f"""
{BASE_PROMPT}

Comment: {row["body"]}

rule: {row["rule"]}
---
{COMPLETE_PHRASE}"""


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # >>> Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=2003).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])

    return dataframe


def build_dataset(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    columns = ["prompt"]
    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: POSITIVE_ANSWER,
                0: NEGATIVE_ANSWER,
            }
        )
        columns.append("completion")

    dataframe = dataframe[columns]
    dataset = Dataset.from_pandas(dataframe)
    return dataset


%%writefile train_fp16.py
import os, math, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import torch
import torch.distributed as dist
from datetime import timedelta
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils import get_dataframe_to_train, build_dataset
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH

# Speed-friendly defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
# QUANT_MODE: "4bit" (default) or "8bit"
QUANT_MODE = os.environ.get("QUANT_MODE", "4bit").lower()
# For 4-bit only:
BNB_4BIT_TYPE = os.environ.get("BNB_4BIT_TYPE", "nf4")          # "nf4" or "fp4"
BNB_4BIT_DQ   = os.environ.get("BNB_4BIT_DQ", "1") == "1"       # double quant: 1/0
# --------------------------------------

def setup_ddp():
    if "RANK" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            timeout=timedelta(minutes=30)
        )
    else:
        world, rank, local_rank = 1, 0, 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    if torch.cuda.is_available() and world > torch.cuda.device_count():
        raise RuntimeError(
            f"WORLD_SIZE={world} but only {torch.cuda.device_count()} visible CUDA device(s). "
            "Match --nproc_per_node to the number of GPUs or set CUDA_VISIBLE_DEVICES accordingly."
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return world, rank, local_rank, device

class CompletionOnlyDataset(Dataset):
    def __init__(self, tokenizer, dataframe, max_length=256, add_eos_to_completion=True):
        self.tok = tokenizer
        self.max_length = max_length
        self.samples = []
        prompts = list(dataframe["prompt"])
        completions = list(dataframe["completion"])

        eos = self.tok.eos_token or ""
        for p, c in zip(prompts, completions):
            p_ids = self.tok.encode(p, add_special_tokens=False)
            c_text = c + (eos if add_eos_to_completion else "")
            c_ids = self.tok.encode(c_text, add_special_tokens=False)
            input_ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + c_ids
            if len(input_ids) > self.max_length:
                excess = len(input_ids) - self.max_length
                input_ids = input_ids[excess:]
                labels = labels[excess:]
            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def left_pad_collate(batch, pad_id, max_length=None):
    max_len = max(len(x["input_ids"]) for x in batch)
    if max_length is not None:
        max_len = min(max_len, max_length)
    input_ids, labels, attn = [], [], []
    for ex in batch:
        ids = ex["input_ids"][-max_len:]
        labs = ex["labels"][-max_len:]
        pad_len = max_len - len(ids)
        input_ids.append([pad_id]*pad_len + ids)
        labels.append([-100]*pad_len + labs)
        attn.append([0]*pad_len + [1]*len(ids))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }

def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def cosine_with_warmup_lambda(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def main():
    world, rank, local_rank, device = setup_ddp()

    # Data
    df = get_dataframe_to_train(DATA_PATH)
    train_ds_hf = build_dataset(df)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ---- Quantized base (bitsandbytes), fp16 compute only ----
    if QUANT_MODE == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        if rank == 0: print("Using bitsandbytes: 8-bit (LLM.int8)")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_TYPE,
            bnb_4bit_use_double_quant=BNB_4BIT_DQ,
            bnb_4bit_compute_dtype=torch.float16,  # force fp16
        )
        if rank == 0: print(f"Using bitsandbytes: 4-bit ({BNB_4BIT_TYPE}, double_quant={BNB_4BIT_DQ})")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )

    # Disable cache during training & prefer SDPA if available
    if hasattr(base, "config"):
        base.config.use_cache = False
        try:
            base.config._attn_implementation = "sdpa"
        except Exception:
            pass
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tok.pad_token_id

    # Prepare for k-bit LoRA training (keeps fp16 compute)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    if rank == 0:
        tr, tot = count_trainable_params(model)
        print(f"Trainable params: {tr:,} / {tot:,} ({100*tr/tot:.4f}%)")

    max_length = 256
    torch_ds = CompletionOnlyDataset(tok, train_ds_hf.to_pandas(), max_length=max_length)
    if world > 1:
        sampler = DistributedSampler(torch_ds, num_replicas=world, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    per_device_batch_size = 8
    loader = DataLoader(
        torch_ds,
        batch_size=per_device_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: left_pad_collate(b, pad_id=tok.pad_token_id, max_length=max_length),
        drop_last=False,
    )

    # Optimizer & scheduler (trainable params only)
    lr, wd = 1.5e-4, 0.01
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=wd,
            fused=(torch.cuda.is_available() and torch.__version__ >= "2.0")
        )
    except TypeError:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    num_epochs = 1
    steps_per_epoch = len(loader)
    total_steps = num_epochs * steps_per_epoch

    warmup_steps = 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: cosine_with_warmup_lambda(s, warmup_steps, total_steps)
    )

    # fp16 autocast (T4-safe; no bf16)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    model.train()

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False, broadcast_buffers=False
        )

    global_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = 0.0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            running += loss.item()

            scaler.unscale_(optimizer)
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            if rank == 0 and global_step % 10 == 0:
                avg = running / 10.0
                print(f"[epoch {epoch+1}] step {global_step}/{total_steps} | "
                      f"lr={scheduler.get_last_lr()[0]:.6e} | loss={avg:.4f}")
                running = 0.0

        if rank == 0:
            print(f"Epoch {epoch+1} done in {(time.time()-t0)/60:.1f} min")

    if rank == 0:
        os.makedirs(LORA_PATH, exist_ok=True)
        to_save = model.module if hasattr(model, "module") else model
        to_save.save_pretrained(LORA_PATH)
        tok.save_pretrained(LORA_PATH)
        print(f"Saved LoRA + tokenizer to {LORA_PATH}")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()


#!QUANT_MODE=8bit CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --master_addr=127.0.0.1 --master_port=29513 --nproc_per_node=2 train_fp16.py


%%writefile inference_fp16_ddp.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, torch.distributed as dist
import pandas as pd
from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from utils import build_dataset
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH, POSITIVE_ANSWER, NEGATIVE_ANSWER

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
QUANT_MODE = os.environ.get("QUANT_MODE", "4bit").lower()
BNB_4BIT_TYPE = os.environ.get("BNB_4BIT_TYPE", "nf4")
BNB_4BIT_DQ   = os.environ.get("BNB_4BIT_DQ", "1") == "1"
# --------------------------------------

# Variants to aggregate (same spirit as train.py)
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]

def _first_token_ids(tok, txt_or_list):
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_list] if isinstance(txt_or_list, str) else list(txt_or_list)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def bucket_by_length(tok, prompts, max_length=512):
    lens = tok(prompts, return_length=True, truncation=True, max_length=max_length)
    order = sorted(range(len(prompts)), key=lambda k: lens["length"][k], reverse=True)
    return order

def main():
    # DDP-style sharding
    if "RANK" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    else:
        world, rank, local_rank = 1, 0, 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Quantized base (bnb), fp16 compute only
    if QUANT_MODE == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        if rank == 0: print("Using bitsandbytes: 8-bit (LLM.int8)")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_TYPE,
            bnb_4bit_use_double_quant=BNB_4BIT_DQ,
            bnb_4bit_compute_dtype=torch.float16,  # force fp16
        )
        if rank == 0: print(f"Using bitsandbytes: 4-bit ({BNB_4BIT_TYPE}, double_quant={BNB_4BIT_DQ})")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )

    # Load LoRA *without* merging (keep quantized)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    # Prefer SDPA if available
    try:
        model.config._attn_implementation = "sdpa"
    except Exception:
        pass

    # --- Aggregate first-token IDs for Yes/No (matches train.py approach) ---
    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)

    # Fallback to the constants if variants somehow mapped to nothing
    if not yes_ids:
        yes_ids = _first_token_ids(tok, POSITIVE_ANSWER)
    if not no_ids:
        no_ids = _first_token_ids(tok, NEGATIVE_ANSWER)

    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    df = pd.read_csv(f"{DATA_PATH}/test.csv")
    ds = build_dataset(df)
    prompts = list(ds["prompt"])
    rows    = list(df["row_id"])
    rules   = list(df["rule"])

    idx = list(range(len(prompts)))
    shard_idx     = idx[rank::world]
    shard_prompts = [prompts[i] for i in shard_idx]
    shard_rows    = [rows[i]    for i in shard_idx]
    shard_rules   = [rules[i]   for i in shard_idx]

    order = bucket_by_length(tok, shard_prompts, max_length=512)
    shard_prompts = [shard_prompts[i] for i in order]
    shard_rows    = [shard_rows[i]    for i in order]
    shard_rules   = [shard_rules[i]   for i in order]

    out_ids, out_probs, out_rules = [], [], []
    bs = 16
    max_len = 512

    for i in range(0, len(shard_prompts), bs):
        batch_prompts = shard_prompts[i:i+bs]
        enc = tok(
            batch_prompts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        # fp16 autocast (T4-safe; no bf16)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(**enc).logits[:, -1, :]

            # Select only the Yes/No first-token columns and compute in float32 for stability
            sel = logits[:, tgt_ids].to(torch.float32)
            logp = torch.log_softmax(sel, dim=-1)

        # Aggregate log-probs across all "Yes" and all "No" variants
        if yes_idx:
            ylp = torch.logsumexp(logp[:, yes_idx], dim=-1)
        else:
            ylp = torch.full((logp.size(0),), -1e9, device=logp.device)

        if no_idx:
            nlp = torch.logsumexp(logp[:, no_idx], dim=-1)
        else:
            nlp = torch.full((logp.size(0),), -1e9, device=logp.device)

        prob_yes = torch.softmax(torch.stack([ylp, nlp], dim=-1), dim=-1)[:, 0].float().cpu().tolist()

        out_probs.extend(prob_yes)
        out_ids.extend(shard_rows[i:i+bs])
        out_rules.extend(shard_rules[i:i+bs])

    if world > 1:
        gi, gp, gr = [None]*world, [None]*world, [None]*world
        dist.all_gather_object(gi, out_ids)
        dist.all_gather_object(gp, out_probs)
        dist.all_gather_object(gr, out_rules)
        if rank == 0:
            all_ids   = [x for L in gi for x in L]
            all_p     = [x for L in gp for x in L]
            all_rules = [x for L in gr for x in L]

            # Per-rule ranks scaled to [0,1]
            out_df = pd.DataFrame({"row_id": all_ids, "rule": all_rules, "rule_violation": all_p})
            r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
            n = out_df.groupby("rule")["rule_violation"].transform("size")
            denom = (n - 1).where(n > 1, 1)  # avoid /0 for singletons
            out_df["rule_violation"] = (r - 1) / denom

            out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission6.csv", index=False)
        dist.destroy_process_group()
    else:
        out_df = pd.DataFrame({"row_id": out_ids, "rule": out_rules, "rule_violation": out_probs})
        r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
        n = out_df.groupby("rule")["rule_violation"].transform("size")
        denom = (n - 1).where(n > 1, 1)
        out_df["rule_violation"] = (r - 1) / denom
        out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission6.csv", index=False)

if __name__ == "__main__":
    main()


#!QUANT_MODE=8bit CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --master_addr=127.0.0.1 --master_port=29514 --nproc_per_node=2 inference_fp16_ddp.py
#!head submission6.csv


%%writefile constants.py
BASE_MODEL_PATH = "/kaggle/input/berts/transformers/default/1/ettin-encoder-400m"

LORA_PATH = "output_deberta/"  # just a folder name; now stores full finetuned model
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."


%%writefile utils.py
import pandas as pd
from datasets import Dataset
from constants import POSITIVE_ANSWER, NEGATIVE_ANSWER, COMPLETE_PHRASE, BASE_PROMPT

def build_prompt(row):
    # Kept for parity; not used by the DeBERTa pipeline
    return f"""
{BASE_PROMPT}

Comment: {row["body"]}

rule: {row["rule"]}
---
{COMPLETE_PHRASE}"""

def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # upsample target block (test examples) by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples once more (2x extra copies -> appears 3x total)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    dataframe = dataframe.sample(frac=1.0, random_state=3001).reset_index(drop=True)
    dataframe = dataframe.drop(columns=["source"])
    return dataframe

def build_classification_dataframe(df, tok_sep_token="</s>"):
    """
    Build a dataframe for classification using text = rule + [SEP] + comment.
    """
    df = df.copy()
    sep = tok_sep_token if tok_sep_token else "</s>"
    df["text"] = df["rule"].astype(str) + sep + df["body"].astype(str)
    cols = ["text"]
    if "rule_violation" in df:
        cols.append("rule_violation")
    return df[cols]

def hf_dataset_from_dataframe(df):
    return Dataset.from_pandas(df.reset_index(drop=True))


%%writefile train_deberta_fp16.py
import os, math, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import torch
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

from utils import get_dataframe_to_train, build_classification_dataframe
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH

# Speed-friendly defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
EPOCHS     = int(os.environ.get("EPOCHS", "1"))
LR         = float(os.environ.get("LR", "2e-5"))
WD         = float(os.environ.get("WD", "0.01"))
WARMUP     = float(os.environ.get("WARMUP_RATIO", "0.00"))  # fraction of total steps
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1"
# --------------------------------------

def setup_ddp():
    if "RANK" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            timeout=timedelta(minutes=30)
        )
    else:
        world, rank, local_rank = 1, 0, 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return world, rank, local_rank, device

class TxtClsDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)

    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        if self.labels is None:
            return {"text": self.texts[i]}
        return {"text": self.texts[i], "label": float(self.labels[i])}

def collate_fn_builder(tokenizer, max_length):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "label" in batch[0]:
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32).unsqueeze(-1)
            out["labels"] = labels
        return out
    return _fn

def main():
    world, rank, local_rank, device = setup_ddp()

    # Data
    raw_df = get_dataframe_to_train(DATA_PATH)

    # Tokenizer first, to know SEP token for concatenation
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=False)
    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"

    df = build_classification_dataframe(raw_df, tok_sep_token=sep_token)

    train_ds = TxtClsDataset(df["text"], df["rule_violation"])
    if world > 1:
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    collate_fn = collate_fn_builder(tokenizer, MAX_LENGTH)
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Model
    cfg = AutoConfig.from_pretrained(BASE_MODEL_PATH, num_labels=1, problem_type=None)  # we'll set loss manually
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH, config=cfg)
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    model.to(device)

    # Optimizer / Scheduler
    # Only full-params (no LoRA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = EPOCHS * len(loader) // max(1, GRAD_ACCUM)
    warmup_steps = int(WARMUP * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss
    bce = torch.nn.BCEWithLogitsLoss()

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.train()

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False
        )

    global_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = 0.0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)  # shape (B,1)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits  # (B,1)
                loss = bce(logits, labels)

            loss = loss / max(1, GRAD_ACCUM)
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += loss.item() * max(1, GRAD_ACCUM)

            if rank == 0 and global_step > 0 and global_step % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[epoch {epoch+1}] step {global_step}/{total_steps} | lr={lr_now:.6e} | loss={running/10.0:.4f}")
                running = 0.0

        if rank == 0:
            print(f"Epoch {epoch+1} done in {(time.time()-t0)/60:.1f} min")

    # Save (rank 0)
    if rank == 0:
        os.makedirs(LORA_PATH, exist_ok=True)  # reuse var name for output dir
        to_save = model.module if hasattr(model, "module") else model
        to_save.save_pretrained(LORA_PATH)
        tokenizer.save_pretrained(LORA_PATH)
        print(f"Saved model + tokenizer to {LORA_PATH}")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()


#!CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --master_addr=127.0.0.1 --master_port=29513 --nproc_per_node=2 train_deberta_fp16.py


%%writefile inference_deberta_fp16_ddp.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, torch.distributed as dist
import pandas as pd
from datetime import timedelta
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from utils import build_classification_dataframe
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))

def setup_ddp():
    if "RANK" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    else:
        world, rank, local_rank = 1, 0, 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return world, rank, local_rank, device

def bucket_by_length(tok, texts, max_length=512):
    lens = tok(texts, return_length=True, truncation=True, max_length=max_length)
    order = sorted(range(len(texts)), key=lambda k: lens["length"][k], reverse=True)
    return order

def main():
    world, rank, local_rank, device = setup_ddp()

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, use_fast=True)
    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"

    cfg = AutoConfig.from_pretrained(LORA_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LORA_PATH, config=cfg)
    model.eval()
    model.to(device)

    # Load test
    df = pd.read_csv(f"{DATA_PATH}/test.csv")

    # --- robust column handling ---
    # choose the comment/body column
    if "body" in df.columns:
        text_col = "body"
    elif "comment" in df.columns:
        text_col = "comment"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise KeyError(f"Could not find a text column. Available columns: {list(df.columns)}")

    if "rule" not in df.columns:
        raise KeyError("Expected 'rule' column in test.csv but did not find it.")

    # Build the classification dataframe using rule + [SEP] + body/comment
    tmp = df.rename(columns={text_col: "body"})  # utils expects 'body'
    cls_df = build_classification_dataframe(tmp, tok_sep_token=sep_token)

    texts = list(cls_df["text"])
    rows  = list(df["row_id"]) if "row_id" in df.columns else list(range(len(df)))
    rules = list(df["rule"])

    # DDP shard
    idx = list(range(len(texts)))
    shard_idx   = idx[rank::world]
    shard_texts = [texts[i] for i in shard_idx]
    shard_rows  = [rows[i]  for i in shard_idx]
    shard_rules = [rules[i] for i in shard_idx]

    order = bucket_by_length(tokenizer, shard_texts, max_length=MAX_LENGTH)
    shard_texts = [shard_texts[i] for i in order]
    shard_rows  = [shard_rows[i]  for i in order]
    shard_rules = [shard_rules[i] for i in order]

    out_ids, out_probs, out_rules = [], [], []

    for i in range(0, len(shard_texts), BATCH_SIZE):
        batch_texts = shard_texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(**enc).logits  # (B,1)
            probs = torch.sigmoid(logits).squeeze(-1)  # (B,)

        out_probs.extend(probs.float().cpu().tolist())
        out_ids.extend(shard_rows[i:i+BATCH_SIZE])
        out_rules.extend(shard_rules[i:i+BATCH_SIZE])

    if world > 1:
        gi, gp, gr = [None]*world, [None]*world, [None]*world
        dist.all_gather_object(gi, out_ids)
        dist.all_gather_object(gp, out_probs)
        dist.all_gather_object(gr, out_rules)
        if rank == 0:
            all_ids   = [x for L in gi for x in L]
            all_p     = [x for L in gp for x in L]
            all_rules = [x for L in gr for x in L]

            out_df = pd.DataFrame({"row_id": all_ids, "rule": all_rules, "rule_violation": all_p})
            # Per-rule ranks scaled to [0,1]
            r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
            n = out_df.groupby("rule")["rule_violation"].transform("size")
            denom = (n - 1).where(n > 1, 1)
            out_df["rule_violation"] = (r - 1) / denom

            out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission7.csv", index=False)
        dist.destroy_process_group()
    else:
        out_df = pd.DataFrame({"row_id": out_ids, "rule": out_rules, "rule_violation": out_probs})
        r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
        n = out_df.groupby("rule")["rule_violation"].transform("size")
        denom = (n - 1).where(n > 1, 1)
        out_df["rule_violation"] = (r - 1) / denom
        out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission7.csv", index=False)

if __name__ == "__main__":
    main()


#!CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --master_addr=127.0.0.1 --master_port=29514 --nproc_per_node=2 inference_deberta_fp16_ddp.py
#!head submission7.csv


#import pandas as pd
#df1 = pd.read_csv("submission1.csv")
#df2 = pd.read_csv("submission2.csv")
#df3 = pd.read_csv("submission3.csv")
#df4 = pd.read_csv("submission4.csv")
#df5 = pd.read_csv("submission5.csv")
#df6 = pd.read_csv("submission6.csv")
#df7 = pd.read_csv("submission7.csv")
#df1["rule_violation"] = 0.25*df1["rule_violation"] + 0.25*df2["rule_violation"] + 0.2*df3["rule_violation"] + 0.15*df4["rule_violation"] + 0.1*df5["rule_violation"] + 0.1*df6["rule_violation"] + 0.1*df7["rule_violation"]
#df1.to_csv("submission.csv", index=False)
#print(df1.head(10))

