import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple
from textwrap import dedent
import re

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# -------------------------
# Config
# -------------------------
@dataclass
class InferenceConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_path: str = "train.csv"
    is_submission: bool = False
    id_column: str = "row_id"        # used only when is_submission=True

    text_column: str = "body"
    rule_column: str = "rule"
    subreddit_column: str = "subreddit"
    pos_example_cols: Tuple[str, str] = ("positive_example_1", "positive_example_2")
    neg_example_cols: Tuple[str, str] = ("negative_example_1", "negative_example_2")
    label_column: str = "rule_violation"   # used only when is_submission=False

    # Prompting
    answer_prefix: str = "Answer:"
    yes_texts: List[str] = ("yes", "Yes")
    no_texts: List[str]  = ("no", "No")

    # Template-based prompt (you can replace this with your own function)
    prompt_template: str = (
        "You are a moderation assistant. Decide if the COMMENT violates the RULE.\n\n"
        "SUBREDDIT: {subreddit}\n"
        "RULE: {rule}\n\n"
        "Here are examples of violating comments:\n"
        "1) {pos1}\n"
        "2) {pos2}\n\n"
        "Here are examples of allowed comments:\n"
        "1) {neg1}\n"
        "2) {neg2}\n\n"
        "Now evaluate the new COMMENT:\n"
        "{body}\n\n"
        "Respond with a single word 'yes' or 'no'.\n"
        "{answer_prefix} "
    )

    # Logit shaping (applied BEFORE probability extraction)
    temperature: float = 1.0         # 1.0 = no scaling
    top_k: int = 0                   # 0 = disabled
    top_p: float = 1.0               # 1.0 = disabled

    # Runtime
    batch_size: int = 2
    max_examples: int = 1024         # ignored when is_submission=True
    seed: int = 42
    device: str = "auto"             # "auto", "cuda", "cpu", or "mps"

    # Quantization (optional, requires bitsandbytes)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "auto"        # "auto", "float16", "bfloat16", "float32"

    # Tokenization / lengths
    max_input_length: int = 4096     # truncate if needed

    # I/O
    output_csv: str = "/kaggle/working/submission.csv"   # will default to "submission.csv" if is_submission=True and not overridden
    print_examples: int = 5             # only used when is_submission=False


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_dtype(name: str):
    name = (name or "auto").lower()
    if name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(name, None)

def pick_device(pref: str) -> str:
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return pref

def expand_label_variants(texts: List[str]) -> List[str]:
    """
    For robustness across tokenizers, add common variants: leading space, trailing period,
    trailing newline, with/without capitalization.
    """
    variants = set()
    for t in texts:
        base = t
        for v in [base, base + ".", base + "\n", " " + base, " " + base + ".", " " + base + "\n"]:
            variants.add(v)
    return sorted(variants)

def ids_for_variants(tokenizer, variants: List[str]) -> List[int]:
    """
    Map string variants to token IDs that would likely be emitted as the FIRST token.
    Take the first token ID of each variant’s encoding.
    """
    ids = set()
    for v in variants:
        enc = tokenizer(v, add_special_tokens=False, return_tensors=None)
        if enc["input_ids"]:
            ids.add(enc["input_ids"][0])  # first token ID
    return sorted(ids)

def apply_logit_shaping(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    logits: [B, V]
    Returns shaped logits (masked for top-k/top-p if enabled), scaled by temperature.
    """
    # Temperature
    if temperature and temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # Top-k
    if top_k and top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(1, topk_idx, topk_vals)
        logits = mask

    # Top-p (nucleus)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        cutoff = (cumulative_probs > top_p).float()
        cutoff[..., 0] = 0.0  # always keep top-1
        sorted_logits = torch.where(cutoff.bool(), torch.full_like(sorted_logits, float('-inf')), sorted_logits)
        # Back to original order
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(1, sorted_indices, sorted_logits)

    return logits


def replace_urls(text: str) -> str:
    # Regex pattern for URLs (http, https, www, etc.)
    url_pattern = re.compile(
        r'(https?://\S+|www\.\S+)',
        re.IGNORECASE
    )
    return url_pattern.sub("<URL>", text)

def build_prompt(row: pd.Series, cfg: InferenceConfig) -> str:
    pos1 = row.get(cfg.pos_example_cols[0], "") or ""
    pos2 = row.get(cfg.pos_example_cols[1], "") or ""
    neg1 = row.get(cfg.neg_example_cols[0], "") or ""
    neg2 = row.get(cfg.neg_example_cols[1], "") or ""
    body = row.get(cfg.text_column, "")
    body = replace_urls(body)

    return cfg.prompt_template.format(
        subreddit=row.get(cfg.subreddit_column, ""),
        rule=row.get(cfg.rule_column, ""),
        pos1=pos1,
        pos2=pos2,
        neg1=neg1,
        neg2=neg2,
        body=row.get(cfg.text_column, ""),
        answer_prefix=cfg.answer_prefix
    )


# -------------------------
# Core: scoring
# -------------------------
def score_batch(
    prompts: List[str],
    model,
    tokenizer,
    yes_ids: List[int],
    no_ids: List[int],
    cfg: InferenceConfig
) -> List[float]:
    """
    Returns list of P(violation) for each prompt using binary-normalized next-token probability.
    """
    device = pick_device(cfg.device)
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=cfg.max_input_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        # logits: [B, T, V] -> take last prompt token position for each example
        input_lengths = (enc["attention_mask"].sum(dim=1) - 1)  # index of last token
        batch_logits = out.logits[torch.arange(out.logits.size(0)), input_lengths]  # [B, V]
        shaped = apply_logit_shaping(batch_logits, cfg.temperature, cfg.top_k, cfg.top_p)
        probs = torch.softmax(shaped, dim=-1)  # [B, V]

        yes_prob = probs[:, yes_ids].sum(dim=-1) if len(yes_ids) else torch.zeros(probs.size(0), device=device)
        no_prob  = probs[:,  no_ids].sum(dim=-1) if len(no_ids)  else torch.zeros(probs.size(0), device=device)
        denom = yes_prob + no_prob
        eps = 1e-12
        p_violation = torch.where(denom > eps, yes_prob / (denom + eps), yes_prob)
        return p_violation.detach().cpu().tolist()


# -------------------------
# Loading model
# -------------------------
def load_model_and_tokenizer(cfg: InferenceConfig):
    device = pick_device(cfg.device)
    dtype = resolve_dtype(cfg.torch_dtype)

    kwargs = dict()
    if cfg.load_in_8bit or cfg.load_in_4bit:
        # bitsandbytes quantization
        kwargs["device_map"] = "auto"
        if cfg.load_in_8bit:
            kwargs["load_in_8bit"] = True
        if cfg.load_in_4bit:
            kwargs["load_in_4bit"] = True
    else:
        # standard load
        kwargs["torch_dtype"] = dtype
        if device == "cuda":
            kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    # ensure a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    if device in ("cpu", "mps") and not (cfg.load_in_8bit or cfg.load_in_4bit):
        model = model.to(device)

    return model, tokenizer


# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(cfg: InferenceConfig):
    set_seed(cfg.seed)

    # Load data
    df = pd.read_csv(cfg.dataset_path)

    # Submission vs. Train/Dev behavior
    if cfg.is_submission:
        # Ensure ID column exists
        if cfg.id_column not in df.columns:
            raise ValueError(
                f"ID column '{cfg.id_column}' not found in dataset. "
                "Pass --id_column to match your test.csv (default 'row_id')."
            )
        # Use all rows for submission; ignore sampling
        # Configure output file if user didn't override
        if cfg.output_csv == "preds_llm.csv":
            cfg.output_csv = "submission.csv"
    else:
        # Expect label column for AUC
        if cfg.label_column not in df.columns:
            raise ValueError(f"Label column '{cfg.label_column}' not found in dataset.")
        # Sample if needed
        if cfg.max_examples and cfg.max_examples < len(df):
            df = df.sample(cfg.max_examples, random_state=cfg.seed).reset_index(drop=True)

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Prepare yes/no token id sets
    yes_variants = expand_label_variants(list(cfg.yes_texts))
    no_variants  = expand_label_variants(list(cfg.no_texts))
    yes_ids = ids_for_variants(tokenizer, yes_variants)
    no_ids  = ids_for_variants(tokenizer, no_variants)

    if not yes_ids or not no_ids:
        print("[WARN] Could not derive token IDs for YES/NO variants. "
              "Consider adjusting --yes_texts/--no_texts (e.g., add ' yes', 'Yes.', '\\nyes').")

    # Build prompts
    prompts = [build_prompt(row, cfg) for _, row in df.iterrows()]

    # Batched scoring
    preds: List[float] = []
    for i in tqdm(range(0, len(prompts), cfg.batch_size), desc="Scoring"):
        batch_prompts = prompts[i:i + cfg.batch_size]
        batch_scores = score_batch(batch_prompts, model, tokenizer, yes_ids, no_ids, cfg)
        preds.extend(batch_scores)

    # Output
    if cfg.is_submission:
        sub = pd.DataFrame({
            cfg.id_column: df[cfg.id_column].values,
            "rule_violation": preds
        })
        # Kaggle expects exactly: row_id, rule_violation
        if cfg.id_column != "row_id":
            sub.rename(columns={cfg.id_column: "row_id"}, inplace=True)
        sub.to_csv(cfg.output_csv, index=False)
        print(f"\nSaved submission file to: {cfg.output_csv}")
    else:
        df_out = df.copy()
        df_out["pred_violation"] = preds
        # AUC
        y_true = df_out[cfg.label_column].astype(int).values
        auc = roc_auc_score(y_true, df_out["pred_violation"].values)
        print(f"\nAUC: {auc:.6f}")

        # Show a few examples for sanity check
        if cfg.print_examples > 0:
            print("\n--- Sample predictions ---")
            for idx in range(min(cfg.print_examples, len(df_out))):
                r = df_out.iloc[idx]
                body_preview = (r.get(cfg.text_column, "") or "").replace("\n", " ")
                if len(body_preview) > 300:
                    body_preview = body_preview[:300] + "..."
                print(f"[{idx}] y={r[cfg.label_column]}  p_yes={r['pred_violation']:.4f}")
                print(f"RULE: {r.get(cfg.rule_column, '')}")
                print(f"BODY: {body_preview}")
                print("-" * 60)

        # Save rich preds (optional)
        df_out.to_csv(cfg.output_csv, index=False)
        print(f"\nSaved predictions to: {cfg.output_csv}")


cfg = InferenceConfig(
    model_name="/kaggle/input/shieldgemma-2b-ported/transformers/base/1",
    dataset_path="/kaggle/input/jigsaw-agile-community-rules/test.csv",
    answer_prefix='',
    prompt_template=dedent("""
      You are a policy expert trying to help determine whether a user prompt is in
      violation of the defined safety policies.

      <start_of_turn>
      Human Question: {body}
      <end_of_turn>

      Our safety principle is defined in the below:

      * "{rule}"

      ===

      Does the human question violate the above principle? Your answer must start with
      'Yes' or 'No'. And then walk through step by step to be sure we answer
      correctly.
    """),
    max_examples=100,
    batch_size=1,
    print_examples=10,
    is_submission=True,
    id_column='row_id'
)


run_pipeline(cfg)




