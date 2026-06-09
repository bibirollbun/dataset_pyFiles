import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print("PID:", os.getpid(), "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

import torch
assert torch.cuda.is_available()
print("Visible device count:", torch.cuda.device_count())
torch.cuda.set_device(0)
print("Using:", torch.cuda.get_device_name(0))

# Offline mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import math
import numpy as np
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from typing import List

# =========================
# KONFIGURASI
# =========================
BASE_MODEL_PATH = "/kaggle/input/qwen3-8b-unsloth-bnb-4bit/transformers/default/1"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."

# Training settings
MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 1
NUM_EPOCHS = 1
LEARNING_RATE = 1.5e-4
LORA_R = 16
LORA_ALPHA = 32

# Inference settings
INFER_BATCH_SIZE = 8
INFER_MAX_LENGTH = 512

# =========================
# DATA PREPARATION
# =========================
def get_training_dataframe(data_path: str) -> pd.DataFrame:
    """
    Load dan prepare data untuk training
    - Gabungkan train.csv dengan test examples
    - Upsample test examples 3x untuk better generalization
    """
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")
    
    flatten = []
    
    # Base training data
    base = train_df[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)
    
    # Extract positive dan negative examples dari test
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_df = test_df[[col, "rule"]].copy()
            sub_df = sub_df.rename(columns={col: "body"})
            sub_df["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_df["source"] = "test_examples"
            flatten.append(sub_df)
    
    # Combine dan remove duplicates
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)
    
    # Upsample test examples (muncul 3x total)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows, test_rows], axis=0, ignore_index=True)
    
    # Shuffle
    dataframe = dataframe.sample(frac=1.0, random_state=1009).reset_index(drop=True)
    dataframe = dataframe.drop(columns=["source"])
    
    print(f"Total training samples: {len(dataframe)}")
    print(f"Positive samples: {dataframe['rule_violation'].sum()}")
    print(f"Negative samples: {len(dataframe) - dataframe['rule_violation'].sum()}")
    
    return dataframe


def make_conversations_dataset(df: pd.DataFrame) -> Dataset:
    """Convert dataframe ke format conversations untuk Qwen"""
    df = df.copy()
    df["completion"] = df["rule_violation"].map({1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER})
    
    convos = []
    for _, row in df.iterrows():
        convos.append([
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user", "content": f"Comment: {row['body']}\n\nrule: {row['rule']}"},
            {"role": "assistant", "content": str(row["completion"])},
        ])
    
    return Dataset.from_dict({"conversations": convos})


def build_text_dataset(tokenizer, conv_dataset: Dataset):
    """Apply chat template ke conversations"""
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, 
                tokenize=False, 
                add_generation_prompt=False, 
                enable_thinking=False
            )[:-11]  # Remove trailing tokens
            for conv in convos
        ]
        return {"text": texts}
    
    return conv_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=conv_dataset.column_names,
    )


# =========================
# TRAINING
# =========================
def train_model():
    """Training function dengan Unsloth + LoRA"""
    print("\n" + "="*50)
    print("MULAI TRAINING")
    print("="*50 + "\n")
    
    # Load model dengan 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto detect
        load_in_4bit=True,
        local_files_only=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1009,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Setup tokenizer
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    
    # Prepare data
    df = get_training_dataframe(DATA_PATH)
    conv_dataset = make_conversations_dataset(df)
    train_dataset = build_text_dataset(tokenizer, conv_dataset)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
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
    
    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user",
        response_part="<think>\n\n</think>\n\n",
    )
    
    print("\nMulai training...")
    trainer.train()
    print("\nTraining selesai!\n")
    
    return model, tokenizer


# =========================
# INFERENCE UTILITIES
# =========================
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No", "NO", "N", "no", "False"]

def _first_token_ids(tok, txt_or_texts) -> List[int]:
    """Get unique first token IDs untuk text variants"""
    texts = [txt_or_texts] if isinstance(txt_or_texts, str) else list(txt_or_texts)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)


def _encode_batch(tok, bodies, rules, device, max_len=512):
    """Manual left padding + truncation untuk batch inference"""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    
    # Tokenize setiap sample
    seqs = []
    for body, rule in zip(bodies, rules):
        msgs = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user", "content": f"Comment: {body}\n\nrule: {rule}"},
        ]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        
        # Left truncation
        if len(ids) > max_len:
            ids = ids[-max_len:]
        
        seqs.append(torch.tensor(ids, dtype=torch.long))
    
    # Left padding
    B = len(seqs)
    T = min(max_len, max(len(x) for x in seqs)) if seqs else 1
    
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    
    for i, ids in enumerate(seqs):
        L = min(T, len(ids))
        input_ids[i, T - L : T] = ids[-L:]
        attention_mask[i, T - L : T] = 1
    
    # Position IDs
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    position_ids = position_ids.to(dtype=torch.long)
    
    return {
        "input_ids": input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "position_ids": position_ids.to(device, non_blocking=True),
    }


# =========================
# INFERENCE
# =========================
@torch.inference_mode()
def run_inference(model, tokenizer):
    """Run inference dan generate submission"""
    print("\n" + "="*50)
    print("MULAI INFERENCE")
    print("="*50 + "\n")
    
    tok = tokenizer
    
    # Setup token IDs untuk Yes/No
    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids = _first_token_ids(tok, NEGATIVE_VARIANTS)
    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx = [tgt_ids.index(t) for t in no_ids]
    
    # Load test data
    test_df = pd.read_csv(f"{DATA_PATH}/test.csv")
    bodies = list(test_df["body"])
    rules = list(test_df["rule"])
    rowids = list(test_df["row_id"])
    
    N = len(bodies)
    
    # Sort by length untuk efficient batching
    approx_lens = [
        (len(tok.encode(b, add_special_tokens=False)) +
         len(tok.encode(r, add_special_tokens=False)))
        for b, r in zip(bodies, rules)
    ]
    sorted_idx = sorted(range(N), key=lambda i: min(approx_lens[i], INFER_MAX_LENGTH))
    
    FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device
    
    probs_yes = [None] * N
    
    # Batch inference
    for i in range(0, N, INFER_BATCH_SIZE):
        if i % 100 == 0:
            print(f"Processing batch {i}/{N}...")
        
        batch_indices = sorted_idx[i:i+INFER_BATCH_SIZE]
        bb = [bodies[j] for j in batch_indices]
        rr = [rules[j] for j in batch_indices]
        
        enc = _encode_batch(tok, bb, rr, device=device, max_len=INFER_MAX_LENGTH)
        
        # Forward pass
        out = model(**enc, use_cache=True)
        step_scores = out.logits[:, -1, :]
        sel = step_scores[:, tgt_ids]
        logp = torch.log_softmax(sel.to(torch.float32), dim=-1)
        
        # Aggregate Yes/No probabilities
        y_logp = (torch.logsumexp(logp[:, yes_idx], dim=-1)
                  if yes_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        n_logp = (torch.logsumexp(logp[:, no_idx], dim=-1)
                  if no_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        p_yes = torch.softmax(torch.stack([y_logp, n_logp], dim=-1), dim=-1)[:, 0]
        
        for k, j in enumerate(batch_indices):
            probs_yes[j] = float(p_yes[k].item())
    
    # Build per-rule ranked scores [0, 1]
    df_scores = pd.DataFrame({
        "row_id": rowids,
        "rule": rules,
        "prob": probs_yes,
    })
    
    grp = df_scores.groupby("rule")
    rank = grp["prob"].rank(method="average", ascending=True)
    n = grp["prob"].transform("size")
    score = (rank - 1.0) / np.maximum(n - 1.0, 1.0)
    df_scores["rule_violation"] = score
    
    out_df = df_scores[["row_id", "rule_violation"]].sort_values("row_id").reset_index(drop=True)
    
    # Save dengan full precision (semua angka di belakang koma)
    out_df.to_csv("submission.csv", index=False, float_format='%.18f')
    
    print("\nInference selesai!")
    print(f"Submission saved to submission.csv")
    print("\nFirst 10 predictions:")
    print(out_df.head(10))
    
    return out_df


# =========================
# MAIN
# =========================
def main():
    print("\n" + "="*50)
    print("QWEN MODEL - COMPLETE PIPELINE")
    print("="*50)
    
    # Training
    model, tokenizer = train_model()
    
    # Inference
    submission = run_inference(model, tokenizer)
    
    print("\n" + "="*50)
    print("PIPELINE SELESAI!")
    print("="*50 + "\n")
    
    return submission


if __name__ == "__main__":
    main()

