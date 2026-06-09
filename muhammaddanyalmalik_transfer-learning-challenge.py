import pandas as pd

# =============================
# 1. List your 10 submission files
# =============================
submission_files = [
    "/kaggle/input/spectra-flow-an-advanced-raman-spectroscopy-ml/submission.csv",
    "/kaggle/input/pca-and-lr-is-all-u-need-pls-upvote/submission.csv",
    "/kaggle/input/spectra-flow-sav-gol-tuned/submission.csv",
    "/kaggle/input/pca-lr-extraregressiontree/submission.csv",
]

# =============================
# 2. Define weights (must sum to 1.0)
# Example: give more weight to better models
# =============================
weights = [0.45, 0.15, 0.15, 0.25]


# Sanity check
assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1!"

# =============================
# 3. Load submissions
# =============================
submissions = [pd.read_csv(f) for f in submission_files]

# Ensure all IDs match
for i, sub in enumerate(submissions):
    assert (sub['ID'].values == submissions[0]['ID'].values).all(), f"Mismatch in submission {i+1}"

# =============================
# 4. Ensemble (weighted average)
# =============================
final = submissions[0].copy()
target_cols = [c for c in final.columns if c != "ID"]

# Start with zeros
for col in target_cols:
    final[col] = 0

# Add weighted contributions
for sub, w in zip(submissions, weights):
    for col in target_cols:
        final[col] += w * sub[col]

# =============================
# 5. Save final submission
# =============================
final_filename = "/kaggle/working/submission1.csv"
final.to_csv(final_filename, index=False)

print(f"✅ Final ensemble submission saved: {final_filename}")
print(final.head())


# # qwen_tabular_sft.py
# import os
# import math
# import json
# import pandas as pd
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer, AutoConfig, AutoModelForCausalLM,
#     Trainer, TrainingArguments, DataCollatorForLanguageModeling
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import torch
# import numpy as np
# from sklearn.model_selection import train_test_split

# # ---------- CONFIG ----------
# MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # HF model repo name; change if needed
# SYSTEM_PROMPT = ("You are a precise numeric predictor. "
#                  "Input: spectral features as key=value pairs. "
#                  "Output: three comma-separated numeric concentrations in order: "
#                  "Glucose, Sodium Acetate, Magnesium Sulfate. "
#                  "Return only the three numbers separated by commas, no text. "
#                  "Round to 6 decimals.")
# TRAIN_CSV = "transfer_plate.csv"
# TEST_CSV  = "96_samples.csv"
# SAMPLE_SUB = "sample_submission.csv"
# OUTPUT_JSONL = "sft_train.jsonl"
# OUTPUT_DIR = "./qwen_lora_out"
# BATCH_SIZE = 8
# EPOCHS = 3
# LR = 2e-4
# WEIGHT_DECAY = 0.0
# SEED = 42
# # ----------------------------

# torch.manual_seed(SEED)
# np.random.seed(SEED)

# # ---------- helper: row -> prompt+completion ----------
# def row_to_prompt_completion(row, feature_cols, target_cols):
#     # construct compact feature string; use key=value pairs
#     feats = ", ".join(f"{c}={row[c]:.6f}" for c in feature_cols)
#     prompt = SYSTEM_PROMPT + "\n\nInput: " + feats + "\nOutput: "
#     # completion should be exactly the numbers, comma-separated
#     completion = ", ".join(f"{row[t]:.6f}" for t in target_cols)
#     # For causal LM SFT, we provide the prompt + completion for supervised training.
#     # We will use the concatenated text: prompt + completion
#     full = prompt + completion
#     return {"text": full}

# # ---------- load data ----------
# train_df = pd.read_csv(TRAIN_CSV)
# test_df = pd.read_csv(TEST_CSV)
# sample_sub = pd.read_csv(SAMPLE_SUB)

# # Identify columns: adapt if your CSV has other column names
# target_cols = ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]
# feature_cols = [c for c in train_df.columns if c not in ["ID"] + target_cols]

# # Create SFT dataset (JSONL)
# records = []
# for _, r in train_df.iterrows():
#     records.append(row_to_prompt_completion(r, feature_cols, target_cols))

# # Save JSONL for inspection
# with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
#     for rec in records:
#         f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# # ---------- create HuggingFace Dataset ----------
# ds = Dataset.from_list(records)
# # split small val set
# ds = ds.train_test_split(test_size=0.05, seed=SEED)
# train_ds = ds["train"]
# eval_ds = ds["test"]

# # ---------- tokenizer and model ----------
# tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
# # ensure tokenizer has pad token
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# def tokenize_fn(examples):
#     return tokenizer(examples["text"], truncation=True, max_length=512)

# train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
# eval_tok  = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# # Load model with 8-bit/4-bit optimizations if you want (here standard)
# config = AutoConfig.from_pretrained(MODEL)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL,
#     device_map="auto",            # or use 'cuda' with accelerate
#     torch_dtype=torch.float16,    # float16 usually for speed
#     low_cpu_mem_usage=True
# )

# # Prepare for k-bit if you plan to use bitsandbytes (optional)
# model = prepare_model_for_kbit_training(model)

# # ---------- PEFT / LoRA setup ----------
# lora_config = LoraConfig(
#     r=8,                        # rank
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # common for causal models
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(model, lora_config)

# # Resize token embeddings if we added pad token
# model.resize_token_embeddings(len(tokenizer))

# # ---------- training ----------
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     num_train_epochs=EPOCHS,
#     learning_rate=LR,
#     weight_decay=WEIGHT_DECAY,
#     logging_steps=50,
#     evaluation_strategy="steps",
#     eval_steps=200,
#     save_strategy="no",       # set to "steps" or "epoch" if you want checkpoints
#     fp16=True,
#     push_to_hub=False,
#     seed=SEED
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_tok,
#     eval_dataset=eval_tok,
#     data_collator=data_collator,
# )

# trainer.train()
# # Save PEFT adapter
# model.save_pretrained(OUTPUT_DIR + "/peft_adapter")
# tokenizer.save_pretrained(OUTPUT_DIR + "/tokenizer")

# # ---------- inference on test ----------
# # Reload model with adapter for inference (if needed)
# from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.float16)
# peft_model = PeftModel.from_pretrained(base, OUTPUT_DIR + "/peft_adapter")
# peft_model = peft_model.to("cuda")

# tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR + "/tokenizer", use_fast=False)

# def predict_row(row):
#     feats = ", ".join(f"{c}={row[c]:.6f}" for c in feature_cols)
#     prompt = SYSTEM_PROMPT + "\n\nInput: " + feats + "\nOutput: "
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     out = peft_model.generate(**inputs, max_new_tokens=40, do_sample=False)
#     text = tokenizer.decode(out[0], skip_special_tokens=True)
#     # strip prompt then get completion
#     completion = text[len(prompt):].strip()
#     # keep only first line / first token chunk
#     # parse three numbers separated by comma
#     try:
#         nums = [float(x.strip()) for x in completion.split(",")[:3]]
#     except Exception as e:
#         # fallback: zeros
#         nums = [0.0, 0.0, 0.0]
#     # return dict matching sample submission columns
#     return nums

# preds = []
# for _, r in test_df.iterrows():
#     g, a, m = predict_row(r)
#     preds.append({"ID": r["ID"], "Glucose": g, "Sodium Acetate": a, "Magnesium Sulfate": m})

# final_df = pd.DataFrame(preds)
# final_df.to_csv("submission_best_model.csv", index=False)
# print("Saved submission_best_model.csv")

