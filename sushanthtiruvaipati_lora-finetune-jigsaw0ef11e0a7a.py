!pip install -q transformers trl peft accelerate deepspeed bitsandbytes


!pip install -q optimum auto-gptq


"""
Qwen2.5 LoRA Fine-tuning - Simple Version
Run this after installing: transformers trl peft accelerate deepspeed bitsandbytes
"""

import pandas as pd
import numpy as np
import random
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers.utils import is_torch_bf16_gpu_available
import warnings

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# Configuration
BASE_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/0.5b-instruct-gptq-int4/1"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
OUTPUT_DIR = "./lora_output"

BASE_PROMPT = "You are given a comment from reddit and a rule. Your task is to classify whether the comment violates the rule. Only respond Yes/No."

print("Starting LoRA fine-tuning...")

# Build prompt function
def build_prompt(row):
    return f"""
{BASE_PROMPT}

Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}
Examples:
1) {row["positive_example"]}
Answer: Yes

2) {row["negative_example"]}
Answer: No

---
Comment: {row["body"]}
Answer:"""

# Create training data
print("\nCreating training data...")
train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
test_df = pd.read_csv(f"{DATA_PATH}/test.csv").sample(frac=0.5, random_state=42)

flatten = []

# Process train data
train_df["positive_example"] = np.where(
    np.random.rand(len(train_df)) < 0.5,
    train_df["positive_example_1"],
    train_df["positive_example_2"]
)
train_df["negative_example"] = np.where(
    np.random.rand(len(train_df)) < 0.5,
    train_df["negative_example_1"],
    train_df["negative_example_2"]
)
flatten.append(train_df[["body", "rule", "subreddit", "rule_violation", "positive_example", "negative_example"]])

# Augment from test data
for vtype in ["positive", "negative"]:
    for i in [1, 2]:
        sub = test_df[["rule", "subreddit", "positive_example_1", "positive_example_2",
                      "negative_example_1", "negative_example_2"]].copy()

        if vtype == "positive":
            sub["body"] = sub[f"positive_example_{i}"]
            sub["positive_example"] = sub[f"positive_example_{3-i}"]
            sub["negative_example"] = np.where(np.random.rand(len(sub)) < 0.5,
                                              sub["negative_example_1"], sub["negative_example_2"])
            sub["rule_violation"] = 1
        else:
            sub["body"] = sub[f"negative_example_{i}"]
            sub["negative_example"] = sub[f"negative_example_{3-i}"]
            sub["positive_example"] = np.where(np.random.rand(len(sub)) < 0.5,
                                              sub["positive_example_1"], sub["positive_example_2"])
            sub["rule_violation"] = 0

        flatten.append(sub[["body", "rule", "subreddit", "rule_violation", "positive_example", "negative_example"]])

df = pd.concat(flatten, axis=0).drop_duplicates(ignore_index=True)
print(f"Total training samples: {len(df)}")

# Create dataset
df["prompt"] = df.apply(build_prompt, axis=1)
df["completion"] = df["rule_violation"].map({1: "Yes", 0: "No"})
dataset = Dataset.from_pandas(df[["prompt", "completion"]])

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Training config
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=is_torch_bf16_gpu_available(),
    fp16=not is_torch_bf16_gpu_available(),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=True,
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",
    completion_only_loss=True,
    packing=False,
    remove_unused_columns=False,
)

# Create trainer and train
print("\nTraining...")
trainer = SFTTrainer(
    model=BASE_MODEL_PATH,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

print(f"\n✓ Training complete! Model saved to {OUTPUT_DIR}")
print("Upload this folder as a Kaggle dataset for inference.")

