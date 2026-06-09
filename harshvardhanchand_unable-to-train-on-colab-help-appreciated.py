!pip -q install condacolab
import condacolab
condacolab.install()


from google.colab import drive
drive.mount('/content/drive')



!uv pip install --system --no-index --find-links='/content/drive/MyDrive/kaggle/working/whls' \
    'trl==0.21.0' \
    'optimum==1.27.0' \
    'auto-gptq==0.7.1' \
    'bitsandbytes==0.46.1' \
    'deepspeed==0.17.4' \
    'logits-processor-zoo==0.2.1' \
    'vllm==0.10.0' \
    'clean-text' \
    'peft' \
    'accelerate' \
    'datasets'


import sys
import os

print("ğŸ�� Python executable running this script:")
print(sys.executable)

print("\nğŸ“¦ Python path:")
print(sys.path)


import os
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import Dataset
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.utils import is_torch_bf16_gpu_available
from peft import LoraConfig, TaskType, get_peft_model
from trl import DataCollatorForCompletionOnlyLM



# Main configuration parameters
WANDB = False  # Enable/disable Weights & Biases logging
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"  # Pre-trained model to fine-tune
IS_DEBUG = True  # Debug mode with small dataset
N_FOLDS = 5  # Number of cross-validation folds
EPOCH = 1  # Training epochs
LR = 1e-4  # Learning rate
TRAIN_BS = 8 #8  # Training batch size
GRAD_ACC_NUM = 1 #1  # Gradient accumulation steps
EVAL_BS = 8  # Evaluation batch size
FOLD = 0  # Current fold to train
SEED = 42  # Random seed for reproducibility

# Derive experiment name and paths
EXP_ID = "jigsaw-lora-finetune-baseline"
if IS_DEBUG:
    EXP_ID += "_debug"
EXP_NAME = EXP_ID + f"_fold{FOLD}"
COMPETITION_NAME = "jigsaw-kaggle"
OUTPUT_DIR = f"/content/drive/MyDrive/kaggle/output/{EXP_NAME}/" # f"/kaggle/output/{EXP_NAME}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_OUTPUT_PATH = f"{OUTPUT_DIR}/trained_model"


# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/synthetic_rebalanced.csv")
if IS_DEBUG:
    # Use a small subset for debugging
    df = df.sample(50, random_state=SEED).reset_index(drop=True)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"  # Important for causal language models

# Define system prompt for the classification task
SYS_PROMPT = """
    You are an expert moderator bot for Reddit. Your task is to accurately determine whether a given Reddit comment violates a specific subreddit rule. You will be provided with the subreddit name, the rule, and several examples of both rule violations and non-violations. Analyze the comment carefully in the context of the provided rule and examples. Your final response MUST be either 'Yes' or 'No'. A 'Yes' indicates a clear violation, and a 'No' indicates no violation.
    """

prompts = []
for i, row in df.iterrows():
    text = f"""
r/{row.subreddit}
Rule: {row.rule}

1) {row.positive_example_1}
Violation: Yes

2) {row.negative_example_1}
Violation: No

3) {row.negative_example_2}
Violation: No

4) {row.positive_example_2}
Violation: Yes

5) {row.body}
"""

    # Format as a chat conversation using the model's template
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    ) + "Answer:"
    prompts.append(prompt)

# Add the formatted prompts to the dataframe
df["text"] = prompts
df["label"] = df["rule_violation"].apply(lambda x: "Yes" if x == 1 else "No")

# Append the label to create completion-based training examples
df["text"] = df["text"] + df["label"]

# Tokenize the examples
def preprocess_row(row, tokenizer) -> dict:
    item = tokenizer(row["text"], add_special_tokens=False, truncation=False)
    return item

def preprocess_df(df, tokenizer) -> pd.DataFrame:
    items = []
    for _, row in df.iterrows():
        items.append(preprocess_row(row, tokenizer))
    df = pd.concat([
        df,
        pd.DataFrame(items)
    ], axis=1)
    return df

df = preprocess_df(df, tokenizer)


# Create a PyTorch dataset class
class ClassifyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> dict:
        row = self.df.iloc[index]

        inputs = {
            "input_ids": row["input_ids"],
        }
        return inputs

# Data collator for completion-only learning
data_collator = DataCollatorForCompletionOnlyLM("Answer:", tokenizer=tokenizer)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # device_map="auto",  # Automatically distribute model across available GPUs
)

# Configure LoRA parameters
lora_config = LoraConfig(
    r=32,  # Rank of the update matrices
    lora_alpha=32,  # Alpha parameter for LoRA scaling
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    task_type=TaskType.CAUSAL_LM,
    bias='none',  # Don't train bias terms
    # Target the attention and MLP modules of the transformer
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Show what percentage of parameters will be trained

# Initialize Weights & Biases for experiment tracking
if WANDB:
    wandb.login()
    wandb.init(project=COMPETITION_NAME, name=EXP_NAME)
    REPORT_TO = "wandb"
else:
    REPORT_TO = "none"


# Split data into train and validation sets
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    if fold == FOLD:
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        break

# Save the split data
df_train.to_pickle(f"{OUTPUT_DIR}/train.pkl")
df_val.to_pickle(f"{OUTPUT_DIR}/val.pkl")


# Set up training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_PATH,
    logging_steps=10,  # Log metrics every 10 steps
    logging_strategy="steps",
    eval_strategy="no",  # No evaluation during training
    save_strategy="steps",
    save_steps=0.1,  # Save checkpoint after 10% of training steps
    save_total_limit=10,  # Keep only the 10 most recent checkpoints
    num_train_epochs=EPOCH,
    optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
    lr_scheduler_type="linear",
    warmup_ratio=0.1,  # Warm up learning rate over 10% of steps
    learning_rate=LR,
    weight_decay=0.01,

    # Use BF16 if available, otherwise FP16
    bf16=is_torch_bf16_gpu_available(),
    fp16=not is_torch_bf16_gpu_available(),

    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACC_NUM,
    gradient_checkpointing=True,  # Save memory with gradient checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=False,
    report_to=REPORT_TO,
    seed=42,
    remove_unused_columns=False,  # Keep all columns in the dataset
)

# Initialize trainer
trainer = Trainer(
    model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=ClassifyDataset(df_train),
    eval_dataset=ClassifyDataset(df_val),
    data_collator=data_collator,
)

# Start training
trainer_output = trainer.train()

# Save the final model
trainer.save_model(MODEL_OUTPUT_PATH)

