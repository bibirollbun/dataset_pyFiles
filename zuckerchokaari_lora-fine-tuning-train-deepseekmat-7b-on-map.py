!pip install bitsandbytes


import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType


# Configuration
MODEL_NAME = "/kaggle/input/qwen2.5-math-7b-instruct/transformers/default/1"
MAX_LENGTH = 256
IS_DEBUG = True  # Debug mode with a small dataset
TRAIN_RATIO = 0.9  # Training data ratio for simple split
EPOCH = 3  # Training epochs
LR = 5e-5  # Learning rate
TRAIN_BS = 8  # Training batch size
GRAD_ACC_NUM = 4  # Gradient accumulation steps
EVAL_BS = 4  # Evaluation batch size
SEED = 42  # Random seed for reproducibility

EXP_NAME = "MAP_lora_finetune"
OUTPUT_DIR = f"./output/{EXP_NAME}/"
MODEL_OUTPUT_PATH = f"{OUTPUT_DIR}trained_model"


# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.001,
    task_type=TaskType.SEQ_CLS,
    bias='none',
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
)


def format_input(row):
    x = "Correctness: This is Correct answer."
    if not row['is_correct']:
        x = "Correctness: This is Incorrect answer."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )


def prepare_data():
    # Load data
    train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

    # Prepare label encoder
    le = LabelEncoder()
    train['Misconception'] = train['Misconception'].fillna('NA')
    train['target'] = train['Category'] + ':' + train['Misconception']
    train['label'] = le.fit_transform(train['target'])

    n_classes = len(le.classes_)
    print(f"Train shape: {train.shape} with {n_classes} target classes")

    # Identify correct answers
    idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
    correct = train.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])
    correct = correct[['QuestionId', 'MC_Answer']]
    correct['is_correct'] = 1

    train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
    train['is_correct'] = train['is_correct'].fillna(0)

    # Format input text
    train['text'] = train.apply(format_input, axis=1)

    return train, le, n_classes


def compute_map3(eval_pred):
    """Compute MAP@3 metric for evaluation"""
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]

    map3 = 0.0
    for i in range(len(labels)):
        if top3[i, 0] == labels[i]:
            map3 += 1.0
        elif top3[i, 1] == labels[i]:
            map3 += 1.0 / 2
        elif top3[i, 2] == labels[i]:
            map3 += 1.0 / 3
    map3 /= len(labels)

    acc = accuracy_score(labels, np.argmax(probs, axis=1))
    return {"accuracy": acc, "map@3": map3}


def tokenize_function(tokenizer):
    def tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None
        )
        tokenized["labels"] = batch["label"]
        return tokenized
    return tokenize


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory created at: {OUTPUT_DIR}")

# Prepare data
train_df, le, n_classes = prepare_data()


if IS_DEBUG:
    print("Running in DEBUG mode with 50 samples.")
    train_df = train_df.sample(50, random_state=SEED).reset_index(drop=True)


# Split data (9:1 train/validation split)
print(f"Using simple {TRAIN_RATIO:.0%}:{1-TRAIN_RATIO:.0%} train/validation split.")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].values,
    train_df['label'].values,
    test_size=(1-TRAIN_RATIO),
    random_state=SEED
)
print(f"Training samples: {len(train_texts)} ({len(train_texts)/len(train_df):.1%})")
print(f"Validation samples: {len(val_texts)} ({len(val_texts)/len(train_df):.1%})")


# Create datasets
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})

val_dataset = Dataset.from_dict({
    'text': val_texts,
    'label': val_labels
})


# Load tokenizer and model
print(f"Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=n_classes,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto"
)

# Configure model for pad token
model.config.pad_token_id = tokenizer.pad_token_id


# Apply LoRA
model = get_peft_model(model, lora_config)
print("LoRA configured. Trainable parameters:")
model.print_trainable_parameters()


# Tokenize datasets
tokenize_fn = tokenize_function(tokenizer)
train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


# Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_PATH,
    logging_steps=30,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=6,
    num_train_epochs=EPOCH,
    optim="paged_adamw_8bit",
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    learning_rate=LR,
    weight_decay=0.01,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACC_NUM,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=False,
    seed=SEED,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    report_to=[],
)


# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)


# Train model
print("Starting training...")
trainer.train()


# Save model and tokenizer
print(f"Saving final model to {MODEL_OUTPUT_PATH}")
trainer.save_model(MODEL_OUTPUT_PATH)
tokenizer.save_pretrained(MODEL_OUTPUT_PATH)


# Save label encoder
import pickle
with open(f'{OUTPUT_DIR}/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Script finished successfully!")

