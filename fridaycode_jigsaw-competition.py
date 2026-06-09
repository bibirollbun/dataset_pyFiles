%%time
import random
import os
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior if possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration parameters
SEED = 42
EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
MAX_LEN = 512

set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use GPUs if available

model_name = "/kaggle/input/jigsaw-competition-models/transformers/default/3"
tokenizer = AutoTokenizer.from_pretrained(model_name)


%%time
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

def make_prompt(row):
    return (f"[RULE]: {row['rule']}\n"
            f"[SUBREDDIT TOPIC]: {row['subreddit']}\n\n"
            f"[COMMENT]: {row['body']}\n\n"
            "[POSITIVE EXAMPLES]:\n"
            f"1. {row['positive_example_1']}\n"
            f"2. {row['positive_example_2']}\n\n"
            "[NEGATIVE EXAMPLES]:\n"
            f"1. {row['negative_example_1']}\n"
            f"2. {row['negative_example_2']}\n\n"
            "[QUESTION]: Does the comment violate the rule?\n[ANSWER]:")
    
# Construct prompt text for each row
train_df['text'] = train_df.apply(make_prompt, axis=1)
test_df['text']  = test_df.apply(make_prompt, axis=1)

# Use a stratified 80/20 split to hold out a validation set
train_subset, val_subset = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df["rule_violation"])
train_subset["label"] = train_subset["rule_violation"].astype(float)
val_subset["label"]   = val_subset["rule_violation"].astype(float)

# Prepare HuggingFace Datasets
features = ['text', 'label']
train_ds = Dataset.from_pandas(train_subset[features])
val_ds   = Dataset.from_pandas(val_subset[features])

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)
train_ds.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])
val_ds.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])


%%time
# Define metric computation (ROC AUC)
from sklearn.metrics import roc_auc_score
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits).float()).numpy().flatten()
    auc = roc_auc_score(labels, probs)
    return {"roc_auc": auc}

# Load pretrained model with a regression head (single output, sigmoid later)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

training_args = TrainingArguments(
    output_dir="./Jigsaw_Agile_v3",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_ratio=0.1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=100,
    report_to="none",
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# trainer.train()
results = trainer.evaluate()
print("Validation ROC-AUC:", results['eval_roc_auc'])
# trainer.save_model("Jigsaw_DeBERTa_v3")
# tokenizer.save_pretrained("Jigsaw_DeBERTa_v3")


# import shutil

# shutil.make_archive("a", 'zip', "/kaggle/working/Jigsaw_DeBERTa_v3")
# from IPython.display import FileLink
# FileLink(r'a.zip')


%%time

test_ds = Dataset.from_pandas(test_df[['text']])
test_ds = test_ds.map(tokenize, batched=True)
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

preds = trainer.predict(test_ds)
probs = torch.sigmoid(torch.tensor(preds.predictions).view(-1)).numpy()
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "rule_violation": probs
})
submission.to_csv("submission.csv", index=False)
submission.head()

