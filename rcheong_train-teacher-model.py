!pip install torch==2.3.0 torchvision torchaudio --quiet
!pip install bitsandbytes==0.43.3 peft==0.11.1 --quiet
!pip install transformers==4.41.2 accelerate==0.31.0 tokenizers==0.19.1 huggingface-hub==0.23.2 safetensors==0.4.2 --quiet


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    PreTrainedTokenizerBase
)
import os
import copy
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from datasets import load_from_disk
@dataclass
class Config:
    checkpoint: str = "microsoft/deberta-v3-small"
    max_length: int = 512
    n_epochs: int = 2
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    lr: float = 3e-5
    output_dir: str = "output_deberta"
    batch_size: int = 8

config = Config()
print(f"Using model: {config.checkpoint}")


from transformers import PreTrainedTokenizerBase
from collections import Counter

class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.total_batches = 0
        self.label_counter = Counter()

    def __call__(self, batch: dict) -> dict:
        self.total_batches += 1
        prompt = ["<prompt>: " + (p or "") for p in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + (a or "") for a in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + (b or "") for b in batch["response_b"]]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]

        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True, padding=True)

        labels = []
        for a_win, b_win, tie in zip(batch["winner_model_a"], batch["winner_model_b"], batch["winner_tie"]):
            if a_win == 1:
                labels.append(0)
            elif b_win == 1:
                labels.append(1)
            else:
                labels.append(2)
            self.label_counter[labels[-1]] += 1

        tokenized["labels"] = labels
        return tokenized



#Load the pretrained Hugging Face tokenizer (handles vocab + special tokens)
hf_tokenizer = AutoTokenizer.from_pretrained(
    config.checkpoint)

tokenizer = CustomTokenizer(hf_tokenizer, config.max_length)


tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
print(type(tokenizer))
custom_tokenizer = CustomTokenizer(tokenizer, config.max_length)
print(type(custom_tokenizer))

path = "/kaggle/input/lmsys-chatbot-arena/train.csv"
df = pd.read_csv(path)
ds = Dataset.from_pandas(df)

tokenized_ds = ds.map(custom_tokenizer, batched=True)
tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Optional: quick sanity check
print("Label counts:", custom_tokenizer.label_counter)








split_ds = tokenized_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split_ds["train"]
eval_ds = split_ds["test"]


!pip install evaluate --quiet


from evaluate import load
import numpy as np

# Load accuracy metric
accuracy = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}

#

model = AutoModelForSequenceClassification.from_pretrained(
    config.checkpoint,
    num_labels=3
).to("cuda")
print(type(tokenizer))
data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

training_args = TrainingArguments(
    output_dir="output_deberta_v2",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,              # more epochs
    learning_rate=2e-5,              # smaller LR
    warmup_ratio=0.1,                # small LR ramp-up
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,     # keeps best checkpoint
    metric_for_best_model="accuracy",
    report_to="none",
    logging_steps=50
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)



trainer.train()


save_dir = "/kaggle/working/teacher_model_v1"
os.makedirs(save_dir, exist_ok=True)

# Save model, tokenizer, and training args
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)



from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

# Put teacher in eval mode
model.eval()

data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
loader = DataLoader(eval_ds, batch_size=16, collate_fn=data_collator)

correct, total = 0, 0
for batch in tqdm(loader, desc="Evaluating teacher accuracy"):
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

acc = correct / total
print(f"\n✅ Teacher Accuracy on eval set: {acc:.4f}")


