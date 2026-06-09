import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from peft import PeftModel

# --- Load model + tokenizer ---
model_name = "/kaggle/input/llama-3-1-8b-fix/Llama-3.1-8b-Fix/Lora"
base_model = "/kaggle/input/llama-3.1/transformers/8b/2"
MAX_LEN = 190

tokenizer = AutoTokenizer.from_pretrained(model_name)

with open(os.path.join(model_name, "label_encoder.json")) as f:
    label_data = json.load(f)
target_classes = label_data["classes"]
n_classes = len(target_classes)
idx2label = {i: lbl for i, lbl in enumerate(target_classes)}

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=n_classes,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# --- Preprocess test data ---
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
train.Misconception = train.Misconception.fillna("NA")
train["target"] = train.Category + ":" + train.Misconception

idx = train.apply(lambda row: row.Category.split("_")[0], axis=1) == "True"
correct = train.loc[idx].copy()
correct["c"] = correct.groupby(["QuestionId", "MC_Answer"]).MC_Answer.transform("count")
correct = correct.sort_values("c", ascending=False)
correct = correct.drop_duplicates(["QuestionId"])
correct = correct[["QuestionId", "MC_Answer"]]
correct["is_correct"] = 1

test = test.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
test.is_correct = test.is_correct.fillna(0)

def format_input(row):
    x = "Yes" if row["is_correct"] else "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

test["text"] = test.apply(format_input, axis=1)

# Dataset + tokenization
ds_test = Dataset.from_pandas(test[["text"]])
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)
ds_test = ds_test.map(tokenize, batched=True)

ds_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

# --- Inference Test Data---
dataloader = DataLoader(ds_test, batch_size=12)
all_logits = []

model.eval()
with torch.inference_mode():
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits.detach().cpu()
        all_logits.append(logits)

logits = torch.cat(all_logits, dim=0).float()
probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
top3 = np.argsort(-probs, axis=1)[:, :3]

top3_labels = [[idx2label[i] for i in row] for row in top3]
joined_preds = [" ".join(row) for row in top3_labels]

# --- Save submission ---
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()
print("✅ Saved submission.csv")

