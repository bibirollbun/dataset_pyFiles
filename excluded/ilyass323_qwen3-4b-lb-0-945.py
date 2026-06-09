import os
import torch
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from scipy.special import softmax

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test  = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False).drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

def format_input(row):
    x = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

test['text'] = test.apply(format_input, axis=1)

# --- Dataset ---
ds_test = Dataset.from_pandas(test[['row_id', 'text']])


model_name = "/kaggle/input/qwen3-4b-full-map-competition"
# --- Load label encoder mapping ---
with open(os.path.join(model_name, "label_encoder.json")) as f:
    label_data = json.load(f)
target_classes = label_data["classes"]
idx2label = {i: lbl for i, lbl in enumerate(target_classes)}
label2idx = {lbl: i for i, lbl in enumerate(target_classes)}

n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")



def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model.config.pad_token_id = tokenizer.pad_token_id

ds_test = ds_test.map(tokenize, batched=True)


test_args = TrainingArguments(
    output_dir="./",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=16,
    fp16=True,
    bf16=False,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=test_args,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

predictions = trainer.predict(ds_test)
probs = softmax(predictions.predictions, axis=1)

# --- Top-3 predictions ---
top3 = np.argsort(-probs, axis=1)[:, :3]
top3_labels = np.vectorize(idx2label.get)(top3)
joined_preds = [" ".join(row) for row in top3_labels]

# --- Save submission ---
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
print(sub.head())

