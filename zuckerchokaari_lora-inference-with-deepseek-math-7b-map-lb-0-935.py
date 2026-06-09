import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from peft import PeftModel


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


# Model and LoRA paths
base_model_path = "/kaggle/input/deepseek-math-7b-instruct/transformers/default/1"
lora_path = "/kaggle/input/map_lora_finetune_deepseek/transformers/default/1/trained_model"


# Load data
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# Prepare label encoder
le = LabelEncoder()
train['Misconception'] = train['Misconception'].fillna('NA')
train['target'] = train['Category'] + ':' + train['Misconception']
train['label'] = le.fit_transform(train['target'])

n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")


# Prepare correct answers mapping
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

print(f"Correct answers mapping created with {len(correct)} entries")


# Merge with test data
test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
test['is_correct'] = test['is_correct'].fillna(0)

# Format input text
test['text'] = test.apply(format_input, axis=1)

print("Sample formatted text:")
print(test['text'].iloc[0])


# Create dataset
ds_test = Dataset.from_pandas(test)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")


# Load base model
print("Loading base model...")
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_path,
    num_labels=n_classes,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print(f"Base model loaded with {n_classes} output classes")


# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()  # Merge LoRA weights with base model

# Update model config
model.config.pad_token_id = tokenizer.pad_token_id

print("LoRA adapter loaded and merged")


# Tokenize function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

# Tokenize test data
ds_test = ds_test.map(tokenize, batched=True)

print("Test data tokenized")


# Setup training arguments for inference
test_args = TrainingArguments(
    output_dir="./",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=16,
    bf16=False,
    fp16=True,
    report_to='none'
)

# Create trainer
trainer = Trainer(
    model=model,
    args=test_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

print("Trainer initialized")


# Run predictions
print("Running inference...")
predictions = trainer.predict(ds_test)

print(f"Predictions shape: {predictions.predictions.shape}")
print(f"Sample predictions: {predictions.predictions[0][:10]}")


# Get top 3 predictions
top3 = np.argsort(-predictions.predictions, axis=1)[:, :3]
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels_cat = decoded_labels.reshape(top3.shape)

print(f"Top 3 predictions shape: {top3_labels_cat.shape}")
print(f"Sample top 3 predictions: {top3_labels_cat[:3]}")


# Format predictions
joined_preds = []
for preds in top3_labels_cat:
    joined_preds.append(" ".join(preds))

print(f"Formatted {len(joined_preds)} predictions")
print(f"Sample formatted prediction: {joined_preds[0]}")


# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)

print(f"Submission saved with {len(sub)} predictions")
print(sub.head())

