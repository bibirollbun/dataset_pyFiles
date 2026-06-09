TRAINING_MODEL_PATH = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-large"  # your model path
TRAINING_MAX_LENGTH = 720
OUTPUT_DIR = '/kaggle/working/'  # your output path


!pip install seqeval evaluate -q


pip install -U datasets


import json
import argparse
from itertools import chain
from functools import partial

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
import evaluate
from datasets import Dataset, features
import numpy as np


import json
import random
import pandas as pd
ai_data = json.load(open('/kaggle/input/gendata/labeled_gen_dataset.json'))
orig_data = json.load(open('/kaggle/input/pii-detection-removal-from-educational-data/train.json'))

print(type(orig_data))
print(type(ai_data))
data = orig_data + ai_data

# Shuffle the combined data randomly
#random.shuffle(data)
print(len(data))



all_labels = sorted(list(set(chain(*[x["labels"] for x in orig_data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v:k for k,v in label2id.items()}

print(id2label)


target = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
    'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
    'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
]


def tokenize(example, tokenizer, label2id):
    text = []

    # these are at the character level
    labels = []
    targets = []

    for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):

        text.append(t)
        labels.extend([l]*len(t))
        
        if l in target:
            targets.append(1)
        else:
            targets.append(0)
        # if there is trailing whitespace
        if ws:
            text.append(" ")
            labels.append("O")

    tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation=True, max_length=TRAINING_MAX_LENGTH)
    
    target_num = sum(targets)
    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:

        # CLS token
        if start_idx == 0 and end_idx == 0: 
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length,
        "target_num": target_num,
        "group": 1 if target_num>0 else 0
    }


# Check for missing keys
required_keys = ["full_text", "document", "tokens", "trailing_whitespace", "labels"]
for item in data:
    for key in required_keys:
        if key not in item:
            print(f"Missing key '{key}' in item: {item}")
            break

# Check data types
for i, item in enumerate(data[:100]):  # Check first 100 items
    if not isinstance(item["full_text"], str):
        print(f"Item {i}: full_text is not string - {type(item['full_text'])}")
    if not isinstance(item["tokens"], list):
        print(f"Item {i}: tokens is not list - {type(item['tokens'])}")
    if not isinstance(item["labels"], (list, str, bytes)):
        print(f"Item {i}: labels is invalid type - {type(item['labels'])}")


from datasets import concatenate_datasets
tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [x["document"] for x in data], #problematic 
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    "provided_labels": [x["labels"] for x in data],
})



%%time
ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id}, num_proc=2)
ds = ds.class_encode_column("group")


x = ds[0]

for t,l in zip(x["tokens"], x["provided_labels"]):
    if l != "O":
        print((t,l))

print("*"*100)

for t, l in zip(tokenizer.convert_ids_to_tokens(x["input_ids"]), x["labels"]):
    if id2label[l] != "O":
        print((t,id2label[l]))


from seqeval.metrics import recall_score, precision_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5*5) * recall * precision / (5*5*precision + recall)
    
    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }
    return results


model = AutoModelForTokenClassification.from_pretrained(
    TRAINING_MODEL_PATH,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)


FREEZE_EMBEDDINGS = False
FREEZE_LAYERS = 6

if FREEZE_EMBEDDINGS:
    print('Freezing embeddings.')
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False
        
if FREEZE_LAYERS>0:
    print(f'Freezing {FREEZE_LAYERS} layers.')
    for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False


# may want to try to balance classes in splits
final_ds = ds.train_test_split(test_size=0.2, seed=42) # cannot use stratify_by_column='group'
final_ds


args = TrainingArguments(
    output_dir=OUTPUT_DIR, 
    fp16=True,
    #warmup_steps=100,
    learning_rate=1e-5,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    report_to="none",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    lr_scheduler_type='cosine',
    metric_for_best_model="f1",
    greater_is_better=True,
    weight_decay=0.01
)
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=final_ds["train"], 
    eval_dataset=final_ds["test"], 
    data_collator=collator, 
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, all_labels=all_labels),
)


%%time
trainer.train()


trainer.save_model(OUTPUT_DIR)
torch.cuda.empty_cache()


!pip install --upgrade ipywidgets==7.6.0  # Explicitly install v7.6.0 (matches frontend)
!jupyter nbextension enable --py widgetsnbextension
from huggingface_hub import notebook_login




notebook_login()


tokenizer.push_to_hub("Svetlana-isch/pii_deberta_with_gen")


import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

# Set your HF token & username as environment variables
os.environ["HF_TOKEN"] = user_secrets.get_secret("tkn1")
# Replace with your username)
os.environ["HF_USERNAME"] = "Svetlana-isch"


model.push_to_hub("Svetlana-isch/gen_model")
tokenizer.push_to_hub("Svetlana-isch/gen_model")

