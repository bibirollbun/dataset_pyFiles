# CONFIG
MODEL_PATHS = [
    "/kaggle/input/pii-1850-mixtral-fold2-12deberta_large/pytorch/default/1/deberta_large/checkpoint-1175",
    "/kaggle/input/pii-1850-mixtral-fold2-12deberta_large/pytorch/default/2/deberta_large/checkpoint-1943",
    "/kaggle/input/pii-1850-mixtral-fold2-12deberta_large/pytorch/default/3/deberta_large/checkpoint-4706",
]

THRESHOLDS = [0.98, 
              0.99, 
              0.998,
             ]

LABEL_CONFIG_PATH = "/kaggle/input/pii-1850-mixtral-fold2-12deberta_large/pytorch/default/1/deberta_large/config.json"
TEST_DATA_PATH = "/kaggle/input/pii-detection-removal-from-educational-data/test.json"
OUTPUT_SUBMISSION = "submission.csv"

# IMPORTS
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)
from datasets import Dataset
from itertools import chain



# TOKENIZER & TOKENIZATION FUNCTIONS

def tokenize(example, tokenizer, max_length=512, stride=128):
    encoded = tokenizer(
        example['tokens'],
        is_split_into_words=True,
        return_overflowing_tokens=True,
        stride=stride,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    encoded['wids'] = []
    for i in range(len(encoded['overflow_to_sample_mapping'])):
        word_ids = encoded.word_ids(i)
        encoded['wids'].append([w if w is not None else -1 for w in word_ids])

    result = []
    for i in range(len(encoded["input_ids"])):
        result.append({
            "tokens": example['tokens'],
            "input_ids": encoded["input_ids"][i],
            "token_type_ids": encoded["token_type_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "document": example["document"],
            "wids": encoded["wids"][i]
        })
    return result

def prepare_test_dataset(tokenizer, test_path):
    data = json.load(open(test_path))
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [x["document"] for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    })

    result = []
    for i in range(len(ds)):
        result.extend(tokenize(ds[i], tokenizer))
    
    return Dataset.from_dict({
        "tokens": [x["tokens"] for x in result],
        "input_ids": [x["input_ids"] for x in result],
        "token_type_ids": [x["token_type_ids"] for x in result],
        "attention_mask": [x["attention_mask"] for x in result],
        "document": [x["document"] for x in result],
        "wids": [x["wids"] for x in result]
    })



# LOAD TOKENIZER AND PREPARE DATASET
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[0])
res_ds = prepare_test_dataset(tokenizer, TEST_DATA_PATH)

# LOAD LABEL MAP
config = json.load(open(LABEL_CONFIG_PATH))
id2label = config["id2label"]

# PREDICT WITH EACH MODEL
all_preds = []

for path, threshold in zip(MODEL_PATHS, THRESHOLDS):
    model = AutoModelForTokenClassification.from_pretrained(path)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
    
    args = TrainingArguments(
        ".", 
        per_device_eval_batch_size=1, 
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=collator
    )
    
    predictions = trainer.predict(res_ds).predictions
    pred_softmax = softmax(predictions, axis=-1)

    preds = pred_softmax.argmax(-1)
    preds_without_O = pred_softmax[:, :, :12].argmax(-1)
    O_preds = pred_softmax[:, :, 12]

    local_preds_final = np.where(O_preds < threshold, preds_without_O, preds)
    all_preds.append(local_preds_final)



# VOTING FUNCTION
def smallest_freq_elem(array):
    freq_count = np.bincount(array)
    idx_of_max_count = np.where(freq_count == np.max(freq_count))[0][0]
    return np.min(np.where(freq_count == freq_count[idx_of_max_count])[0])

# APPLY VOTE
preds_final = np.apply_along_axis(smallest_freq_elem, axis=0, arr=all_preds)

# BUILD SUBMISSION
triplets = set()
document, token, label, token_str = [], [], [], []

for i in range(len(res_ds)):
    row = res_ds[i]
    for j in range(len(row['input_ids'])):
        if row['wids'][j] != -1:
            label_pred = id2label[str(preds_final[i][j])]
            token_id = row['wids'][j]
            token_value = row['tokens'][token_id]

            if label_pred != "O":
                key = (row['document'], token_id, token_value)
                if key not in triplets:
                    triplets.add(key)
                    document.append(row['document'])
                    token.append(token_id)
                    label.append(label_pred)
                    token_str.append(token_value)

df = pd.DataFrame({
    "document": document,
    "token": token,
    "label": label,
    "token_str": token_str,
})
df["row_id"] = list(range(len(df)))
df[["row_id", "document", "token", "label"]].to_csv(OUTPUT_SUBMISSION, index=False)

df.head(10)  # quick check


