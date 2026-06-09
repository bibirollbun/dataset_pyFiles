TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH = 512
TRAINING_STRIDE = 256
OUTPUT_DIR = "deberta_large"

import json
import argparse
from itertools import chain
from functools import partial

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from datasets import Dataset
import numpy as np
import os
from collections import defaultdict
import pandas as pd
from typing import Dict
import gc

print("GPU available:", torch.cuda.is_available())



gc.collect()


data = json.load(open("/kaggle/input/pii-detection-removal-from-educational-data/train.json"))

# Assign fold
for i in range(len(data)):
    data[i]["fold"] = data[i]["document"] % 4

# Split train/val
train_data = [d for d in data if d["fold"] != 2]
val_data = [d for d in data if d["fold"] == 2]

# Add synthetic data (except those with I-USERNAME labels)
#add_mixtral_data = json.load(open("/kaggle/input/mixtral-original-prompt/Fake_data_1850_218.json"))
#add_mixtral_data = json.load(open("/kaggle/input/extra-data1/mpware_mixtral8x7b_v1.1.json"))
add_mixtral_data = json.load(open("/kaggle/input/pii-dd-mistral-generated/mixtral-8x7b-v1.json"))

print("Add mixtral data:", len(add_mixtral_data))

for d in add_mixtral_data:
    if "I-USERNAME" not in d["labels"]:
        train_data.append(d)



#import random

# Réduction (80%) après enrichissement
#random.seed(42)
#train_data = random.sample(train_data, int(0.8 * len(train_data)))
#val_data = random.sample(val_data, int(0.8 * len(val_data)))

#print(f"Réduction terminée : {len(train_data)} train, {len(val_data)} val")


all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

train_ds = Dataset.from_dict({
    "document": [str(x["document"]) for x in train_data],
    "tokens": [x["tokens"] for x in train_data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in train_data],
    "provided_labels": [x["labels"] for x in train_data],
})

val_ds = Dataset.from_dict({
    "document": [str(x["document"]) for x in val_data],
    "tokens": [x["tokens"] for x in val_data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in val_data],
    "provided_labels": [x["labels"] for x in val_data],
})



def get_labels(word_ids, word_labels):
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(label2id[word_labels[word_idx]])
    return label_ids

def tokenize(example, tokenizer, label2id, max_length=TRAINING_MAX_LENGTH, stride=TRAINING_STRIDE):
    encoded = tokenizer(
        example['tokens'],
        is_split_into_words=True,
        return_overflowing_tokens=True,
        stride=stride,
        max_length=max_length,
        padding="max_length",
        truncation=True
    )

    encoded['labels'], encoded['wids'] = [], []
    for i in range(len(encoded['overflow_to_sample_mapping'])):
        word_ids = encoded.word_ids(i)
        label_ids = get_labels(word_ids, example['provided_labels'])
        encoded['labels'].append(label_ids)
        encoded['wids'].append([w if w is not None else -1 for w in word_ids])

    res = []
    for i in range(len(encoded["input_ids"])):
        wids = encoded["wids"][i]
        labels = encoded["labels"][i]
        tokens, provided_labels, prev = [], [], -1
        for w in wids:
            if w != -1 and w != prev:
                tokens.append(example['tokens'][w])
                provided_labels.append(example['provided_labels'][w])
                prev = w
        res.append({
            "tokens": example['tokens'],
            "provided_labels": provided_labels,
            "input_ids": encoded["input_ids"][i],
            "token_type_ids": encoded["token_type_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "document": example["document"],
            "wids": wids,
            "labels": labels,
            "length": len(encoded["input_ids"][i]),
        })
    return res



train_result, val_result = [], []

for i in range(len(train_ds)):
    train_result.extend(tokenize(train_ds[i], tokenizer, label2id))

for i in range(len(val_ds)):
    val_result.extend(tokenize(val_ds[i], tokenizer, label2id))

train_res_ds = Dataset.from_dict({k: [x[k] for x in train_result] for k in train_result[0]})
val_res_ds = Dataset.from_dict({k: [x[k] for x in val_result] for k in val_result[0]})



df = pd.DataFrame(val_ds)
df['labels'] = df['provided_labels']
ref_df = df[['document', 'tokens', 'labels']].explode(['tokens', 'labels']).reset_index(drop=True).rename(columns={'tokens': 'token', 'labels': 'label'})
ref_df['token'] = ref_df.groupby('document').cumcount()

reference_df = ref_df[ref_df['label'] != 'O'].copy().reset_index().rename(columns={'index': 'row_id'})
reference_df = reference_df[['row_id', 'document', 'token', 'label']]



from typing import Dict, Tuple, List, Any

class PRFScore:
    def __init__(self, tp=0, fp=0, fn=0):
        self.tp, self.fp, self.fn = tp, fp, fn

    def __iadd__(self, other: 'PRFScore'):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    def f5(self) -> float:
        p, r = self.precision(), self.recall()
        beta = 5
        return (1 + beta**2) * p * r / (beta**2 * p + r + 1e-100)

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision(), "r": self.recall(), "f5": self.f5()}


def parse_predictions(predictions: np.ndarray, id2label: Dict[int, str], res_ds: List[Dict[str, Any]], threshold: float) -> pd.DataFrame:
    softmax_preds = np.exp(predictions) / np.sum(np.exp(predictions), axis=2, keepdims=True)
    preds = predictions.argmax(-1)
    preds_no_O = softmax_preds[:, :, :12].argmax(-1)
    final_preds = np.where(softmax_preds[:, :, 12] < threshold, preds_no_O, preds)

    triplets = set()
    rows = []

    for i, sample in enumerate(res_ds):
        for j in range(len(sample['input_ids'])):
            wid = sample['wids'][j]
            if wid == -1:
                continue
            label = id2label[final_preds[i][j]]
            token_str = sample['tokens'][wid]
            triplet = (sample['document'], wid, token_str)
            if label != "O" and triplet not in triplets:
                rows.append({
                    "eval_row": i,
                    "document": sample['document'],
                    "token": wid,
                    "label": label,
                    "token_str": token_str,
                })
                triplets.add(triplet)

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    df["row_id"] = range(len(df))
    return df


def compute_metrics(
    p: Tuple[np.ndarray, np.ndarray],
    id2label: Dict[int, str],
    valid_ds: List[Dict[str, Any]],
    valid_df: pd.DataFrame,
    threshold: float = 0.95
) -> Dict[str, float]:
    logits, _ = p
    pred_df = parse_predictions(logits, id2label, valid_ds, threshold)

    reference_set = {(r.document, r.token, r.label) for r in valid_df.itertuples()}
    prediction_set = {(r.document, r.token, r.label) for r in pred_df.itertuples()}

    score_per_type = defaultdict(PRFScore)

    for pred in prediction_set:
        label = pred[2]
        label_type = label[2:] if label != 'O' else 'O'
        if pred in reference_set:
            score_per_type[label_type].tp += 1
            reference_set.remove(pred)
        else:
            score_per_type[label_type].fp += 1

    for doc, tok, label in reference_set:
        label_type = label[2:] if label != 'O' else 'O'
        score_per_type[label_type].fn += 1

    total_score = PRFScore()
    for score in score_per_type.values():
        total_score += score

    results = {
        "ents_p": total_score.precision(),
        "ents_r": total_score.recall(),
        "f5": total_score.f5(),
        "ents_per_type": {
            label: score.to_dict() for label, score in score_per_type.items() if label != 'O'
        }
    }

    # Flatten nested metrics
    flat_results = {}
    for key, val in results.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat_results[f"{key}_{subkey}"] = subval
        else:
            flat_results[key] = val

    return flat_results


from transformers import TrainerCallback

class PrintMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"\nStep {state.global_step} - Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(logs)


model = AutoModelForTokenClassification.from_pretrained(
    TRAINING_MODEL_PATH,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=True,
    learning_rate=2e-5,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    report_to="none",
    #eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=20,
    lr_scheduler_type='cosine',
    metric_for_best_model="f5",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    auto_find_batch_size=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_res_ds,
    eval_dataset=val_res_ds,
    data_collator=collator,
    tokenizer=tokenizer,
    #callbacks=[PrintMetricsCallback()],
    compute_metrics=partial(compute_metrics, id2label=id2label, valid_ds=val_res_ds, valid_df=reference_df)
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)



# Inference on validation set
preds = trainer.predict(val_res_ds)
preds


# Evaluate across thresholds
print('Computing final metrics...')
final_metrics = {
    f'final_f5_at_{threshold}': compute_metrics(
        (preds.predictions, None), id2label, val_res_ds, reference_df, threshold=threshold
    )['f5']
    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998]
}
print(final_metrics)



best_thresh = max(final_metrics, key=final_metrics.get)
print("Best threshold and F5-score:")
print(best_thresh, final_metrics[best_thresh])


