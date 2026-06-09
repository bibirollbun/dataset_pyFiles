model_names = [
    "distilbert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
    "bert-large-uncased",
    "roberta-large",
    "microsoft/deberta-v3-large"
]


import pandas as pd
import os

train_dir = "./data/train"

ref_df = pd.read_csv("./data/train.csv")
train_df = pd.DataFrame(columns=["text", "labels"])

for _, row in ref_df.iterrows():
    id = row["id"]
    real_text_id = row["real_text_id"]

    file_prefix = f"article_{id:04d}"

    file_path_dir = os.path.join(train_dir, file_prefix)

    file_1_path = os.path.join(file_path_dir, "file_1.txt")
    file_2_path = os.path.join(file_path_dir, "file_2.txt")

    with open(file_1_path, "r", encoding="utf-8") as f:
        file_1_text = f.read().strip()
    with open(file_2_path, "r", encoding="utf-8") as f:
        file_2_text = f.read().strip()

    if real_text_id == 1:
        train_df = pd.concat(
            [train_df, pd.DataFrame({"text": [file_1_text], "labels": [0]})],
            ignore_index=True,
        )
        train_df = pd.concat(
            [train_df, pd.DataFrame({"text": [file_2_text], "labels": [1]})],
            ignore_index=True,
        )
    else:
        train_df = pd.concat(
            [train_df, pd.DataFrame({"text": [file_1_text], "labels": [1]})],
            ignore_index=True,
        )
        train_df = pd.concat(
            [train_df, pd.DataFrame({"text": [file_2_text], "labels": [0]})],
            ignore_index=True,
        )


from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import os
import pandas as pd

def prepare_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=1).detach().cpu().numpy()
    preds = np.argmax(probs, axis=1)

    probs_class1 = probs[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs_class1)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "aucroc": auc,
    }

def run_test(idx,model,tokenizer):

    test_df = pd.DataFrame(columns=["id", "real_text_id"])
    test_dir = "./data/test"

    for i in range(len(os.listdir(test_dir))):
        file_prefix = f"article_{i:04d}"
        file_path_dir = os.path.join(test_dir, file_prefix)

        file_1_path = os.path.join(file_path_dir, "file_1.txt")
        file_2_path = os.path.join(file_path_dir, "file_2.txt")

        with open(file_1_path, "r", encoding="utf-8") as f:
            file_1_text = f.read().strip()
        with open(file_2_path, "r", encoding="utf-8") as f:
            file_2_text = f.read().strip()

        inputs_1 = tokenizer(
            file_1_text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to("cuda")
        inputs_2 = tokenizer(
            file_2_text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to("cuda")

        inputs_1.pop("token_type_ids", None)
        inputs_2.pop("token_type_ids", None)

        with torch.no_grad():
            outputs_1 = model(**inputs_1)
            outputs_2 = model(**inputs_2)

        logits_1 = outputs_1["logits"]
        logits_2 = outputs_2["logits"]

        probs_1 = F.softmax(logits_1, dim=1).detach().cpu().numpy()
        probs_2 = F.softmax(logits_2, dim=1).detach().cpu().numpy()

        human_prob_file1 = probs_1[0][0]
        human_prob_file2 = probs_2[0][0]

        real_text_id = 1 if human_prob_file1 > human_prob_file2 else 2

        test_df = pd.concat(
            [
                test_df,
                pd.DataFrame({"id": [i], "human_prob_file1": [human_prob_file1], "human_prob_file2": [human_prob_file2], "real_text_id": [real_text_id]}),
            ],
            ignore_index=True,
        )

        test_df.to_csv(f"submission_{idx}.csv")

def train_model_and_test(idx,model_name):
    tokenizer, model = prepare_model_and_tokenizer(model_name)
    dataset = Dataset.from_pandas(train_df)
    data = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = data["train"]
    val_dataset = data["test"]

    def preprocess(batch):
        return tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=512,
        )

    train_dataset = train_dataset.map(preprocess, batched=True)
    val_dataset = val_dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=f"./{model_name}-kaggle",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        eval_steps=None,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=3e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    run_test(idx,model,tokenizer)


for idx,model_name in enumerate(model_names):
    train_model_and_test(idx,model_name)


import pandas as pd
from collections import Counter
import glob

csv_files = sorted(glob.glob("submission_*.csv"))

dfs = [pd.read_csv(file) for file in csv_files]

final_df = pd.DataFrame()
final_df["id"] = dfs[0]["id"]

def majority_vote(values):
    return Counter(values).most_common(1)[0][0]


predictions = pd.concat([df["real_text_id"] for df in dfs], axis=1)

final_df["real_text_id"] = predictions.apply(majority_vote, axis=1)

final_df.to_csv("final_submission.csv", index=False)

print("Voting complete. Saved to final_submission.csv")


csv_files = sorted(glob.glob("submission_*.csv"))
dfs = [pd.read_csv(file) for file in csv_files]

final_df = pd.DataFrame()
final_df["id"] = dfs[0]["id"]

prob_file1 = pd.concat([df["human_prob_file1"] for df in dfs], axis=1).mean(axis=1)
prob_file2 = pd.concat([df["human_prob_file2"] for df in dfs], axis=1).mean(axis=1)

final_df["human_prob_file1_avg"] = prob_file1
final_df["human_prob_file2_avg"] = prob_file2
final_df["real_text_id"] = (prob_file1 > prob_file2).astype(int) + 1  

final_df[["id", "real_text_id"]].to_csv("ensemble_submission.csv", index=False)

print("Ensemble complete! Final file saved as ensemble_submission.csv")

# worse than voting

