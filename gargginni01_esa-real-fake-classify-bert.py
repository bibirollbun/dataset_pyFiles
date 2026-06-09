import pandas as pd

# Read CSV into DataFrame
train_df_target = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')

# Convert entire DataFrame to dictionary
dict_records = train_df_target.to_dict(orient="records")  # list of row dicts
dict_list = train_df_target.to_dict(orient="list")        # dict of columns -> lists
dict_series = train_df_target.to_dict(orient="series")    # dict of columns -> Series
dict_split = train_df_target.to_dict(orient="split")      # {index, columns, data}
dict_index = train_df_target.to_dict(orient="index")      # {row_index -> {col: value}}

# Example: row-wise dictionary
# print(dict_records[:2])  # print first 2 rows as list of dictionaries
print(dict_index)


from pathlib import Path
import pandas as pd

# out_df = pd.DataFrame([], columns=["id", "real_text_id"])
# input_df = pd.DataFrame([], columns=["text", "labels"])

directory = Path("/kaggle/input/fake-or-real-the-impostor-hunt/data/train")

# Recursively list all files
all_files_recursive = list(directory.rglob("*.txt"))
print("Recursive:", all_files_recursive[1])
result_list = []
for file in all_files_recursive:
    file = str(file)
    split_file = file.split("/")
    file_name = split_file[-1]
    article_id = int(list(split_file[-2].split("_"))[-1])
    text_id = int(list(file_name.split("_"))[-1].split(".")[-2])
    
    with open(file, 'rb') as f:
        text = str(f.read())

    label = 1 if dict_index[article_id]['real_text_id'] == text_id else 0
    result_list.append([article_id, text, label])

train = pd.DataFrame(result_list, columns=["id", "text", "labels"])
print(train)


from pathlib import Path
import pandas as pd

# out_df = pd.DataFrame([], columns=["id", "real_text_id"])
# input_df = pd.DataFrame([], columns=["text", "labels"])

directory = Path("/kaggle/input/fake-or-real-the-impostor-hunt/data/test")

# Recursively list all files
all_files_recursive = list(directory.rglob("*.txt"))
print("Recursive:", all_files_recursive[1])
result_list = []
for file in all_files_recursive:
    file = str(file)
    split_file = file.split("/")
    file_name = split_file[-1]
    article_id = int(list(split_file[-2].split("_"))[-1])
    text_id = int(list(file_name.split("_"))[-1].split(".")[-2])
    
    with open(file, 'rb') as f:
        text = str(f.read())

    # label = 1 if dict_index[article_id]['real_text_id'] == text_id else 0
    result_list.append([f"{article_id}_{text_id}", text])

test = pd.DataFrame(result_list, columns=["id", "text"])
print(test)


%pip install evaluate


import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
# torch._dynamo.config.suppress_errors = True
# -----------------------
# 1. Load Data
# -----------------------
# data_dir = Path("/kaggle/input/playground-series-s5e8")
# train_path = data_dir / "train.csv"
# test_path = data_dir / "test.csv"
# sample_path = data_dir / "sample_submission.csv"
out_path = "./501_submission.csv"

# train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
# test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
# train = train_augmented
# train = train[:100]
# test = test[:25]

TARGET = "labels"
ID_COL = "id"

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# -----------------------
# 2. Prepare Text Feature
# -----------------------

# Train/Validation split
train_df, valid_df = train_test_split(train, test_size=0.2, stratify=train[TARGET], random_state=42)

# Convert to HuggingFace Dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[["text", TARGET]]),
    "validation": Dataset.from_pandas(valid_df[["text", TARGET]]),
    "test": Dataset.from_pandas(test[["text", ID_COL]])
})
# -----------------------
# 3. Tokenizer
# -----------------------
# MODEL = "distilbert-base-uncased"
# MODEL = "answerdotai/ModernBERT-base"
# model_names = [
#     "answerdotai/ModernBERT-base",
#         'distilbert-base-uncased',
        
#         'roberta-base',
#         # 'microsoft/deberta-v3-base',
#         'bert-large-uncased',
#         'roberta-large',
#         'microsoft/deberta-v3-large',
#         "huawei-noah/TinyBERT_General_4L_312D"
#     ]
# model_names = [
#         # "huawei-noah/TinyBERT_General_4L_312D",
#         # "microsoft/deberta-v3-base",
#         "answerdotai/ModernBERT-large",
#         # "answerdotai/ModernBERT-base",
#         # "microsoft/deberta-v3-large",
# ]

model_names = [
        # 'distilbert-base-uncased',
        # "answerdotai/ModernBERT-large",
        # 'roberta-base',
        # 'microsoft/deberta-v3-base',
        'bert-large-uncased',
        'roberta-large',
        # 'microsoft/deberta-v3-large'
    ]

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# Metrics
from evaluate import load
accuracy = load("accuracy")
roc_auc = load("roc_auc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average='binary')
    bal_acc = balanced_accuracy_score(labels, preds)
    acc = accuracy.compute(predictions=preds, references=labels)
    auc = roc_auc.compute(prediction_scores=logits[:,1], references=labels)
    return {"f1":f1, "balanced_accuracy": bal_acc, "accuracy": acc["accuracy"], "roc_auc": auc["roc_auc"]}

BATCH_SIZE = 256
trained_model_dict = {}
trained_tokenizer_dict = {}
trained_trainer_obj = {}

for MODEL in model_names:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.map(tokenize, batched=True)
    
    # Set format for PyTorch
    dataset['train'].set_format("torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=False)
    dataset['validation'].set_format("torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=False)
    
    
    # -----------------------
    # 4. Model
    # -----------------------
    num_labels = len(train[TARGET].unique())  # should be 2 for binary classification
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
    
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 
    # Move model to GPU
    model.to(device)
    model.eval()
    
    # -----------------------
    # 5. Training Setup
    # -----------------------
    training_args = TrainingArguments(
        output_dir="./results",
        
        # ✅ Evaluation & Saving
        eval_strategy="epoch",       # run eval every epoch
        save_strategy="epoch",             # save checkpoint every epoch
        load_best_model_at_end=True,       # keep best model based on metric
        metric_for_best_model="balanced_accuracy",        # optimize for F1
        save_total_limit=2,                # keep only last 2 checkpoints
        
        # ✅ Optimization
        learning_rate=2e-5,                # slightly higher LR may help with small data
        per_device_train_batch_size=8,     # smaller batch for stability
        per_device_eval_batch_size=16,
        num_train_epochs=15,                # reduce to avoid overfitting
        # weight_decay=0.05,                 # stronger regularization
        # warmup_ratio=0.1,                  # gradual LR increase
        gradient_accumulation_steps=2,     # effective batch size
        
        # ✅ Logging
        logging_dir="./logs",
        logging_strategy="epoch",
        report_to="none",                  # no external reporting
        
        # ✅ Reproducibility
        seed=42,
        dataloader_drop_last=True
    )
    
    
    # -----------------------
    # 6. Trainer
    # -----------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # -----------------------
    # 7. Train
    # -----------------------
    trainer.train()
    trained_model_dict[MODEL] = model
    trained_tokenizer_dict[MODEL] = tokenizer
    trained_trainer_obj[MODEL] = trainer

# -----------------------
# 8. Predictions on Test
# -----------------------
all_probs = []
weight_models = {}
test_data_model_to_pred_dict = {}
test_texts = test["text"].tolist()

num_models = len(trained_model_dict)
weights = 1 / num_models

for model_m in trained_model_dict.keys():
    test_data_model_to_pred_dict[model_m] = []
    weight_models[model_m] = weights

with torch.no_grad():
    for model_m in trained_model_dict.keys():
        for i in range(0, len(test_texts), BATCH_SIZE):
            batch_texts = test_texts[i:i+BATCH_SIZE]
            batch_encodings = trained_tokenizer_dict[model_m](
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Move batch tensors to GPU
            batch_encodings = {k: v.to(device) for k, v in batch_encodings.items()}
            outputs = trained_model_dict[model_m](**batch_encodings)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            # all_probs.extend(probs.cpu().numpy())  # move back to CPU for storage
            test_data_model_to_pred_dict[model_m].extend(probs.cpu().numpy())

# compute weighted sum across keys
all_probs = [sum(test_data_model_to_pred_dict[k][i] * weight_models[k] for k in test_data_model_to_pred_dict) for i in range(len(next(iter(test_data_model_to_pred_dict.values()))))]

# print(all_probs)

# Convert to numpy array
all_probs = np.array(all_probs)
        
# -----------------------
# 9. Save Submission
# -----------------------


submission_testing = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: all_probs   # threshold at 0.5
})
# print(submission_testing)
# Split id_label into id and label
submission_testing[['id', 'real_text_id']] = submission_testing['id'].str.split('_', expand=True)
submission_testing['labels'] = submission_testing['labels'].astype(float)

# Keep only the row with max score per id
df_max = submission_testing.loc[submission_testing.groupby('id')['labels'].idxmax()].reset_index(drop=True)

df_max = df_max.drop(columns=["labels"])

# print(df_max)
df_max.to_csv(out_path, index=False)
print(f"✅ Saved submission.csv : {out_path}")


import matplotlib.pyplot as plt

# Extract training & eval loss
train_loss = []
eval_loss = []
epochs = []
for model_name in trained_trainer_obj.keys():
    for log in trained_trainer_obj[model_name].state.log_history:
        if "loss" in log.keys():
            train_loss.append(log["loss"])
            epochs.append(log["epoch"])
        if "eval_loss" in log.keys():
            eval_loss.append(log["eval_loss"])
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[:len(train_loss)], train_loss, label="Training Loss")
    plt.plot(epochs[:len(eval_loss)], eval_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} : Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()





