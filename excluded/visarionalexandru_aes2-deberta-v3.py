import pandas as pd

# load dataset

df = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv")


# replace non-breaking spaces with spaces

def replace_xa0s(full_text):
    return full_text.replace("\xa0", " ")

df["full_text"] = df["full_text"].apply(replace_xa0s)

# transform column

df.rename(columns={"score": "labels"}, inplace=True)
df["labels"] = df["labels"].astype(float)


from sklearn.model_selection import train_test_split

# split dataset

df_train, df_val = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    stratify=df["labels"]
)


LOAD_DIR = "/kaggle/input/aes2-deberta-v3-finetuned/deberta"
OUT_DIR = "deberta"
CHECKPOINT = "microsoft/deberta-v3-base"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.0
SCHEDULER = "linear"


from transformers import AutoTokenizer
from tokenizers import AddedToken

if LOAD_DIR:
    tokenizer = AutoTokenizer.from_pretrained(LOAD_DIR)

else:
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    
    # add DeBERTa ignored tokens
    
    tokenizer.add_tokens([AddedToken("\n", normalized=False)])
    tokenizer.add_tokens([AddedToken("  ", normalized=False)])
    
    tokenizer.save_pretrained(OUT_DIR)

def tokenize_fn(example):
    return tokenizer(example["full_text"], truncation=True, max_length=MAX_SEQ_LENGTH)


from datasets import Dataset

ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)

tokenized_ds_train = ds_train.map(tokenize_fn, batched=True)
tokenized_ds_train = tokenized_ds_train.remove_columns(["essay_id", "full_text"])
tokenized_ds_val = ds_val.map(tokenize_fn, batched=True)
tokenized_ds_val = tokenized_ds_val.remove_columns(["essay_id", "full_text"])


from sklearn.metrics import cohen_kappa_score

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    qwk = cohen_kappa_score(preds.round().clip(1, 6).astype(int), labels, weights="quadratic")
    return {"qwk": qwk}


from transformers import AutoConfig

if LOAD_DIR:
    config = AutoConfig.from_pretrained(LOAD_DIR)

else:
    config = AutoConfig.from_pretrained(CHECKPOINT)
    
    config.attention_probs_dropout_prob = 0.0
    config.hidden_dropout_prob = 0.0
    config.num_labels = 1  # as regression
    
    config.save_pretrained(OUT_DIR)


from transformers import AutoModelForSequenceClassification

if LOAD_DIR:
    model = AutoModelForSequenceClassification.from_pretrained(LOAD_DIR, config=config)

else:
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, config=config)


from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=SCHEDULER,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="qwk",
    greater_is_better=True,
    warmup_ratio=WARMUP_RATIO,
    report_to="none",
    fp16=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


if not LOAD_DIR:
    trainer.train()
    
    model.save_pretrained(OUT_DIR)


eval_results = trainer.evaluate()
eval_results


import pandas as pd
from datasets import Dataset

# dataset preparation

df_test = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv")
df_test["full_text"] = df_test["full_text"].apply(replace_xa0s)

# create dataset

ds_test = Dataset.from_pandas(df_test)

tokenized_ds_test = ds_test.map(tokenize_fn, batched=True)
tokenized_ds_test = tokenized_ds_test.remove_columns(["essay_id", "full_text"])

# prediction

preds = trainer.predict(tokenized_ds_test).predictions
preds = preds.round().clip(1, 6).astype(int)

# save csv

df_test["score"] = preds
df_test[["essay_id", "score"]].to_csv("submission.csv", index=False)

