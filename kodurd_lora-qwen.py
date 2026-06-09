import os
import tempfile
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, log_loss
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    TaskType,
    PeftModel
)

VER = 5


# paths
main_path = '/kaggle/input/map-charting-student-math-misunderstandings'
train_data = main_path +'/train.csv'
test_data = main_path + '/test.csv'
submit_example = main_path + '/sample_submission.csv'


@dataclass
class Config:
    
    output_dir: str = f"output-{VER}"
    model_path = "/kaggle/input/qwen2-math-1-5b"
    adapter_path = "/kaggle/input/qwen2-math-1-5b-lora"

    freeze_layers: int = 22
        
    lora_r: int = 16
    lora_alpha: float = 4 
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    warmup_steps: int = 20
    lr: float = 2e-4
    n_epochs = 2
    optim_type: str = "adamw_torch"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = 4


config = Config()


df_train = pd.read_csv(train_data)
df_train.head()


class SafeLabelEncoder(LabelEncoder):
    def __init__(self):
        super().__init__()
        self.unknown_class = None

    def fit(self, y):
        super().fit(y)
        self.unknown_class = max(self.classes_) + "_UNK"
        self.classes_ = np.append(self.classes_, self.unknown_class)
        return self

    def transform(self, y):
        y = np.array(y)
        mask = np.isin(y, self.classes_)
        y[~mask] = self.unknown_class
        return super().transform(y)

    def inverse_transform(self, y):
        return super().inverse_transform(y)


def camel_to_snake(name: str) -> str:
    new_word = []
    n = len(name)

    for index, ch in enumerate(name):
        if ch.isupper():
            if index > 0:
                prev = name[index - 1]
                next_ch = name[index + 1] if index + 1 < n else ''
                if prev not in ['_', '-'] and (prev.islower() or (next_ch.islower() if next_ch else False)):
                    new_word.append('_')
            new_word.append(ch.lower())
        else:
            new_word.append(ch)
    
    return ''.join(new_word)

def rename_col_df(df: pd.DataFrame) -> pd.DataFrame:
    old_columns = df.columns
    new_columns = [
        camel_to_snake(col)
        for col in df.columns
    ]

    df.columns = new_columns
    return df

def encoding_label(df: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    le = SafeLabelEncoder()
    df['misconception'] = df.misconception.fillna('NA')
    df['target'] = df['category'] + ":" + df['misconception']
    df['label'] = le.fit_transform(df['target'])

    if pipeline is not None:  # сохраним encoder в pipeline
        pipeline.label_encoder = le
    return df

def add_correct_answer(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.apply(lambda row: row['category'].split('_')[0], axis=1) == 'True'
    correct = df.loc[idx].copy()
    correct['c'] = correct.groupby(['question_id', 'mc_answer'])['mc_answer'].transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['question_id'])
    correct = correct[['question_id', 'mc_answer']]
    correct['is_correct'] = 1

    df = df.merge(correct, on=['question_id', 'mc_answer'], how='left')
    df['is_correct'] = df['is_correct'].fillna(0)
    df['is_correct'] = df.apply(
        lambda x: "yes" 
        if x['is_correct'] == 1 
        else "no", axis=1
    )
    return df

def add_formatted_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет столбец 'text' с объединённым описанием вопроса, ответа,
    правильности и объяснения студента.
    """
    def format_input(row):
        x = "Yes" if row["is_correct"] == "yes" or row["is_correct"] == 1 else "No"
        return (
            f"Question: {row['question_text']}\n"
            f"Answer: {row['mc_answer']}\n"
            f"Correct? {x}\n"
            f"Student Explanation: {row['student_explanation']}"
        )

    df["text"] = df.apply(format_input, axis=1)
    return df


class DataFramePipeline:
    def __init__(self):
        self.steps = []
        self.label_encoder = None 

    def add_step(self, func, name=None):
        step_name = name or func.__name__
        self.steps.append((step_name, func))
        return self

    def run(self, df, verbose=True):
        for name, func in self.steps:
            if verbose:
                print(f"[Pipeline] Running step: {name}")
            try:
                df = func(df, pipeline=self)
            except TypeError:
                df = func(df)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Step '{name}' must return DataFrame, got {type(df)}")
        return df

    def describe(self):
        print("Pipeline steps:")
        for i, (name, _) in enumerate(self.steps):
            print(f"{i+1}. {name}")
        return self

    def __call__(self, df, verbose=True):
        return self.run(df, verbose=verbose)


pipe = DataFramePipeline()
pipe.add_step(rename_col_df)\
    .add_step(encoding_label)\
    .add_step(add_correct_answer)\
    .add_step(add_formatted_text)

df_processed = pipe(df_train)
df_processed.head()


num_classes = len(pipe.label_encoder.classes_)
print(f"Классов: {num_classes}")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(pipe.label_encoder, f)


train_df, val_df = train_test_split(df_processed, test_size=0.2, random_state=42)
cols = ['text', 'label']
train_ds = Dataset.from_pandas(train_df[cols])
val_ds = Dataset.from_pandas(val_df[cols])


print(train_ds)           
print(train_ds[0])        
print(train_ds[:5])


offload_dir = tempfile.mkdtemp()

tokenizer = AutoTokenizer.from_pretrained(
    config.model_path, 
    local_files_only=True,
)
base_model = AutoModelForSequenceClassification.from_pretrained(
    config.model_path,
    num_labels=num_classes,
    torch_dtype=torch.float32,
    device_map="auto",
    local_files_only=True,
    offload_folder=offload_dir
)


# num_layers = len(base_model.model.layers)
# print(num_layers)
# layers_to_transform = [i for i in range(num_layers) if i >= 0]
# layers_to_transform


# inference
model = PeftModel.from_pretrained(
    base_model,
    config.adapter_path,
    local_files_only=True,
    is_trainable=False,
    offload_folder=offload_dir
)


# train
# lora_config = LoraConfig(
#     r=config.lora_r,
#     lora_alpha=config.lora_alpha,
#     # only target self-attention
#     target_modules=["q_proj", "k_proj", "v_proj",
#                     "down_proj","up_proj","o_proj","gate_proj"],
#     # layers_to_transform = [i for i in range(num_layers) if i >= config.freeze_layers],
#     layers_to_transform=layers_to_transform,
#     lora_dropout=config.lora_dropout,
#     bias=config.lora_bias,
#     task_type=TaskType.SEQ_CLS,
#     modules_to_save=["score","classifier_head1", "classifier_head2"]
# )

# model = get_peft_model(base_model, lora_config)
# model.print_trainable_parameters()


# def compute_map3(eval_pred):
#     logits, labels = eval_pred
#     logits = np.array(logits)
#     labels = np.array(labels)

#     probs = softmax(logits, axis=-1)

#     top3 = np.argsort(-probs, axis=1)[:, :3]
#     map3 = 0.0
#     for i, label in enumerate(labels):
#         if label in top3[i]:
#             rank = np.where(top3[i] == label)[0][0]
#             map3 += 1.0 / (rank + 1)
#     map3 /= len(labels)

#     log_loss_value = log_loss(labels, probs, labels=np.arange(num_classes))

#     return {"eval_log_loss": log_loss_value, "eval_map@3": map3}


tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


# training_args = TrainingArguments(
#     output_dir = config.output_dir,
#     overwrite_output_dir=True,
#     report_to="none",
#     num_train_epochs=config.n_epochs,
#     per_device_train_batch_size=config.per_device_train_batch_size,
#     gradient_accumulation_steps=config.gradient_accumulation_steps,
#     per_device_eval_batch_size=config.per_device_eval_batch_size,
#     logging_steps=10,
#     eval_strategy="epoch",
#     save_strategy="no",
#     optim=config.optim_type,
#     # fp16=True,
#     learning_rate=config.lr,
#     warmup_steps=config.warmup_steps,
#     metric_for_best_model='log_loss',
#     greater_is_better=False, 
#     fp16=False,      
#     bf16=False
# )


# if model.config.pad_token_id is None:
#     model.config.pad_token_id = model.config.eos_token_id

# small_train_ds = train_ds.select(range(20))
# small_val_ds = val_ds.select(range(10))

# trainer = Trainer(
#     args=training_args, 
#     model=model,
#     processing_class=tokenizer,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     compute_metrics=compute_map3,
#     data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
# )

# trainer.train()


# model.save_pretrained(config.output_dir)
# tokenizer.save_pretrained(config.output_dir)


le = pipe.label_encoder
df_test = pd.read_csv(test_data)
print(df_test.shape)

df_test = rename_col_df(df_test)

idx = df_train.apply(lambda row: row['category'].split('_')[0], axis=1) == 'True'
correct = df_train.loc[idx].copy()
correct['c'] = correct.groupby(['question_id', 'mc_answer'])['mc_answer'].transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['question_id'])
correct = correct[['question_id', 'mc_answer']]
correct['is_correct'] = 1

df_test = df_test.merge(correct, on=['question_id', 'mc_answer'], how='left')
df_test['is_correct'] = df_test['is_correct'].fillna(0)
df_test['is_correct'] = df_test.apply(
    lambda x: "yes" 
    if x['is_correct'] == 1 
    else "no", axis=1
)

df_test = add_formatted_text(df_test)
df_test.head()


model.eval()


from torch.utils.data import DataLoader

ds_test = Dataset.from_pandas(df_test[['text']])
ds_test = ds_test.map(tokenize, batched=True)
ds_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader = DataLoader(ds_test, batch_size=2, collate_fn=data_collator)


if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
    
all_probs = []

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

probs = np.vstack(all_probs)


top3 = np.argsort(-probs, axis=1)[:, :3]
flat_top3 = top3.flatten()
try:
    decoded_labels = le.inverse_transform(flat_top3)
except:
    decoded_labels = ["Unknown"] * len(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

joined_preds = [" ".join(row) for row in top3_labels]

sub = pd.DataFrame({
    "row_id": df_test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()

