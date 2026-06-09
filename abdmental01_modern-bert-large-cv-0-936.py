%%time

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
import re
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import warnings 
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=2
model_n = "/kaggle/input/modernbert-large-cv936"
EPOCHS = 4

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


%%time

le = LabelEncoder()

train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)

def extra_fe(train):
    train['explanation_len'] = train['StudentExplanation'].fillna('').apply(len)
    train['mc_frac_count'] = train['StudentExplanation'].fillna('').apply(
        lambda x: len(re.findall(r'FRAC_\d+_\d+|\\frac', x))
    )
    train['number_count'] = train['StudentExplanation'].fillna('').apply(
        lambda x: len(re.findall(r'\b\d+\b', x))
    )
    train['operator_count'] = train['StudentExplanation'].fillna('').apply(
        lambda x: len(re.findall(r'[\+\-\*/=]', x))
    )
    train['mc_answer_len'] = train['MC_Answer'].fillna('').apply(len)
    train['question_len'] = train['QuestionText'].fillna('').apply(len)
    train['explanation_to_question_ratio'] = train['explanation_len'] / (train['question_len'] + 1)
    return train

train = extra_fe(train)
test = extra_fe(test)

print(f"Train shape: {train.shape} with {n_classes} target classes")

idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

tokenizer = AutoTokenizer.from_pretrained(model_n)
MAX_LEN = 256

def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This answer is incorrect."

    extra = (
        f"Additional Info: "
        f"The explanation has {row['explanation_len']} characters "
        f"and includes {row['mc_frac_count']} fraction(s)."
    )

    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}\n"
        f"{extra}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )

lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]

L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort(lengths)

train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)

test['text'] = test.apply(format_input,axis=1)

ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

%time

model_F = AutoModelForSequenceClassification.from_pretrained(
    model_n,
    num_labels=n_classes,
    reference_compile=False,
)

training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    save_total_limit=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=EPOCHS,
    gradient_accumulation_steps=1, 
    load_best_model_at_end=True,
    metric_for_best_model="map@3",
    greater_is_better=True,
    fp16=True,   
    bf16=False,  
    report_to="none",
    logging_dir="./logs",
)

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]
    match = (top3 == labels[:, None])

    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}

trainer = Trainer(
    model=model_F,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)


%%time

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


%%time

top3 = np.argsort(-probs, axis=1)[:, :3] 

flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

joined_preds = [" ".join(row) for row in top3_labels]

sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()

