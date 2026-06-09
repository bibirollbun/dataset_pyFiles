import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


model_name = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-large'


train_path = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
test_path = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
#load csv
import pandas as pd
train_df = pd.read_csv(train_path)


# --- Data Loading and Preprocessing ---
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
import re

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)

# --- Feature Engineering ---
train['explanation_len'] = train['StudentExplanation'].fillna('').apply(len)
train['mc_frac_count'] = train['StudentExplanation'].fillna('').apply(lambda x: len(re.findall(r'FRAC_\d+_\d+|\\frac', x)))
train['number_count'] = train['StudentExplanation'].fillna('').apply(lambda x: len(re.findall(r'\b\d+\b', x)))
train['operator_count'] = train['StudentExplanation'].fillna('').apply(lambda x: len(re.findall(r'[\+\-\*/=]', x)))
train['mc_answer_len'] = train['MC_Answer'].fillna('').apply(len)
train['question_len'] = train['QuestionText'].fillna('').apply(len)
train['explanation_to_question_ratio'] = train['explanation_len'] / (train['question_len'] + 1)




# --- Input Formatting ---
def format_input(row):
    x = "This answer is correct." if row.get('is_correct', 0) else "This answer is incorrect."
    extra = (f"Additional Info: The explanation has {row['explanation_len']} characters and includes {row['mc_frac_count']} fraction(s).")
    return (f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"{x}\n"
            f"Student Explanation: {row['StudentExplanation']}\n"
            f"{extra}")
train['text'] = train.apply(format_input, axis=1)



#example use:
formated_input = train['text'].iloc[0]
print(formated_input)


# --- Tokenization and Dataset Preparation ---
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification,TrainingArguments,Trainer

from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

# Use the manually included model path
model_name = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-large'
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
MAX_LEN = 256
train_df, val_df = train_test_split(train, test_size=0.05, random_state=42)
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)

# --- Model Setup ---
model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)



training_args = TrainingArguments(
    output_dir='./bert_output',
    do_train=True,
    do_eval=True,
    eval_strategy="steps", 
    save_strategy="steps",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none",
)



# --- Custom Metric (MAP@3) ---
import numpy as np
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

# --- Trainer Setup and Training ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)
trainer.train()

# --- Inference and Submission ---
# Prepare test set features and format
for col in ['explanation_len','mc_frac_count','number_count','operator_count','mc_answer_len','question_len','explanation_to_question_ratio']:
    test[col] = test['StudentExplanation'].fillna('').apply(len) if col == 'explanation_len' else 0
# If you have correctness info, merge and format as above
# test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
# test.is_correct = test.is_correct.fillna(0)
test['text'] = test.apply(format_input, axis=1)
ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)
predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
top3 = np.argsort(-probs, axis=1)[:, :3]
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)
joined_preds = [" ".join(row) for row in top3_labels]



sub = pd.DataFrame({
    "row_id": test.get('row_id', pd.Series(range(len(test)))),
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()

