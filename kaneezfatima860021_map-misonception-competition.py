# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
#model_name = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-xsmall"
#model_name = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-small'
#model_name = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-base'
model_name = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-large'
EPOCHS = 4

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)

import re

# Enhanced feature extraction
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

print(f"Train shape: {train.shape} with {n_classes} target classes")

train.head()


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


from IPython.display import display, Math, Latex

# Get MC Answer rankings
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count', axis=1)
tmp = tmp.sort_values(['QuestionId', 'rank'])


# Display each question with its answer choices
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId == q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId == q].MC_Answer.values
    labels = "ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}"))
    display(Latex(f"MC Answers: {choice_str}"))



import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
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
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


L = (np.array(lengths) > MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort(lengths)



# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.05, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


from transformers import DebertaV2ForSequenceClassification

model = DebertaV2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes
)


training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    num_train_epochs=EPOCHS,

    # Keep batch size small for memory
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,  # Simulates batch_size * 4


    # Must enable mixed precision to reduce VRAM usage
    #fp16=True,

    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
)


# CUSTOM MAP@3 METRIC

from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

trainer.train()


import joblib

trainer.save_model(f"{DIR}/best")
_ = joblib.dump(le, f"{DIR}/label_encoder.joblib")


tokenizer = AutoTokenizer.from_pretrained(f"{DIR}/best")
model = DebertaV2ForSequenceClassification.from_pretrained(
    f"{DIR}/best",
    num_labels=n_classes  
)
training_args = TrainingArguments(report_to="none")
trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args)
le = joblib.load(f"{DIR}/label_encoder.joblib")


test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test.shape )
test.head()


import re

# Enhanced feature extraction for test set
test['explanation_len'] = test['StudentExplanation'].fillna('').apply(len)
test['mc_frac_count'] = test['StudentExplanation'].fillna('').apply(
    lambda x: len(re.findall(r'FRAC_\d+_\d+|\\frac', x))
)
test['number_count'] = test['StudentExplanation'].fillna('').apply(
    lambda x: len(re.findall(r'\b\d+\b', x))
)
test['operator_count'] = test['StudentExplanation'].fillna('').apply(
    lambda x: len(re.findall(r'[\+\-\*/=]', x))
)
test['mc_answer_len'] = test['MC_Answer'].fillna('').apply(len)
test['question_len'] = test['QuestionText'].fillna('').apply(len)
test['explanation_to_question_ratio'] = test['explanation_len'] / (test['question_len'] + 1)


test.head()


# Add 'is_correct' info to test set using same logic
test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

# Format inputs using the same prompt template
test['text'] = test.apply(format_input, axis=1)

# Preview the resulting structure
test.head()



ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :3]

# Decode indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join top 3 labels into a space-separated string per row
joined_preds = [" ".join(row) for row in top3_labels]

# Create submission DataFrame
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})

# Save to CSV
sub.to_csv("submission.csv", index=False)
sub.head()


