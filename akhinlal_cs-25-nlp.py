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
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')


iskaggle


creds = '{"username":"akhinlal","key":"9d7761cee33e1a403f24351c358a7142"}'


# for working with paths in Python, I recommend using `pathlib.Path`
from pathlib import Path

cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)


path = Path('/kaggle/input/nlp-cs-2025')


path


!ls {path}


import pandas as pd


df = pd.read_csv(path/'train_submission.csv')


df.head()


df.columns


df["Text"][10]


df.describe(include='object')


df.groupby("Label").sample()


df.rename({"Text" : "input"}, axis = 1, inplace = True)


df.loc[df[df["Label"].isna()].index.to_list(), "Label"] = "NA"


df.describe(include='object')


from datasets import ClassLabel

class_label = ClassLabel(names=list(df["Label"].unique()))
encoded_labels = [class_label.str2int(label) for label in df["Label"]]


len(encoded_labels), len(set(encoded_labels))


df["labels"] = encoded_labels


df.head()


df = df.drop(["Usage", "Label"], axis = 1)


df.head()


label_counts = df["labels"].value_counts()
valid_labels = label_counts[label_counts > 1].index
df_filtered = df[df["labels"].isin(valid_labels)]


df_filtered.shape


sample_size = 18000


from sklearn.model_selection import train_test_split

# Assuming your dataset is a pandas DataFrame with a column "label"
sampled_df, _ = train_test_split(df_filtered, train_size=sample_size, stratify=df_filtered["labels"], random_state=42)


sampled_df.shape


from datasets import Dataset,DatasetDict

ds = Dataset.from_pandas(sampled_df) #changed to sampled_df


ds


# model_nm = 'microsoft/deberta-v3-small'
model_nm = "distilbert-base-uncased"


from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)


def tok_func(x): return tokz(x["input"], truncation=True, padding=True, max_length=512, return_tensors="pt")


tok_ds = ds.map(tok_func, batched=True)


# tok_ds = tok_ds.rename_columns({'Label':'labels'})


# tok_ds = tok_ds.rename_columns({"labels" : "ne_labels"})


import matplotlib.pyplot as plt
from collections import Counter

# Sample list
elements = tok_ds["labels"]

# Count occurrences of each unique element
counts = Counter(elements)

# Extract keys and values for plotting
labels, values = zip(*counts.items())


tok_ds


# Plot bar chart
plt.bar([class_label.int2str(label) for label in labels], values, )
plt.xlabel("Elements")
plt.ylabel("Count")
plt.title("Element Count Bar Chart")
plt.show()


eval_df = pd.read_csv(path/'test_without_labels.csv')
eval_df.describe()


eval_df.rename({"Text" : "input"}, axis = 1, inplace = True)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


from numpy.random import normal,seed,uniform


tok_ds


# tok_ds = tok_ds.remove_columns(["Usage", "Label"])
tok_ds.set_format("torch")


dds = tok_ds.train_test_split(test_size = 0.1, seed=42)
dds


eval_df.head()


eval_df = eval_df.drop(["Usage"], axis = 1)


# eval_df_sample = eval_df.sample(n = sample_size, random_state = 42)


# eval_ds = Dataset.from_pandas(eval_df_sample).map(tok_func, batched=True)
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)


# eval_ds = eval_ds.rename_columns({'Text':'input'})


def corr(x,y): return np.corrcoef(x,y)[0][1]


def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}


def accuracy(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return ((np.array(predictions) == np.array(labels)).mean())


def accuracy_d(eval_pred): return {'accuracy': accuracy(eval_pred)}


from transformers import TrainingArguments,Trainer


import torch
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

torch.cuda.memory_stats(device=None)  # Optional: Check memory stats before reset
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()


dds


bs = 64
epochs = 4
lr = 8e-5
args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    eval_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2, #evaluation_strategy changed to eval_strategy
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')


from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokz)

# train_dataloader = DataLoader(dds["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
# eval_dataloader = DataLoader(dds["test"], batch_size=8, collate_fn=data_collator)


model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=390) # We have 390 labels 
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  processing_class=tokz, compute_metrics=accuracy_d) # changed the computation metric


trainer.train()


trainer


trainer.predict(eval_ds)


trainer.save_model(r'/kaggle/my_save_model1')


pwd


!ls


import os
import subprocess
from IPython.display import FileLink, display

def download_file(path, download_file_name):
    os.chdir('/kaggle/working/')
    zip_name = f"/kaggle/working/{download_file_name}.zip"
    command = f"zip {zip_name} {path} -r"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Unable to run zip command!")
        print(result.stderr)
        return
    display(FileLink(f'{download_file_name}.zip'))


folder_loc =  r'/kaggle/my_save_model1'
download_file(folder_loc, 'out')


eval_ds


predictions = trainer.predict(eval_ds)


preds = predictions[0]


len(preds.argmax(axis = 1))


pred_list = preds.argmax(axis = 1)


pred_list


pred_str = class_label.int2str(pred_list)


sub_df = pd.DataFrame()


sub_df["Label"] = pred_str


sub_df.head()


sub_df.reset_index(inplace = True)


sub_df["index"] = sub_df["index"] + 1


sub_df.rename({"index": "ID"}, axis = 1, inplace = True)


sub_df.to_csv('/kaggle/working/outputs/submission1.csv')


folder_loc


download_file('/kaggle/working/outputs/submission1.csv', "sub1")




