DEBUG = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
from tqdm import tqdm as tqdm_slim
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')
%load_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
sns.set_palette('bwr')
SNS_CMAP = 'bwr'
plt.style.use("dark_background")
plt.rcParams['grid.color'] = '#444444'
colors = sns.palettes.color_palette(SNS_CMAP)
pd.options.mode.chained_assignment = None

def clrd(text: str, color: str = None, con: bool = None, c1:str = 'ok', c2:str = 'error')->str:
    text = str(text)
    color_codes = {
        'ok': '\033[1;92m',
        'error': '\033[91m',
        'warning': '\033[93m',
        'success': '\033[92m',
        'status': '\033[95m',
        'special': '\033[94m',
        'log': '\033[96m',
        'reset': '\033[0m',
    }
    if con is not None:
        color = c1 if con else c2
    color_code = color_codes.get(color, color_codes['reset'])
    return f"{color_code}{text}{color_codes['reset']}"


def read_texts_from_dir(dir_path):
    """
    Reads 'file_1.txt' and 'file_2.txt' from each subfolder in the directory
    and returns a DataFrame with columns: id, file_1, file_2
    """
    data = []
    error_count, last_error = 0, None
    for folder_name in tqdm(sorted(os.listdir(dir_path))):
        folder_path = os.path.join(dir_path, folder_name)
        pos_path = os.path.join(folder_path, 'file_1.txt')
        neg_path = os.path.join(folder_path, 'file_2.txt')

        try:
            with open(pos_path, 'r', encoding='utf-8') as f1:
                    text1 = f1.read().strip()
            with open(neg_path, 'r', encoding='utf-8') as f2:
                    text2 = f2.read().strip()

            index = int(folder_name.split('_')[-1])
            data.append((index, text1, text2))

        except (FileNotFoundError, ValueError, OSError) as e:
            error_count+=1
            last_error=e
            if globals().get('DEBUG', False):
                raise e

    print(f"Read {clrd(len(data), 'ok')} records with {clrd(error_count, 'error')} errors")
    if error_count > 0:
        print(clrd("Last Error: ", 'warn'), last_error)
    return pd.DataFrame(data, columns=['id', 'file_1', 'file_2'])

train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
test_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

df_train = read_texts_from_dir(train_path)
df_test = read_texts_from_dir(test_path)

df_train = df_train.merge(pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"), how='inner', on='id')
df_train.head()


df = df_train.copy()
real_texts = df.apply(lambda row: row[f'file_{row.real_text_id}'], axis=1)
fake_texts = df.apply(lambda row: row[f'file_{1 if row.real_text_id == 2 else 2}'], axis=1)

real_lens = real_texts.str.len()
fake_lens = fake_texts.str.len()
ratios = real_lens / fake_lens

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'Length': pd.concat([real_lens, fake_lens], ignore_index=True),
    'Type': ['Real'] * len(real_lens) + ['Fake'] * len(fake_lens)
})

# # Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Left: Histogram of text lengths
sns.histplot(data=plot_df, x='Length', hue='Type', ax=ax[0], bins=10, kde=True, palette='Pastel2', multiple='stack')
ax[0].set_title('Text Length Distribution')
ax[0].set_xlabel('Text Length')
ax[0].set_ylabel('Count')

# Right: Histogram of real/fake length ratios
sns.kdeplot(ratios, ax=ax[1], color='yellow')
ax[1].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Ratio = 1')
ax[1].set_title('Real / Fake Text Length Ratio')
ax[1].set_xlabel('Ratio (real_len / fake_len)')
ax[1].set_ylabel('Count')
ax[1].legend()
ax[1].set_title('Real / Fake Text Length Ratio')
ax[1].set_xlabel('Ratio')
ax[1].set_ylabel('Count')

plt.tight_layout()
plt.show()


from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def get_word_counts(texts, remove_stopwords=True):
    all_words = ' '.join(texts).lower().split()
    if remove_stopwords:
        all_words = [w for w in all_words if w.isalpha() and w not in stop_words]
    else:
        all_words = [w for w in all_words if w.isalpha()]
    return Counter(all_words)

real_counts = get_word_counts(real_texts)
fake_counts = get_word_counts(fake_texts)
real_minus_fake = {word: real_counts[word] - fake_counts.get(word, 0) 
                   for word in real_counts if real_counts[word] > fake_counts.get(word, 0)}
fake_minus_real = {word: fake_counts[word] - real_counts.get(word, 0) 
                   for word in fake_counts if fake_counts[word] > real_counts.get(word, 0)}

wc_real = WordCloud(width=1000, height=400, background_color='white', colormap='Greens')
wc_fake = WordCloud(width=1000, height=400, background_color='white', colormap='Reds')

fig, axs = plt.subplots(2, 1, figsize=(18, 10))

axs[0].imshow(wc_real.generate_from_frequencies(real_minus_fake), interpolation='bilinear')
axs[0].set_title("Words Favored in Real Texts", fontsize=18)
axs[0].axis('off')

axs[1].imshow(wc_fake.generate_from_frequencies(fake_minus_real), interpolation='bilinear')
axs[1].set_title("Words Favored in Fake (LLM) Texts", fontsize=18)
axs[1].axis('off')

plt.tight_layout()
plt.show()


fake_flags = ['â˜‰', 'à¦¿', 'àµ‡', 'àµ�', 'áŸ’', 'á�¶', 'áŸ†', 'ï¿½', 'à¦¾', 'à§ˆ', 'à¤¾', 'à±‡', 'à±�', 'Û”', 'Ö·', 'à¥�', 'à°¿', 'ï¼�', 'ã€�', 'àª¾', 'à«€',
              'à«‡', 'á€»', 'á€¹', 'á€¬', 'à´¿', '\u200b', 'à¯�', 'à·Š', 'à®¾', 'à³�', 'à³�', 'ğŸ˜°', 'àµ€', 'Ù‘', 'à¥‹', 'à¥�', 'à¤�', 'à¥ˆ', 'à¤¿', 'à²¾',
              'à¥‡', 'à¸´', 'à§‡', 'à¨¾', 'á€¯', 'à´¾', 'àª¿', 'à¥°', 'à²¿', 'à§�', 'à«�', 'âˆ€', 'ï¼š', 'á�¾', 'à¸·', 'à¹ˆ', 'à¯�', 'à·“', 'à¸¶', 'à±‚',
              '\u0ab1', 'à°¾', '\u200c', 'à¦‚', 'à³ˆ', 'à©‹', 'à¥Œ', 'â‚¬', 'ã€‚', 'á€±', 'â—¸', 'à¥‚', 'à¤‚', 'à¯‡', '\u05f6', 'à·’',
              '\u0ba5', 'à¸µ', 'à±�', 'à¹Œ', 'Ã·', 'à¯†', 'à¥€', 'Ò†', 'Ëš', 'Ìˆ', 'à¥¤', 'ğŸ¥¶', '\u05cd', 'à±Š', 'ã€�', 'ã€‹', 'à¹‰', 'â†�', 'à¸¹',
              'à§‹', 'à«�', 'à®¿', 'à±‹', 'à¨¼', 'à«‹', 'â‰¥', 'ØŒ', 'àµ�', 'à´‚', 'ï¼Œ', 'à»‰', 'â €', 'ã€‘', 'à¼‹', 'á�»', 'Ì®', 'ğŸ�¦', 'à±€', 'à«‚',
              'àª‚', 'ï¼‰', 'à¤¼', 'à©‡', 'Â¡', 'ã€�', 'á�¹', 'à·�', 'à³†', 'à¸¸', 'à§�', 'ï¼Ÿ', 'à¸±', 'à±†', 'à­�', 'àµ‹', 'à³€', 'ï¼�', 'à¯‹',
              'á�¼', 'à·�', 'Â´', 'â�”', 'à¹‡', 'àµƒ', 'Õ›', 'ğŸ•Š', 'â‰¤', 'àµ‚', 'Ö¸', 'à¦¼', 'à¤€', 'à²‚', 'à¨¿', 'Ù”', 'ï¼›', 'á€·', 'á€º', 'áŸ„',
              'áŸ‡', 'Ù�', 'à¯€', 'â‰•', 'ï¼ˆ', 'ã�ˆ', 'à¹‹', 'àµˆ', 'á€²', '\xad', 'à¥’', 'áŸƒ', 'à§ƒ', 'Ù�', '\x04', 'ï¼»', 'à³‡']

print(clrd("Fake Flagged Special Characters ===> ", 'warning'), fake_flags[:30], "...")

def contains_special_char(text):
    for ch in text:
        if ch in fake_flags:
            return True
    return False

def count_special_texts(texts):
    return sum(contains_special_char(text) for text in texts)

real_count = count_special_texts(real_texts)
fake_count = count_special_texts(fake_texts)
print(f"Real Train texts with special chars: {real_count} / {len(real_texts)}")
print(f"Fake Train texts with special chars: {fake_count} / {len(fake_texts)}")

df_test['flag'] = df_test.apply(lambda x: contains_special_char(x['file_1'])+
                                contains_special_char(x['file_2']), axis=1)
print(f"Test records where no file has flag chars: {clrd(df_test['flag'].value_counts()[0], 'ok')}")
print(f"Test records where one file has flag chars: {clrd(df_test['flag'].value_counts()[1], 'error')}")


# !pip install -q peft bitsandbytes evaluate accelerate
!pip install -U -q transformers datasets evaluate


#1.1) imports
# import transformers
from datasets import Dataset, load_dataset
# from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    # BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding)

# import bitsandbytes as bnb
import evaluate


#1.2) hf login
# from huggingface_hub import notebook_login
# notebook_login()

# from kaggle_secrets import UserSecretsClient
# HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")


#1.3) Seperate real and fake texts and truncate
MAX_CHARS = 10000
train_texts = []
labels = []

for _, row in df_train.iterrows():
    train_texts.append(row['file_1'][:MAX_CHARS])
    labels.append(1 if row['real_text_id'] == 2 else 0)
    
    train_texts.append(row['file_2'][:MAX_CHARS])
    labels.append(1 if row['real_text_id'] == 1 else 0)  

train_data = {
    "text": train_texts,
    "label": labels
}
if globals().get('DEBUG', False):
    epochs = 1
    train_data = {
        "text": train_texts[:5],
        "labels": labels[:5]
    }
    
#1.4) Create Huggingface Datasets
data = Dataset.from_dict(train_data)


#2.1) Tokenize Dataset
checkpoint  = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                          # token=HF_TOKEN,
                                         )

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

data = data.map(preprocess_function, batched=True)
data


#2.2) Load Batches
if globals().get('DEBUG', False):
    assert 'labels' in data[0].keys(), "Dataset format might be incorrect"
    
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc") # F1 and Accuracy
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#2.3 Load Model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=2,
                                                          # token=HF_TOKEN,
                                                          )


from datetime import datetime

today = datetime.now().date()
result_dir = os.path.join(r"/kaggle/working/", f"distilbert_{today}")
ngpus = 1
device = None
epochs = 5
# WANDB_START_METHOD="thread" 

if globals().get('DEBUG', False):
    epochs = 2

print(f"Saving to {result_dir}")
print(f"Epochs: {epochs}")
# !nvidia-smi -L


training_args = TrainingArguments(
    output_dir=result_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_strategy="epoch",
    # logging_steps = 10,
    # eval_strategy = "epoch",
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    # eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics = compute_metrics,
)

trainer.train()


import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference_pipeline(text):
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        inputs = inputs.to(device)
        logits = model(**inputs).logits
        logits = logits.detach().to('cpu')
        probs = F.softmax(logits, dim=1)  
        return probs


df_test["logits1"] = df_test.apply(lambda x: float(inference_pipeline(x["file_1"][:MAX_CHARS])[0][0]), axis=1)
df_test["logits2"] = df_test.apply(lambda x: float(inference_pipeline(x["file_2"][:MAX_CHARS])[0][0]), axis=1)
df_test.to_csv(os.path.join(r"/kaggle/working/", "pred.csv"), index=False)

def get_real_idx(logits_1, logits_2):
    if(logits_1 > logits_2):
        return 1
    else:
        return 2

df_test['real_text_id'] = df_test.apply(lambda x: get_real_idx(x['logits1'], x['logits2']), axis=1)


# import transformers
from datasets import Dataset, load_dataset
# from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    # BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding)


import re
import unicodedata

def clean_text(text):
    # normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # remove non-printable characters
    text = re.sub(r"[^\x20-\x7E]", "", text)
    # remove emojis & symbols
    text = re.sub(r"[^\w\s.,!?\"']", "", text)
    # remove repeated garbage (e.g., $$$$, !!!!)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"(\W)\1{2,}", r"\1", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

MAX_CHARS = 10000
train_texts1, train_texts2 = [], []
labels = []

for _, row in df_train.iterrows():
    txt1 = clean_text(row['file_1'])[:MAX_CHARS]
    train_texts1.append(txt1)

    txt2 = clean_text(row['file_2'])[:MAX_CHARS]
    train_texts2.append(txt2)
    
    labels.append(1 if row['real_text_id'] == 1 else 0)

    #ensure similarity
    train_texts1.append(txt2)
    train_texts2.append(txt1)
    labels.append(0 if row['real_text_id'] == 1 else 1)

train_data = {
    "text1": train_texts1,
    "text2": train_texts2,
    "label": labels
}
if globals().get('DEBUG', False):
    epochs = 1
    train_data = {
    "text1": train_texts1[:5],
    "text2": train_texts2[:5],
    "label": labels[:5]
}
    
#1.4) Create Huggingface Datasets
data = Dataset.from_dict(train_data)


checkpoint  = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                          # token=HF_TOKEN,
                                         )

def preprocess_function(examples):
    return tokenizer(examples["text1"], examples["text2"], truncation=True, max_length=512)

data = data.map(preprocess_function, batched=True)
data


#2.2) Load Batches
if globals().get('DEBUG', False):
    assert 'labels' in data[0].keys(), "Dataset format might be incorrect"
    
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc") # F1 and Accuracy
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#2.3 Load Model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=2,
                                                          # token=HF_TOKEN,
                                                          )


from datetime import datetime

today = datetime.now().date()
result_dir = os.path.join(r"/kaggle/working/", f"distilbert_{today}")
ngpus = 1
device = None
epochs = 3
# WANDB_START_METHOD="thread" 

if globals().get('DEBUG', False):
    epochs = 2

print(f"Saving to {result_dir}")
print(f"Epochs: {epochs}")
# !nvidia-smi -L


training_args = TrainingArguments(
    output_dir=result_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_strategy="epoch",
    # logging_steps = 10,
    # eval_strategy = "epoch",
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    # eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics = compute_metrics,
)

trainer.train()


import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference_pipeline(text1, text2):
    inputs = tokenizer(text1, text2, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        inputs = inputs.to(device)
        logits = model(**inputs).logits
        logits = logits.detach().to('cpu')
        probs = F.softmax(logits, dim=1)  
        return probs


df_sub = df_test[['id', 'real_text_id']]
df_sub.to_csv("submission.csv", index=False)
df_sub


import transformers
import datasets
import torch

libs = [transformers, datasets, torch]
for lib in libs:
    print(f"{lib.__version__} ====> {lib.__name__}")
!python --version

