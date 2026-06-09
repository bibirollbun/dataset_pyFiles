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


import torch
from torch import Tensor
import torch.nn.functional as F

device = torch.device('cuda')
model_id = '/kaggle/input/wsdm-xlm-roberta-base-rt-cp-2978'


from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=['classifier']
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForSequenceClassification.from_pretrained(model_id, quantization_config=quantization_config, num_labels=1)


from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel

bnb_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    inference_mode=False,
    target_modules=['query','key','value','self.output.dense','intermediate.dense', 'output.dense']
    )

base_model = prepare_model_for_kbit_training(base_model)
model = PeftModel.from_pretrained(base_model, '/kaggle/input/wsdm-xlm-roberta-base-rt-cp-2978/xlm-roberta-base-wsdm-adapter', is_trainable=True)
model.to(device)


model.print_trainable_parameters()


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id


import re
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def preprocess(row):
    row.prompt = re.sub(r'\s+', ' ', row.prompt)
    row.response_a = re.sub(r'\s+', ' ', row.response_a)
    row.response_b = re.sub(r'\s+', ' ', row.response_b)
    row['chosen'] = row.prompt[:500] + '\n' + (row.response_a if row.winner == 'model_a' else row.response_b)
    row['rejected'] = row.prompt[:500] + '\n' + (row.response_a if row.winner != 'model_a' else row.response_b)
    return row

df = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')[40000:]
df = df.apply(preprocess, axis=1)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

train_indices, val_indices = list(skf.split(df, df.winner))[4]

train_df, val_df = df.iloc[train_indices], df.iloc[val_indices]

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

len(train_df), len(val_df)


max_length = min(tokenizer.model_max_length, 1900)


from datasets import Dataset

def tokenize(row):
    chosen = tokenizer(
        text=row['chosen'],
        truncation=True,
        padding=True,
        max_length=max_length
    )

    rejected = tokenizer(
        text=row['rejected'],
        truncation=True,
        padding=True,
        max_length=max_length
    )

    return {
        'input_ids_chosen': chosen['input_ids'],
        'attention_mask_chosen': chosen['attention_mask'],
        'input_ids_rejected': rejected['input_ids'],
        'attention_mask_rejected': rejected['attention_mask']
    }

ds = Dataset.from_pandas(train_df)
ds = ds.map(tokenize, num_proc=4, batched=True) \
    .remove_columns(['id', 'prompt', 'response_a', 'response_b', 'winner', 'model_a', 'model_b', 'language', 'chosen', 'rejected'])
ds.set_format('torch')
ds = ds.train_test_split(0.2)
ds


torch.cuda.empty_cache()


from trl import RewardTrainer, RewardConfig

training_args = RewardConfig(
    output_dir="outputs",
    eval_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=400,
    warmup_steps=1,
    logging_dir="./logs",
    logging_steps=500,
    report_to='none',
    optim='paged_adamw_8bit',
    fp16=True,
    save_total_limit=5,
    remove_unused_columns=False,
    lr_scheduler_type='cosine',
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
)

model.config.use_cache = False
trainer.train()


model.save_pretrained('xlm-roberta-large-rt-wsdm')


from tqdm import tqdm

@torch.amp.autocast('cuda', dtype=torch.float16)
@torch.no_grad()
def inference(df, model, device):
    answers = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        response_a_inputs = tokenizer(row.prompt[:500] + '\n' + row.response_a, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True).to(device)
        response_b_inputs = tokenizer(row.prompt[:500] + '\n' + row.response_b, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True).to(device)
        a_outputs = model(**response_a_inputs).logits
        b_outputs = model(**response_b_inputs).logits
        answer = torch.tensor([a_outputs, b_outputs]).argmax(-1)
        answers.append('model_a' if answer == 0 else 'model_b')

    df['pred'] = answers
    return df


from sklearn.metrics import accuracy_score

val_df = inference(val_df, model, device)
accuracy_score(val_df.winner, val_df.pred)

