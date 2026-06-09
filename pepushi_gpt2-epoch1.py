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


# ライブラリのインポート
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# 1. データセットの準備（ここでは例としてwikitextを使用）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
# ここで、必要に応じて自分のエッセイデータセットなどを用意する

# 2. トークナイザーのロード
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # パディングトークンを設定

# 3. データセットの前処理関数の定義
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. モデルのロード（事前学習済みのGPT-2）
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 5. トレーニング設定の定義
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    run_name="my_experiment_01",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,         # 例として1エポック
    weight_decay=0.01,
    save_total_limit=1,
    fp16=True,                  # GPU環境の場合
)

# 6. Trainerの作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 7. ファインチューニングの実行
trainer.train()

# 8. ファインチューニング後のモデルを保存
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")


from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ファインチューニング済みモデルのロード
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

# エッセイ生成パイプラインの作成
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=180,
    temperature=0.7,
    top_p=0.7,
    do_sample=True
)

# 例: 与えられたトピックに対してエッセイを生成
topic = "Discuss the impact of remote work on urban culture."
prompt = f"Topic: {topic} (provide your response in 60 words).\nPlease write in a formal, scholarly manner."
generated = pipe(prompt)
print(generated[0]['generated_text'])




