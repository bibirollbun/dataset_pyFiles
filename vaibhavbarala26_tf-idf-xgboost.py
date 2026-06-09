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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")


train


n_rules      = train['rule'].nunique()
n_subreddits = train['subreddit'].nunique()
print(f"Number of unique rules:      {n_rules}")
print(f"Number of unique subreddits: {n_subreddits}")

# Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='rule_violation', data=train)
plt.title('Distribution of Rule Violation (0 = no, 1 = yes)')
plt.show()

# Overall violation rate
rate_overall = train['rule_violation'].mean()
print(f"Overall violation rate: {rate_overall:.3f}")


rule_stats = train.groupby('rule')['rule_violation'] \
    .agg(['mean', 'count']) \
    .rename(columns={'mean':'violation_rate','count':'num_samples'}) \
    .sort_values('num_samples', ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x='violation_rate', y=rule_stats.index, data=rule_stats.reset_index())
plt.title('Violation Rate by Rule')
plt.xlabel('Violation Rate')
plt.ylabel('Rule')
plt.show()


sub_stats = train.groupby('subreddit')['rule_violation'] \
    .agg(['mean','count']) \
    .rename(columns={'mean':'violation_rate','count':'num_samples'}) \
    .sort_values('num_samples', ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.scatterplot(x='num_samples', y='violation_rate', data=sub_stats.reset_index(), s=100)
for _, row in sub_stats.reset_index().iterrows():
    plt.text(row['num_samples']+10, row['violation_rate'], row['subreddit'])
plt.title('Top 10 Subreddits: Sample Count vs Violation Rate')
plt.xlabel('Number of Samples')
plt.ylabel('Violation Rate')
plt.show()


train['char_len'] = train['body'].str.len()
train['word_len'] = train['body'].str.split().map(len)

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].hist(train['char_len'], bins=50)
axes[0].set_title('Character Length Distribution')
axes[0].set_xlabel('Number of Characters')

axes[1].hist(train['word_len'], bins=50)
axes[1].set_title('Word Count Distribution')
axes[1].set_xlabel('Number of Words')

plt.tight_layout()
plt.show()


# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# #Load base model and tokenizer
# # model1 = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
# # tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# # # Save them locally
# # model1.save_pretrained("roberta-base-local")
# # tokenizer.save_pretrained("roberta-base-local")
# # from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # Load local model and tokenizer
# model = AutoModelForSequenceClassification.from_pretrained("/kaggle/working/roberta-base-local", local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/roberta-base-local", local_files_only=True)



from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification

# Set path
model_dir = "/kaggle/working/roberta-base-local"

# Load config, tokenizer, model manually
config = RobertaConfig.from_json_file(f"{model_dir}/config.json")
tokenizer = RobertaTokenizer.from_pretrained(model_dir, local_files_only=True)
model = RobertaForSequenceClassification.from_pretrained(model_dir, config=config, local_files_only=True)



train["input_text"] = train["body"]+"Rule"+train["rule"]


from datasets import Dataset, Features, Value

# 1. Convert to HuggingFace dataset
dataset = Dataset.from_pandas(train[['input_text', 'rule_violation']])

# 2. Tokenize
def preprocess(example):
    return tokenizer(example['input_text'], truncation=True, padding='max_length', max_length=256)

dataset = dataset.map(preprocess, batched=True)

# 3. Rename label column
dataset = dataset.rename_column("rule_violation", "labels")

# ✅ 4. Force labels to float
dataset = dataset.cast_column("labels", Value("float32"))



args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    logging_dir="./logs",             # ✅ logs directory
    logging_steps=10,                 # ✅ log every 10 steps
    disable_tqdm=False,  
    report_to="none",  # ✅ disables wandb
# ✅ make sure tqdm is not disabled
)



from sklearn.metrics import roc_auc_score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    return {"roc_auc": roc_auc_score(labels, probs)}
    

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset.shuffle(seed=42),  # Subsample for quick test
    eval_dataset=None,
    #compute_metrics=compute_metrics,
)

trainer.train()


test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")


from datasets import Dataset, Features, Value
test["input_text"] = test["body"]+"Rule"+test["rule"]
# 1. Convert to HuggingFace dataset
dataset = Dataset.from_pandas(test[['input_text', 'rule_violation']])

# 2. Tokenize
def preprocess(example):
    return tokenizer(example['input_text'], truncation=True, padding='max_length', max_length=256)

dataset = dataset.map(preprocess, batched=True)

# 3. Rename label column
dataset = dataset.rename_column("rule_violation", "labels")

# ✅ 4. Force labels to float
dataset = dataset.cast_column("labels", Value("float32"))



pred = trainer.predict(dataset)


test


probs = 1 / (1 + np.exp(-pred.predictions))   
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": probs.flatten(),  # Make sure it's a 1D array
})
submission.to_csv("submission.csv" , index=False)


!ls /kaggle/working/roberta-base-local




