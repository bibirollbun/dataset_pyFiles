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


local_dir = '/kaggle/input/cardiffnlptwitter-roberta-base-sentiment-latest/transformers/default/1/model'


import pandas as pd
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer
from typing import Union
import html
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from scipy.special import softmax


TRAIN_DATA_PATH = r'/kaggle/input/jigsaw-agile-community-rules/train.csv'
TEST_DATA_PATH = r'/kaggle/input/jigsaw-agile-community-rules/test.csv'

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

RANDOM_STATE = 42


g = torch.manual_seed(RANDOM_STATE)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'File {path} not found')
    df = pd.read_csv(path)
    print(f'Data shape: {df.shape}', end='\n\n' + '-'*50+ '\n')
    print('Data info: ', df.info(), sep='\n', end='\n\n' + '-'*50+ '\n')
    print('Data head: ', df.head(), sep='\n', end='\n\n' + '-'*50+ '\n')
    return df


train_df = load_data(TRAIN_DATA_PATH)


testt_df = load_data(TEST_DATA_PATH)


def check_null_values(df: pd.DataFrame) -> pd.DataFrame:
    null_info = pd.DataFrame({
        'Column': df.columns,
        'Null_Count': df.isnull().sum().values,
        'Null_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    null_info = null_info[null_info['Null_Count'] > 0].sort_values('Null_Count', ascending=False)

    if null_info.empty:
        print("No null values found in the dataframe!")
    else:
        print(f"Found null values in {len(null_info)} columns:\n")
        print(null_info.to_string(index=False))

    return null_info


check_null_values(train_df)


check_null_values(testt_df)


def check_class_distribution(df: pd.DataFrame, target_col: str = 'rule_violation'):
    if target_col not in df.columns:
        print(f"Column '{target_col}' not found in dataframe!")
        return None

    dist = df[target_col].value_counts().sort_index()
    dist_pct = df[target_col].value_counts(normalize=True).sort_index() * 100

    dist_df = pd.DataFrame({
        'Class': dist.index,
        'Count': dist.values,
        'Percentage': dist_pct.values.round(2)
    })

    print(f"Class distribution for '{target_col}':\n")
    print(dist_df.to_string(index=False))

    return dist_df


check_class_distribution(train_df)


plt.figure(figsize=(8, 6))
train_df['rule_violation'].value_counts().sort_index().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Class Distribution - Rule Violation')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.show()



tokenizer = AutoTokenizer.from_pretrained(local_dir)


import pandas as pd
from transformers import AutoTokenizer

def create_strategy(df: pd.DataFrame, text_cols: list[str], tokenizer) -> pd.DataFrame:
    new_df = df.copy()

    missing_cols = [col for col in text_cols if col not in new_df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    for col in text_cols:
        new_df[col] = new_df[col].fillna('').astype(str)
        print(f"Column '{col}' has {new_df[col].isna().sum()} NaN values")

    new_df['input_text'] = new_df[text_cols].astype(str).agg(tokenizer.eos_token.join, axis=1)

    print(f"\nCreated input_text column using the optimal method. Sample:")
    print(new_df['input_text'].head(3), end='\n\n' + '-'*50+ '\n')

    return new_df


first_strategy = create_strategy(train_df, ['body', 'rule', 'positive_example_1', 'negative_example_1'], tokenizer)
second_strategy = create_strategy(train_df, ['body', 'rule'], tokenizer)


def get_token_distribution(df: pd.DataFrame, text_col: str, tokenizer, sample_size: int = None):
    if sample_size:
        df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    else:
        df_sample = df

    token_lengths = []
    for text in df_sample[text_col]:
        if pd.notna(text):
            tokens = tokenizer.encode(str(text), add_special_tokens=True)
            token_lengths.append(len(tokens))
        else:
            token_lengths.append(0)

    stats = {
        'mean': round(sum(token_lengths) / len(token_lengths), 2),
        'median': sorted(token_lengths)[len(token_lengths) // 2],
        'min': min(token_lengths),
        'max': max(token_lengths),
        'std': round((sum((x - sum(token_lengths) / len(token_lengths)) ** 2 for x in token_lengths) / len(
            token_lengths)) ** 0.5, 2)
    }

    print(f"Token Distribution for '{text_col}' column:")
    print(f"Sample Size: {len(df_sample)}")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value}")


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(token_lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']}")
    plt.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']}")
    plt.title('Token Length Distribution')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot(token_lengths, vert=True)
    plt.title('Token Length Boxplot')
    plt.ylabel('Token Count')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


get_token_distribution(first_strategy, 'input_text', tokenizer, sample_size=5000)


get_token_distribution(second_strategy, 'input_text', tokenizer, sample_size=5000)


def preprocess_for_roberta(df: pd.DataFrame, col: str) -> pd.DataFrame:
    new_df = df.copy()

    processed_texts = []
    for text in new_df[col]:
        text = str(text)

        text = html.unescape(text)
        text = re.sub(r'@\w+', '@user', text)
        text = re.sub(r'https?://\S+|www\.\S+', 'http', text)
        text = re.sub(r'\s+', ' ', text).strip()

        processed_texts.append(text)

    new_df[col] = processed_texts
    return new_df


clear_first_strategy = preprocess_for_roberta(first_strategy, 'input_text')
clear_secong_strategy = preprocess_for_roberta(second_strategy, 'input_text')


class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_texts, val_texts, train_labels, val_labels = train_test_split(
    clear_first_strategy['input_text'], clear_first_strategy['rule_violation'], test_size=0.2, random_state=RANDOM_STATE
)


train_dataset = JigsawDataset(
    texts=train_texts,
    labels=train_labels,
    tokenizer=tokenizer
)

val_dataset = JigsawDataset(
    texts=val_texts,
    labels=val_labels,
    tokenizer=tokenizer
)


print(f"train_dataset size: {len(train_dataset)}")
print(f"val_dataset size: {len(val_dataset)}")


model = AutoModelForSequenceClassification.from_pretrained(local_dir, num_labels=2, ignore_mismatched_sizes=True)


def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions

    preds = np.argmax(logits, axis=1)

    probs = softmax(logits, axis=1)
    positive_class_probs = probs[:, 1]

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    mcc = matthews_corrcoef(labels, preds)
    roc_auc = roc_auc_score(labels, positive_class_probs)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'mcc': mcc,
        'roc_auc': roc_auc
    }


from transformers import TrainingArguments, EarlyStoppingCallback

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",     
    save_strategy="epoch",
    save_total_limit=1,              
    load_best_model_at_end=True,     
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    report_to="none",
    save_safetensors=False,        
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


print("Eğitim başlıyor...")
trainer.train()

print("\nEğitim tamamlandı. En iyi modelin validation sonuçları:")
eval_results = trainer.evaluate()
print(eval_results)


first_strategy_test = create_strategy(testt_df, ['body', 'rule', 'positive_example_1', 'negative_example_1'], tokenizer)
clear_first_strategy_test = preprocess_for_roberta(first_strategy_test, 'input_text')


get_token_distribution(first_strategy_test, 'input_text', tokenizer, sample_size=5000)


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }



test_dataset = InferenceDataset(
    texts=clear_first_strategy_test['input_text'], 
    tokenizer=tokenizer,
    max_len=128
)


prediction_output = trainer.predict(test_dataset)

logits = prediction_output.predictions

probs = softmax(logits, axis=1)

positive_class_probs = probs[:, 1]

sample_submission_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

sample_submission_df['rule_violation'] = positive_class_probs

sample_submission_df.to_csv("submission.csv", index=False)

print("submission.csv dosyası başarıyla oluşturuldu!")
print(sample_submission_df.head())





