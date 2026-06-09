import pandas as pd

train_df = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
train_df['text']


!pip install --upgrade transformers huggingface_hub



import nltk
nltk.download('punkt_tab')


import nltk
nltk.download('all')


train_df['label'].value_counts()


import re

import string

def clean_text(text):
    """
    LÃ m sáº¡ch text cho phÃ¢n loáº¡i jailbreak/benign

    Args:
        text: Chuá»—i text cáº§n lÃ m sáº¡ch

    Returns:
        Chuá»—i text Ä‘Ã£ Ä‘Æ°á»£c lÃ m sáº¡ch
    """
    if pd.isna(text) or text.strip() == '':
        return ''

    # 1. Chuyá»ƒn vá»� lowercase
    text = text.lower()



    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    text = re.sub(r'\s+', ' ', text).strip()

    return text


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
# Removed nltk.download('punkt_tab')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    """
    Lemmatize text.

    Args:
        text: Input text.

    Returns:
        Lemmatized text.
    """
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


contractions_dict = {
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not"
}

def expand_contractions(text, contractions_dict=contractions_dict):
    """
    Expand contractions in text.

    Args:
        text: Input text.
        contractions_dict: Dictionary of contractions and their expansions.

    Returns:
        Text with contractions expanded.
    """
    words = text.split()
    new_words = [contractions_dict.get(word, word) for word in words]
    return ' '.join(new_words)


# Separate benign and jailbreak samples
benign_df = train_df[train_df['label'] == 'benign'].copy()
jailbreak_df = train_df[train_df['label'] == 'jailbreak'].copy()

# Calculate the number of samples needed for augmentation
num_benign = len(benign_df)
num_jailbreak = len(jailbreak_df)
num_to_augment = num_benign - num_jailbreak

if num_to_augment > 0:
    print(f"Augmenting {num_to_augment} jailbreak samples to match benign count.")
    # Randomly sample jailbreak samples to augment
    jailbreak_samples_to_augment = jailbreak_df.sample(n=num_to_augment, replace=True, random_state=42)

    # Apply augmentation (expand contractions and lemmatize) to the text before cleaning
    # This is because the original cleaning might have removed characters needed for contractions or lemmatization
    jailbreak_samples_to_augment['augmented_text'] = jailbreak_samples_to_augment['text'].apply(expand_contractions).apply(lemmatize_text)

    # Apply the clean_text function to the newly augmented text
    jailbreak_samples_to_augment['cleaned_text'] = jailbreak_samples_to_augment['augmented_text'].apply(clean_text)

    # Combine original, augmented jailbreak, and benign data
    augmented_train_df = pd.concat([train_df, jailbreak_samples_to_augment.drop(columns=['augmented_text'])], ignore_index=True)
else:
    print("Jailbreak samples are already more than or equal to benign samples. No augmentation needed for balancing.")
    augmented_train_df = train_df

print("Original train set size:", len(train_df))
print("Augmented train set size:", len(augmented_train_df))
print("Label distribution in augmented data:\n", augmented_train_df['label'].value_counts())

train_df = augmented_train_df # Update train_df to use the augmented data


# Removed redundant cleaning step
train_df['cleaned_text'] = train_df['text'].apply(clean_text)

# Mapping label benign to 0 and jailbreak to 1
train_df['label'] = train_df['label'].replace({'benign': 0, 'jailbreak': 1})


# Removed redundant label mapping
# train_df['label'] = train_df['label'].replace({'benign': 0, 'jailbreak': 1})

print(train_df['label'].value_counts())


train_df['label'].value_counts()


train_df.to_csv("/kaggle/working/cleaned_train.csv", index=False)
df = pd.read_csv("/kaggle/working/cleaned_train.csv")


import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score



from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base") # Changed model name

def tokenize_function(examples):
    # Ensure each item in the batch is a string
    cleaned_texts = [text if isinstance(text, str) else '' for text in examples['cleaned_text']]
    return tokenizer(cleaned_texts, padding='max_length', truncation=True, max_length=512)


def compute_metrics(eval_pred):
    from scipy.special import softmax

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    probs = softmax(logits, axis=1)[:, 1]

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'roc_auc': roc_auc_score(labels, probs)
    }


from sklearn.model_selection import StratifiedKFold
import numpy as np
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding # Import necessary classes
from datasets import Dataset # Import Dataset
import torch # Import torch

test_df = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros((len(test_df), 5))
# ==========================
# 5ï¸�âƒ£ Train tá»«ng fold
# ==========================
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
    print(f"\n===== ğŸ§© Fold {fold+1} / 5 =====")

    # Ensure label column is int64 before creating Dataset
    train_df['label'] = train_df['label'].astype('int64')

    # Ensure cleaned_text column contains only strings and handle potential NaNs
    train_df['cleaned_text'] = train_df['cleaned_text'].apply(lambda x: x if isinstance(x, str) else '')

    # Táº¡o dataset fold - Exclude 'label' column initially
    train_dataset = Dataset.from_pandas(train_df.iloc[train_idx][['cleaned_text']])
    val_dataset = Dataset.from_pandas(train_df.iloc[val_idx][['cleaned_text']])

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['cleaned_text'])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=['cleaned_text'])

    # Explicitly handle labels as NumPy arrays
    train_labels = train_df.iloc[train_idx]['label'].values.astype(np.int64) # Convert to numpy array
    val_labels = train_df.iloc[val_idx]['label'].values.astype(np.int64) # Convert to numpy array

    # Add labels back to the tokenized datasets
    tokenized_train = tokenized_train.add_column("labels", train_labels)
    tokenized_val = tokenized_val.add_column("labels", val_labels)


    tokenized_train.set_format('torch')
    tokenized_val.set_format('torch')


    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=2,
        id2label={0: 'benign', 1: 'jailbreak'},
        label2id={'benign': 0, 'jailbreak': 1}
    )

    # Define Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # TrainingArguments riÃªng cho tá»«ng fold
    training_args = TrainingArguments(
        output_dir=f'./results/fold_{fold}',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer, # Keep tokenizer here as it's used by DataCollator
        data_collator=data_collator, # Explicitly provide data collator
        compute_metrics=compute_metrics
    )

    # Huáº¥n luyá»‡n
    trainer.train()

    # Dá»± Ä‘oÃ¡n trÃªn táº­p validation (OOF)
    val_logits = trainer.predict(tokenized_val).predictions
    val_probs = torch.softmax(torch.tensor(val_logits), dim=1)[:, 1].numpy()
    oof_preds[val_idx] = val_probs

    # Dá»± Ä‘oÃ¡n trÃªn test
    test_df['cleaned_text'] = test_df['text'].apply(clean_text)
    test_dataset = Dataset.from_pandas(test_df[['Id', 'cleaned_text']])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=['cleaned_text'])

    tokenized_test.set_format('torch')
    preds = trainer.predict(tokenized_test).predictions
    test_probs = torch.softmax(torch.tensor(preds), dim=1)[:, 1].numpy()
    test_preds[:, fold] = test_probs


final_test_pred = test_preds.mean(axis=1)



submission = pd.DataFrame({
    "Id": test_df["Id"],
    "TARGET": final_test_pred
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… File submission.csv Ä‘Ã£ Ä‘Æ°á»£c táº¡o vÃ  lÆ°u thÃ nh cÃ´ng!")


