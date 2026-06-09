# Install the necessary libraries
!pip uninstall -y datasets pyarrow && pip install --no-cache-dir -U datasets transformers


import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import gc
import os


import re

# A dictionary of common English contractions to expand
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def advanced_clean_text(text):
    # 1. Lowercase the text
    text = text.lower()
    
    # 2. Expand contractions
    for word, new_word in CONTRACTIONS.items():
        text = text.replace(word, new_word)
        
    # 3. Remove URLs (your original step)
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    
    # 4. Remove HTML tags (your original step)
    text = re.sub(r'<.*?>', '', text)
    
    # 5. Add space around punctuation to separate them as tokens
    text = re.sub(r'([,.!?"\\\'()])', r' \\1 ', text)
    
    # 6. Remove non-alphanumeric characters (keeps letters, numbers, and core punctuation)
    text = re.sub(r'[^a-zA-Z0-9\\s.,!?"\\\']', '', text)
    
    # 7. Normalize repeated characters (e.g., "soooo goood" -> "soo good")
    text = re.sub(r'(.)\\1{2,}', r'\\1\\1', text)
    
    # 8. Remove extra whitespace (your original step)
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

# --- 1. Load Data ---
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')

# APPLY THE CLEANING FUNCTION
print("Cleaning text data with advanced rules...")
train_df['text'] = train_df['text'].apply(advanced_clean_text)
test_df['text'] = test_df['text'].apply(advanced_clean_text)

print("Cleaning complete.")
display(train_df.head())

# Map labels to integers
label_map = {'benign': 0, 'jailbreak': 1}
train_df['label'] = train_df['label'].map(label_map)
# --- 2. Configuration ---
MODEL_NAME = 'microsoft/deberta-v3-large'

# --- 3. Tokenization ---
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the number of tokens for each text entry
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
token_lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in train_df['text']]

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

test_dataset = Dataset.from_pandas(test_df)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Plot the distribution
print("Analyzing token length distribution...")
sns.histplot(token_lengths)
plt.title('Distribution of Text Lengths in Tokens')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.show()

# Print descriptive statistics
print("\\nStatistics for token lengths:")
print(pd.Series(token_lengths).describe())
"""
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

test_dataset = Dataset.from_pandas(test_df)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
"""


from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = FocalLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


from torch.optim import AdamW
from transformers import get_scheduler

# --- 4. Cross-Validation Training Loop with Custom Optimizers ---
N_SPLITS = 3 # Number of folds

oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
    print(f"\n===== Starting Fold {fold+1}/{N_SPLITS} =====")
    
    train_fold_df = train_df.iloc[train_idx]
    val_fold_df = train_df.iloc[val_idx]
    
    train_fold_dataset = Dataset.from_pandas(train_fold_df)
    val_fold_dataset = Dataset.from_pandas(val_fold_df)
    
    tokenized_train = train_fold_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_fold_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold+1}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        logging_dir=f'./logs_fold_{fold+1}',
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True,
        eval_strategy="epoch", 
        logging_strategy="epoch",
    )
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    num_training_steps = (len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
    lr_scheduler = get_scheduler(
        name="cosine_with_restarts", 
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        optimizers=(optimizer, lr_scheduler)
    )
    
    trainer.train()
    
    print(f"Predicting on test data for fold {fold+1}...")
    fold_preds = trainer.predict(tokenized_test)
    logits = fold_preds.predictions
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    test_predictions += probabilities / N_SPLITS
    
    # Clean up to free up disk space
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    os.system(f'rm -rf ./results_fold_{fold+1}')
    os.system(f'rm -rf ./logs_fold_{fold+1}')

print("\n✅ All folds complete! Creating submission file...")


# --- 5. Create Submission File ---
submission_df = pd.DataFrame({'Id': test_df['Id'], 'TARGET': test_predictions})
submission_df.to_csv('submission.csv', index=False)
display(submission_df.head())

