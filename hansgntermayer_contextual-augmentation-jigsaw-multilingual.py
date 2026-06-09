!pip install -q transformers[torch] datasets nlpaug
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import nlpaug.augmenter.word as naw

# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
train_df = train_df[['comment_text', 'toxic']].dropna()

# Class distribution analysis
toxic_df = train_df[train_df['toxic'] == 1]
non_toxic_df = train_df[train_df['toxic'] == 0]

print(f"Original counts - Toxic: {len(toxic_df)}, Non-toxic: {len(non_toxic_df)}")


# Original counts
toxic_count = len(toxic_df)
non_toxic_count = len(non_toxic_df)
required_toxic = non_toxic_count - toxic_count

aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute",
    aug_max=10,
    aug_p=0.6,
    batch_size=256,
    device='cuda'
)

# Batch processing function
def batch_augment(texts, aug, num_variants):
    """Generate multiple variants for a batch of texts"""
    return [aug.augment(text) for text in texts for _ in range(num_variants)]

# Calculate needed variants per sample
remaining = required_toxic
augmented_toxic = []
batch_size = 512

for i in range(0, len(toxic_df), batch_size):
    batch_texts = toxic_df['comment_text'].iloc[i:i+batch_size].tolist()
    
    # Calculate variants needed from this batch
    variants_needed = min(remaining // (len(toxic_df) // batch_size), 8)
    variants_needed = max(variants_needed, 1)
    
    # Augment batch
    try:
        augmented_batch = batch_augment(batch_texts, aug, variants_needed)
        augmented_toxic.extend(augmented_batch)
        remaining -= len(augmented_batch)
        
        print(f"Generated {len(augmented_batch)} samples | Remaining: {remaining}")
        
        if remaining <= 0:
            break
            
    except Exception as e:
        print(f"Error in batch {i}: {str(e)}")
        continue

# Final balanced dataset
balanced_toxic = pd.DataFrame({
    'comment_text': toxic_df['comment_text'].tolist() + augmented_toxic[:required_toxic],
    'toxic': 1
})

balanced_df = pd.concat([balanced_toxic, non_toxic_df]).sample(frac=1, random_state=42)
balanced_df['comment_text'] = balanced_df['comment_text'].astype(str)


balanced_df.to_csv('/kaggle/working/balanced_dataset.csv', index=False)


'''
train_df = balanced_df

val_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
val_df = val_df[['comment_text', 'toxic']]

test_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
test_df_labels = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test_labels.csv')

test_df['toxic'] = test_df_labels['toxic']
test_df['comment_text'] = test_df['content']
test_df = test_df[['comment_text', 'toxic']]

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
'''


'''
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

# Initialize tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Tokenization function
def tokenize_function(batch):
    tokenized = tokenizer(
        batch['comment_text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

    tokenized['labels'] = batch['toxic']
    return tokenized

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[['comment_text', 'toxic']])
val_dataset = Dataset.from_pandas(val_df[['comment_text', 'toxic']])
test_dataset = Dataset.from_pandas(test_df[['comment_text', 'toxic']])

train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1024)
val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=1024)
test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=1024)

# Metric computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    roc_auc = roc_auc_score(labels, probs)
    accuracy = accuracy_score(labels, preds)
    
    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy
    }

# Model configuration
model = XLMRobertaForSequenceClassification.from_pretrained(
    'xlm-roberta-base',
    num_labels=2
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy='no',
    load_best_model_at_end=False,
    metric_for_best_model='roc_auc',
    greater_is_better=True,
    fp16=True,
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()
'''


'''
# Evaluate on test dataset
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test ROC-AUC: {test_results['eval_roc_auc']:.4f}")
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
'''

